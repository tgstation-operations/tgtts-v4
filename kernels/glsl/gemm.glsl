// The products the local stacks' kernels share, over the few tokens a stack runs at once: a
// row-major f16 weight `w` of K columns against M tokens staged in `xs` a chunk of KC columns
// at a time, both eight to a uvec4. A cluster of CLUSTER lanes takes ROWS rows, each lane
// reading 16 contiguous bytes of each row per step, and sums every row against every token in
// f32: each weight is read once for all the tokens.
//
// The including kernel declares K, KC, M, ROWS, the SLOTS of ROWS rows there are and the `w`
// buffer, then includes this, defines `stage`, `rows_of` and `epilogue`, and calls `run`.
#extension GL_KHR_shader_subgroup_clustered : require

// A literal, as glslang takes a cluster size; it divides every subgroup size there is.
const uint CLUSTER = 16;
const uint CLUSTERS = WORKGROUP / CLUSTER;
const uint STEPS = KC / 8 / CLUSTER;
const uint CHUNKS = K / KC;
// The tokens each lane of a cluster is left holding by `mine`.
const uint PER_LANE = (M + CLUSTER - 1) / CLUSTER;
// The sizes of a lane's part of a chunk of ROWS rows and of its sums, named once: glslang
// tells apart arrays sized by two spellings of the same expression.
const uint LOADS = ROWS * STEPS;
const uint SUMS = ROWS * M;
// The uvec4 of a chunk of the tokens each invocation stages.
const uint STAGED = (M * (KC / 8) + WORKGROUP - 1) / WORKGROUP;

shared uvec4 xs[M * (KC / 8)];

// Writes chunk `chunk` of every token into xs, `[M][KC / 8]`.
void stage(uint chunk);

// Where in the tokens, `[M][K / 8]`, the uvec4 `v` of xs comes from in chunk `chunk`.
uint staged_at(uint v, uint chunk) {
    return v / (KC / 8) * (K / 8) + chunk * (KC / 8) + v % (KC / 8);
}

uint cluster_lane() {
    return gl_SubgroupInvocationID % CLUSTER;
}

// This invocation's cluster among the workgroup's.
uint cluster_id() {
    return gl_SubgroupID * (SUBGROUP / CLUSTER) + gl_SubgroupInvocationID / CLUSTER;
}

// The tokens whose products `accumulate` interleaves, so that consecutive FMAs add to
// different sums rather than wait on each other.
const uint GROUP = 4;

// Adds the products of this lane's part of a chunk of ROWS rows, `ws`, with every token's
// part of it in xs to acc, `[ROWS][M]`.
void accumulate(uvec4 ws[LOADS], inout float acc[SUMS]) {
    uint lane = cluster_lane();
    [[unroll]] for (uint s = 0; s < STEPS; s++) {
        vec4 wlo[ROWS], whi[ROWS];
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            unpack8(ws[r * STEPS + s], wlo[r], whi[r]);
        }
        [[unroll]] for (uint first = 0; first < M; first += GROUP) {
            vec4 lo[GROUP], hi[GROUP];
            [[unroll]] for (uint g = 0; g < GROUP; g++) {
                if (first + g < M) {
                    unpack8(xs[(first + g) * (KC / 8) + s * CLUSTER + lane], lo[g], hi[g]);
                }
            }
            [[unroll]] for (uint e = 0; e < 8; e++) {
                [[unroll]] for (uint g = 0; g < GROUP; g++) {
                    [[unroll]] for (uint r = 0; r < ROWS; r++) {
                        uint i = r * M + first + g;
                        if (first + g < M) {
                            float x = e < 4 ? lo[g][e % 4] : hi[g][e % 4];
                            float v = e < 4 ? wlo[r][e % 4] : whi[r][e % 4];
                            acc[i] = fma(v, x, acc[i]);
                        }
                    }
                }
            }
        }
    }
}

// The rows of w in set `slot` of the SLOTS sets of ROWS rows: the including kernel's.
void rows_of(uint slot, out uint rows[ROWS]);

// What the including kernel makes of the sums of rows `rows` with every token, `[ROWS][M]`,
// in every lane of the cluster.
void epilogue(uint rows[ROWS], float acc[SUMS]);

void load_chunk(uint rows[ROWS], uint chunk, out uvec4 ws[LOADS]) {
    uint lane = cluster_lane();
    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        [[unroll]] for (uint s = 0; s < STEPS; s++) {
            ws[r * STEPS + s] = w[rows[r] * (K / 8) + chunk * (KC / 8) + s * CLUSTER + lane];
        }
    }
}

// The rows of `slot` of the SLOTS sets of ROWS rows times every token, chunk by chunk: each
// chunk of the rows is loaded while the one before is summed, the first before the tokens are
// staged, so that the weights stream while the workgroup computes.
void sum_rows(uint slot, uint summed, inout float acc[SUMS]) {
    uint rows[ROWS];
    rows_of(slot, rows);
    [[unroll]] for (uint j = 0; j < SUMS; j++) {
        acc[j] = 0.0;
    }
    uvec4 next[LOADS];
    load_chunk(rows, 0, next);
    for (uint c = 0; c < summed; c++) {
        uvec4 ws[LOADS];
        [[unroll]] for (uint j = 0; j < LOADS; j++) {
            ws[j] = next[j];
        }
        if (c + 1 < summed) {
            load_chunk(rows, c + 1, next);
        }
        if (summed > 1) {
            barrier();
            stage(c);
            barrier();
        }
        accumulate(ws, acc);
    }
    [[unroll]] for (uint j = 0; j < SUMS; j++) {
        acc[j] = subgroupClusteredAdd(acc[j], CLUSTER);
    }
    epilogue(rows, acc);
}

// Every set of rows the workgroup takes. A workgroup that sums one chunk stages the tokens once,
// and each of its subgroups then takes every NumWorkGroups-th of the subgroups' worth of slots
// from its own on, apart from the others, so that the work is shared out a subgroup's worth at
// a time. One that sums several restages them chunk by chunk, all its clusters together, for
// every NumWorkGroups-th of the workgroups' worth of slots from its own on.
//
// A slot's first chunk is not loaded while the slot before is summed: the registers that would
// take are what lets the compiler issue the tokens' loads from shared memory early.
void run() {
    float acc[SUMS];
    if (CHUNKS == 1) {
        barrier();
        stage(0);
        barrier();
        const uint PER_SUBGROUP = SUBGROUP / CLUSTER;
        uint cluster = gl_SubgroupInvocationID / CLUSTER;
        uint stride = gl_NumWorkGroups.x * SUBGROUPS;
        uint units = SLOTS / PER_SUBGROUP;
        for (uint unit = gl_WorkGroupID.x * SUBGROUPS + gl_SubgroupID; unit < units;
             unit += stride) {
            sum_rows(unit * PER_SUBGROUP + cluster, 1, acc);
        }
    } else {
        for (uint block = gl_WorkGroupID.x; block < SLOTS / CLUSTERS;
             block += gl_NumWorkGroups.x) {
            sum_rows(block * CLUSTERS + cluster_id(), CHUNKS, acc);
        }
    }
}

// Row r's sums for this lane's tokens, lane + CLUSTER t for t < PER_LANE: past M for some
// lanes when CLUSTER does not divide M.
void mine(float acc[SUMS], uint r, out float sums[PER_LANE]) {
    uint lane = cluster_lane();
    [[unroll]] for (uint t = 0; t < PER_LANE; t++) {
        sums[t] = 0.0;
        [[unroll]] for (uint j = 0; j < CLUSTER; j++) {
            if (t * CLUSTER + j < M && lane == j) {
                sums[t] = acc[r * M + t * CLUSTER + j];
            }
        }
    }
}
