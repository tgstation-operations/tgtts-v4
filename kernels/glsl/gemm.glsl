// The products the local stacks' kernels share, over the few tokens a stack runs at once: a
// row-major f16 weight `w` of K columns against M tokens staged in `xs` a chunk of KC columns
// at a time, both eight to a uvec4. A cluster of CLUSTER lanes takes ROWS rows, each lane
// reading 16 contiguous bytes of each row per step, and sums every row against every token in
// f32: each weight is read once for all the tokens.
//
// The including kernel declares K, KC, M, ROWS and BLOCKS and the `w` buffer, then includes
// this, defines `stage`, `rows_of` and `epilogue`, and calls `run`.
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

// Adds the products of this lane's part of a chunk of ROWS rows, `ws`, with every token's
// part of it in xs to acc, `[ROWS][M]`.
void accumulate(uvec4 ws[LOADS], inout float acc[SUMS]) {
    uint lane = cluster_lane();
    [[unroll]] for (uint s = 0; s < STEPS; s++) {
        vec4 wlo[ROWS], whi[ROWS];
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            unpack8(ws[r * STEPS + s], wlo[r], whi[r]);
        }
        [[unroll]] for (uint m = 0; m < M; m++) {
            vec4 lo, hi;
            unpack8(xs[m * (KC / 8) + s * CLUSTER + lane], lo, hi);
            [[unroll]] for (uint r = 0; r < ROWS; r++) {
                float a = acc[r * M + m];
                [[unroll]] for (uint e = 0; e < 4; e++) {
                    a = fma(wlo[r][e], lo[e], a);
                }
                [[unroll]] for (uint e = 0; e < 4; e++) {
                    a = fma(whi[r][e], hi[e], a);
                }
                acc[r * M + m] = a;
            }
        }
    }
}

// The rows of w that this cluster takes in block `block`: the including kernel's.
void rows_of(uint block, out uint rows[ROWS]);

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

// The workgroup's BLOCKS blocks of rows, every NumWorkGroups-th from its own, each times every
// token, chunk by chunk: each chunk of the rows is loaded while the one before is summed, so
// that the weights stream while the workgroup computes. The tokens are staged once if they
// fit in one chunk, and chunk by chunk for every block if not.
void run() {
    uint rows[ROWS];
    rows_of(gl_WorkGroupID.x, rows);
    uvec4 next[LOADS];
    load_chunk(rows, 0, next);
    for (uint i = 0; i < BLOCKS; i++) {
        uint block = gl_WorkGroupID.x + i * gl_NumWorkGroups.x;
        uint these[ROWS];
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            these[r] = rows[r];
        }
        float acc[SUMS];
        [[unroll]] for (uint j = 0; j < SUMS; j++) {
            acc[j] = 0.0;
        }
        for (uint chunk = 0; chunk < CHUNKS; chunk++) {
            uvec4 ws[LOADS];
            [[unroll]] for (uint j = 0; j < LOADS; j++) {
                ws[j] = next[j];
            }
            if (chunk + 1 < CHUNKS) {
                load_chunk(rows, chunk + 1, next);
            } else if (i + 1 < BLOCKS) {
                rows_of(block + gl_NumWorkGroups.x, rows);
                load_chunk(rows, 0, next);
            }
            if (CHUNKS > 1 || i == 0) {
                barrier();
                stage(chunk);
                barrier();
            }
            accumulate(ws, acc);
        }
        [[unroll]] for (uint j = 0; j < SUMS; j++) {
            acc[j] = subgroupClusteredAdd(acc[j], CLUSTER);
        }
        epilogue(these, acc);
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
