// The matrix-vector products shared by the GEMV kernels: a row-major f16 weight `w` of K
// columns against BATCH f16 vectors, both eight to a uvec4. A subgroup takes ROWS rows at
// once, each lane reading 16 contiguous bytes of each row per step, and sums each row with
// every vector in f32, so that a weight is read once for all of them. A vector's sums are taken
// in the same order whatever BATCH is: it comes out as it does alone.
//
// A batch of BATCH * GROUPS vectors is shared out over GROUPS workgroups side by side, which take
// the same rows of w, BATCH vectors each: the rows a workgroup reads after its neighbour are
// still in cache. A workgroup stages its vectors in `xs` COLS columns at a time: all K of them
// unless SPAN says otherwise, which the host sets to a divisor of K when BATCH vectors of K do
// not fit. One vector is staged by the whole workgroup, and more each by a subgroup of its own.
//
// A lane may load its first AHEAD steps of its rows before the kernel's prologue (`prefetch`),
// so that they are in flight while the prologue runs rather than issued after its barrier; the
// steps of a later chunk are loaded before the chunk is staged. When a weight is loaded does not
// change what is summed, or in what order.
//
// The including kernel declares the `w` buffer, K, ROWS, AHEAD, BATCH, GROUPS and SPAN, then
// includes this and defines `x_at`.

const uint COLS = SPAN == 0 ? K : SPAN;
const uint CHUNKS = K / COLS;
const uint STEPS = K / 8 / SUBGROUP;
const uint CHUNK_STEPS = COLS / 8 / SUBGROUP;
const uint EARLY = AHEAD < STEPS ? AHEAD : STEPS;
// Sizes named once: glslang tells apart arrays sized by two spellings of the same expression.
const uint STAGED = BATCH * (COLS / 8);
const uint SUMS = BATCH * ROWS;
// The vectors a subgroup stages, when there are more than one.
const uint MINE = (BATCH + SUBGROUPS - 1) / SUBGROUPS;

shared uvec4 xs[STAGED];

// One entry at least, as an array must have.
uvec4 early[EARLY > 0 ? EARLY : 1][ROWS];

// Loads the first EARLY steps of the rows `rows` of w into `early`.
void prefetch(uint rows[ROWS]) {
    uint lane = gl_SubgroupInvocationID;
    [[unroll]] for (uint s = 0; s < EARLY; s++) {
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            early[s][r] = w[rows[r] * (K / 8) + s * SUBGROUP + lane];
        }
    }
}

// The block of rows of w the workgroup takes, and the first of the batch's vectors it takes.
uint row_block() {
    return gl_WorkGroupID.x / GROUPS;
}

uint first_vector() {
    return gl_WorkGroupID.x % GROUPS * BATCH;
}

// The uvec4 `v` of the batch's vector `b` as xs takes it, `v` counted along the whole of K: the
// subgroup's vector `j` of those it stages, or 0 when the workgroup stages one together.
uvec4 x_at(uint j, uint b, uint v);

// Writes chunk `chunk` of the workgroup's vectors into xs, `[BATCH][COLS / 8]`.
void stage(uint chunk) {
    if (BATCH == 1) {
        for (uint v = gl_LocalInvocationID.x; v < COLS / 8; v += WORKGROUP) {
            xs[v] = x_at(0, first_vector(), chunk * (COLS / 8) + v);
        }
        return;
    }
    [[unroll]] for (uint j = 0; j < MINE; j++) {
        uint b = gl_SubgroupID + j * SUBGROUPS;
        if (b >= BATCH) {
            continue;
        }
        for (uint v = gl_SubgroupInvocationID; v < COLS / 8; v += SUBGROUP) {
            xs[b * (COLS / 8) + v] = x_at(j, first_vector() + b, chunk * (COLS / 8) + v);
        }
    }
}

// Sums of the rows `rows` of w with each of the workgroup's vectors, each in every lane of the
// subgroup: `sums[b * ROWS + r]` is row r's with its vector b. Stages the vectors, so every
// invocation of the workgroup calls it.
void dot_rows(uint rows[ROWS], out float sums[SUMS]) {
    float acc[SUMS];
    [[unroll]] for (uint j = 0; j < SUMS; j++) {
        acc[j] = 0.0;
    }
    uint lane = gl_SubgroupInvocationID;
    [[unroll]] for (uint c = 0; c < CHUNKS; c++) {
        uvec4 ws[CHUNK_STEPS][ROWS];
        [[unroll]] for (uint t = 0; t < CHUNK_STEPS; t++) {
            uint s = c * CHUNK_STEPS + t;
            [[unroll]] for (uint r = 0; r < ROWS; r++) {
                if (s < EARLY) {
                    ws[t][r] = early[s][r];
                } else if (c > 0) {
                    ws[t][r] = w[rows[r] * (K / 8) + s * SUBGROUP + lane];
                }
            }
        }
        if (c > 0) {
            barrier();
        }
        stage(c);
        barrier();
        [[unroll]] for (uint t = 0; t < CHUNK_STEPS; t++) {
            uint s = c * CHUNK_STEPS + t;
            vec4 wlo[ROWS], whi[ROWS];
            [[unroll]] for (uint r = 0; r < ROWS; r++) {
                uvec4 q = ws[t][r];
                if (c == 0 && s >= EARLY) {
                    q = w[rows[r] * (K / 8) + s * SUBGROUP + lane];
                }
                unpack8(q, wlo[r], whi[r]);
            }
            [[unroll]] for (uint b = 0; b < BATCH; b++) {
                vec4 lo, hi;
                unpack8(xs[b * (COLS / 8) + t * SUBGROUP + lane], lo, hi);
                [[unroll]] for (uint r = 0; r < ROWS; r++) {
                    acc[b * ROWS + r] += dot(wlo[r], lo) + dot(whi[r], hi);
                }
            }
        }
    }
    [[unroll]] for (uint j = 0; j < SUMS; j++) {
        sums[j] = subgroupAdd(acc[j]);
    }
}
