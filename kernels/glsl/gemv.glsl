// The matrix-vector product shared by the GEMV kernels: a row-major f16 weight `w` of K
// columns, eight to a uvec4, against the f16 vector in `xs`. A subgroup takes ROWS rows at
// once, each lane reading 16 contiguous bytes of each row per step, and sums in f32.
//
// The including kernel declares the `w` buffer, K and ROWS, then includes this.

shared f16vec4 xs[K / 4];

const uint STEPS = K / 8 / SUBGROUP;

void store_x(uint i, float16_t v) {
    xs[i / 4][i % 4] = v;
}

// Sums of the rows `rows` of w with xs, each in every lane of the subgroup.
void dot_rows(uint rows[ROWS], out float sums[ROWS]) {
    float acc[ROWS];
    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        acc[r] = 0.0;
    }
    uint lane = gl_SubgroupInvocationID;
    [[unroll]] for (uint s = 0; s < STEPS; s++) {
        uint i = s * SUBGROUP + lane;
        vec4 lo = vec4(xs[2 * i]);
        vec4 hi = vec4(xs[2 * i + 1]);
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            uvec4 q = w[rows[r] * (K / 8) + i];
            acc[r] += dot(vec4(unpackHalf2x16(q.x), unpackHalf2x16(q.y)), lo)
                + dot(vec4(unpackHalf2x16(q.z), unpackHalf2x16(q.w)), hi);
        }
    }
    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        sums[r] = subgroupAdd(acc[r]);
    }
}

void load_plain() {
    for (uint i = gl_LocalInvocationID.x; i < K; i += WORKGROUP) {
        store_x(i, x[i]);
    }
    barrier();
}
