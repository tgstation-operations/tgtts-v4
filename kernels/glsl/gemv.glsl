// The matrix-vector product shared by the GEMV kernels: a row-major f16 weight `w` of K
// columns against the f16 vector in `xs`, both eight to a uvec4. A subgroup takes ROWS rows at
// once, each lane reading 16 contiguous bytes of each row per step, and sums in f32.
//
// The including kernel declares the `w` buffer, K and ROWS, then includes this.

shared uvec4 xs[K / 8];

const uint STEPS = K / 8 / SUBGROUP;

// Sums of the rows `rows` of w with xs, each in every lane of the subgroup.
void dot_rows(uint rows[ROWS], out float sums[ROWS]) {
    float acc[ROWS];
    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        acc[r] = 0.0;
    }
    uint lane = gl_SubgroupInvocationID;
    [[unroll]] for (uint s = 0; s < STEPS; s++) {
        uint i = s * SUBGROUP + lane;
        vec4 lo, hi;
        unpack8(xs[i], lo, hi);
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            vec4 wlo, whi;
            unpack8(w[rows[r] * (K / 8) + i], wlo, whi);
            acc[r] += dot(wlo, lo) + dot(whi, hi);
        }
    }
    [[unroll]] for (uint r = 0; r < ROWS; r++) {
        sums[r] = subgroupAdd(acc[r]);
    }
}

// xs = x, a uvec4 view of the input vector.
void load_plain() {
    for (uint v = gl_LocalInvocationID.x; v < K / 8; v += WORKGROUP) {
        xs[v] = x[v];
    }
    barrier();
}
