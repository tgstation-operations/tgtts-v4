// RMSNorm as upstream's MiniCPMRMSNorm computes it: the mean square of x in f32, and x times
// its reciprocal root times the gain g, rounded to f16 once.
//
// The including kernel declares K and the `x` and `g` buffers as uvec4 views, eight f16 to
// each, then includes this.

shared float norm_partial[SUBGROUPS];

// The reciprocal root of the mean square of x plus eps, in every invocation of the workgroup.
float rms_scale(float eps) {
    float squares = 0.0;
    for (uint v = gl_LocalInvocationID.x; v < K / 8; v += WORKGROUP) {
        vec4 lo, hi;
        unpack8(x[v], lo, hi);
        squares += dot(lo, lo) + dot(hi, hi);
    }
    squares = subgroupAdd(squares);
    if (gl_SubgroupInvocationID == 0) {
        norm_partial[gl_SubgroupID] = squares;
    }
    barrier();
    float total = 0.0;
    for (uint s = 0; s < SUBGROUPS; s++) {
        total += norm_partial[s];
    }
    return 1.0 / sqrt(total / float(K) + eps);
}

// Eight normed values from x[v], packed.
uvec4 normed8(uint v, float scale) {
    vec4 lo, hi, glo, ghi;
    unpack8(x[v], lo, hi);
    unpack8(g[v], glo, ghi);
    return pack8(lo * scale * glo, hi * scale * ghi);
}
