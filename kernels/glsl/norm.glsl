// RMSNorm as upstream's MiniCPMRMSNorm computes it: the mean square of x in f32, and x times
// its reciprocal root times the gain g, rounded to f16 once. x holds vectors of K, each normed
// on its own.
//
// The including kernel declares K and the `x` and `g` buffers as uvec4 views, eight f16 to
// each, then includes this.

shared float norm_partial[SUBGROUPS];

// The reciprocal root of the mean square of vector b of x plus eps, in every invocation of the
// workgroup.
float rms_scale(uint b, float eps) {
    float squares = 0.0;
    for (uint v = gl_LocalInvocationID.x; v < K / 8; v += WORKGROUP) {
        vec4 lo, hi;
        unpack8(x[b * (K / 8) + v], lo, hi);
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

// rms_scale(b, eps) as a workgroup `width` wide sums it, by one subgroup alone taking the
// subgroups of that workgroup in turn, in every invocation of the subgroup.
float subgroup_rms_scale(uint b, float eps, uint width) {
    float total = 0.0;
    for (uint first = 0; first < width; first += SUBGROUP) {
        float squares = 0.0;
        for (uint v = first + gl_SubgroupInvocationID; v < K / 8; v += width) {
            vec4 lo, hi;
            unpack8(x[b * (K / 8) + v], lo, hi);
            squares += dot(lo, lo) + dot(hi, hi);
        }
        total += subgroupAdd(squares);
    }
    return 1.0 / sqrt(total / float(K) + eps);
}

// Eight normed values from uvec4 v of vector b of x, packed.
uvec4 normed8(uint b, uint v, float scale) {
    vec4 lo, hi, glo, ghi;
    unpack8(x[b * (K / 8) + v], lo, hi);
    unpack8(g[v], glo, ghi);
    return pack8(lo * scale * glo, hi * scale * ghi);
}
