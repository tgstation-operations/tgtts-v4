// RMSNorm as upstream's MiniCPMRMSNorm computes it: the mean square of x in f32, and x times
// its reciprocal root times the gain g, rounded to f16 once.
//
// The including kernel declares the `x` and `g` buffers and K, then includes this.

shared float norm_partial[SUBGROUPS];

// The reciprocal root of the mean square of x plus eps, in every invocation of the workgroup.
float rms_scale(float eps) {
    float squares = 0.0;
    for (uint i = gl_LocalInvocationID.x; i < K; i += WORKGROUP) {
        float v = float(x[i]);
        squares += v * v;
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

float16_t normed(uint i, float scale) {
    return float16_t(float(x[i]) * scale * float(g[i]));
}
