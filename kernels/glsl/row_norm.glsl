// RMSNorm over one row, as norm.glsl computes it over one vector. The including kernel
// declares K, the width of a row, which 8 WORKGROUP must divide.

const uint STEPS = K / 8 / WORKGROUP;

shared float row_partial[SUBGROUPS];

// The reciprocal root of the mean square of the row plus eps, in every invocation, from each
// invocation's sum of the squares of its values.
float row_scale(float squares, float eps) {
    squares = subgroupAdd(squares);
    if (gl_SubgroupInvocationID == 0) {
        row_partial[gl_SubgroupID] = squares;
    }
    barrier();
    float total = 0.0;
    for (uint s = 0; s < SUBGROUPS; s++) {
        total += row_partial[s];
    }
    return 1.0 / sqrt(total / float(K) + eps);
}
