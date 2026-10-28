// gemv.glsl's vectors staged through RMSNorm, as norm.glsl normalizes them. Each vector's mean
// square is summed in the order a workgroup of NORM_WIDTH sums it, batch 1's, so that a vector
// comes out as it does alone however wide the workgroup is.
//
// The including kernel declares NORM_WIDTH, includes gemv.glsl and norm.glsl, then this, and
// calls norm_scales before dot_rows.

const bool COOPERATIVE = BATCH == 1 && WORKGROUP == NORM_WIDTH;

// The one vector's scale, when the workgroup sums it together; each vector's otherwise.
float scale;
shared float scales[BATCH];

// Every vector's reciprocal root mean square, plus eps.
void norm_scales(float eps) {
    if (COOPERATIVE) {
        scale = rms_scale(0, eps);
        return;
    }
    for (uint b = gl_SubgroupID; b < BATCH; b += SUBGROUPS) {
        float s = subgroup_rms_scale(b, eps, NORM_WIDTH);
        if (gl_SubgroupInvocationID == 0) {
            scales[b] = s;
        }
    }
    barrier();
}

uvec4 x_at(uint b, uint v) {
    return normed8(b, v, COOPERATIVE ? scale : scales[b]);
}
