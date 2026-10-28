// gemv.glsl's vectors staged through RMSNorm, as norm.glsl normalizes them. Each vector's mean
// square is summed in the order a workgroup of NORM_WIDTH sums it, batch 1's, so that a vector
// comes out as it does alone however wide the workgroup is: by the workgroup together when it is
// that wide and takes one vector, and otherwise by each subgroup alone for the vectors it stages.
//
// The including kernel declares NORM_WIDTH, includes gemv.glsl and norm.glsl, then this, and
// calls norm_scales before dot_rows.

const bool COOPERATIVE = BATCH == 1 && WORKGROUP == NORM_WIDTH;

// The reciprocal root mean square, plus eps, of each vector the subgroup stages.
float scales[MINE];

void norm_scales(float eps) {
    if (COOPERATIVE) {
        scales[0] = rms_scale(first_vector(), eps);
        return;
    }
    [[unroll]] for (uint j = 0; j < MINE; j++) {
        uint b = BATCH == 1 ? 0 : gl_SubgroupID + j * SUBGROUPS;
        if (b < BATCH) {
            scales[j] = subgroup_rms_scale(first_vector() + b, eps, NORM_WIDTH);
        }
    }
}

uvec4 x_at(uint j, uint b, uint v) {
    return normed8(b, v, scales[j]);
}
