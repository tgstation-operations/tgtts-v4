// What the audio VAE decoder's kernels share. The decoder runs over a chunk of latent frames at
// a time, its tensors time-major, a row of channels per step, all of it f32.
//
// A causal convolution reaches back REACH of its inputs, past the chunk's start into the chunks
// before. It keeps them in a ring of REACH rows, the input at step g of the stream in row
// g % REACH, zeros before the stream's start. Those inputs are the convolution's own, after any
// prologue, so a stream's first chunk sees zeros there as a whole sequence's padding does.
//
// The including kernel declares the buffers `chunk` (frames_before, frames), `x` (the chunk's
// inputs), `ring` and `prologue_` (per channel: scale, bias, Snake's alpha and the reciprocal of
// alpha + 1e-9), then includes this.

layout(constant_id = 0) const uint WORKGROUP = 256;
layout(local_size_x_id = 0) in;
// Input rows per latent frame, input channels, and the rows a convolution reaches back.
layout(constant_id = 1) const uint UP = 1;
layout(constant_id = 2) const uint C = 64;
layout(constant_id = 3) const uint REACH = 0;
layout(constant_id = 4) const bool PROLOGUE = false;

// The input rows in this chunk.
uint rows() {
    return frames * UP;
}

// The ring row of the chunk's first input, the rows before it being in the rows before that.
// The host keeps frames_before below a multiple of every REACH. A convolution of REACH 0 has no
// ring, and reads none of it.
uint ring_start() {
    uint reach = max(REACH, 1u);
    return frames_before % reach * UP % reach;
}

// The ring row holding the input `back` rows before the chunk's first, 1 <= back <= REACH.
uint ring_row(uint back) {
    uint reach = max(REACH, 1u);
    return (ring_start() + reach - back) % reach;
}

// sin(x) within a few ulp for |x| to 1e5, unlike the builtin, whose precision Vulkan bounds
// only absolutely: Snake divides its square by alphas as small as 1e-4. x is reduced by
// multiples of pi/2 in three parts, then sin or cos is a polynomial on [-pi/4, pi/4] (Cephes').
float sine(float x) {
    float q = roundEven(x * 0.636619747);
    float r = fma(q, -1.57079637, x);
    r = fma(q, 4.37113883e-08, r);
    r = fma(q, 1.71512451e-15, r);
    float z = r * r;
    float s = fma(fma(fma(-1.95152959e-4, z, 8.33216087e-3), z, -1.66666546e-1), z * r, r);
    float c = fma(fma(fma(2.44331571e-5, z, -1.38873163e-3), z, 4.16666457e-2), z * z,
                  fma(-0.5, z, 1.0));
    int quadrant = int(q) & 3;
    float v = (quadrant & 1) == 0 ? s : c;
    return (quadrant & 2) == 0 ? v : -v;
}

// x * scale + bias, then Snake: x + sin(alpha x)^2 / (alpha + 1e-9), as upstream's
// SampleRateConditionLayer and Snake1d compute them.
float affine_snake(float x, vec4 p) {
    x = x * p.x + p.y;
    float s = sine(p.z * x);
    return x + p.w * (s * s);
}

// The convolution's input at row r of the chunk, from the ring if r is negative, with the
// prologue applied to what is not in the ring yet; zero past the chunk's rows.
float input_at(int r, uint c) {
    if (r < 0) {
        return ring[ring_row(uint(-r)) * C + c];
    }
    if (uint(r) >= rows()) {
        return 0.0;
    }
    float v = x[uint(r) * C + c];
    return PROLOGUE ? affine_snake(v, prologue_[c]) : v;
}
