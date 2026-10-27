// gemm.glsl's tokens staged through RMSNorm, as norm.glsl normalizes one: each token's mean
// square in f32, and the token times its reciprocal root times the gain g, rounded to f16 once.
//
// The including kernel declares K, KC, M, EPS and the `x` and `g` buffers as uvec4 views, eight
// f16 to each, and includes gemm.glsl, then this, and stages its tokens with stage_normed.
// Token m is row TOKEN_ROW(m) of x, m unless the kernel defines it otherwise.

#ifndef TOKEN_ROW
#define TOKEN_ROW(m) (m)
#endif

// The tokens each subgroup sums the squares of, and the uvec4 of a token each lane reads.
const uint NORMED = (M + SUBGROUPS - 1) / SUBGROUPS;
const uint SQUARED = (K / 8 + SUBGROUP - 1) / SUBGROUP;

shared float token_scale[M];

// Every token's reciprocal root mean square, into token_scale. Every load is issued before
// any is used: the tokens come from another workgroup's writes, a trip to L2 each.
void norm_scales() {
    float squares[NORMED];
    [[unroll]] for (uint t = 0; t < NORMED; t++) {
        uint m = t * SUBGROUPS + gl_SubgroupID;
        squares[t] = 0.0;
        [[unroll]] for (uint i = 0; i < SQUARED; i++) {
            uint v = i * SUBGROUP + gl_SubgroupInvocationID;
            if (m < M && v < K / 8) {
                vec4 lo, hi;
                unpack8(x[TOKEN_ROW(m) * (K / 8) + v], lo, hi);
                squares[t] += dot(lo, lo) + dot(hi, hi);
            }
        }
    }
    [[unroll]] for (uint t = 0; t < NORMED; t++) {
        uint m = t * SUBGROUPS + gl_SubgroupID;
        float total = subgroupAdd(squares[t]);
        if (m < M && subgroupElect()) {
            token_scale[m] = 1.0 / sqrt(total / float(K) + EPS);
        }
    }
    barrier();
}

// Stages chunk `chunk` of the normed tokens into xs, once norm_scales has run.
void stage_normed(uint chunk) {
    uvec4 tokens[STAGED], gains[STAGED];
    [[unroll]] for (uint i = 0; i < STAGED; i++) {
        uint v = i * WORKGROUP + gl_LocalInvocationID.x;
        if (v < M * (KC / 8)) {
            uint at = staged_at(v, chunk);
            tokens[i] = x[TOKEN_ROW(at / (K / 8)) * (K / 8) + at % (K / 8)];
            gains[i] = g[at % (K / 8)];
        }
    }
    [[unroll]] for (uint i = 0; i < STAGED; i++) {
        uint v = i * WORKGROUP + gl_LocalInvocationID.x;
        if (v < M * (KC / 8)) {
            float scale = token_scale[v / (KC / 8)];
            vec4 lo, hi, glo, ghi;
            unpack8(tokens[i], lo, hi);
            unpack8(gains[i], glo, ghi);
            xs[v] = pack8(lo * scale * glo, hi * scale * ghi);
        }
    }
}
