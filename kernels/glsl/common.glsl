// What every kernel here declares: 16-bit storage, and f16 conversions that round to nearest
// even and keep subnormals, as the CPU's do. Weights converted from bf16 hold f16 subnormals,
// which a flush would zero.
//
// A kernel rounds to f16 where it stores a tensor, the vector a GEMV multiplies included, and
// computes in f32 between: an f16 rounding inside a kernel, of a value no other kernel sees,
// is one a driver may fold away (RADV folds float(float16_t(v)) into what follows).
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_spirv_intrinsics : require

spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4467], 4462, 16);
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4464], 4459, 16);

// The subgroup size the pipeline is required to run at, and its workgroup width.
layout(constant_id = 0) const uint SUBGROUP = 64;
layout(constant_id = 1) const uint WORKGROUP = 256;
layout(local_size_x_id = 1) in;

const uint SUBGROUPS = WORKGROUP / SUBGROUP;

// Eight f16 values packed in a uvec4, as f32, and back, rounding to nearest even.
void unpack8(uvec4 q, out vec4 lo, out vec4 hi) {
    lo = vec4(unpackHalf2x16(q.x), unpackHalf2x16(q.y));
    hi = vec4(unpackHalf2x16(q.z), unpackHalf2x16(q.w));
}

uvec4 pack8(vec4 lo, vec4 hi) {
    return uvec4(
        packFloat2x16(f16vec2(lo.xy)),
        packFloat2x16(f16vec2(lo.zw)),
        packFloat2x16(f16vec2(hi.xy)),
        packFloat2x16(f16vec2(hi.zw)));
}
