// The SPIR-V target every GPU kernel here is compiled for: what both GPUs it
// runs on offer under Vulkan 1.4, RADV (Mesa) on the Radeon 8060S, gfx1151,
// and NVIDIA's driver 595.71.05 on the RTX 4070 Ti SUPER -- SPIR-V 1.6; f16
// and 8-bit arithmetic and storage; the shapes of VK_KHR_cooperative_matrix
// with f16 inputs that both have, 16x16x16 at subgroup scope with an f16 or
// f32 accumulator; 48 KiB of workgroup memory; 1,024 invocations a workgroup.
// Subgroups are of the one size the kernel's pipeline requires: RADV can
// require 32 or 64, NVIDIA only 32.
//
// The version, capabilities and extensions are what the target allows, not
// what a kernel declares: the build's spirv-update-vce pass gives each module
// the minimal set it actually uses.
//
// A template and a transform library: the build substitutes @SUBGROUP@, then
// runs @attach_spirv_target after outlining, and it annotates every gpu.module
// with this environment, which convert-gpu-to-spirv then reads.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @attach_spirv_target(%root: !transform.any_op {transform.readonly}) {
    %modules = transform.structured.match ops{["gpu.module"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %env = transform.param.constant #spirv.target_env<
        #spirv.vce<v1.6,
            [Shader, Float16, Int8, Int16, StorageBuffer8BitAccess,
             StorageBuffer16BitAccess, GroupNonUniform, VulkanMemoryModel, CooperativeMatrixKHR],
            [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage,
             SPV_KHR_storage_buffer_storage_class, SPV_KHR_vulkan_memory_model, SPV_KHR_cooperative_matrix]>,
        api = Vulkan,
        #spirv.resource_limits<
            max_compute_shared_memory_size = 49152,
            max_compute_workgroup_invocations = 1024,
            max_compute_workgroup_size = [1024, 1024, 64],
            subgroup_size = @SUBGROUP@,
            min_subgroup_size = @SUBGROUP@,
            max_subgroup_size = @SUBGROUP@,
            cooperative_matrix_properties_khr = [
                #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16,
                    a_type = f16, b_type = f16, c_type = f16, result_type = f16,
                    acc_sat = false, scope = #spirv.scope<Subgroup>>,
                #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16,
                    a_type = f16, b_type = f16, c_type = f32, result_type = f32,
                    acc_sat = false, scope = #spirv.scope<Subgroup>>]>>
        -> !transform.any_param
    transform.annotate %modules "spirv.target_env" = %env
        : !transform.any_op, !transform.any_param
    transform.yield
  }
}
