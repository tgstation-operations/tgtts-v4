// The SPIR-V target every GPU kernel here is compiled for: RADV (Mesa) on the
// Radeon 8060S, gfx1151, as Vulkan 1.4 reports it -- SPIR-V 1.6; f16 and 8-bit
// arithmetic and storage; VK_KHR_cooperative_matrix at its one shape on this
// generation, 16x16x16 at subgroup scope with f16 inputs and an f16 or f32
// accumulator; subgroups of 32 or 64 invocations under subgroup size control,
// 64 by default for compute; 64 KiB of workgroup memory; 1,024 invocations a
// workgroup.
//
// The version, capabilities and extensions are what the target allows, not
// what a kernel declares: the build's spirv-update-vce pass gives each module
// the minimal set it actually uses.
//
// A transform library: the build runs @attach_spirv_target after outlining,
// and it annotates every gpu.module with this environment, which
// convert-gpu-to-spirv then reads.
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
        api = Vulkan, AMD:IntegratedGPU,
        #spirv.resource_limits<
            max_compute_shared_memory_size = 65536,
            max_compute_workgroup_invocations = 1024,
            max_compute_workgroup_size = [1024, 1024, 1024],
            subgroup_size = 64,
            min_subgroup_size = 32,
            max_subgroup_size = 64,
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
