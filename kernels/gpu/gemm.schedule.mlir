// How gemm.mlir's linalg.matmul becomes a cooperative-matrix kernel.
//
// Each workgroup computes a 128 x 128 tile of C with four subgroups of 64
// invocations (256 in all), each subgroup a 64 x 64 quarter of it held as
// sixteen 16 x 16 accumulators for the whole K loop. Each iteration of that
// loop covers 32 of K: eight 16 x 16 tiles of A and eight of B, loaded
// straight from the storage buffers, and 32 multiply-adds at RADV's one
// cooperative-matrix shape, 16 x 16 x 16. No workgroup memory: the two
// subgroups that share rows of A each load them, and likewise for B.
//
// A transform library with two entry points, run by the build in this order:
// @gemm_schedule on the tensor-level function, then outlining and
// convert-vector-to-gpu, then @gemm_abi on the outlined kernel.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @gemm_schedule(%root: !transform.any_op {transform.readonly}) {
    %fill = transform.structured.match ops{["linalg.fill"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // C in 128 x 128 tiles, one per workgroup; then 64 x 64 per subgroup,
    // numbered linearly (subgroup = invocation / 64, row-major over the 2 x 2).
    // The zero fill follows the matmul into both, so each subgroup starts its
    // own accumulators at zero rather than reading C.
    %mm_wg, %wg = transform.structured.tile_using_forall %matmul tile_sizes [128, 128]
        (mapping = [#gpu.block<y>, #gpu.block<x>])
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill_wg, %wg2 = transform.structured.fuse_into_containing_op %fill into %wg
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm_sg, %sg = transform.structured.tile_using_forall %mm_wg tile_sizes [64, 64]
        (mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>])
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill_sg, %sg2 = transform.structured.fuse_into_containing_op %fill_wg into %sg
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // K in steps of 32.
    %mm_k, %kloop = transform.structured.tile_using_for %mm_sg tile_sizes [0, 0, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Vectorize: the 64 x 64 x 32 matmul becomes one vector.contract, and the
    // fill a transfer_write of zeros. Fold f16 -> f32 extensions into the
    // contraction (the f32-accumulating instance would otherwise multiply f32
    // matrices, which RADV does not offer), then unroll everything to the
    // native 16 x 16 x 16.
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %vfunc = transform.structured.vectorize_children_and_apply_patterns %func
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vfunc {
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.gpu.unroll_vectors_subgroup_mma [16, 16, 16]
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // The sixteen accumulators leave the K loop's body and become its loop
    // values; reading them back from the zero fill folds to a constant.
    %loops = transform.structured.match ops{["scf.for"]} in %vfunc
        : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %loops : !transform.any_op
    transform.apply_patterns to %vfunc {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %vfunc : !transform.any_op

    // Bufferize straight into the arguments, then put the loops on the GPU:
    // the grid follows the workgroup loop's trip counts (N / 128 across,
    // M / 128 down), and subgroups are 64 invocations, the size @gemm_abi
    // records for the runtime.
    transform.bufferization.eliminate_empty_tensors %vfunc : !transform.any_op
    %bfunc = transform.bufferization.one_shot_bufferize %vfunc
        : (!transform.any_op) -> !transform.any_op
    %launch = transform.gpu.map_forall_to_blocks %bfunc generate_gpu_launch
        : (!transform.any_op) -> !transform.any_op
    %mapped = transform.gpu.map_nested_forall_to_threads %launch block_dims = [256, 1, 1]
        sync_after_distribute = false warp_size = 64 : (!transform.any_op) -> !transform.any_op
    %final = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %final {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %final : !transform.any_op
    transform.yield
  }

  // The workgroup size, as SPIR-V's LocalSize, and the subgroup size the
  // schedule above divided invocations by. SPIR-V for Vulkan cannot carry a
  // required subgroup size (that is pipeline state), so it is recorded on the
  // kernel for build.py to put in the manifest.
  transform.named_sequence @gemm_abi(%root: !transform.any_op {transform.readonly}) {
    %kernel = transform.structured.match ops{["gpu.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %abi = transform.param.constant #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>
        -> !transform.any_param
    transform.annotate %kernel "spirv.entry_point_abi" = %abi
        : !transform.any_op, !transform.any_param
    %subgroup = transform.param.constant 64 : i32 -> !transform.any_param
    transform.annotate %kernel "kernel.required_subgroup_size" = %subgroup
        : !transform.any_op, !transform.any_param
    transform.yield
  }
}
