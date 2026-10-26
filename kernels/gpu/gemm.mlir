// C = A * B^T, a linear layer's projection:
//
//   A  @M@ x @K@ f16, row-major: the activations, one row per position
//   B  @N@ x @K@ f16, row-major: the weights as a checkpoint lays them out,
//      one row per output feature
//   C  @M@ x @N@ @ACC@, row-major: one row per position
//
// accumulating in @ACC@: each product and each partial sum is an @ACC@ value.
//
// A template: the build substitutes @M@, @N@, @K@ and @ACC@ (f16 or f32)
// before compiling, because the GPU schedule needs static shapes, and the CPU
// oracle instantiates it at a small shape. Nothing else differs between
// instances.
func.func @gemm(%a: memref<@M@x@K@xf16>, %b: memref<@N@x@K@xf16>, %c: memref<@M@x@N@x@ACC@>) {
  %ta = bufferization.to_tensor %a restrict : memref<@M@x@K@xf16> to tensor<@M@x@K@xf16>
  %tb = bufferization.to_tensor %b restrict : memref<@N@x@K@xf16> to tensor<@N@x@K@xf16>
  %zero = arith.constant 0.0 : @ACC@
  %empty = tensor.empty() : tensor<@M@x@N@x@ACC@>
  %init = linalg.fill ins(%zero : @ACC@) outs(%empty : tensor<@M@x@N@x@ACC@>) -> tensor<@M@x@N@x@ACC@>
  %res = linalg.matmul
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>]
      ins(%ta, %tb : tensor<@M@x@K@xf16>, tensor<@N@x@K@xf16>)
      outs(%init : tensor<@M@x@N@x@ACC@>) -> tensor<@M@x@N@x@ACC@>
  bufferization.materialize_in_destination %res in restrict writable %c
      : (tensor<@M@x@N@x@ACC@>, memref<@M@x@N@x@ACC@>) -> ()
  return
}
