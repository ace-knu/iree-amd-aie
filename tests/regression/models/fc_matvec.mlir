// FC (M=1 matvec): on the GPU this takes the SPIR-V SubgroupReduce pipeline.
// With all-ones inputs every output element is K = 4096; a subgroup size that
// does not match the generated code gives 8192.
func.func @fc(%a: tensor<1x4096xf32>, %b: tensor<4096x1000xf32>) -> tensor<1x1000xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x1000xf32>
  %z = linalg.fill ins(%cst : f32) outs(%e : tensor<1x1000xf32>) -> tensor<1x1000xf32>
  %r = linalg.matmul ins(%a, %b : tensor<1x4096xf32>, tensor<4096x1000xf32>) outs(%z : tensor<1x1000xf32>) -> tensor<1x1000xf32>
  return %r : tensor<1x1000xf32>
}
