func.func @mm(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x128xf32>
  %z = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
  %r = linalg.matmul ins(%a, %b : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%z : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %r : tensor<128x128xf32>
}
