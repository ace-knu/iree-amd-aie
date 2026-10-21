// v9 회귀 V3: int8 matmul 3136x64x576 → i32 (npu4 벡터화 int8 MAC). 10/21 세션 1 챕터 04 rnd_test.sh의 모델과 같은 형태.
func.func @mm(%a: tensor<3136x576xi8>, %b: tensor<576x64xi8>) -> tensor<3136x64xi32> {
  %z = arith.constant 0 : i32
  %e = tensor.empty() : tensor<3136x64xi32>
  %f = linalg.fill ins(%z : i32) outs(%e : tensor<3136x64xi32>) -> tensor<3136x64xi32>
  %r = linalg.matmul ins(%a, %b : tensor<3136x576xi8>, tensor<576x64xi8>) outs(%f : tensor<3136x64xi32>) -> tensor<3136x64xi32>
  return %r : tensor<3136x64xi32>
}
