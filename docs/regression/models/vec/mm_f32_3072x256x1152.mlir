// v9 회귀 V1·V2: f32 matmul 3072x256x1152. 컴파일 시 팀 demote(--iree-amdaie-demote-contraction-inputs-to-bf16)로 bf16 입력 matmul이 되고,
// npu4 벡터화(bfp16 MAC)로 코드젠된다. 10/21 세션 1 챕터 04 rnd_test.sh의 모델과 같은 형태.
func.func @mm(%a: tensor<3072x1152xf32>, %b: tensor<1152x256xf32>) -> tensor<3072x256xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<3072x256xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<3072x256xf32>) -> tensor<3072x256xf32>
  %r = linalg.matmul ins(%a, %b : tensor<3072x1152xf32>, tensor<1152x256xf32>) outs(%f : tensor<3072x256xf32>) -> tensor<3072x256xf32>
  return %r : tensor<3072x256xf32>
}
