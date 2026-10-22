// v11 회귀 V1: f32 matmul 4096x64x32 (규민 memtile join 레이스 재현 모양 M=4096, K=32, N=64; d2846c0d). 컴파일 시 팀 demote(--iree-amdaie-demote-contraction-inputs-to-bf16)로 bf16 입력 matmul이 되고,
// npu4 벡터화(bfp16 MAC)로 코드젠된다. 10/21 세션 1 챕터 04 rnd_test.sh의 모델과 같은 형태.
func.func @mm(%a: tensor<4096x32xf32>, %b: tensor<32x64xf32>) -> tensor<4096x64xf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<4096x64xf32>
  %f = linalg.fill ins(%z : f32) outs(%e : tensor<4096x64xf32>) -> tensor<4096x64xf32>
  %r = linalg.matmul ins(%a, %b : tensor<4096x32xf32>, tensor<32x64xf32>) outs(%f : tensor<4096x64xf32>) -> tensor<4096x64xf32>
  return %r : tensor<4096x64xf32>
}
