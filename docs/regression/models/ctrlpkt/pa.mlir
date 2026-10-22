// v10 회귀 V7(ctrlpkt_pa_3routes): 2026-10-22 세션 6. 컨트롤 패킷이면 entry 3개의 base PDI(배선)가 3종 -> 배선 묶음별 컨텍스트 3개.
// 고치기 전(S13 이전) 빌드는 이 모델을 컨트롤 패킷으로 돌리면 틀렸다(corr 0.229). 판정: 컨트롤 패킷 끔 출력과 비트 동일 + ctx MISS 3.
func.func @pa(%a: tensor<1024x32xf32>, %w1: tensor<32x64xf32>, %w2: tensor<64x512xf32>, %w3: tensor<512x32xf32>) -> tensor<1024x32xf32> {
  %z = arith.constant 0.0 : f32
  %e1024x64 = tensor.empty() : tensor<1024x64xf32>
  %f1024x64 = linalg.fill ins(%z : f32) outs(%e1024x64 : tensor<1024x64xf32>) -> tensor<1024x64xf32>
  %e1024x512 = tensor.empty() : tensor<1024x512xf32>
  %f1024x512 = linalg.fill ins(%z : f32) outs(%e1024x512 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %e1024x32 = tensor.empty() : tensor<1024x32xf32>
  %f1024x32 = linalg.fill ins(%z : f32) outs(%e1024x32 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %u0 = linalg.matmul ins(%a, %w1 : tensor<1024x32xf32>, tensor<32x64xf32>) outs(%f1024x64 : tensor<1024x64xf32>) -> tensor<1024x64xf32>
  %v0 = linalg.matmul ins(%u0, %w2 : tensor<1024x64xf32>, tensor<64x512xf32>) outs(%f1024x512 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %x0 = linalg.matmul ins(%v0, %w3 : tensor<1024x512xf32>, tensor<512x32xf32>) outs(%f1024x32 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  return %x0 : tensor<1024x32xf32>
}
