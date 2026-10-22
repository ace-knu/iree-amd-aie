// v10 회귀 V7(ctrlpkt_chain4): 2026-10-22 세션 5·6. 모양 2가지 matmul 4개, 컨트롤 패킷이면 base PDI 1종 -> 컨텍스트 1개 공유.
// 판정: 컨트롤 패킷 끔 출력과 비트 동일 + ctx MISS 1.
func.func @chain4(%a: tensor<128x128xf32>, %w1: tensor<128x256xf32>, %w2: tensor<256x128xf32>) -> tensor<128x128xf32> {
  %z = arith.constant 0.0 : f32
  %e1 = tensor.empty() : tensor<128x256xf32>
  %f1 = linalg.fill ins(%z : f32) outs(%e1 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %e2 = tensor.empty() : tensor<128x128xf32>
  %f2 = linalg.fill ins(%z : f32) outs(%e2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %u0 = linalg.matmul ins(%a, %w1 : tensor<128x128xf32>, tensor<128x256xf32>) outs(%f1 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %v0 = linalg.matmul ins(%u0, %w2 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%f2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %u1 = linalg.matmul ins(%v0, %w1 : tensor<128x128xf32>, tensor<128x256xf32>) outs(%f1 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %v1 = linalg.matmul ins(%u1, %w2 : tensor<128x256xf32>, tensor<256x128xf32>) outs(%f2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %v1 : tensor<128x128xf32>
}
