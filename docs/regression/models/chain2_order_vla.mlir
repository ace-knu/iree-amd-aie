// chain2_npufirst.mlir 에서 디바이스 인덱스만 바꾼 변형 (docs/2026-09-28/03). 컴파일 순서: vulkan,local,amdxdna
#gpu = #hal.device.affinity<@__device_0>
#npu = #hal.device.affinity<@__device_2>

func.func @chain2(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>,
                  %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %zero = arith.constant 0.0 : f32

  // matmul_1 → NPU
  %a_npu = flow.tensor.transfer %a : tensor<128x128xf32> to #npu
  %b_npu = flow.tensor.transfer %b : tensor<128x128xf32> to #npu
  %e1 = tensor.empty() : tensor<128x128xf32>
  %f1 = linalg.fill ins(%zero : f32) outs(%e1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %m1 = linalg.matmul ins(%a_npu, %b_npu : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%f1 : tensor<128x128xf32>) -> tensor<128x128xf32>

  // matmul_2 → GPU  (m1을 NPU에서 GPU로 직접)
  %m1_gpu = flow.tensor.transfer %m1 : tensor<128x128xf32> to #gpu
  %c_gpu  = flow.tensor.transfer %c  : tensor<128x128xf32> to #gpu
  %e2 = tensor.empty() : tensor<128x128xf32>
  %f2 = linalg.fill ins(%zero : f32) outs(%e2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %m2 = linalg.matmul ins(%m1_gpu, %c_gpu : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%f2 : tensor<128x128xf32>) -> tensor<128x128xf32>

  return %m2 : tensor<128x128xf32>
}
