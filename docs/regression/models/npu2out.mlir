// NPU 가 만든 경계 값 2개를 GPU 가 함께 읽는 모델 (docs/2026-09-29/1_…/03 시도 (b)).
// 라운드로빈(a,v,l): d0 -> NPU, d1 -> GPU, d2 -> NPU, d3 -> GPU (flow 에서 확인).
//   m0 = x @ y   (NPU)      x = 1, y = 1, z = 2 (128x128)
//   m1 = x @ z   (GPU)      m0 = 128, m1 = 256, m2 = 256
//   m2 = y @ z   (NPU)      r  = m0 @ m2 = 128 * 256 * 128 = 4194304
//   r  = m0 @ m2 (GPU)      (m0 를 두 번 읽으면 2097152, m2 를 두 번 읽으면 8388608)
// 반환 (r, m1). 첫 결과 r = 4194304, 둘째 m1 = 256
module {
  func.func @npu2out(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.vtensor<[128,128],f32>) -> (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) attributes {torch.onnx_meta.ir_version = 13 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %1 = torch.operator "onnx.MatMul"(%arg0, %arg2) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %2 = torch.operator "onnx.MatMul"(%arg1, %arg2) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %3 = torch.operator "onnx.MatMul"(%0, %2) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    return %3, %1 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>
  }
}
