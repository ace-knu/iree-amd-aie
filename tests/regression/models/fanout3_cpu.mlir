// 3후보 경계 버퍼 모델, CPU 생산자판 (docs/2026-09-28/05).
// s = softmax(a) (CPU) 를 NPU matmul(s, b) 와 GPU matmul(s, c) 가 함께 읽는다.
// 기대 (a=b=c=1, 128x128): s = 1/128, m1 = s x b = 128 * (1/128) = 1, m2 = s x c = 1
module {
  func.func @fanout3_cpu(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.vtensor<[128,128],f32>) -> (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) attributes {torch.onnx_meta.ir_version = 13 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %1 = torch.operator "onnx.MatMul"(%0, %arg1) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %2 = torch.operator "onnx.MatMul"(%0, %arg2) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    return %1, %2 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>
  }
}
