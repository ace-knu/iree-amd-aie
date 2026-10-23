// 3후보 경계 버퍼 모델 (docs/2026-09-28/05). m0 를 NPU(생산), GPU(matmul), CPU(softmax) 가 모두 사용.
// 기대 (a=b=c=1, 128x128): m0 = 128, m1 = m0 x c = 128*128 = 16384, s = softmax(행 전부 128) = 1/128 = 0.0078125
module {
  func.func @fanout3(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.vtensor<[128,128],f32>) -> (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) attributes {torch.onnx_meta.ir_version = 13 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %1 = torch.operator "onnx.MatMul"(%0, %arg2) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %2 = torch.operator "onnx.Softmax"(%0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    return %1, %2 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>
  }
}
