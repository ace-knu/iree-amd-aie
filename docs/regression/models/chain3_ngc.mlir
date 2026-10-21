// v9 회귀 V6: NPU → GPU → CPU 직결 체인 (10/20 경계 버퍼 표 "미시험 2"). y = softmax((x @ w1) @ w2, axis=1)
// 자동 배정(a,v,l 순서, 가속기 라운드로빈)이면 matmul1 → NPU, matmul2 → GPU, softmax → CPU (2026-10-22 세션 2 챕터 02에서 stream IR로 확인)
module {
  func.func @chain3(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> attributes {torch.onnx_meta.ir_version = 13 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> 
    %1 = torch.operator "onnx.MatMul"(%0, %arg2) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> 
    %2 = torch.operator "onnx.Softmax"(%1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    return %2 : !torch.vtensor<[128,128],f32>
  }
}
