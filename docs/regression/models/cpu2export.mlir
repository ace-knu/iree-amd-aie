// CPU 가 한 execute 영역에서 만든 출력 2개가 한 외부 할당에 packing 되어 subview 로 export 되는 모델
// (docs/2026-09-29/1_…/03 시도 (c)). lead=vulkan 이면 툴링이 CPU 출력을 Vulkan copy_buffer BRIDGE 로 옮긴다.
// 배정(a,v,l): m0 = x@x -> NPU, s1 = softmax(y) -> CPU, s2 = z*z -> CPU
//   x = 1, y = 1, z = 2 (128x128): s2 = 4, s1 = 1/128 = 0.0078125, m0 = 128
// 반환 (s2, s1, m0). 첫 결과 s2 = 4 (s1 자리를 읽으면 0.0078125)
module {
  func.func @cpu2export(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.vtensor<[128,128],f32>) -> (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) attributes {torch.onnx_meta.ir_version = 13 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MatMul"(%arg0, %arg0) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %1 = torch.operator "onnx.Softmax"(%arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    %2 = torch.operator "onnx.Mul"(%arg2, %arg2) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32>
    return %2, %1, %0 : !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>
  }
}
