// cg_softmax (회귀 v10): CPU softmax 5번 -> GPU matmul -> CPU softmax -> GPU matmul, 512x512 f32. 자동 배정(contraction -> GPU).
// CPU·GPU가 같이 쓰는 transient의 GPU 실행이 아직 도는 CPU 작업 뒤 alloca 신호를 기다리는 모양(GoogLeNet CPU+GPU와 같음).
module {
  func.func @cg_softmax(%arg0: !torch.vtensor<[512,512],f32>, %arg1: !torch.vtensor<[512,512],f32>, %arg2: !torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> attributes {torch.onnx_meta.ir_version = 13 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %s0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> 
    %s1 = torch.operator "onnx.Softmax"(%s0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> 
    %s2 = torch.operator "onnx.Softmax"(%s1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> 
    %s3 = torch.operator "onnx.Softmax"(%s2) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> 
    %s4 = torch.operator "onnx.Softmax"(%s3) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> 
    %m0 = torch.operator "onnx.MatMul"(%s4, %arg1) : (!torch.vtensor<[512,512],f32>, !torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> 
    %s5 = torch.operator "onnx.Softmax"(%m0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> 
    %m1 = torch.operator "onnx.MatMul"(%s5, %arg2) : (!torch.vtensor<[512,512],f32>, !torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> 
    return %m1 : !torch.vtensor<[512,512],f32>
  }
}
