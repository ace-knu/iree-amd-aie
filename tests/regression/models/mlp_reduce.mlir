module {
  func.func @mlp_reduce(%arg0: !torch.vtensor<[128,128],f32>, %arg1: !torch.vtensor<[128,128],f32>, %arg2: !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> attributes {torch.onnx_meta.ir_version = 13 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> 
    %1 = torch.operator "onnx.Softmax"(%0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> 
    %2 = torch.operator "onnx.MatMul"(%1, %arg2) : (!torch.vtensor<[128,128],f32>, !torch.vtensor<[128,128],f32>) -> !torch.vtensor<[128,128],f32> 
    return %2 : !torch.vtensor<[128,128],f32>
  }
}
