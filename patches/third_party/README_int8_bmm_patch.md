# int8 `aten.bmm` 패치 (torch-mlir)

`bert` 브랜치의 컴파일러 수정만으로는 **int8 배치 matmul 모델을 컴파일할 수 없습니다.**
torch-mlir 에 `aten.bmm` 의 int8 양자화 경로가 없어서(2D `aten.mm` 은 이미 있음),
아래 패치를 따로 적용해야 합니다.

이 패치가 브랜치에 안 들어간 이유: torch-mlir 서브모듈의 리모트가 우리 포크가 아니라
upstream `iree-org/torch-mlir` 이라 푸쉬할 수 없습니다.

## 파일

`0001-torch-mlir-int8-aten-bmm.patch`

## 적용 위치

```
<repo>/third_party/iree/third_party/torch-mlir      ← 3단 서브모듈
```

## 적용

```bash
cd third_party/iree/third_party/torch-mlir
git am ../../../../_local/share/0001-torch-mlir-int8-aten-bmm.patch
#   또는
git apply /path/to/0001-torch-mlir-int8-aten-bmm.patch
```

베이스는 upstream `origin/main` 에 있는 `d2768f87`
("[TorchToLinalg] Add lowering for AtenEluBackwardOp (#4545)") 이므로,
그 근처 리비전이면 깨끗이 붙습니다.

적용 후 IREE 를 다시 빌드해야 합니다(torch-mlir 은 컴파일러에 링크됨).

## 무엇을 바꾸나 (2개 파일, +66줄)

| 파일 | 증감 |
|---|---|
| `lib/Dialect/Torch/Transforms/FuseQuantizedOps.cpp` | +2 |
| `lib/Conversion/TorchToLinalg/Linear.cpp` | +64 |

### 왜 필요한가

`QuantizeOperandsPastCommutingOps` 와 `QuantizeAccumulator` 가 `AtenMmOp`/`AtenMatmulOp` 에는
있었지만 **`AtenBmmOp` 대응물이 없었습니다.** ONNX 의 배치 MatMul 은 `aten.matmul` 이 아니라
**`aten.bmm` 으로 임포트**되므로, `FuseQuantizedOpsPass` 에 매칭되는 패턴이 없어
**양자화가 조용히 통째로 건너뛰어졌습니다.** `ConvertAtenBmmOp` 도 zero-point 를 보지 않고
항상 f32 `linalg.BatchMatmulOp` 를 만들었습니다.

### 무엇을 추가하나

- `FuseQuantizedOps.cpp`: `QuantizeOperandsPastCommutingOps<AtenBmmOp, 2>` (입력측 융합) 과
  `QuantizeAccumulator<AtenBmmOp>` (출력측 재스케일). 후자가 없으면 결과가 f32 로 선언된 채로
  남아 `convertTensorToElementType` 이 스케일된 dequantize 대신 단순 `sitofp` 를 해서
  거의 모든 원소가 포화됩니다.
- `Linear.cpp`: `ConvertAtenBmmOp::matchAndRewrite` 에 zero-point 처리 확장.

## 확인 방법

패치 후 int8 양자화된 배치 matmul 모델을 임포트하면 `input` 단계에서
`linalg.quantized_matmul` (i8, i8, i32, i32) 이 나와야 합니다. 패치 전에는
평범한 f32 `linalg.matmul` + 명시적 dequantize 가 나옵니다.

```bash
build/tools/iree-compile <model>_int8.mlir -o /tmp/x.mlir --compile-to=input
grep quantized_matmul /tmp/x.mlir
```

## 알려진 한계

**per-channel(축별) 가중치 스케일은 아직 안 됩니다.** `linalg.quantized_matmul` 이 zero-point 를
스칼라 `i32` 로만 받아서 per-channel 을 표현할 수 없고, 그래서 per-axis 양자화 모델은 f32 로
폴백한 뒤 AMD-AIE 백엔드에서 컴파일 실패합니다
(`AMDAIEBufferizeToAllocation` — "expected only one target op, found 2").
현재 조사 중이며, 양자화는 **per-tensor 로 하십시오**
(`quantize_static(..., per_channel=False)` — 기본값).
