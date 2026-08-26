# Pass 예시 모음 (모델 → linalg → dispatch)

`docs/2026-08-16_frontend_lowering_passes.md`(이하 "본 문서화")가 각 pass의 **위치와 목적**을 정리한
문서라면, 이 문서는 **각 pass가 IR을 실제로 어떻게 바꾸는지**를 작은 예시로 보여주는 보조 자료다. pass
조사 결과를 이해/공유하는 용도.

## 읽기 전에

- 이 저장소에 로컬 빌드(`build/tools/iree-compile`)가 없어서, 아래 IR은 **실제 컴파일러 출력을 캡처한
  게 아니라 각 pass의 소스 코드(주석/구현)를 근거로 손으로 구성한 예시**다. 문법은 최대한 정확히
  맞췄지만, 실제 dump와 텍스트가 한 글자도 다르지 않다고 보장하지 않는다.
- 예시는 VGG16을 극단적으로 축소한 두 가지 조각을 계속 재사용한다:
  - **conv 조각**: `Conv2d(Cin=3, Cout=8, k=3, pad=1)` → `+bias` → `ReLU` (VGG16의 첫 conv 블록과 동일한
    연산 시퀀스, shape만 1x3x4x4로 축소)
  - **FC 조각**: `matmul(1x512, 512x1000)` (VGG16 마지막 FC층과 동일하게 N=1000 — NPU pack-peel 타일
    배수(예: 32)로 안 나눠떨어지는 걸 보여주기 위한 조각)
- 빌드가 준비되면 `docs/2026-07-06_env_setup/DEBUG_PIPELINE.md`의 `scripts/debug/pipeline_dump.py
  --pass <pass-cli-name>`으로 **진짜 before/after IR**을 뽑아 이 문서와 대조해볼 것. 그게 이 문서의
  다음 단계다.
- 각 절 제목의 `(§N)`은 `2026-08-16_frontend_lowering_passes.md`의 해당 절을 가리킨다.

## 요약 표

| Pass | Phase | 한 줄 요약 | 아래 절 |
| --- | --- | --- | --- |
| `ConvertTorchToLinalg` | 1.input | torch dialect → linalg-on-tensors | §1 |
| `iree-preprocessing-convert-conv-to-channels-last` | 3.preprocessing | NCHW → NHWC | §2.1 |
| bf16 demote (AMD-AIE) | 3.preprocessing | f32 matmul/conv 입력 → bf16 (named op 상태에서) | §2.2 |
| `ConvertConv2DToImg2Col` | 4.global-optimization | conv → im2col-generic + matmul | §3.1 |
| `DetachElementwiseFromNamedOps`(+through-reshape) | 4.global-optimization | bias를 conv/matmul의 init에서 분리 | §3.2 |
| `GeneralizeLinalgNamedOps` | 4.global-optimization | named op(`linalg.matmul`) → `linalg.generic` | §3.3 |
| `FormDispatchRegionsPass`(+no-fuse) | 5.dispatch-creation | linalg → `flow.dispatch.region`, contraction root는 fusion 제외 | §4.1 |
| `ConvertDispatchRegionsToWorkgroupsPass` | 5.dispatch-creation | region → `flow.dispatch.workgroups` | §4.2 |
| `AMDAIEAssignDeviceAffinities`(2차) | 6.flow | 각 dispatch에 `stream.affinity` | §5.1 |
| `AMDAIEPadContractionDispatches` | 6.flow | N=1000 → 1024로 패딩 + host pad/crop dispatch | §5.2 |
| `AMDAIESplitLargeContractionDispatches` | 6.flow | 패딩된 매트멀을 N-chunk로 split + host concat | §5.3 |

---

## §1. `ConvertTorchToLinalg` — torch → linalg (Phase 1)

torch-mlir이 `linalg` named op을 만들 때, bias는 별도 add가 아니라 **conv의 누산 초기값(`outs`)에
브로드캐스트해서 넣는** 형태로 내려오는 게 흔한 패턴이다(그래야 뒤에서 `DetachElementwiseFromNamedOps`가
분리할 대상이 생긴다, §3.2 참고).

**Before** (torch dialect, `import_onnx` 결과):
```mlir
%conv = torch.aten.convolution %input, %weight, %bias, %stride, %padding, %dilation,
    %false, %empty_list, %int1
    : !torch.vtensor<[1,3,4,4],f32>, !torch.vtensor<[8,3,3,3],f32>, !torch.vtensor<[8],f32>,
      !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int
    -> !torch.vtensor<[1,8,4,4],f32>
%relu = torch.aten.relu %conv : !torch.vtensor<[1,8,4,4],f32> -> !torch.vtensor<[1,8,4,4],f32>
```

**After** (`ConvertTorchToLinalg` + `ConvertElementwiseToLinalg`, linalg-on-tensors):
```mlir
// bias를 (N,H,W)로 브로드캐스트해서 conv의 초기 누산값으로 사용
%bias_bcast = linalg.generic {
  indexing_maps = [affine_map<(n,c,h,w) -> (c)>, affine_map<(n,c,h,w) -> (n,c,h,w)>],
  iterator_types = ["parallel","parallel","parallel","parallel"]
} ins(%bias : tensor<8xf32>) outs(%empty : tensor<1x8x4x4xf32>) {
  ^bb0(%b: f32, %o: f32):
    linalg.yield %b : f32
} -> tensor<1x8x4x4xf32>

%conv = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
  ins(%input, %weight : tensor<1x3x4x4xf32>, tensor<8x3x3x3xf32>)
  outs(%bias_bcast : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>   // bias 위에 누산

%relu = linalg.generic {
  indexing_maps = [affine_map<(n,c,h,w) -> (n,c,h,w)>, affine_map<(n,c,h,w) -> (n,c,h,w)>],
  iterator_types = ["parallel","parallel","parallel","parallel"]
} ins(%conv : tensor<1x8x4x4xf32>) outs(%empty2 : tensor<1x8x4x4xf32>) {
  ^bb0(%in: f32, %o: f32):
    %zero = arith.constant 0.0 : f32
    %r = arith.maximumf %in, %zero : f32
    linalg.yield %r : f32
} -> tensor<1x8x4x4xf32>
```

핵심: `torch.aten.convolution` 하나가 **`linalg.generic`(bias 브로드캐스트) + `linalg.conv_2d_nchw_fchw`
+ `linalg.generic`(relu)** 세 개로 펼쳐졌다. 이 세 op이 이후 모든 pass가 다루는 기본 단위가 된다.

## §2. Phase 3 — Preprocessing

### §2.1 `convert-conv-to-channels-last` — NCHW → NHWC

ONNX conv는 NCHW인데, 뒤에서 쓸 im2col 경로(§3.1)는 NHWC를 기대한다.

**Before**: §1의 `%conv`(`tensor<1x8x4x4xf32>`, NCHW)

**After**:
```mlir
%input_nhwc = linalg.transpose ins(%input : tensor<1x3x4x4xf32>)
  outs(%empty3 : tensor<1x4x4x3xf32>) permutation = [0, 2, 3, 1]   // NCHW -> NHWC
%weight_hwcf = linalg.transpose ins(%weight : tensor<8x3x3x3xf32>)
  outs(%empty4 : tensor<3x3x3x8xf32>) permutation = [2, 3, 1, 0]   // FCHW -> HWCF

%bias_bcast = linalg.generic {...} ins(%bias : tensor<8xf32>) outs(%empty5 : tensor<1x4x4x8xf32>) {...}

%conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
  ins(%input_nhwc, %weight_hwcf : tensor<1x4x4x3xf32>, tensor<3x3x3x8xf32>)
  outs(%bias_bcast : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>   // 채널이 마지막 차원으로

%relu = linalg.generic {...} ins(%conv : tensor<1x4x4x8xf32>) outs(...) {...}  // shape만 NHWC로 바뀜
```

핵심: `linalg.conv_2d_nchw_fchw` → `linalg.conv_2d_nhwc_hwcf`로 op 이름 자체가 바뀌고(named op이라 layout이
op identity에 박혀있음), 앞뒤로 `linalg.transpose`가 새로 생긴다. 이 transpose들이 나중에 §5.1에서 host(CPU)로
배치되는 대상이다.

### §2.2 bf16 demote (AMD-AIE `extendPreprocessingPassPipeline`)

**Before**: §2.1의 `%conv` (아직 `linalg.conv_2d_nhwc_hwcf`, named op 상태 — 이게 이 pass가 GlobalOptimization의
`GeneralizeLinalgNamedOps`(§3.3)보다 먼저 돌아야 하는 이유).

**After**:
```mlir
%input_bf16 = arith.truncf %input_nhwc : tensor<1x4x4x3xf32> to tensor<1x4x4x3xbf16>
%weight_bf16 = arith.truncf %weight_hwcf : tensor<3x3x3x8xf32> to tensor<3x3x3x8xbf16>

%conv = linalg.conv_2d_nhwc_hwcf {...}
  ins(%input_bf16, %weight_bf16 : tensor<1x4x4x3xbf16>, tensor<3x3x3x8xbf16>)
  outs(%bias_bcast : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>   // 누산(accumulate)은 f32 그대로
```

핵심: **입력만** bf16이고 **누산 타입(outs)은 f32 그대로**다("정밀도는 깎이지만 오버플로/누적오차는
f32로 관리"). `linalg.conv_2d_nhwc_hwcf`라는 op 이름이 그대로 유지되는 것도 포인트 — 이 pass가 아직
named op 형태일 때 패턴 매치하기 때문에 가능한 일이고, 이게 §3.3(`GeneralizeLinalgNamedOps`)보다 먼저
돌아야 하는 이유를 그대로 보여준다.

## §3. Phase 4 — GlobalOptimization

### §3.1 `ConvertConv2DToImg2Col` — conv → im2col-generic + matmul

NPU는 표준 NHWC conv codegen이 없어서, conv를 "슬라이딩 윈도우를 펼친 행렬"(im2col) × "펼친 필터"의
matmul로 바꾼다.

**Before**: §2.2의 `%conv` (`linalg.conv_2d_nhwc_hwcf`, bf16 입력 / f32 누산)

**After**:
```mlir
// 1. 필터를 2D로 collapse: [KH,KW,C,F] -> [KH*KW*C, F]
%filter_2d = tensor.collapse_shape %weight_bf16 [[0,1,2],[3]]
  : tensor<3x3x3x8xbf16> into tensor<27x8xbf16>

// 2. im2col: 출력 픽셀(OH*OW)마다 그 위치의 KHxKWxC 윈도우를 한 행으로 복사
//    (실제로는 linalg.generic이 인덱스 계산으로 슬라이딩 윈도우를 읽어 채움 — 여기선 개념만 표현)
%img2col = linalg.generic {
  indexing_maps = [affine_map<(n,ohw,khwc) -> (n, ohw, khwc)>],  // 단순화: 실제 map은 h/w/kh/kw 인덱스 계산 포함
  iterator_types = ["parallel","parallel","parallel"]
} outs(%empty6 : tensor<1x16x27xbf16>) {
  ^bb0(%o: bf16):
    // 입력 위치 (h+kh-1, w+kw-1, c)에서 값을 읽어옴 (패딩 영역은 0)
    linalg.yield %read : bf16
} -> tensor<1x16x27xbf16>   // [N, OH*OW=4*4, KH*KW*C=3*3*3]

// 3. matmul: [N, OH*OW, KHWC] x [KHWC, F] -> [N, OH*OW, F]
%matmul_2d = linalg.matmul
  ins(%img2col, %filter_2d : tensor<1x16x27xbf16>, tensor<27x8xbf16>)
  outs(%bias_bcast_2d : tensor<1x16x8xf32>) -> tensor<1x16x8xf32>

// 4. 다시 4D로 expand: [N, OH*OW, F] -> [N, OH, OW, F]
%conv_result = tensor.expand_shape %matmul_2d [[0],[1,2],[3]]
  : tensor<1x16x8xf32> into tensor<1x4x4x8xf32>
```

핵심: **하나의 named conv op이 `linalg.generic`(im2col 패킹) + `linalg.matmul` 두 개로 쪼개졌다.** 이
`linalg.matmul`이 바로 pack-peel codegen이 인식하는 "contraction root"이고, VGG16의 13개 conv 레이어
전부 이 형태로 NPU에 도달한다(§0의 "conv를 지원하는 유일한 경로"가 이것).

### §3.2 `DetachElementwiseFromNamedOps` (+ `-through-reshape`)

matmul의 `outs`가 bias 브로드캐스트로 채워져 있으면 pack-peel이 요구하는 "순수 matmul(`fill(0)→matmul`)"
형태가 아니다. 이 pass가 bias를 다시 분리해낸다 — `-through-reshape` 없이는 §3.1의 `tensor.expand_shape`
때문에 bias가 안 보여서 실패하는 게 이 옵션이 필요한 이유다.

**Before**: §3.1의 `%matmul_2d` (bias가 `outs`에 이미 섞여 들어간 상태)

**After**:
```mlir
%zero = arith.constant 0.0 : f32
%zero_init = linalg.fill ins(%zero : f32) outs(%empty7 : tensor<1x16x8xf32>) -> tensor<1x16x8xf32>

%matmul_2d = linalg.matmul
  ins(%img2col, %filter_2d : tensor<1x16x27xbf16>, tensor<27x8xbf16>)
  outs(%zero_init : tensor<1x16x8xf32>) -> tensor<1x16x8xf32>   // 이제 순수 matmul: fill(0) -> matmul

%biased_2d = linalg.generic {
  indexing_maps = [affine_map<(n,ohw,f) -> (n,ohw,f)>, affine_map<(n,ohw,f) -> (f)>,
                   affine_map<(n,ohw,f) -> (n,ohw,f)>],
  iterator_types = ["parallel","parallel","parallel"]
} ins(%matmul_2d, %bias : tensor<1x16x8xf32>, tensor<8xf32>) outs(%empty8 : tensor<1x16x8xf32>) {
  ^bb0(%m: f32, %b: f32, %o: f32):
    %sum = arith.addf %m, %b : f32
    linalg.yield %sum : f32
} -> tensor<1x16x8xf32>
```

핵심: matmul 자신은 `linalg.fill(0) → linalg.matmul`이라는, `AMDAIEPadContractionDispatches.cpp`가
가정하는 딱 그 "plain matmul" 형태가 됐고, bias-add는 **별도의 `linalg.generic`**으로 빠져나왔다. 이
분리가 없으면 §5의 AMD-AIE pad/split pass들이 아예 매치를 못 한다.

### §3.3 `GeneralizeLinalgNamedOps` — named op → generic

**Before**: §3.2의 `%matmul_2d` (`linalg.matmul`, named op)

**After**:
```mlir
%matmul_2d = linalg.generic {
  indexing_maps = [affine_map<(n,ohw,f,k) -> (n,ohw,k)>,   // img2col
                   affine_map<(n,ohw,f,k) -> (k,f)>,       // filter
                   affine_map<(n,ohw,f,k) -> (n,ohw,f)>],  // output
  iterator_types = ["parallel","parallel","parallel","reduction"]
} ins(%img2col, %filter_2d : tensor<1x16x27xbf16>, tensor<27x8xbf16>)
  outs(%zero_init : tensor<1x16x8xf32>) {
  ^bb0(%a: bf16, %b: bf16, %acc: f32):
    %a_f32 = arith.extf %a : bf16 to f32
    %b_f32 = arith.extf %b : bf16 to f32
    %mul = arith.mulf %a_f32, %b_f32 : f32
    %sum = arith.addf %acc, %mul : f32
    linalg.yield %sum : f32
} -> tensor<1x16x8xf32>
```

핵심: `linalg.matmul`이라는 **op 이름/identity**가 사라지고 `linalg.generic` + 명시적 `affine_map`/reduction
body로 풀렸다. 이후로는 "이게 matmul이다"라는 정보가 op 이름이 아니라 **구조(indexing map + body가
multiply-accumulate 패턴)**로만 남는다 — §5.1의 `AMDAIEAssignDeviceAffinities`가
`linalg::isaContractionOpInterface` + `hasMultiplyAccumulateBody`로 이 구조를 다시 인식해내는 이유가
바로 이거다(named op이 아니라 구조를 본다).

## §4. Phase 5 — DispatchCreation: dispatch가 실제로 생기는 지점

### §4.1 `FormDispatchRegionsPass` (+ `no-fuse-into-contraction-conv-roots`)

**Before**: §3.3의 `%matmul_2d`(=conv, contraction root) + §3.2의 `%biased_2d`(bias) + relu(§1 형태 그대로).
세 op 모두 아직 top-level 함수 안에 나열된 평범한 linalg 연산.

**플래그 없이(upstream 기본 동작, 참고용 — 이 저장소는 이 경로를 안 씀)**:
```mlir
%region = flow.dispatch.region -> (tensor<1x4x4x8xf32>) {
  %conv = linalg.generic {...} ins(...) outs(%zero_init) {...} -> tensor<1x16x8xf32>   // matmul
  %biased = linalg.generic {...} ins(%conv, %bias) outs(...) {...} -> tensor<1x16x8xf32>
  %relu = linalg.generic {...} ins(%biased) outs(...) {...} -> tensor<1x4x4x8xf32>
  flow.return %relu : tensor<1x4x4x8xf32>
}
```
세 op이 **하나의 dispatch**로 묶임 — 그런데 pack-peel codegen은 이 dispatch를 못 받는다(§11의
"matmul/transpose만 지원" 제약).

**플래그 켜짐(`--iree-dispatch-creation-no-fuse-into-contraction-conv-roots`, 이 저장소의 실제 동작)**:
```mlir
// matmul만 담긴 dispatch — contraction root에는 아무것도 안 붙음
%region0 = flow.dispatch.region -> (tensor<1x16x8xf32>) {
  %matmul = linalg.generic {...} ins(%img2col, %filter_2d) outs(%zero_init) {...} -> tensor<1x16x8xf32>
  flow.return %matmul : tensor<1x16x8xf32>
}

// bias+relu는 서로 fusion돼서(둘 다 elementwise, contraction root가 아니므로 이 플래그의 영향을 안 받음)
// 별도 dispatch 하나로 묶임
%region1 = flow.dispatch.region -> (tensor<1x4x4x8xf32>) {
  %biased = linalg.generic {...} ins(%region0, %bias) outs(...) {...} -> tensor<1x16x8xf32>
  %relu = linalg.generic {...} ins(%biased) outs(...) {...} -> tensor<1x4x4x8xf32>
  flow.return %relu : tensor<1x4x4x8xf32>
}
```

핵심: matmul이 **혼자만 있는** dispatch가 되고, bias+relu는 서로 fusion되어 **matmul과는 분리된** 별도
dispatch가 된다(플래그는 "contraction root에 뭘 붙이는 것"만 막지, elementwise들끼리 fusion되는 건 막지
않음). `%region1`이 §5.1에서 host(CPU)로 배치되는 대상이다.

### §4.2 `ConvertDispatchRegionsToWorkgroupsPass`

**Before**: §4.1의 `%region0`(`flow.dispatch.region`)

**After**:
```mlir
%exe0 = flow.dispatch.workgroups(%img2col, %filter_2d) : (tensor<1x16x27xbf16>, tensor<27x8xbf16>) -> tensor<1x16x8xf32> = (
    %img2col_arg: !flow.dispatch.tensor<readonly:tensor<1x16x27xbf16>>,
    %filter_arg: !flow.dispatch.tensor<readonly:tensor<27x8xbf16>>,
    %out_arg: !flow.dispatch.tensor<writeonly:tensor<1x16x8xf32>>) {
  %img2col_val = flow.dispatch.tensor.load %img2col_arg : !flow.dispatch.tensor<readonly:tensor<1x16x27xbf16>> -> tensor<1x16x27xbf16>
  %filter_val = flow.dispatch.tensor.load %filter_arg : !flow.dispatch.tensor<readonly:tensor<27x8xbf16>> -> tensor<27x8xbf16>
  %zero_init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<1x16x8xf32>) -> tensor<1x16x8xf32>
  %matmul = linalg.generic {...} ins(%img2col_val, %filter_val) outs(%zero_init) {...} -> tensor<1x16x8xf32>
  flow.dispatch.tensor.store %matmul, %out_arg : tensor<1x16x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x16x8xf32>>
  flow.return
}
```

핵심: **암묵적 캡처(region이 바깥 SSA 값을 그냥 참조)가 명시적 `readonly`/`writeonly` 바인딩 + explicit
`load`/`store`로 바뀌었다.** 이게 §6(Flow)에서 `flow.executable`로 outline되는 최종 형태이고, §5의 AMD-AIE
pass들이 실제로 조작하는 IR 구조다(operand/binding을 새로 추가하거나 shape을 바꾸는 게 이 형태 위에서
일어남).

## §5. Phase 6 — Flow: AMD-AIE 3개 pass (§6.2에서 "dispatch 이후"로 분류한 것)

### §5.1 `AMDAIEAssignDeviceAffinities` (2차 호출)

**Before**: §4.2의 `%exe0`(matmul dispatch) + bias/relu dispatch(§4.1의 `%region1`이 workgroups로 변환된 것)

**After**:
```mlir
%exe0 = flow.dispatch.workgroups(...) attributes {stream.affinity = #hal.device.affinity<@device_npu>} : ... { ... }

%exe1 = flow.dispatch.workgroups(...) attributes {stream.affinity = #hal.device.affinity<@device_cpu>} : ... { ... }
```

핵심: `executableIsContractionOrConv`가 `%exe0` 안의 `linalg.generic`을 구조적으로 훑어(§3.3에서 이미
generic으로 풀렸으므로 이름이 아니라 body/indexing-map으로 판별) contraction임을 찾아내 `@device_npu`를,
`%exe1`(bias+relu, contraction 없음)엔 `@device_cpu`를 부여한다.

### §5.2 `AMDAIEPadContractionDispatches` — FC 조각(N=1000)으로 예시

**Before**: FC 레이어의 matmul dispatch. `M=1, K=512, N=1000`, pack-peel 타일 배수를 32라 하면 K=512는
16×32라 나눠떨어지지만 **N=1000은 31.25×32로 안 나눠떨어짐.**

```mlir
%exe_fc = flow.dispatch.workgroups(%x, %w) attributes {stream.affinity = #hal.device.affinity<@device_npu>}
  : (tensor<1x512xbf16>, tensor<512x1000xbf16>) -> tensor<1x1000xf32> = (...) {
  %zero_init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<1x1000xf32>) -> tensor<1x1000xf32>
  %matmul = linalg.generic {...} ins(%x_val, %w_val) outs(%zero_init) {...} -> tensor<1x1000xf32>
  flow.dispatch.tensor.store %matmul, %out_arg : ...
  flow.return
}
```

**After**:
```mlir
// host(CPU) dispatch: N을 1000 -> 1024(32의 배수)로 제로 패딩
%w_padded = flow.dispatch.workgroups(%w) attributes {stream.affinity = #hal.device.affinity<@device_cpu>}
  : (tensor<512x1000xbf16>) -> tensor<512x1024xbf16> = (...) {
  %padded = tensor.pad %w_val low[0, 0] high[0, 24] { ^bb0(...): tensor.yield %zero_bf16 : bf16 } : tensor<512x1000xbf16> to tensor<512x1024xbf16>
  flow.dispatch.tensor.store %padded, %out_arg : ...
  flow.return
}

// NPU dispatch: 실행부가 패딩된 shape(1024)을 보도록 재작성됨
%exe_fc = flow.dispatch.workgroups(%x, %w_padded) attributes {stream.affinity = #hal.device.affinity<@device_npu>}
  : (tensor<1x512xbf16>, tensor<512x1024xbf16>) -> tensor<1x1024xf32> = (...) {
  %zero_init = linalg.fill ins(%c0 : f32) outs(%empty : tensor<1x1024xf32>) -> tensor<1x1024xf32>
  %matmul = linalg.generic {...} ins(%x_val, %w_padded_val) outs(%zero_init) {...} -> tensor<1x1024xf32>
  flow.dispatch.tensor.store %matmul, %out_arg : ...
  flow.return
}

// host(CPU) dispatch: 결과를 다시 1000으로 crop
%result = flow.dispatch.workgroups(%exe_fc) attributes {stream.affinity = #hal.device.affinity<@device_cpu>}
  : (tensor<1x1024xf32>) -> tensor<1x1000xf32> = (...) {
  %cropped = tensor.extract_slice %in[0, 0] [1, 1000] [1, 1] : tensor<1x1024xf32> to tensor<1x1000xf32>
  flow.dispatch.tensor.store %cropped, %out_arg : ...
  flow.return
}
```

핵심: **NPU dispatch 자체는 여전히 "순수 matmul" 형태를 유지**하면서(pack-peel의 제약을 만족), 패딩/크롭은
전부 **별도의 host dispatch**로 처리한다. K(reduction dim)는 제로 패딩이면 결과가 그대로 보존되지만,
N(출력 dim)은 패딩된 만큼 결과에 여분의 열이 생기므로 마지막에 crop이 필요하다 — 그래서 pad는
"앞뒤 dispatch 한 쌍"으로 나타난다.

### §5.3 `AMDAIESplitLargeContractionDispatches`

**Before**: §5.2의 `%exe_fc` (N=1024로 패딩된 matmul, `transpose_b` 형태의 큰 weight)

**After** (N을 4개의 256-chunk로 분할한다고 가정):
```mlir
%w0 = tensor.extract_slice %w_padded[0, 0]   [512, 256] [1, 1] : tensor<512x1024xbf16> to tensor<512x256xbf16>
%w1 = tensor.extract_slice %w_padded[0, 256] [512, 256] [1, 1] : tensor<512x1024xbf16> to tensor<512x256xbf16>
%w2 = tensor.extract_slice %w_padded[0, 512] [512, 256] [1, 1] : tensor<512x1024xbf16> to tensor<512x256xbf16>
%w3 = tensor.extract_slice %w_padded[0, 768] [512, 256] [1, 1] : tensor<512x1024xbf16> to tensor<512x256xbf16>

%r0 = flow.dispatch.workgroups(%x, %w0) attributes {stream.affinity = #hal.device.affinity<@device_npu>}
  : (tensor<1x512xbf16>, tensor<512x256xbf16>) -> tensor<1x256xf32> = (...) { ... }
%r1 = flow.dispatch.workgroups(%x, %w1) ... -> tensor<1x256xf32> = (...) { ... }
%r2 = flow.dispatch.workgroups(%x, %w2) ... -> tensor<1x256xf32> = (...) { ... }
%r3 = flow.dispatch.workgroups(%x, %w3) ... -> tensor<1x256xf32> = (...) { ... }

// host(CPU) dispatch: 4개의 결과를 N 방향으로 다시 이어붙임
%concat = flow.dispatch.workgroups(%r0, %r1, %r2, %r3) attributes {stream.affinity = #hal.device.affinity<@device_cpu>}
  : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x1024xf32> = (...) {
  %c0 = tensor.insert_slice %r0_val into %empty[0, 0]   [1, 256] [1, 1] : tensor<1x256xf32> into tensor<1x1024xf32>
  %c1 = tensor.insert_slice %r1_val into %c0[0, 256]    [1, 256] [1, 1] : tensor<1x256xf32> into tensor<1x1024xf32>
  %c2 = tensor.insert_slice %r2_val into %c1[0, 512]    [1, 256] [1, 1] : tensor<1x256xf32> into tensor<1x1024xf32>
  %c3 = tensor.insert_slice %r3_val into %c2[0, 768]    [1, 256] [1, 1] : tensor<1x256xf32> into tensor<1x1024xf32>
  flow.dispatch.tensor.store %c3, %out_arg : ...
  flow.return
}
```

핵심: weight 하나가 **N 방향으로 4개의 독립 dispatch**로 쪼개진다(각각 별도의 L3→L2 shim DMA를 씀 — 이게
"주소 한계 안에 들어오게" 하는 목적). 4개 결과를 다시 이어붙이는 concat은 §5.2의 crop처럼 host(CPU)
dispatch로 처리된다. 이 뒤에 §5.2의 crop(1024→1000)이 이어 붙는다 — 실제로는 pad → split → (NPU 4개
matmul) → concat → crop 순서로 dispatch 그래프가 만들어진다.

---

## 참고

- `docs/2026-08-16_frontend_lowering_passes.md` — 각 pass의 위치/목적/전체 파이프라인 지도.
- `docs/device_placement.md` — §5.1 device affinity의 설계 상세.
- `docs/2026-07-06_env_setup/DEBUG_PIPELINE.md` — 실제 IR을 뽑아 이 문서의 예시와 대조하는 방법.
- 소스 근거: `ConvertConv2DToImg2Col.cpp`(§3.1), `DetachElementwiseFromNamedOps.cpp`(§3.2),
  `DispatchCreation/FormDispatchRegions.cpp` + `Passes.cpp`(§4.1), `AMDAIEAssignDeviceAffinities.cpp`(§5.1),
  `AMDAIEPadContractionDispatches.cpp`(§5.2, §5.3) — 전부 `third_party/iree/compiler/src/iree/compiler/...`
  또는 `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/...` 아래.
