# 프론트엔드 로워링 패스 조사 (모델 → linalg → dispatch 직전)

VGG-16(ONNX)을 기준으로, `iree-compile` 한 번 실행 안에서 **모델이 linalg-on-tensors로 내려오기까지, 그리고
linalg 레벨에서 dispatch(`flow.dispatch.region`)로 묶이기 직전까지** 어떤 pass들이 어떤 순서로 도는지 정리한다.
목적은 이 구간(우리 팀 책임 범위 = dispatch 이전)에서 프론트엔드 최적화 pass를 새로 추가하거나 기존 pass를
분석할 때 참고할 지도를 만드는 것.

- 조사 기준 커밋: 이 브랜치(`vgg16-onnx`) HEAD, IREE submodule `third_party/iree`.
- 실행 근거: `models/vgg16/README.md`의 컴파일 커맨드, `docs/2026-07-06_env_setup/DEBUG_PIPELINE.md`의 phase
  덤프 목록(`--dump-compilation-phases-to`), `compiler/plugins/target/AMD-AIE/iree-amd-aie/PluginRegistration.cpp`.
- 이종 device(CPU+NPU) 배치 자체의 상세 설계는 이 문서에서 반복하지 않고 `docs/device_placement.md`를 가리킨다.

## 0. 결론 먼저 — "dispatch 이전"의 정확한 경계

`iree-compile`의 전체 파이프라인은 `IREEVMPipelinePhase` enum 순서로 진행되고,
`--dump-compilation-phases-to`가 저장하는 파일명(`N.<phase>.mlir`)이 그대로 이 순서다
(`third_party/iree/compiler/src/iree/compiler/Pipelines/Pipelines.cpp`):

```
1.input                 ─┐
2.abi                    │  ← 이 문서의 "모델 → linalg" 구간
3.preprocessing          │
4.global-optimization   ─┘  ← linalg 레벨. 여기까지 나온 IR은 순수 linalg-on-tensors + util/flow 최소 골격
──────────────────────────────────────────────────────────────
5.dispatch-creation      ← ★ 실제 dispatch region(`flow.dispatch.region`)이 여기서 만들어진다
──────────────────────────────────────────────────────────────
6.flow                   ← dispatch region을 flow.executable로 outline (dispatch "형성"은 이미 끝난 뒤)
7.stream / 8-10.executable-* / 11.hal / 12.vm
```

**중요한 정정**: IREE 구버전에서는 "Flow" 페이즈가 dispatch region 형성 자체를 담당했지만, 지금 이 IREE 버전은
`DispatchCreation`이라는 별도 페이즈로 분리되어 있다(`Pipelines.cpp:374-407`). 즉:

- linalg 레벨 IR이 최종 확정되는 지점은 **4.global-optimization 끝**이다.
- **5.dispatch-creation**이 linalg 연산들을 fuse/tiling 판단해서 `flow.dispatch.region` → `flow.dispatch.workgroups`로
  묶는, 말 그대로 "dispatch를 만드는" 페이즈다. 우리 팀 책임 범위(dispatch 이전)는 **정확히 여기서 끝난다.**
- **6.flow**는 이미 만들어진 dispatch region을 `flow.executable`로 outline하고 이름 붙이는 페이즈일 뿐, 새 dispatch
  경계를 만들지 않는다.

AMD-AIE 플러그인은 두 지점에 훅을 건다(`PluginRegistration.cpp:56-88`):

| 훅 | 실행 시점 | dispatch 형성 기준 |
| --- | --- | --- |
| `extendPreprocessingPassPipeline` | 3.preprocessing 안, **GlobalOptimization보다도 전** | **dispatch 이전** (linalg 이전이기도 함) |
| `extendFlowTransformPassPipeline` | 6.flow 안, IREE 자체 Flow 파이프라인(outlining) **뒤** | **dispatch 형성(5.dispatch-creation) 이후.** 이름은 "Flow"지만 이미 `flow.dispatch`가 존재하는 시점 — per-dispatch affinity, pad, split처럼 "다 만들어진 dispatch를 다시 손보는" 성격의 패스들이다 |

즉 이 저장소에서 실제로 "dispatch 이전"에 걸리는 AMD-AIE 고유 코드는 `extendPreprocessingPassPipeline` 쪽
(bf16 demote, device affinity 조기 주입) 뿐이고, `extendFlowTransformPassPipeline` 쪽(뒤에서 다룸)은 이름과 달리
dispatch 형성 **이후** 구간이라는 점이 향후 pass를 설계할 때 헷갈리기 쉬운 지점이라 먼저 명시한다.

## 1. Phase 1 — Input: ONNX → torch dialect → linalg-on-tensors

여기가 "모델 → linalg"의 핵심 구간이다. 두 단계로 나뉜다.

### 1a. `iree-compile` 밖: ONNX → torch dialect (텍스트 `.mlir` 생성)

`python3 -m iree.compiler.tools.import_onnx models/vgg16/vgg16-12.onnx -o /tmp/vgg.mlir`
(`models/vgg16/README.md:43`)이 `iree-compile` 실행 전에 torch-mlir의 ONNX 임포터를 호출해 ONNX 그래프를
`torch` dialect(주로 `torch.aten.*`, 필요시 `torch.operator`)로 직접 방출한다. `DEBUG_PIPELINE.md:118`이
`1.input.mlir`을 "frontend: torch->linalg+ABI"라고 명시하는 것과 일치 — `iree-compile`에 들어가는 시점에
이미 `torch` dialect라는 뜻.

### 1b. `iree-compile` 안: Input 페이즈

`Pipelines.cpp:92-142`. 순서:

1. `hooks.pipelineExtensions->extendInputConversionPreprocessingPassPipeline(...)` — 플러그인 훅(이 저장소에서는
   AMD-AIE/XDNA-OPLIB 둘 다 오버라이드하지 않음, no-op).
2. Input dialect 자동 감지(`auto_detect`) → `InputConversion::createAutoInputConversionPipelinePass`가 모듈에
   어떤 dialect가 있는지 보고 맞는 입력 파이프라인을 선택. 여기서는 `torch` dialect가 감지되어
   Torch 입력 플러그인의 파이프라인이 선택된다.
3. **Torch → linalg 본체**: `createTorchToIREEPipeline`
   (`third_party/iree/compiler/plugins/input/Torch/InputConversion/Passes.cpp:29-97`). 실질적으로 이 pass
   리스트가 "모델을 linalg로 내리는" pass들이다:
   - `BindSymbolicShapes` / (옵션)`SetStrictSymbolicShapes` — 동적 shape 힌트를 SSA로 구체화.
   - `torch::Torch::RecomposeComplexOps` — 분해되어 있던 복합 연산 재조립.
   - `BitCastTensor`
   - `torch::Torch::ReduceOpVariants` — value/비-value 세만틱 변형들을 정규형으로.
   - `TorchConversion::ConvertCustomQuantOp`
   - (옵션) `torch::Torch::TorchShapeRefinementPipeline` — shape 추론/정제 서브파이프라인.
   - (옵션, 기본 on) `torch::Torch::DecomposeComplexOps` — aten 고수준 연산을 backend-legal 연산 집합으로 분해.
   - `torch::Torch::FuseQuantizedOps`, `Canonicalizer`, `torch::Torch::ScalarizeShapes`
   - `ConvertTorchToTMTensor` → `ConvertTMTensorToLinalgExt`(IREE 자체 확장) — scan/sort/scatter류를 LinalgExt로.
   - `ConvertTorchToTensor` → `ConvertTorchUnstructuredToLinalgExt`
   - **`ConvertTorchToLinalg`** — 실질적인 aten → `linalg.*` (+ `tensor.*`) 로워링 본체. VGG16의 conv/relu/matmul/
     maxpool/flatten이 여기서 `linalg.conv_2d_*`, `linalg.generic`(relu), `linalg.matmul`(Gemm), pooling 등으로 변환.
   - `CSE` → `ConvertTorchToSCF` → `ConvertTorchToArith` → `ConvertTorchConversionToMLProgram`
   - cleanup: `Canonicalizer`, `memref::ResolveShapedTypeResultDims`, `CSE`
   - `Inliner` — torch의 일반 함수 호출은 이 시점에 인라인되어야 함.
   - `FuncConversion` → `SymbolDCE`
   - `TorchConversion::FinalizingBackendTypeConversion` — torch 타입을 linalg-on-tensors backend contract 타입으로 확정.
4. `InputConversion::buildCommonInputConversionPassPipeline` — dialect 무관 공통 정리:
   `SanitizeModuleNames`, `ImportMLProgram`, `IREEImportPublic`, `ConvertPrimitiveType` 등
   (`third_party/iree/compiler/src/iree/compiler/InputConversion/Common/Passes.cpp`).

**이 시점(phase 1 끝)에 이미 IR은 linalg-on-tensors다.** 이후 phase 2(ABI), 3(Preprocessing), 4(GlobalOptimization)는
전부 linalg 레벨 위에서 도는 whole-program 최적화/정규화이지 더 이상 "모델 dialect → linalg" 변환이 아니다.

## 2. Phase 2 — ABI

`IREE::ABI::buildTransformPassPipeline` — entry function에 대해 외부에서 부를 수 있는 wrapper(인자/리턴 marshaling,
동기/비동기 invocation model 선택)를 생성. VGG16 자체의 텐서 연산에는 영향 없음, 함수 시그니처 레벨 처리.

이 직후(하지만 여전히 phase 3 이전) `IREE::HAL::buildHALDeviceAssignmentPassPipeline`이 실행되어
`--iree-hal-target-device=npu=amdxdna` / `cpu=local` 같은 CLI 선언을 모듈에 device global로 박아 넣는다
(`Pipelines.cpp:174-184`). VGG16 레시피가 CPU+NPU 두 device를 선언하는 것이 바로 이 지점의 입력.

## 3. Phase 3 — Preprocessing (여기부터가 "linalg 이후, dispatch 이전"의 시작)

`Preprocessing::buildPreprocessingPassPipeline` (`third_party/iree/compiler/src/iree/compiler/Preprocessing/Passes.cpp:97-115`), 세 단계 우선순위:

1. **커맨드라인 지정 파이프라인** (`buildPreprocessingPassPipelineFromCommandLine`) — 최우선.
   VGG16 레시피의 `--iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-convert-conv-to-channels-last)'`
   가 여기서 실행됨: ONNX conv는 NCHW인데 뒤에서 쓸 im2col 경로는 NHWC를 기대하므로 이 시점에 channels-last로
   변환. (transform-spec/PDL 파일 지정 방식도 있지만 이 레시피는 안 씀.)
2. **플러그인 확장** — `pipelineExtensions->extendPreprocessingPassPipeline(passManager)`. 여기서
   **AMD-AIE 플러그인 고유 pass 두 개**가 들어간다 (`PluginRegistration.cpp:56-71`):
   - (옵션, `--iree-amdaie-demote-contraction-inputs-to-bf16`)
     `GlobalOptimization::createDemoteContractionInputsPass(BF16, All)` — matmul/conv 입력을 f32→bf16으로
     **named op(`linalg.matmul`/`linalg.conv_*`) 상태일 때** 미리 demote한다. 이유가 명시되어 있음: 뒤(4.
     global-optimization)의 `GeneralizeLinalgNamedOps`가 named op을 `linalg.generic`으로 풀어버리면, IREE
     upstream에 이미 있는 동일 기능 demote pass(`GlobalOptimization` 파이프라인 내부, `Passes.cpp:169`)가
     더 이상 이 op들을 매치 못 하기 때문 — 그래서 AMD-AIE는 **더 이른 시점(Preprocessing)**에 선점 실행한다.
     npu4는 f32 벡터 경로가 없어 이 demote가 사실상 필수(VGG16처럼 f32 ONNX 모델을 돌리려면).
   - `AMDAIE::createAMDAIEAssignDeviceAffinitiesPass()` — **1차 호출** (뒤 6.flow에서 2차 호출됨, §5 참고).
     이 시점엔 아직 dispatch가 하나도 없으므로 개별 dispatch에 affinity를 매기진 않고, "CPU device global이
     한 번도 참조 안 되면 나중에 SymbolDCE로 지워질 것"을 막기 위해 device topology(`stream.topology`)를
     조기 주입해 CPU device global을 symbol로 붙잡아 두는 역할만 한다. amd-aie + llvm-cpu 두 device가 모두
     선언된 경우에만 동작(그 외엔 no-op). 상세 정책/테이블은 `docs/device_placement.md` 참고.
3. `createAttrBasedPipelinePass` — 개별 op에 `preprocessing_pipeline` attribute로 박힌 임의 파이프라인 실행(옵션 경로).

XDNA-OPLIB 프리프로세싱 플러그인(`compiler/plugins/preprocessing/XDNA-OPLIB/`)도 같은 훅
(`extendPreprocessingPassPipeline`)을 통해 `XDNAOPLIBHelloWorld` pass 하나를 추가하도록 배선되어 있지만,
현재는 이름 그대로 hello-world 스켈레톤이고 `PluginActivationPolicy::Explicit`이라 명시적으로 켜야 활성화된다
— VGG16 레시피에서는 쓰지 않음. **새 프론트엔드 pass를 추가할 때 이미 준비된 확장 지점**이라는 점만 기억해두면 됨.

## 4. Phase 4 — GlobalOptimization (linalg 레벨 whole-program 최적화, 순수 IREE upstream)

`GlobalOptimization::buildGlobalOptimizationPassPipeline`
(`third_party/iree/compiler/src/iree/compiler/GlobalOptimization/Passes.cpp:99-`). AMD-AIE 플러그인이 직접
끼어드는 지점은 없고(전부 upstream pass), VGG16 레시피 플래그 다수가 이 페이즈의 옵션이다. 주요 pass(순서대로):

- (옵션) `ImportParametersPass` — 외부 파라미터 파일 inline.
- `WarnOnUninitializedValues`(옵션) / `StripDebugOps`(옵션) / `OptimizeIntArithmetic`
- `LinalgQuantizedConvToConv` / `LinalgQuantizedMatmulToMatmul`
- **(`--iree-global-opt-use-im2col-for-convs`) `ConvertConv2DToImg2Col`** — VGG16이 NPU에 도달하는 유일한 conv
  경로. `linalg.conv_2d_*`를 im2col(`tensor.extract_slice`/reshape 등) + `linalg.matmul`로 변환. NPU는 표준
  NCHW conv codegen을 지원하지 않으므로 이 pass가 사실상 필수.
- `Canonicalize` / `RemoveZeroExtentTensors` / `DetachElementwiseFromNamedOps` / `SimplifyDepthwiseConv`
- `EraseUnusedLinalgOperands`
- `ExpandTensorShapes` — 텐서 shape을 SSA 값으로 펼쳐 전체 프로그램 단위로 shape 동치를 더 잘 잡음(뒤 fusion 품질에 영향).
- `ConvertElementwiseToLinalg` / `RaiseSpecialOps` / `DecomposeConcat` / `GeneralizeLinalgNamedOps`
  (여기서 named conv/matmul이 `linalg.generic`으로 풀림 — §3의 AMD-AIE demote가 이보다 먼저 도는 이유가 이 지점)
  / (옵션) `InsertTensorBarriers`
- `FoldUnitExtentDims` → `FoldReshapesIntoTensorBarriers`
- **upstream `DemoteContractionInputsPass`** (기본 옵션은 no-op에 가까움; VGG16은 AMD-AIE 쪽에서 이미 먼저 처리했으므로
  이 자리에서 다시 손댈 대상이 없음) → (옵션) `FuseDequantizationMatmul`
- (옵션, 기본 on) `PropagateLinalgTransposePass` — transpose를 그래프 상에서 밀어올리는 pass. VGG16 레시피는 별도
  지정 안 하지만 관련 GlobalOptimizationOptions(`aggressiveTransposePropagation` 등)이 여기로 연결됨.
- `ConvertStridedContractionToContractionPass`
- (데이터 타일링 켜진 경우만) `AnnotateDataTilingHints` → `SetEncoding` → `MaterializeHomogeneousEncodings` →
  `SimplifyPackUnpack` → `DataLayoutPropagation` — VGG16은 이 경로를 안 씀(NPU pack-peel 코드젠이 encoding
  기반이 아니라 뒤(dispatch-creation) 단계 전용 옵션으로 처리됨).
- `GeneralizeLinalgNamedOps`(2차) / `GlobalLoopInvariantCodeMotion` / cleanup / `SimplifyGlobalAccesses` /
  `ApplyPatterns` / `FoldGlobals` / `IPO`
- (옵션) `constExprHoisting` → `buildGlobalOptExprHoistingPassPipeline` — 상수 하위 표현식을 initializer로 호이스팅.
- (옵션) const-eval 서브파이프라인 — JIT으로 상수 폴딩.
- (옵션, `numericPrecisionReduction`) `InferNumericNarrowingPass` → `OptimizeNumericsPass` →
  `CleanupNumericNarrowingPass` — VGG16은 안 씀(비트폭 좁히기 분석/최적화, bf16 demote와는 다른 경로).
- **(무조건 실행)** `Canonicalize` → `CSE` → `RaiseSpecialOpsPass`(2차) — const-eval 이후 새로 생긴 raise
  기회를 한 번 더 훑는다. **이전 버전 문서에 누락됐던 부분** — 이 페이즈에서 항상 도는 pass 중 하나.
- (옵션) `ExportParametersPass` — 상수를 parameter archive로 분리 export. VGG16은 안 씀.
- (옵션) `GenerateSplatParameterArchivePass` — 상수를 splat(0으로 채운) archive로 대체. VGG16은 안 씀.

> **이 절의 스코프에 대한 정정**: 위 목록은 `buildGlobalOptimizationPassPipeline` 함수 전체
> (`Passes.cpp:99-290`)를 순서대로 옮긴 것이고, 순서 자체는 실제 실행 순서와 동일하다. 다만 `(옵션)`으로
> 표시된 항목들은 **"파이프라인 구조상 존재하는 pass"**이지 **"VGG16 레시피가 실제로 켜는 pass"**가
> 아니다 — 이 문서는 파이프라인 지도이고, VGG16이 실제로 밟는 경로는 그 지도 위의 부분집합이다. VGG16
> 레시피 플래그로 실제 활성화되는 건 `useIm2colForConvs`(§4 본문의 `ConvertConv2DToImg2Col`)와 fork
> 전용 `detach-elementwise-through-reshape` 둘뿐이고, `dataTiling`/`constEval`/`numericPrecisionReduction`/
> `parameterExport`/`parameterSplat` 등은 전부 꺼진 채로 지나간다. `(무조건 실행)` 표시가 없는 항목은
> 전부 이런 조건부다.

`--iree-global-opt-detach-elementwise-through-reshape`(VGG16 레시피가 쓰는 fork 전용 플래그, `.gitmodules`의
`ace-knu/iree`)는 이 페이즈의 `DetachElementwiseFromNamedOps`류 로직을 im2col reshape 너머까지 통과시켜, conv의
bias-add가 im2col reshape에 가려 fusion 대상에서 누락되지 않게 하는 변형이다. 정확한 위치는 upstream
`GlobalOptimization` 소스에 fork가 patch를 얹은 형태이므로, 상세 diff가 필요하면
`git log --oneline -- third_party/iree`에서 이 플래그 도입 커밋을 확인할 것.

**이 페이즈 끝(`4.global-optimization.mlir`)이 "linalg 레벨 IR"의 최종 확정 지점**이다 — 이 뒤로는 더 이상
whole-program 형태의 linalg 변환이 아니라, "이 linalg 연산들을 몇 개의 dispatch로 어떻게 묶을 것인가"라는
전혀 다른 질문(§5)으로 넘어간다.

## 5. Phase 5 — DispatchCreation: 여기서 "dispatch"가 실제로 생긴다

`DispatchCreation::buildDispatchCreationPassPipeline`
(`third_party/iree/compiler/src/iree/compiler/DispatchCreation/Passes.cpp`). 우리 팀 책임 범위의 끝이자, "dispatch
이전"이라는 경계선이 그어지는 페이즈. 요약만 남긴다(상세 조사는 이 페이즈부터가 이미 dispatch 형성 로직 그 자체라
이 문서의 범위 밖):

- `InjectTensorTracingPass` → (조건부) `TensorPadToTensorInsertSlicePass`
- IPO 고정점 반복(`FixedPointIteratorPass` 안에서 cleanup)
- `FusionPreprocessingPass` → **`addDispatchRegionCreationPreprocessingPasses`**: elementwise fusion, reshape
  bubble-up/sink, split-reduction, `TransposeGenericOps`, `PropagateEncodings`, const-expr hoisting
- **`addDispatchRegionCreationPasses`**: `FormScalarDispatchesPass` → **`FormDispatchRegionsPass`**(진짜 dispatch
  region 생성 지점) → intra-dispatch elementwise fusion → `CloneProducersIntoDispatchRegions` →
  `CollapseDimensionsPass` → (데이터 타일링 옵션 시) `SetEncodingPass` 등 → `RemoveTensorBarriers`
- `ConvertDispatchRegionsToWorkgroupsPass` → `ConvertTensorToFlowPass` → workgroup count region 생성

VGG16 레시피의 `--iree-dispatch-creation-no-fuse-into-contraction-conv-roots`가 여기서 소비된다
(`FormDispatchRegionsPass`의 옵션) — amd-aie codegen이 elementwise가 융합된 contraction dispatch를 아직 타일링
못 하므로, contraction/conv를 dispatch root로 삼을 때 producer/consumer fusion을 강제로 끈다.

## 6. Phase 6 — Flow: "이름은 Flow지만 dispatch는 이미 다 만들어진 뒤" (AMD-AIE 2차 훅)

`IREE::Flow::buildFlowTransformPassPipeline`
(`third_party/iree/compiler/compiler/src/iree/compiler/Dialect/Flow/Transforms/Passes.cpp:131-`)는 이미 만들어진
dispatch region을 `flow.executable`로 outline(`OutlineDispatchRegionsPass`), 주석 달기(`AnnotateDispatchesPass`),
중복 제거(`DeduplicateExecutablesPass`, VGG16은 `--iree-flow-enable-executable-deduplication=false`로 끔 — 상수
weight offset만 다른 두 conv를 합치면 그 offset이 런타임 push constant가 되는데, `amdaie.npu.address_patch`는
static offset만 지원하므로 dedup을 꺼야 함), 그리고(`Passes.cpp:237-276`, **이전 버전 문서에 누락됐던 뒷부분**):

- IPO 고정점 안에서 `OutlineConstantsPass`(상수를 global로 outline) + cleanup
- **(무조건 실행)** `flow.executable` 내부에 nested로 `Canonicalize` → `CSE`
- (옵션) `ReplicateGlobalsPerAffinityPass` — affinity별로 global을 복제. VGG16은 안 씀.
- **(무조건 실행)** `SymbolDCEPass` — 더 이상 안 쓰는 심볼 정리
- (옵션, 디버그용) `DumpDispatchGraphPass` — Graphviz로 dispatch 그래프 출력

등을 수행한다. (§4와 마찬가지로 `(옵션)` 표시는 파이프라인 구조상 존재할 뿐 VGG16이 실제로 켜는 건
아니라는 뜻이고, `(무조건 실행)` 표시가 없는 나머지 항목도 대부분 무조건 실행되는 cleanup류다 — VGG16이
끄는 건 명시적으로 언급된 dedup 하나뿐이다.)

이 IREE 자체 파이프라인이 끝난 **직후**, AMD-AIE 플러그인의 `extendFlowTransformPassPipeline`이 3개 pass를
추가한다 (`PluginRegistration.cpp:73-88`, 상세 설계는 `docs/device_placement.md`):

1. `AMDAIEAssignDeviceAffinitiesPass` — **2차 호출**. 이번엔 실제 `flow.dispatch`가 다 존재하므로, contraction/
   conv dispatch는 amd-aie(NPU) device에, 그 외(transpose/cast/pooling/flatten 등)는 llvm-cpu device에
   `stream.affinity`를 부여 — VGG16이 "conv/matmul은 NPU, pooling/flatten/layout은 CPU"로 이종 실행되는
   근거가 여기.
2. `AMDAIEPadContractionDispatchesPass` — affinity가 확정된 뒤, NPU에 배정된 순수 matmul dispatch의 M/K가
   타겟 pack-peel 타일 배수가 아니면 host 쪽에 `tensor.pad` dispatch를 만들어 제로 패딩하고, 실행부는 패딩된
   shape로 재작성한다. N(출력 채널)이 안 나눠떨어지는 경우(예: VGG16의 1000-class FC)는 계산 뒤 host
   `tensor.extract_slice` dispatch로 잘라낸다.
3. `AMDAIESplitLargeContractionDispatchesPass` — 위에서 패딩된 large-(K,N) transpose_b matmul을 N-청크로 쪼개
   host에서 concat — 큰 weight 하나를 L3→L2로 옮기는 shim DMA가 주소 한계를 넘어 stall하는 것을 막기 위함.

이 세 pass는 전부 **"dispatch가 이미 형성된 뒤 그 dispatch를 재작성/분할"**하는 성격이라, 엄밀히 말하면
"dispatch 이전" 최적화가 아니라 **dispatch 형성 직후 pass**다. 이름이 `extendFlowTransformPassPipeline`이라
"Flow 단계 확장"처럼 보이지만, 이 시점의 Flow 페이즈 자체가 이미 dispatch-creation 다음이라는 점(§0)을
놓치면 "dispatch 이전 vs 이후" 경계를 착각하기 쉽다.

### 6.1 정확한 실행 순서 (phase 6 안에서)

`PluginRegistration.cpp:73-88`에 `addPass`로 나열된 순서 그대로, 다음이 IREE 자체 Flow 파이프라인(outline →
annotate → dedup) **바로 뒤에** 붙는다:

```
... (IREE 자체 Flow pipeline: outline/annotate/dedup 등) ...
→ AMDAIEAssignDeviceAffinitiesPass        (2차 호출, per-dispatch affinity 확정)
→ AMDAIEPadContractionDispatchesPass      (affinity로 타겟 알아낸 뒤 M/K/N 패딩)
→ AMDAIESplitLargeContractionDispatchesPass (패딩된 large-(K,N) matmul을 N-분할)
→ (phase 6 끝, 7.stream으로 진행)
```

셋의 순서 자체가 의존관계다 — Pad는 "이 dispatch가 NPU에 배정됐는지"를 알아야 타일 배수를 판단하므로 반드시
AssignDeviceAffinities 뒤에 와야 하고, Split은 Pad가 만든(패딩된) 결과물을 다시 쪼개는 것이므로 Pad 뒤에 와야 한다.
그래서 순서를 바꿔서 부를 수 없고, 같은 `Passes.td` 파일(`AMDAIEPadContractionDispatches.cpp`)에 두 pass가
나란히 구현돼 있다.

### 6.2 그래서 이 세 pass는 프론트엔드인가?

**아니다.** 이 문서가 정의하는 "프론트엔드"(모델 → linalg, 그리고 dispatch 형성 이전의 linalg-level 최적화)
기준으로 보면 이 셋은 명백히 그 경계 **밖**이다:

- 입력도 출력도 linalg가 아니라 **이미 outline된 `flow.executable` + `flow.dispatch`**다. linalg 연산을 보거나
  고치는 게 아니라 "dispatch 단위"를 보고 고친다(어느 device에 배정할지, operand shape을 얼마나 패딩할지, 몇 개
  sub-dispatch로 쪼갤지).
- 목적도 프론트엔드 최적화(연산 재배치, 융합, 상수 폴딩 같은 것)가 아니라 **NPU 타겟의 codegen/DMA 제약을 우회**하는
  것이다 — pack-peel 타일 배수, shim DMA 주소 한계, CPU/NPU 이종 실행 같은 전적으로 backend/target-specific한 이유.
- 같은 `AMDAIEAssignDeviceAffinitiesPass`라도 §3에서 다룬 **1차 호출(Preprocessing)**은 아직 dispatch가 없는
  시점이라 topology 주입만 하는 사실상 no-op에 가까운 준비 작업이고, 여기 **2차 호출(Flow)**이 실제 배치 결정을
  내리는 본체다 — 같은 pass인데 호출 위치에 따라 "frontend 쪽 준비물" 대 "post-dispatch 본 작업"으로 역할이
  갈린다는 점이 미묘한 부분.

즉 "linalg 최적화 pass를 새로 만든다"는 관점에서는 이 셋을 참고할 필요가 없고, 참고해야 할 건 §3(Preprocessing
훅)·§4(GlobalOptimization) 쪽이다. 반대로 "dispatch가 만들어진 뒤 NPU 타겟에 맞게 손보는 pass"를 만든다면
이 세 개가 정확히 그 자리(§6, `extendFlowTransformPassPipeline`)의 실제 선례다.

## 7. 요약 표

| Phase (dump 번호) | 무엇을 하나 | AMD-AIE 플러그인 개입 | dispatch 기준 | 프론트엔드(모델→linalg) 해당? |
| --- | --- | --- | --- | --- |
| 1. input | ONNX(torch dialect) → linalg-on-tensors (`createTorchToIREEPipeline`) | 없음 | 이전 | 예 (핵심 그 자체) |
| 2. abi | entry function ABI wrapper, HAL device global 주입 | 없음 | 이전 | 아니오 (함수 시그니처 레벨) |
| 3. preprocessing | 사용자 지정 파이프라인 + plugin 확장 | **있음**: bf16 demote(named op 상태에서 선점), device affinity 조기 주입(topology only) | 이전 | 예 (linalg-level 준비) |
| 4. global-optimization | linalg 레벨 whole-program 최적화(im2col, transpose 전파, const hoisting 등) | 없음 | 이전 | 예 |
| **5. dispatch-creation** | **`flow.dispatch.region`/`workgroups` 형성** | 없음 | **= dispatch 형성 그 자체** | 경계선 (linalg의 끝) |
| 6. flow | dispatch region → `flow.executable` outline, dedup | **있음**: per-dispatch affinity(2차), pad, split | **이후** | **아니오** — dispatch/backend 레벨 (§6.2) |

새 pass 3개(`AMDAIEAssignDeviceAffinities`/`AMDAIEPadContractionDispatches`/`AMDAIESplitLargeContractionDispatches`)와
IREE fork에 추가된 옵션 3개(`iree-dispatch-creation-no-fuse-into-contraction-conv-roots`,
`iree-global-opt-detach-elementwise-through-reshape`, `iree-flow-enable-executable-deduplication`)는 모두
Junho Kwak이 VGG16 지원을 위해 2026-07-30~08-05 사이에 추가한 것(`git log --author="Junho Kwak"` 확인).
이 중 GlobalOptimization/DispatchCreation의 옵션 2개만 "linalg 레벨(4번 phase)에서 도는" 진짜 프론트엔드
변경이고, Flow의 세 pass와 executable-dedup 옵션은 위 표처럼 dispatch 형성 이후 구간이다.

## 8. 향후 프론트엔드 pass 작업에 대한 함의

- **새 whole-program/linalg 최적화 pass**를 추가하려면 자연스러운 위치는 3.preprocessing(`extendPreprocessingPassPipeline`,
  이미 AMD-AIE 세션이 쓰고 있는 훅) 또는 XDNA-OPLIB 프리프로세싱 플러그인(현재 hello-world 스켈레톤, `Explicit`
  activation이라 명시적으로 켜야 함) — 둘 다 GlobalOptimization보다 먼저 돌므로 named op(`linalg.matmul`/
  `linalg.conv_*`) 형태가 아직 살아있는 채로 패턴 매치할 수 있다는 게 핵심 이점(§3의 bf16 demote가 그 예시).
- GlobalOptimization 이후(및 DispatchCreation 도중)에는 `GeneralizeLinalgNamedOps`가 이미 여러 번 돌아 대부분
  `linalg.generic`으로 풀려 있으므로, named-op 패턴 매칭이 필요한 분석/변환은 그보다 먼저 끝내야 한다.
  반대로 "융합 이후의 실제 fusion 그룹"을 분석하고 싶다면 5.dispatch-creation 이후 IR(`6.flow.mlir` 이후)을 봐야 한다.
- dispatch 단위 사후 교정(패딩, split, per-dispatch affinity)이 필요한 작업은 `extendFlowTransformPassPipeline`
  자리(6.flow, dispatch 형성 이후)가 이미 확립된 패턴이며 AMDAIEPadContractionDispatches/
  AMDAIESplitLargeContractionDispatches가 실제 예시로 존재한다.
- `scripts/debug/pipeline_dump.py` + `--from-phase`/`--to-phase`(`DEBUG_PIPELINE.md`)로 특정 구간만 재실행하며
  이 문서의 각 phase 경계에서 IR을 직접 비교해볼 수 있다. 새 pass 하나를 추가/수정할 때는 `--pass <pass-cli-name>`으로
  before/after IR을 바로 뽑아 확인하는 워크플로우를 그대로 쓰면 된다.

## 9. 이 변경들은 누가, 왜, "새 pass" 대 "옵션"으로 나누면 어떻게 되나

`git log`로 확인한 사실. upstream(nod-ai/iree-amd-aie)의 마지막 커밋은 2026-06-04(`fddfec1` 등)이고,
2026-07-03(`6338b4b`)부터 이 브랜치의 모든 amd-aie 관련 커밋은 **Junho Kwak**(`junho7513@gmail.com`) 한
사람이 작성했다 — VGG16(과 그 전 MLP 검증 모델)을 붙이기 위한 작업 전체가 이 사람의 것. 원본
iree-amd-aie는 이 시점까지 matmul 위주였고, f32 conv 기반 CNN을 돌리려는 시도가 아래 변경들을 만들었다.

핵심 커밋:

| 날짜 | 커밋 | 내용 |
| --- | --- | --- |
| 07-30 | `53a63bd` | bf16 demote + `AMDAIEAssignDeviceAffinities` 신설 (matmul/conv 첫 성공) |
| 08-03 | `563946e` | `AMDAIEPadContractionDispatches` 신설 (M/K 패딩) |
| 08-04 | `1aa7295` | IREE fork에 `no-fuse-into-contraction-conv-roots` + `detach-elementwise-through-reshape` 옵션 추가 → im2col conv e2e 성공 |
| 08-05 | `e53589e`, `b7dfc0d` | N-split pass 추가 + `executable-deduplication` 옵션 추가 → VGG16 전체 컴파일 성공 |
| 08-06 | `0087a63` | 실험적 workaround 제거 후에도 vgg16-12.onnx corr 1.00000 확인 |

**"pass가 새로 생긴 것"과 "기존 pass에 옵션만 추가된 것"은 다르다** — 실제 소스를 diff해서 구분한 결과:

| 변경 | 분류 | 근거 |
| --- | --- | --- |
| `AMDAIEAssignDeviceAffinities` | **신규 pass** | `git log --diff-filter=A` → `53a63bd`에서 파일 자체가 처음 생김 |
| `AMDAIEPadContractionDispatches` | **신규 pass** | 〃 → `563946e`에서 처음 생김 |
| `AMDAIESplitLargeContractionDispatches` | **신규 pass** | 같은 파일(`AMDAIEPadContractionDispatches.cpp`)에 `e53589e`가 추가 |
| bf16 demote 조기 호출(Preprocessing 훅) | 기존 pass 재사용, **호출 위치·옵션만 변경** | `createDemoteContractionInputsPass`는 GlobalOptimization에 원래 있던 pass(`Passes.cpp:169`); 더 이른 시점에 `BF16,All`로 한 번 더 부른 것뿐 |
| `iree-dispatch-creation-no-fuse-into-contraction-conv-roots` | 기존 pass(`FormDispatchRegionsPass`)에 **새 cl::opt 추가** | `DispatchCreation/Passes.cpp:42-43`, 기존 pass 안에 조건문 하나 추가 |
| `iree-global-opt-detach-elementwise-through-reshape` | 기존 pass(`DetachElementwiseFromNamedOps`)에 **새 cl::opt 추가** | `GlobalOptimization/DetachElementwiseFromNamedOps.cpp:37-38`, 주석: "Off by default so other backends see no change" |
| `iree-flow-enable-executable-deduplication` | 기존 pass(`DeduplicateExecutablesPass`)에 **새 cl::opt 추가** | `Flow/Transforms/Passes.cpp:35-36` |
| `iree-global-opt-use-im2col-for-convs` | **아무것도 새로 안 만듦** — 원래 있던 upstream 옵션을 처음 켠 것 | AIE가 지금까지 conv를 안 다뤄서 아무도 켤 필요가 없었을 뿐 |

즉 "matmul만 되던" 플러그인을 conv 기반 VGG16까지 확장하는 데 **진짜 새 코드(새 pass)가 필요했던 건
device 배치/패딩/split 세 가지뿐**이고, 나머지는 IREE가 이미 갖고 있던 pass들의 옵션을 새로 노출하거나
호출 위치를 바꿔서 해결했다.

## 10. 프론트엔드 쪽에서 할 수 있는 일 — "안 건드리는 게 베스트"는 대체로 맞다

지금까지 나온 실제 문제들(no-fuse 워크어라운드, bf16 벡터화 실패, dispatch 패딩/split, §11의 codegen
단일-root-op 가정)은 전부 pack-peel 타일 크기·AIE 벡터 ISA·shim DMA 주소 한계처럼 **backend/codegen이
알아야 하는 정보**다. 이런 걸 프론트엔드(linalg 레벨, §3·§4)에서 손대면 백엔드 지식을 프론트엔드에
하드코딩하는 꼴이 되고, `models/vgg16/README.md`의 "Known limitations" 섹션도 "실제 해결책은 codegen
쪽"이라고 스스로 인정하고 있다.

다만 리스크 낮고 값어치 있는 프론트엔드 작업 두 가지는 있다:

1. **fail-fast 진단 pass**: 지금은 필수 플래그 하나 빠뜨리면 "컴파일 중간 크래시" 또는 런타임
   `ert state 6`로만 드러난다고 README 스스로 한계로 적어뒀다(§9의 워크어라운드 조합이 통째로 암묵적
   전제 — 하나라도 빠지면 사람이 읽을 진단이 없음). 3.preprocessing 단계에서 "conv가 있는데
   im2col/channels-last/bf16-demote 조합이 안 갖춰졌다" 같은 전제조건을 검증해 사람이 읽을 에러로
   바꾸는 pass — 정확성엔 손 안 대고 진단만 하므로 리스크가 거의 없다.
2. **`device_placement.md`의 미래 확장(§"미래: 지원연산 + 성능 기반 배치 정책")이 필요로 하는
   supportedOps 분석**: "이 연산이 이 accelerator에서 codegen 가능한가"를 linalg 레벨에서 미리
   분류해두면, 지금 front(첫 device 무조건 선택)로 되어 있는 배치 정책이 나중에 실제 지원 연산
   기반으로 넘어갈 때 그 입력이 된다.

이 둘 다 "고치는" 게 아니라 "관찰/검증"이라 안전한 축이다. 반대로 fusion·레이아웃 결정을 프론트엔드에서
미리 손대는 건 지금 아키텍처(§6의 pad/split이 dispatch 이후에 처리)와 중복되거나 충돌할 여지가 크다.

## 11. 실제 codegen 병목: pack-peel의 "dispatch = 단일 root op" 가정

§8의 결론(no-fuse 플래그가 프론트엔드/dispatch-creation 문제가 아니라 codegen 문제)을 소스 레벨로
파고든 결과.

**지금 실제로 벌어지는 일**: `--iree-dispatch-creation-no-fuse-into-contraction-conv-roots`는
"conv-bias-relu가 일단 dispatch로 묶이는데 그게 무의미하다"가 아니라, **애초에 묶이지 못하게 강제로
끈 것**이다(`FormDispatchRegionsPassOptions`에 그대로 전달됨, §5). 그 결과 bias-add/relu는 conv/matmul과
**별도의 `flow.dispatch`**로 남는다. 그런데 `AMDAIEAssignDeviceAffinities.cpp`의 배치 판정 함수는:

```cpp
static bool executableIsContractionOrConv(IREE::Flow::ExecutableOp exe) {
  ...
  innerModule.walk([&](linalg::LinalgOp linalgOp) {
    if (linalg::isaContractionOpInterface(linalgOp) || ...) found = true; ...
  });
  return found;   // dispatch 안에 contraction/conv op가 "하나라도" 있는가만 봄
}
```

dispatch **전체**가 순수 matmul인지가 아니라 "contraction/conv op가 하나라도 포함됐는가"만 검사한다.
그래서 bias-add/relu만 남은 dispatch는 이 조건에서 `false`가 나와 **host(CPU)로 배치된다.** 즉 지금
VGG16은 conv(matmul)만 NPU에서 돌고, 13개 conv 레이어의 bias-add + relu(26개 연산)는 전부 CPU를
왕복하고 있다 — "dispatch가 의미없다" 수준을 넘어 실질적 성능 비용이다.

**codegen이 고쳐지면 프론트엔드는 할 일이 없다**: pack-peel이 "matmul + epilogue(bias+relu)"를 하나의
dispatch로 받아들이게 되면, `no-fuse` 플래그를 끄기만 해도 DispatchCreation의 **기존** fusion 로직이
알아서 conv/matmul + bias + relu를 한 dispatch로 묶고, 위 `executableIsContractionOrConv`도 "그 안에
matmul이 있다"로 그대로 NPU 배치를 내린다 — 코드 수정 지점이 없다. im2col을 쓰는 것도 이 방향에
유리하다: conv를 matmul로 내리면 codegen이 풀 문제가 "conv 반복 공간에 epilogue 융합"이 아니라
"matmul(GEMM) + epilogue 융합"이 되는데, 후자는 GEMM 커널 어디서나 쓰는 표준 패턴이라 훨씬 다루기
쉽다. im2col이든 decompose든 결국 named matmul/conv로 내려와야 dispatch-creation의 contraction-root
인식이 작동하므로, 프론트엔드가 고를 수 있는 레버는 실질적으로 없다.

**병목이 codegen에 있다는 구체적 근거** (`compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/`):

- `KernelDispatch.cpp`의 `getRootOperation(computeOps)` / `setRootConfigImpl` — dispatch 안의 여러
  compute op 중 **정확히 하나의 root op**만 골라 그 op 기준으로 타일링 설정(`setRootConfigForPackPeelPipeline`
  등)을 만든다. 다중-op dispatch를 위한 일반화된 경로가 없다.
- `AMDAIETile.cpp:100` — `// Currently matmul and transpose op are the only ones supported. If its
  not matmul then it is a transpose.` operand 쪽 pack/tile 로직이 이 두 케이스로 하드코딩됨.
- `KernelDispatch.cpp` 253번 줄 근처의 `isMatmulWithElementwiseConsumer` — "matmul + elementwise
  consumer **1개**"까지는 buffer depth 계산에서 이미 고려된 흔적이 있다. 즉 완전히 처음부터 막혀있는
  게 아니라 **부분적으로 시도되다 만 상태**로 보이고, bias+relu처럼 체인된 2개 이상의 epilogue나
  im2col+PackPeel 조합에 대해서는 검증/완성이 안 됐다.
- `AMDAIEPadContractionDispatches.cpp:650` — `// They were validated on exactly one dispatch:
  VGG-16's dense0` — 패딩/split pass 스스로 "VGG-16의 dense0 레이어 하나로만 검증됨"이라고 적어뒀다.

**결론**: 병목은 프론트엔드가 아니라 pack-peel codegen의 "dispatch = 단일 root op" 가정
(`AMDAIETile.cpp`, `KernelDispatch.cpp`의 root-config 선택, 그리고 그 뒤의 `AMDAIETileAndFuse.cpp` 등)에
있다. 이게 풀리기 전까지는 프론트엔드/dispatch-creation 로직이 이미 구조적으로 옳게 동작할 준비가
되어 있으므로(§10), 프론트엔드를 먼저 손대는 것은 실익이 없다 — `models/vgg16/README.md`의 자체 평가
("stopgap", "should disappear as those are fixed")와도 일치한다. 이 codegen 작업을 실제로 시작한다면
`AMDAIETile.cpp` → `KernelDispatch.cpp`(root op 선택) → `AMDAIETileAndFuse.cpp` 순으로 보는 게 자연스럽다.

## 12. Pass 파일 vs 유틸리티 코드 구분법

이 디렉터리(`Transforms/`) 안의 `.cpp` 파일이 실제 MLIR pass인지, 아니면 pass들이 갖다 쓰는 그냥
라이브러리 코드인지는 다음 신호로 구분한다:

| 신호 | 의미 |
| --- | --- |
| `class XxxPass : public impl::XxxBase<XxxPass>` 가 있다 | `.td`로 정의된 진짜 pass. `impl::XxxBase`는 `Passes.td`의 `def Xxx : Pass<"...">`를 빌드 시점에 `mlir-tblgen`이 읽어 자동 생성한 boilerplate(pass 이름, CLI 옵션 바인딩 등)이고, `Passes.td`에 반드시 짝이 있다 |
| `createXxxPass()` 팩토리 함수 + `Passes.h` 선언 | 그 pass를 pipeline에서 부를 때 쓰는 생성 함수 |
| 위 둘이 없고 `namespace { LogicalResult foo(...) {...} }` 형태의 자유 함수만 있다 | pass가 **아님** — 다른 pass가 호출해 쓰는 유틸리티/라이브러리 코드 |

이 문서·대화에서 다룬 파일들로 확인한 결과:

| 파일 | 분류 |
| --- | --- |
| `AMDAIEAssignDeviceAffinities.cpp`, `AMDAIEPadContractionDispatches.cpp`(Split 포함), `AMDAIETile.cpp`, `AMDAIELowerExecutableTarget.cpp` | `.td`로 정의된 pass (`Passes.td`에 `def Xxx : Pass<...>` 있음) |
| `KernelDispatch.cpp`/`.h` | pass 아님 — `AMDAIELowerExecutableTargetPass::runOnOperation()`이 호출하는 유틸리티(root op 선택, 타일 크기 계산 등 순수 로직 모음) |

## 13. 참고 파일

- `third_party/iree/compiler/src/iree/compiler/Pipelines/Pipelines.cpp` — 전체 phase 오케스트레이션.
- `third_party/iree/compiler/plugins/input/Torch/InputConversion/Passes.cpp` — torch → linalg 본체.
- `third_party/iree/compiler/src/iree/compiler/Preprocessing/Passes.cpp`,
  `third_party/iree/compiler/src/iree/compiler/GlobalOptimization/Passes.cpp`,
  `third_party/iree/compiler/src/iree/compiler/DispatchCreation/Passes.cpp`,
  `third_party/iree/compiler/src/iree/compiler/Dialect/Flow/Transforms/Passes.cpp` — 각 phase 본체.
- `compiler/plugins/target/AMD-AIE/iree-amd-aie/PluginRegistration.cpp` — AMD-AIE의 두 파이프라인 훅.
- `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEAssignDeviceAffinities.cpp`,
  `AMDAIEPadContractionDispatches.cpp`(Split 포함) — §6·§9의 신규 pass 본체.
- `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/KernelDispatch.cpp`,
  `AMDAIETile.cpp`, `AMDAIELowerExecutableTarget.cpp` — §11·§12에서 다룬 pack-peel codegen 병목/구조.
- `compiler/plugins/preprocessing/XDNA-OPLIB/` — preprocessing 확장 플러그인 스켈레톤(현재 hello-world).
- `docs/device_placement.md` — device affinity/topology 설계 상세(이 문서 §3, §6의 배경).
- `docs/2026-07-06_env_setup/DEBUG_PIPELINE.md` — phase별 IR 덤프/재개 도구.
- `models/vgg16/README.md` — 이 문서에서 언급한 모든 컴파일 플래그의 출처와 이유.
- `git log --author="Junho Kwak"` (특히 `53a63bd`, `563946e`, `1aa7295`, `e53589e`, `b7dfc0d`, `0087a63`) —
  §9의 타임라인/분류 근거.
