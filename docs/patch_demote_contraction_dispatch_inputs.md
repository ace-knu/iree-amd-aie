# 패치 노트: 장치 배정 후 장치별 contraction 입력 타입 변환

브랜치: `dev` (`786d1d19`) 위 작업 트리, **아직 커밋하지 않음**

장치 배정(device placement)이 끝난 뒤, 각 contraction(matmul/conv) dispatch의 입력을 **그 dispatch가 배정된 장치가 원하는 타입**으로 바꾸는 Flow 단계 패스 `AMDAIEDemoteContractionDispatchInputs`(파일 `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEDemoteContractionDispatchInputs.cpp`)를 추가한 작업의 기록입니다. 무엇이 바뀌었는지, 왜 바꿨는지, 어떻게 검증했는지를 적습니다. 통합 컴파일러 1단계 범위(VGG16 + NPU)이고, 다음 단계에서 이 문서에 같은 형식으로 항목을 덧붙입니다.

## 한눈에 보기

경로의 `$P`는 `compiler/plugins/target/AMD-AIE/iree-amd-aie/`입니다(0절).

| # | 영역 | 내용 | 위치 |
|---|---|---|---|
| 1 | 새 패스 | 패스 `AMDAIEDemoteContractionDispatchInputs`: 장치 배정 직후, 장치가 원하는 타입으로 contraction 입력 변환 | `$P/Transforms/AMDAIEDemoteContractionDispatchInputs.cpp` |
| 2 | 장치 정보 | 새 컴파일 플래그 `--iree-amdaie-contraction-input-type=bf16` → amd-aie target 설정 키 `contraction_input_type = bf16` | `$P/Target/AIETarget.h`, `$P/Target/AIETarget.cpp` |
| 3 | 설정 읽기 | 새 함수 `getConfigContractionInputType()`: 어느 백엔드 target에서든 위 키를 읽음 | `$P/Transforms/Utils/AMDAIEUtils.h/.cpp` |
| 4 | 공용 helper | 새 함수 `getDispatchExecutableTarget()`(모든 백엔드용) 분리, 새 symbol 접두어 2개 등록 | `$P/Transforms/AMDAIEDispatchRewriteUtils.h/.cpp` |
| 5 | 파이프라인 | Flow 단계에서 장치 배정 패스와 bias fold 패스 사이에 새 패스 추가 | `$P/PluginRegistration.cpp` |
| 6 | 테스트 | lit 테스트 파일 (5개 경우) | `$P/Transforms/test/demote_contraction_dispatch_inputs.mlir` |

**결과 요약** (VGG16, CPU+NPU, 사용자 스크립트 옵션):

| | 기존 `--numeric bf16` | 새 패스 |
|---|---|---|
| 변환 시점 | 장치 배정 전 (모델 전체) | 장치 배정 후 (배정된 장치 기준) |
| NPU dispatch 입력·weight | bf16 | bf16 (weight는 바이트까지 동일) |
| 정확도 (f32 CPU 기준) | corr 0.99996, maxabs 0.0479, top-1 일치 | corr 0.99996, maxabs 0.0424, top-1 일치 |
| 실행 (5회 중앙값) | 2.48 s | 2.33 s (실행마다 2.1~3.6 s로 흔들려 사실상 같음) |
| 컴파일 | 105.5 s | 110.2 s |
| FC를 CPU에 배치하면 | CPU matmul도 bf16 | NPU matmul만 bf16, CPU matmul은 f32 |

새 플래그를 주지 않으면 새 패스는 아무것도 하지 않습니다. 기존 플래그 `--iree-amdaie-demote-contraction-inputs-to-bf16`은 지우지 않았으므로 기존 빌드는 그대로입니다.

---

## 0. 이름과 경로

이 문서에는 패스, 함수, 플래그, 설정 키 등 종류가 다른 이름이 섞여 나옵니다. 본문에서는 처음 나올 때 종류를 같이 적고, 위치는 이 절에 모았습니다.

**경로 약어**

| 약어 | 실제 경로 | 무엇 |
|---|---|---|
| `$P` | `compiler/plugins/target/AMD-AIE/iree-amd-aie/` | 우리 AMD-AIE 플러그인 |
| `$IREE` | `third_party/iree/compiler/src/iree/compiler/` | IREE 본체 (upstream, `ace-knu/iree` 포크) |

**이름의 종류**

| 종류 | 뜻 |
|---|---|
| 패스 | 모듈 전체를 훑으며 IR을 고치는 컴파일 단계 하나. `.cpp` 파일 하나에 클래스로 구현하고 `Passes.td`에 등록함. `iree-opt --pass-pipeline='builtin.module(<명령줄 이름>)'`으로 따로 돌릴 수 있음 |
| 함수 | 패스 안이나 공용 파일에 있는 C++ 함수 (`static`이면 그 파일 안에서만 씀) |
| 플래그 | `iree-compile`에 주는 명령줄 옵션 (`--iree-...`) |
| 스크립트 옵션 | `scripts/onnx/*.py`에 주는 옵션 (`--numeric` 등). 스크립트가 플래그로 바꿔 `iree-compile`에 넘김 |
| target 설정 키 | `#hal.executable.target<"amd-aie", ..., { ... }>`의 `{ }` 안 항목. 장치(target)별 정보 |
| 파이프라인 단계 | 패스 여러 개를 묶은 큰 단계: Preprocessing → GlobalOpt → Flow → Stream → HAL. const-eval은 GlobalOpt 안에서 상수 계산을 컴파일 중에 미리 실행하는 부분 |

**패스**

| 패스 (C++ 이름) | 명령줄 이름 | 파일 | 도는 단계 | 이 문서에서 |
|---|---|---|---|---|
| `AMDAIEDemoteContractionDispatchInputs` | `iree-amdaie-demote-contraction-dispatch-inputs` | `$P/Transforms/AMDAIEDemoteContractionDispatchInputs.cpp` | Flow | **새 패스** |
| `DemoteContractionInputsPass` (upstream) | `iree-global-opt-demote-contraction-inputs` | `$IREE/GlobalOptimization/DemoteContractionInputs.cpp` | 우리 플러그인은 Preprocessing에서 부름 | **기존 방식**의 변환 패스 |
| `AMDAIEAssignDeviceAffinities` | `iree-amdaie-assign-device-affinities` | `$P/Transforms/AMDAIEAssignDeviceAffinities.cpp` | Flow | 장치 배정 (dispatch마다 `stream.affinity`를 붙임) |
| `AMDAIEFoldBiasIntoContractionDispatches` | `iree-amdaie-fold-bias-into-contraction-dispatches` | `$P/Transforms/AMDAIEFoldBiasIntoContractionDispatches.cpp` | Flow | bias fold |
| `AMDAIEPadContractionDispatches` | `iree-amdaie-pad-contraction-dispatches` | `$P/Transforms/AMDAIEPadContractionDispatches.cpp` | Flow | contraction pad |
| `AMDAIEPadPoolingDispatches` | `iree-amdaie-pad-pooling-dispatches` | `$P/Transforms/AMDAIEPadContractionDispatches.cpp` (contraction pad와 같은 파일) | Flow | pooling pad |

**함수** (새 패스 파일 안의 함수는 6-1절 표)

| 함수 | 종류 | 위치 |
|---|---|---|
| `createDemoteContractionInputsPass(type, operation)` | upstream 패스를 만들어 주는 함수 | `$IREE/GlobalOptimization/DemoteContractionInputs.cpp:177` |
| `extendPreprocessingPassPipeline`, `extendFlowTransformPassPipeline` | 플러그인 hook 함수. IREE가 단계마다 불러 플러그인이 패스를 끼우게 함 | `$P/PluginRegistration.cpp:56`, `:104` |
| `getConfigContractionInputType` | **새 함수**: target 설정 키 읽기 | `$P/Transforms/Utils/AMDAIEUtils.cpp:226` |
| `getConfigAttr`, `getConfigGemvPadN` 등 | 기존 함수: target 설정 키 읽기 | `$P/Transforms/Utils/AMDAIEUtils.cpp:34`, `:214` |
| `getDispatchExecutableTarget` | **새 함수**: dispatch → target (모든 백엔드) | `$P/Transforms/AMDAIEDispatchRewriteUtils.cpp:17` |
| `getDispatchTarget`, `getDispatchFunc`, `getFullTensorLoad`, `getHostAffinity`, `firstFreeHelperIndex`, `createPaddingDispatch`, `getConstantValue`, `createReplacementConstant`, `eraseDeadGlobals` | 기존 공용 함수 (Flow 패스들이 같이 씀) | `$P/Transforms/AMDAIEDispatchRewriteUtils.cpp` 35, 43, 56, 89, 107, 125, 209, 229, 262행 |
| `createCropDispatch` | 기존 함수 (`static`) | `$P/Transforms/AMDAIEPadContractionDispatches.cpp:260` |
| `deviceRole` | 기존 함수: 백엔드 이름 → Host/Accelerator | `$P/Transforms/Utils/AMDAIEDevicePlacementUtils.cpp:21` |
| `check()` | Python 함수: 옵션 조합 검사 | `scripts/onnx/options.py:136` |

**플래그, 스크립트 옵션, 설정 키**

| 이름 | 종류 | 정의 / 쓰는 곳 |
|---|---|---|
| `--iree-amdaie-contraction-input-type` | **새 플래그** (`none`/`bf16`) | 정의 `$P/Target/AIETarget.h:213`, target에 싣기 `$P/Target/AIETarget.cpp:200` |
| `contraction_input_type` | **새 target 설정 키** | 붙이기 `$P/Target/AIETarget.cpp:201`, 읽기 `$P/Transforms/Utils/AMDAIEUtils.cpp:226` |
| `--iree-amdaie-demote-contraction-inputs-to-bf16` | 기존 플래그 | 정의 `$P/Target/AIETarget.h:206`, 쓰는 곳 `$P/PluginRegistration.cpp:60` |
| `--numeric {f32,bf16}` | 스크립트 옵션 | `scripts/onnx/options.py:99`. `bf16`이면 위 기존 플래그를 붙임 |
| `--iree-amdaie-placement-map` | 기존 플래그 (실험용) | 정의 `$P/Target/AIETarget.h:223`, 장치 배정 패스의 옵션 |
| `--iree-amdaie-core-rounding-mode` | 플래그, **dev에는 없음** | 커밋 `822f0fed`에서 추가 (9절) |

**기타 파일**

| 이름 | 위치 |
|---|---|
| lit 테스트 | `$P/Transforms/test/demote_contraction_dispatch_inputs.mlir` |
| npu4 matmul ukernel 소스 (`matmul.cc`) | `$P/Target/uKernels/npu4/peano/matmul.cc` |
| ONNX 스크립트 | `scripts/onnx/compile.py`, `run.py`, `e2e.py`, `options.py`; 호스트용 `scripts/docker/run-onnx.sh` |

## 1. 배경: 기존 방식의 문제

기존에는 플래그 `--iree-amdaie-demote-contraction-inputs-to-bf16`(스크립트 옵션 `--numeric bf16`)을 주면, 플러그인 hook 함수 `extendPreprocessingPassPipeline`(`$P/PluginRegistration.cpp:56`)이 **Preprocessing 단계**에 upstream 패스 `DemoteContractionInputsPass`(`$IREE/GlobalOptimization/DemoteContractionInputs.cpp`)를 끼워 넣었습니다. 이 패스가 모델의 모든 contraction 입력 앞에 truncf(f32 → bf16)를 넣었습니다.

```cpp
// $P/PluginRegistration.cpp 56–64행 (기존 코드, 지금도 그대로)
void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
  if (options.demoteContractionInputsToBf16) {
    passManager.addPass(GlobalOptimization::createDemoteContractionInputsPass(
        GlobalOptimization::DemoteType::BF16,          // f32 → bf16
        GlobalOptimization::DemoteOperation::All));    // matmul, conv 모두
  }
```

```
[Preprocessing]  패스 DemoteContractionInputsPass  ← 모든 matmul/conv 입력 → bf16
[GlobalOpt]      const-eval (weight의 truncf를 컴파일 중에 실행)
[Flow]           dispatch 생성 → 장치 배정 → bias fold → pad
```

이 시점에는 어느 연산이 어느 장치로 갈지 아직 모릅니다. 그래서 문제가 두 가지 있었습니다.

1. **CPU나 GPU에 배정될 matmul도 bf16이 됩니다.** `$P/Target/AIETarget.h`의 기존 옵션 필드 `demoteContractionInputsToBf16` 주석에도 "module-wide이고 배정 전이라 host에 놓일 contraction도 demote된다"고 적혀 있습니다.
2. **GPU가 하나라도 섞이면 bf16을 쓸 수 없습니다.** GPU(vulkan) 경로가 bf16 contraction을 컴파일하지 못해서, Python 함수 `check()`(`scripts/onnx/options.py:136`)가 "bf16 + vulkan"을 막고 있습니다. 결국 NPU도 f32로 돌아야 했습니다.

목표는 **dispatch 경계는 기본 f32로 두고, 각 장치는 dispatch 안쪽에 필요한 타입만 받는 것**(캡슐화)입니다. 그러려면 변환을 장치 배정 뒤에 결정해야 합니다.

## 2. 설계

### 2-1. 장치가 원하는 타입은 target이 들고 있음

장치마다 executable target 설정에 원하는 contraction 입력 타입을 둡니다.

```mlir
#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {contraction_input_type = bf16, num_cols = 8 : i32, ...}>
```

패스는 장치 이름(npu4 등)을 보지 않고 이 키만 읽습니다. 키가 없는 장치(현재 llvm-cpu, vulkan)는 손대지 않습니다. 다른 가속기도 자기 target에 이 키를 붙이면 같은 패스로 처리됩니다(통합 컴파일러 일반성 원칙).

### 2-2. 변환은 NPU 입력 쪽이 아니라 그 앞에서

변환 방식은 두 가지가 가능합니다.

| | NPU 안에서 변환 (입력 쪽) | NPU 앞에서 변환 (채택) |
|---|---|---|
| 흐름 | DRAM f32 → NPU에서 bf16으로 → matmul | 앞 단계가 bf16을 써 둠 → NPU는 bf16을 읽음 |
| DRAM 트래픽 | f32라 두 배 | bf16이라 절반 |
| NPU L1 | f32 입력이 공간을 두 배로 차지해 타일이 작아짐 | 영향 없음 |
| 백엔드 작업 | NPU codegen이 입력 쪽 변환을 지원해야 함 (현재 미지원) | 없음 |

그래서 NPU dispatch는 처음부터 bf16 입력을 받게 하고, 변환은 그 앞에서 끝냅니다. 같은 RNE 반올림이면 어디서 변환하든 값은 같습니다.

### 2-3. 결정 규칙

NPU(또는 bf16을 원하는 장치)의 dispatch D가 읽는 값 X마다, 위에서부터 처음 맞는 규칙을 씁니다.

| 순서 | 조건 | 처리 | 1단계 구현 | 새 패스 파일의 함수 |
|---|---|---|---|---|
| 1 | X가 컴파일 시점 상수 (weight) | 컴파일할 때 bf16 global을 새로 만듦. f32 global은 다른 곳에서 안 읽으면 지움 | O | `createDemotedConstant` (6-5절) |
| 2 | X가 dispatch P의 결과이고, X를 읽는 곳이 모두 bf16을 원함 | P의 저장 직전에 truncf를 넣어 P가 bf16을 씀 | O (P가 CPU일 때만) | `fuseIntoProducer` (6-6절) |
| 3 | X를 f32로 읽는 곳도 있음 | P가 f32와 bf16 두 개를 출력 | X (다음 단계) | 없음 |
| 4 | 위가 모두 안 됨 (모델 입력 등) | CPU에서 변환만 하는 dispatch를 새로 만듦 | O (안전장치) | `createCastDispatch` (6-7절) |

"bf16을 원한다"는 것은 X를 읽는 dispatch의 장치 target에 `contraction_input_type = bf16`이 있고, 새 패스가 그 dispatch 안의 contraction 입력을 bf16으로 바꿨다는 뜻입니다. 패스는 먼저 소비자(NPU dispatch) 쪽을 바꾸면서 "이 값을 bf16으로 원하는 사용처" 목록을 만들고, 규칙 2에서는 X를 실제로 읽는 곳이 모두 그 목록에 있는지 비교합니다(6-6절).

규칙 2에서 P가 NPU인 경우(NPU epilogue에서 bf16을 씀)는 2단계로 미뤘습니다(3-7절, 9절, 10절 참고).

---

## 3. 타입 변환 흐름 (dispatch 기준)

### 3-1. 파이프라인 위치

**전체 컴파일 과정에서의 위치.** `iree-compile`은 아래 8단계를 차례로 돕니다. 단계 순서는 IREE 본체의 `$IREE/Pipelines/Pipelines.cpp`(함수 `buildIREEPrecompileTransformPassPipeline`, `buildIREEVMTransformPassPipeline`)에 있습니다.

```
 ONNX 모델 (.onnx)
   │  스크립트 compile.py가 MLIR로 import
   ▼
┌──────────────────────── iree-compile ────────────────────────┐
│                                                               │
│  1. Input               torch/onnx op → linalg 변환           │
│     │                   (linalg.matmul, linalg.conv_2d 등장)  │
│     ▼                                                         │
│  2. ABI                 함수 입출력 형태 정리                  │
│     ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 3. Preprocessing    ◀━━ 기존 방식의 타입 변환은 여기     │ │
│  │    · DemoteContractionInputsPass                        │ │
│  │      (모든 matmul/conv 입력 앞에 truncf)                │ │
│  │    · pooling → NHWC                                     │ │
│  │    · AssignDeviceAffinities (장치 정보만 심음)          │ │
│  └─────────────────────────────────────────────────────────┘ │
│     ▼                                                         │
│  4. GlobalOptimization  상수 미리 계산 (weight truncf → bf16  │
│     │                   상수), transpose 정리, 융합 준비      │
│     ▼                                                         │
│  5. DispatchCreation    연산들을 dispatch 단위로 묶음          │
│     │                   (truncf가 앞 dispatch에 합쳐짐)       │
│     ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 6. Flow             플러그인 패스                        │ │
│  │    · AssignDeviceAffinities (dispatch마다 NPU/CPU 결정) │ │
│  │    · DemoteContractionDispatchInputs ◀━━ 새 방식은 여기 │ │
│  │    · FoldBias → PadContraction → PadPooling             │ │
│  └─────────────────────────────────────────────────────────┘ │
│     ▼                                                         │
│  7. Stream              버퍼·실행 순서·장치 간 복사 계획       │
│     ▼                                                         │
│  8. HAL                 dispatch마다 장치 코드 생성            │
│                         (NPU: AIE 코드, CPU: LLVM)            │
└───────────────────────────────────────────────────────────────┘
   │
   ▼
 .vmfb (실행 파일) → iree-run-module로 실행
```

- 3번 Preprocessing은 모델이 linalg로 바뀐 직후입니다. dispatch도 없고 장치도 정해지지 않았습니다. 플러그인 패스는 hook 함수 `extendPreprocessingPassPipeline`(`$P/PluginRegistration.cpp:56`)이 끼우고, 이 hook은 `$IREE/Preprocessing/Passes.cpp:109`에서 불립니다.
- 장치는 6번 Flow의 `AssignDeviceAffinities`에서야 dispatch마다 정해집니다(3번의 같은 이름 패스는 장치 목록만 심고 배치는 하지 않음).
- 그래서 기존 방식은 장치를 모른 채 모든 matmul/conv 입력을 bf16으로 바꿨습니다. 새 패스는 6번 안, 장치 배치 바로 뒤에서 NPU dispatch 입력만 바꿉니다.
- 기존 방식에서 넣은 truncf는 뒤 단계에서 사라집니다. weight 앞의 것은 4번에서 bf16 상수로 미리 계산되고, 나머지는 5번에서 앞 dispatch 끝에 합쳐집니다(3-7절의 `[NPU FC1: matmul → ReLU → truncf]`가 이 경우).

**Preprocessing 이후만 확대하면** 다음과 같습니다.

```
기존:
  [Preprocessing] (--numeric bf16이면 여기서 패스 DemoteContractionInputsPass가 모든 contraction 입력 truncf)
  [Flow] dispatch 생성 → AssignDeviceAffinities → FoldBias → PadContraction → PadPooling

변경 후:
  [Preprocessing] (변환 없음, 모두 f32)
  [Flow] dispatch 생성 → AssignDeviceAffinities → ★DemoteContractionDispatchInputs → FoldBias → PadContraction → PadPooling
                                │                          │
                                │ 모든 flow.dispatch에      │ 장치가 원하는 타입으로 입력 변환
                                │ stream.affinity 부여      │ (bias fold와 pad는 바뀐 타입을 보고 동작)
```

그림의 이름은 모두 **패스**를 줄여 쓴 것입니다. 실제 이름과 파일은 아래와 같습니다. Flow 단계의 이 패스들은 hook 함수 `extendFlowTransformPassPipeline`(`$P/PluginRegistration.cpp:104`)이 이 순서로 끼웁니다.

| 그림의 이름 | 패스 (C++ 이름) | 파일 |
|---|---|---|
| (기존) Preprocessing의 변환 | `DemoteContractionInputsPass` (upstream) | `$IREE/GlobalOptimization/DemoteContractionInputs.cpp` |
| AssignDeviceAffinities | `AMDAIEAssignDeviceAffinities` | `$P/Transforms/AMDAIEAssignDeviceAffinities.cpp` |
| ★DemoteContractionDispatchInputs | `AMDAIEDemoteContractionDispatchInputs` (새) | `$P/Transforms/AMDAIEDemoteContractionDispatchInputs.cpp` |
| FoldBias | `AMDAIEFoldBiasIntoContractionDispatches` | `$P/Transforms/AMDAIEFoldBiasIntoContractionDispatches.cpp` |
| PadContraction | `AMDAIEPadContractionDispatches` | `$P/Transforms/AMDAIEPadContractionDispatches.cpp` |
| PadPooling | `AMDAIEPadPoolingDispatches` | `$P/Transforms/AMDAIEPadContractionDispatches.cpp` |

"dispatch 생성"은 IREE 본체의 DispatchCreation 패스들(`$IREE/DispatchCreation/`)이 하는 일로, 우리가 바꾸지 않았습니다.

새 패스가 bias fold와 pad보다 앞에 있는 이유는 두 패스가 원소 타입을 보고 일하기 때문입니다. pad 배수는 원소 타입에 따라 다르고(예: bf16 ukernel이면 N 배수 128), bias fold는 bias를 weight 원소 타입으로 반올림해 붙입니다.

### 3-2. 패스 안의 흐름

```
runOnOperation
│
├─ 준비: host 장치 확인 (없으면 아무것도 안 함), 각 executable을 부르는 dispatch 목록(callers)
│
├─ 1. 분석
│   ├─ 모든 flow.dispatch D에 대해
│   │    target = getDispatchExecutableTarget(D)          (affinity → device global → target)
│   │    type   = getConfigContractionInputType(target)   (키 없음 → 건너뜀)
│   │    groups[(D의 executable, type)] += D
│   │
│   └─ 그룹마다
│        plan = analyzeFunc(executable, type)
│          · named contraction(곱셈 있는 matmul/conv 계열)이고 입력이 type보다 넓은 float인 op
│          · 입력을 load → binding(%argN)까지 거슬러 올라감
│          · 그 binding을 다른 곳에서도 읽으면 제외 (바뀌어도 아무것도 안 깨지는 것만)
│        executable을 다른 장치 dispatch도 부르면 → 복사본을 만들어 이 그룹만 연결
│        demoteFunc: executable 안의 binding · load · contraction 입력을 type으로 바꿈
│        D의 바뀐 operand마다, reshape를 거슬러 올라간 원래 값(root)으로 요청(Request) 기록
│
└─ 2·3. 결정과 IR 고치기 (root 값마다)
     a. 상수?                              → __demoted_constant_N global (컴파일 시 RNE 변환)
     b. CPU dispatch 결과이고 읽는 곳이 모두 bf16?  → 그 dispatch 안에 truncf, 결과·reshape 타입 bf16
     c. 그 밖                              → cast_executable_N (CPU)를 값 정의 바로 뒤에 생성
     마지막에 안 쓰게 된 reshape · load · global 정리
```

그림의 이름은 모두 **함수**입니다.
- `runOnOperation`은 새 패스 클래스의 진입 함수(패스가 돌 때 IREE가 부름)입니다.
- `analyzeFunc`, `demoteFunc` 등 새 패스 파일 `$P/Transforms/AMDAIEDemoteContractionDispatchInputs.cpp` 안의 함수는 6-1절 표에 행 번호와 함께 있습니다.
- `getDispatchExecutableTarget`(`$P/Transforms/AMDAIEDispatchRewriteUtils.cpp`)와 `getConfigContractionInputType`(`$P/Transforms/Utils/AMDAIEUtils.cpp`)은 다른 파일의 공용 함수입니다(5-3, 5-4절).
- `__demoted_constant_N`, `cast_executable_N`은 함수가 아니라 새 패스가 IR에 만드는 global과 executable의 **이름**입니다(N은 번호).

### 3-3. VGG 첫 conv: 변환 전후 IR

**패스 전 (모두 f32)**

```mlir
// CPU: im2col
%4 = flow.dispatch @dispatch_1::@..._generic_224x224x3x3x3_f32(%3) {stream.affinity = #hal.device.affinity<@d0_local_host>}
       : (tensor<226x226x3xf32>) -> tensor<224x224x3x3x3xf32>
%5 = flow.tensor.reshape %4 : tensor<224x224x3x3x3xf32> -> tensor<50176x27xf32>
%w = util.global.load immutable @__hoisted_tensor_27x64xf32 : tensor<27x64xf32>
// NPU: matmul
%8 = flow.dispatch @dispatch_2::@..._matmul_50176x64x27_f32(%5, %w) {stream.affinity = #hal.device.affinity<@d1_amdxdna_npu4>}
       : (tensor<50176x27xf32>, tensor<27x64xf32>) -> tensor<50176x64xf32>
```

**결정**: `%w`는 상수 → 규칙 1, `%4`는 CPU dispatch 결과이고 NPU matmul만 읽음 → 규칙 2.

**패스 후**

```mlir
// CPU: im2col이 저장 직전에 bf16으로 변환해서 씀 (규칙 2)
%4 = flow.dispatch @dispatch_1::@..._generic_224x224x3x3x3_f32(%3) {...@d0_local_host}
       : (tensor<226x226x3xf32>) -> tensor<224x224x3x3x3xbf16>
%5 = flow.tensor.reshape %4 : tensor<224x224x3x3x3xbf16> -> tensor<50176x27xbf16>
// weight는 컴파일할 때 bf16으로 변환한 새 global (규칙 1)
%w = util.global.load immutable @__demoted_constant_0 : tensor<27x64xbf16>
// NPU: 입력은 bf16, 출력은 f32 그대로
%8 = flow.dispatch @dispatch_2::@..._matmul_50176x64x27_f32(%5, %w) {...@d1_amdxdna_npu4}
       : (tensor<50176x27xbf16>, tensor<27x64xbf16>) -> tensor<50176x64xf32>
```

dispatch 이름 끝의 `_f32`는 dispatch를 만들 때 정해진 이름이라 그대로 남습니다. 동작에는 영향이 없습니다.

im2col executable 안쪽은 이렇게 바뀝니다. 출력을 만드는 generic에 truncf가 합쳐져, 기존 bf16 빌드와 같은 모양이 됩니다.

```mlir
// 전
%2 = linalg.generic {...} ins(%0 : tensor<226x226x3xf32>) outs(%1 : tensor<224x224x3x3x3xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}
iree_tensor_ext.dispatch.tensor.store %2, %arg1 ... -> ...<writeonly:tensor<224x224x3x3x3xf32>>

// 후
%2 = linalg.generic {...} ins(%0 : tensor<226x226x3xf32>) outs(%1 : tensor<224x224x3x3x3xbf16>) {
^bb0(%in: f32, %out: bf16):
  %3 = arith.truncf %in : f32 to bf16
  linalg.yield %3 : bf16
}
iree_tensor_ext.dispatch.tensor.store %2, %arg1 ... -> ...<writeonly:tensor<224x224x3x3x3xbf16>>
```

NPU matmul executable 안쪽은 이렇게 바뀝니다.

```mlir
// 전
%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<50176x27xf32>>
%0 = iree_tensor_ext.dispatch.tensor.load %arg0 ... -> tensor<50176x27xf32>
%4 = linalg.matmul ins(%0, %1 : tensor<50176x27xf32>, tensor<27x64xf32>) outs(%3 : tensor<50176x64xf32>)

// 후
%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<50176x27xbf16>>
%0 = iree_tensor_ext.dispatch.tensor.load %arg0 ... -> tensor<50176x27xbf16>
%4 = linalg.matmul ins(%0, %1 : tensor<50176x27xbf16>, tensor<27x64xbf16>) outs(%3 : tensor<50176x64xf32>)
//   (matmul body는 bf16 입력을 extf로 f32로 늘려 곱하고 f32로 누산)
```

이후 pad 패스가 bf16 기준 배수로 패딩합니다(K 27 → 32, weight N 64 → 128). 결과는 기존 `--numeric bf16` 빌드와 같습니다.

### 3-4. 규칙 4: CPU cast dispatch

모델 입력이거나 f32로도 읽히는 값이면 CPU에 변환 전용 dispatch를 만듭니다(새 패스 파일의 함수 `createCastDispatch`, 6-7절). 원래 f32 값은 다른 사용처를 위해 그대로 둡니다. 아래 `@cast_executable_2::@cast_dispatch_2`는 이 함수가 만든 executable과 그 안 함수의 이름입니다.

```mlir
%1 = flow.dispatch @relu::@relu(%0) {...@cpu} : (tensor<8x2xf32>) -> tensor<8x2xf32>
%c = flow.dispatch @cast_executable_2::@cast_dispatch_2(%1) {...@cpu} : (tensor<8x2xf32>) -> tensor<8x2xbf16>   // 새로 생김
%2 = flow.dispatch @mm::@mm(%c, %w) {...@npu} : (tensor<8x2xbf16>, tensor<2x2xbf16>) -> tensor<8x2xf32>
util.return %1, %2      // %1(f32)은 반환에도 쓰이므로 그대로 남음
```

### 3-5. 같은 executable을 NPU와 다른 장치가 같이 쓸 때

복사는 새 패스 파일의 함수 `copyExecutableFor`가 합니다. `@mm_0`처럼 원래 이름 뒤에 번호를 붙인 새 executable을 만들고, NPU dispatch만 그쪽을 부르게 바꿉니다.

```mlir
// 전: @mm 하나를 NPU와 CPU가 같이 부름
%0 = flow.dispatch @mm::@mm(%a, %w) {...@npu}
%1 = flow.dispatch @mm::@mm(%b, %w) {...@cpu}

// 후: NPU 쪽만 복사본 @mm_0(bf16 입력)으로 연결, CPU 쪽은 원본 @mm(f32) 그대로
%0 = flow.dispatch @mm_0::@mm(%a_bf16, %w_bf16) {...@npu}
%1 = flow.dispatch @mm::@mm(%b, %w) {...@cpu}      // f32 weight global도 이 dispatch를 위해 남음
```

### 3-6. 장치마다 다르게 변환되는 예 (같은 모델 안)

기존 실험용 플래그 `--iree-amdaie-placement-map=~matmul_1x=cpu`(장치 배정 패스 `AMDAIEAssignDeviceAffinities`의 옵션, 정의 `$P/Target/AIETarget.h:223`)로 VGG의 FC matmul 3개만 CPU에 배치한 결과입니다.

| dispatch | 장치 | 입력 타입 |
|---|---|---|
| conv(im2col) matmul 13개 | NPU | bf16 (activation, weight 모두) |
| FC matmul 3개 | CPU | **f32** (activation, weight 모두) |

기존 플래그였다면 CPU FC도 bf16이 됐습니다. GPU도 `contraction_input_type`이 없으니 같은 방식으로 f32로 남습니다(이 빌드에는 Vulkan이 꺼져 있어 GPU로 직접 확인하지는 못했음).

### 3-7. 사용자 스크립트 설정의 FC: 아직 다른 부분

스크립트 옵션(bias fold 포함)에서는 FC1, FC2 dispatch가 NPU에서 `matmul + ReLU` 형태입니다. 그 출력이 다음 FC의 입력이므로 앞 dispatch가 NPU입니다.

```
기존 --numeric bf16:
  [NPU FC1: matmul → ReLU → truncf] --bf16--> DRAM --bf16--> [NPU FC2]

새 패스 (1단계):
  [NPU FC1: matmul → ReLU] --f32--> DRAM --f32--> [CPU cast_executable_N] --bf16--> DRAM --bf16--> [NPU FC2]
```

이 그림은 규칙 2가 **적용되지 못해** 규칙 4로 빠진 경우입니다(규칙 2가 적용된 모습은 3-3절의 im2col).
- FC1의 결과를 읽는 곳은 FC2(NPU)뿐이라 "모두 bf16을 원함"은 맞습니다. 하지만 FC1이 NPU라서 함수 `canWriteElementType`(새 패스 파일 73행, 지금은 host만 허용)에서 걸립니다.
- 기존에는 truncf가 FC1 dispatch 안에 있어서, NPU가 계산 직후 bf16으로 바꿔 메모리에 한 번 썼습니다(쓰기 1, 읽기 1).
- 새 패스는 변환을 별도 CPU dispatch로 하므로, dispatch끼리 데이터를 메모리로만 주고받는 구조상 왕복이 한 번 더 생깁니다(쓰기 2, 읽기 2). FC 출력은 4096개(f32 16 KB)라 데이터 양 자체는 작습니다.

1단계는 규칙 2를 CPU producer에만 적용하므로 FC 뒤에 CPU cast dispatch가 하나씩(총 2개) 생겼습니다. dispatch 수가 87개에서 89개로 늘었습니다. NPU epilogue에 합치는 2단계에서 없앨 대상입니다.

---

## 4. 변경 파일

경로는 모두 `$P` = `compiler/plugins/target/AMD-AIE/iree-amd-aie/` 아래입니다.

| 파일 | 구분 | 변경 |
|---|---|---|
| `$P/Transforms/AMDAIEDemoteContractionDispatchInputs.cpp` | 새 파일 (676줄) | 새 패스 본체 |
| `$P/Transforms/test/demote_contraction_dispatch_inputs.mlir` | 새 파일 (314줄) | lit 테스트 5개 |
| `$P/Target/AIETarget.h` | 수정 (+15) | 옵션 필드 `contractionInputType`, 플래그 `--iree-amdaie-contraction-input-type` |
| `$P/Target/AIETarget.cpp` | 수정 (+10) | target 설정 키 `contraction_input_type` 추가 |
| `$P/Transforms/Utils/AMDAIEUtils.h/.cpp` | 수정 (+7, +7) | 새 함수 `getConfigContractionInputType()` |
| `$P/Transforms/AMDAIEDispatchRewriteUtils.h/.cpp` | 수정 (+12/−5, +11/−4) | 새 함수 `getDispatchExecutableTarget()` 분리, 접두어 2개 추가 |
| `$P/PluginRegistration.cpp` | 수정 (+7) | hook 함수 `extendFlowTransformPassPipeline`에 새 패스 추가 |
| `$P/Transforms/Passes.td`, `Passes.h`, `PassDetail.h` | 수정 (+7, +4, +1) | 패스 등록 |
| `$P/Transforms/CMakeLists.txt`, `$P/Transforms/test/CMakeLists.txt` | 수정 (+1, +1) | 소스와 테스트 등록 |

기존 파일은 12개, 83줄 추가 9줄 삭제입니다. IREE 본체(`$IREE`)는 바꾸지 않았습니다.

> 아래 행 번호는 **이 문서 작성 시점의 작업 트리 기준**입니다.

## 5. 상세: 수정한 기존 파일

### 5-1. `$P/Target/AIETarget.h`: 옵션 필드와 플래그 추가

**① 옵션 필드** (70–75행)

```cpp
69    bool demoteContractionInputsToBf16{false};
70    // Element type the inputs of contraction dispatches placed on this device are
71    // converted to after device placement (`none` keeps them, `bf16`), carried
72    // as `contraction_input_type` on the executable target. Unlike
73    // demoteContractionInputsToBf16 above, only the dispatches placed here are
74    // converted, so a GPU or the CPU sharing the model keeps f32.
75    std::string contractionInputType{"none"};
```

**② 컴파일 플래그** (213–220행)

```cpp
213      binder.opt<std::string>(
214          "iree-amdaie-contraction-input-type", contractionInputType,
215          llvm::cl::cat(category),
216          llvm::cl::desc(
217              "Element type the inputs of contraction dispatches placed on the "
218              "amd-aie device are converted to after device placement (Flow "
219              "phase; accumulation stays f32): `none` (default) or `bf16`. "
220              "Dispatches placed on other devices keep their types."));
```

> **변경 전 → 후**
> - **기존**: 타입 변환 옵션은 `demoteContractionInputsToBf16`(bool) 하나였습니다. 켜면 장치 배정 전에 모델 전체를 bf16으로 바꿨습니다.
> - **추가**: 문자열 옵션 `contractionInputType`(기본 `none`)과 플래그 `--iree-amdaie-contraction-input-type`. "amd-aie 장치에 배정된 dispatch의 입력"만 대상입니다. 값은 `none`, `bf16` 두 가지입니다.
> - **바뀌지 않은 것**: 기존 bool 옵션과 플래그는 그대로입니다. 둘을 같이 주면 기존 플래그가 먼저 모두 bf16으로 바꾸므로 새 패스는 할 일이 없습니다.

### 5-2. `$P/Target/AIETarget.cpp`: target 설정 키 싣기 (198–207행)

```cpp
198      // The element type the Flow demote pass converts the inputs of this
199      // device's contraction dispatches to (absent: kept as they are).
200      if (options.contractionInputType == "bf16") {
201        addConfig("contraction_input_type",
202                  TypeAttr::get(BFloat16Type::get(context)));
203      } else if (options.contractionInputType != "none") {
204        llvm::report_fatal_error(
205            llvm::Twine("Invalid --iree-amdaie-contraction-input-type (") +
206            options.contractionInputType + "), must be `none` or `bf16`");
207      }
```

> **변경 전 → 후**
> - **기존**: amd-aie target 설정에는 `target_device`, `ukernels`, `gemv_pad_n`, `gemv_n1`, `pool_columns_policy`, `num_rows`, `num_cols`가 있었고, 입력 타입 정보는 없었습니다.
> - **추가**: 옵션이 `bf16`이면 `contraction_input_type = bf16`(TypeAttr)을 붙이고, 잘못된 값이면 에러를 냅니다. `none`이면 키를 붙이지 않습니다.
> - **결과**: 장치마다 원하는 타입을 target이 들고 있게 됩니다. pad 패스(`AMDAIEPadContractionDispatches`)가 `gemv_pad_n`을 target에서 읽는 것과 같은 방식입니다.

### 5-3. `$P/Transforms/Utils/AMDAIEUtils.h/.cpp`: 설정 키를 읽는 함수

```cpp
// AMDAIEUtils.h 128–133행
/// The float type the inputs of contraction dispatches placed on this target
/// are converted to (`contraction_input_type` in the executable target config,
/// e.g. `bf16`), or null to keep them as they are. Read per dispatch by the
/// Flow demote pass after device placement; any backend's target may carry it.
FloatType getConfigContractionInputType(
    IREE::HAL::ExecutableTargetAttr targetAttr);

// AMDAIEUtils.cpp 226–231행
FloatType getConfigContractionInputType(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<TypeAttr> attr =
      getConfigAttr<TypeAttr>(targetAttr, "contraction_input_type");
  return attr ? dyn_cast<FloatType>(attr->getValue()) : FloatType();
}
```

> **변경 전 → 후**
> - **기존**: 같은 파일에 `getConfigGemvPadN`, `getConfigGemvN1`, `getConfigPoolColumnsPolicy` 등 target 설정 키를 읽는 함수들이 있었습니다. 모두 템플릿 함수 `getConfigAttr`(`AMDAIEUtils.cpp:34`)로 키를 꺼냅니다.
> - **추가**: 같은 방식으로 `contraction_input_type`을 읽는 함수입니다. 키가 없거나 float 타입이 아니면 null을 돌려줍니다.
> - **일반성**: 백엔드를 확인하지 않습니다. llvm-cpu든 vulkan이든 target에 이 키가 있으면 같은 값을 돌려줍니다.

### 5-4. `$P/Transforms/AMDAIEDispatchRewriteUtils.h/.cpp`: 공용 helper 함수

**① target 조회 함수 분리** (`.cpp` 17–41행)

```diff
-IREE::HAL::ExecutableTargetAttr getDispatchTarget(
+IREE::HAL::ExecutableTargetAttr getDispatchExecutableTarget(
     IREE::Flow::DispatchOp dispatch, ModuleOp module) {
   auto affinity =
       dispatch->getAttrOfType<IREE::HAL::DeviceAffinityAttr>("stream.affinity");
   ...
   if (targets.empty()) return {};
-  IREE::HAL::ExecutableTargetAttr exec = targets.front();
-  if (exec.getBackend().getValue() != "amd-aie") return {};
+  return targets.front();
+}
+
+IREE::HAL::ExecutableTargetAttr getDispatchTarget(
+    IREE::Flow::DispatchOp dispatch, ModuleOp module) {
+  IREE::HAL::ExecutableTargetAttr exec =
+      getDispatchExecutableTarget(dispatch, module);
+  if (!exec || exec.getBackend().getValue() != "amd-aie") return {};
   return exec;
 }
```

> **변경 전 → 후**
> - **기존**: 함수 `getDispatchTarget()` 하나가 "affinity → device global → target" 조회와 "amd-aie인지" 검사를 같이 했습니다. amd-aie가 아니면 null이라, CPU에 배정된 dispatch의 target은 알 수 없었습니다.
> - **변경**: 조회만 하는 `getDispatchExecutableTarget()`(모든 백엔드)을 따로 만들고, `getDispatchTarget()`은 그걸 불러 amd-aie만 걸러냅니다.
> - **이유**: 새 패스는 앞 dispatch가 CPU인지 확인해야 하고(규칙 2), NPU가 아닌 장치의 `contraction_input_type`도 읽을 수 있어야 합니다.
> - **바뀌지 않은 것**: `getDispatchTarget()`의 결과는 전과 같습니다. 이를 쓰는 패스 `AMDAIEPadContractionDispatches`, `AMDAIEPadPoolingDispatches`, `AMDAIEFoldBiasIntoContractionDispatches`의 동작도 같습니다(기존 lit 테스트 통과).

**② 새 symbol 접두어 등록** (함수 `firstFreeHelperIndex`, `.cpp` 113–116행)

```diff
     for (StringRef prefix :
          {"pad_executable_", "crop_executable_", "concat_executable_",
-          "__padded_constant_", "__bias_folded_weight_"}) {
+          "cast_executable_", "__padded_constant_", "__bias_folded_weight_",
+          "__demoted_constant_"}) {
```

> **변경 전 → 후**
> - **기존**: Flow 패스들이 IR에 새로 만드는 helper executable과 global은 5개 접두어 중 가장 큰 번호 다음부터 번호를 매겼습니다.
> - **추가**: 새 패스가 만드는 `cast_executable_N`, `__demoted_constant_N`도 셉니다.
> - **이유**: 새 패스 바로 뒤에 bias fold와 pad 패스가 같은 모듈에서 돌므로, 번호가 겹치면 symbol 이름이 충돌합니다. 헤더 주석(`.h` 65–68행)도 같이 고쳤습니다.

### 5-5. `$P/PluginRegistration.cpp`: 파이프라인에 패스 추가 (116–122행)

```cpp
114      passManager.addPass(
115          AMDAIE::createAMDAIEAssignDeviceAffinitiesPass(placementOptions));
116      // Now that every dispatch has a device, convert the inputs of contraction
117      // dispatches to the element type their device asks for
118      // (--iree-amdaie-contraction-input-type for amd-aie); dispatches on other
119      // devices keep theirs. Before the bias fold and the pads, which size their
120      // work by the element type. No-op when no device asks.
121      passManager.addPass(
122          AMDAIE::createAMDAIEDemoteContractionDispatchInputsPass());
123      // Fold a constant bias fused into an NPU contraction dispatch into K, so
```

> **변경 전 → 후**
> - **기존**: hook 함수 `extendFlowTransformPassPipeline`(104행, IREE가 Flow 단계에서 부름)은 패스를 장치 배정(`AMDAIEAssignDeviceAffinities`) → bias fold(`AMDAIEFoldBiasIntoContractionDispatches`) → contraction pad(`AMDAIEPadContractionDispatches`) → pooling pad(`AMDAIEPadPoolingDispatches`) 순서로 끼웠습니다.
> - **추가**: 장치 배정 직후에 새 패스 `AMDAIEDemoteContractionDispatchInputs`를 끼웁니다(`createAMDAIEDemoteContractionDispatchInputsPass()`는 이 패스를 만드는 함수). 어느 장치도 `contraction_input_type`을 갖지 않으면 즉시 할 일이 없습니다.
> - **바뀌지 않은 것**: hook 함수 `extendPreprocessingPassPipeline`(56행)이 기존 플래그로 upstream 패스 `DemoteContractionInputsPass`를 끼우는 부분은 그대로 있습니다.

### 5-6. 등록 파일 (`$P/Transforms/` 아래)

```diff
# Transforms/Passes.td
+def AMDAIEDemoteContractionDispatchInputs :
+    Pass<"iree-amdaie-demote-contraction-dispatch-inputs", "ModuleOp"> {
+  let summary = "Convert the inputs of contraction dispatches to the element type their device asks for ...";
+  let constructor =
+      "mlir::iree_compiler::AMDAIE::createAMDAIEDemoteContractionDispatchInputsPass()";
+}
# Transforms/Passes.h
+std::unique_ptr<Pass> createAMDAIEDemoteContractionDispatchInputsPass();
# Transforms/PassDetail.h
+#define GEN_PASS_DEF_AMDAIEDEMOTECONTRACTIONDISPATCHINPUTS
# Transforms/CMakeLists.txt
+    "AMDAIEDemoteContractionDispatchInputs.cpp"
# Transforms/test/CMakeLists.txt
+    "demote_contraction_dispatch_inputs.mlir"
```

`iree-opt --pass-pipeline='builtin.module(iree-amdaie-demote-contraction-dispatch-inputs)'`로 패스만 따로 돌릴 수 있습니다.

## 6. 상세: 새 패스 파일 `$P/Transforms/AMDAIEDemoteContractionDispatchInputs.cpp`

기존 Flow 단계 패스(`$P/Transforms/AMDAIEPadContractionDispatches.cpp`, `$P/Transforms/AMDAIEFoldBiasIntoContractionDispatches.cpp`)와 같은 구조입니다. dispatch마다 affinity로 target을 찾고, executable을 직접 고치고, `$P/Transforms/AMDAIEDispatchRewriteUtils.cpp`의 공용 helper 함수(`getFullTensorLoad`, `getConstantValue`, `createReplacementConstant`, `getHostAffinity`, `eraseDeadGlobals`, 위치는 0절)를 씁니다.

### 6-1. 함수 구성

아래는 모두 이 파일 안의 함수입니다. `fuseIntoProducer`와 `runOnOperation`은 패스 클래스 `AMDAIEDemoteContractionDispatchInputsPass`(478행)의 멤버 함수이고, 나머지는 이 파일 안에서만 쓰는 `static` 함수입니다. `FuncPlan`(121행)과 `Request`(471행)는 함수가 아니라 결과를 담는 구조체입니다.

| 단계 | 함수 | 행 | 하는 일 |
|---|---|---|---|
| 판단 | `canWriteElementType` | 73 | 앞 dispatch의 장치가 bf16을 직접 쓸 수 있는가 (지금은 host만) |
| 분석 | `isDemotableContraction` | 88 | 곱셈이 있는 named matmul/conv이고 입력이 대상 타입보다 넓은 float인가 |
| 분석 | `getLoadedBinding` | 107 | contraction 입력 → 전체 load → 읽기 전용 binding |
| 분석 | `analyzeFunc` | 129 | 바꿀 contraction과 binding 목록(`FuncPlan`) |
| 고치기 | `rebuildNamedOp` | 178 | 입력 타입이 바뀐 named op을 다시 만들어 body에 `extf`가 들어가게 함 |
| 고치기 | `demoteFunc` | 203 | executable 안의 binding, load, contraction 입력을 대상 타입으로 |
| 규칙 2 | `createTruncF` | 226 | 별도 elementwise truncf generic 생성 |
| 규칙 2 | `fuseTruncFIntoGeneric` | 250 | 출력을 만드는 generic의 body에 truncf를 합침 |
| 규칙 2 | `getFullTensorStore` | 287 | 결과 binding이 전체 store 하나로 쓰이는지 |
| 규칙 1 | `createDemotedConstant` | 307 | 상수를 RNE로 변환해 새 global |
| 정리 | `eraseDeadReshapeChains` | 343 | 안 쓰게 된 reshape, load, 상수 제거 |
| 규칙 4 | `createCastDispatch` | 370 | CPU 변환 dispatch 생성 |
| 공유 | `copyExecutableFor` | 431 | executable 복사 후 일부 dispatch만 연결 |
| 분석 | `collectLeafUses` | 457 | 값을 읽는 곳을 reshape를 지나 끝까지 모음 |
| 규칙 2 | `fuseIntoProducer` | 491 | 규칙 2 조건 확인과 적용 |
| 진입 | `runOnOperation` | 554 | 분석 → 결정 → 고치기 |

### 6-2. 1단계 분석: 장치별 그룹 (`runOnOperation` 안, 570–582행)

```cpp
570    // 1. Analysis: the dispatches whose device asks for a contraction input
571    // type, grouped by executable and type.
572    llvm::MapVector<std::pair<Operation *, Type>,
573                    SmallVector<IREE::Flow::DispatchOp>>
574        groups;
575    for (IREE::Flow::DispatchOp dispatch : dispatches) {
576      FloatType type = getConfigContractionInputType(
577          getDispatchExecutableTarget(dispatch, module));
578      if (!type) continue;
579      func::FuncOp func = getDispatchFunc(dispatch, module);
580      if (!func) continue;
581      groups[{func.getOperation(), type}].push_back(dispatch);
582    }
```

장치가 원하는 타입을 묻는 곳은 이 한 군데입니다. 장치 이름이나 backend 이름을 비교하지 않으므로 다른 가속기를 붙여도 이 부분은 바뀌지 않습니다.

### 6-3. 바꿀 contraction 찾기: 함수 `analyzeFunc` (129–173행)

```cpp
129  static FuncPlan analyzeFunc(func::FuncOp func, FloatType type) {
130    llvm::SetVector<Operation *> candidates;
131    func.walk([&](linalg::LinalgOp op) {
132      if (!isDemotableContraction(op, type)) return;
133      if (llvm::all_of(op.getDpsInputs(),
134                       [](Value input) { return getLoadedBinding(input); }))
135        candidates.insert(op);
136    });
137    // A binding is converted only if all its readers are converted: drop the
138    // candidates reading a binding (or load) used elsewhere, until stable.
       ...
165    FuncPlan plan;
166    for (Operation *op : candidates) {
       ...
170        plan.bindings.insert(getLoadedBinding(input).getArgNumber());
```

- 대상 op은 upstream 패스 `DemoteContractionInputsPass`(`$IREE/GlobalOptimization/DemoteContractionInputs.cpp`)와 같은 범위(`ContractionOpInterface`/`ConvolutionOpInterface`의 named op)입니다. 곱셈(`arith.mulf`)이 없는 pooling은 뺍니다(`isDemotableContraction`, 88행).
- 입력이 binding 전체를 읽는 load에서 와야 합니다. binding 타입을 바꾸면 그 binding을 읽는 다른 연산이 깨질 수 있으므로, **binding을 읽는 곳이 모두 바꿀 contraction일 때만** 바꿉니다. 조건에 안 맞는 입력은 f32 그대로 둡니다. 결과는 맞고, 빨라지지만 않을 뿐입니다.
- 후보를 빼는 반복(151–164행)은 한 바퀴에서 뺄 후보를 먼저 모은 뒤, 순회가 끝나고 한꺼번에 뺍니다. 처음에는 순회하면서 바로 지웠는데, `SetVector`는 내부가 배열이라 원소를 지우면 뒤 원소가 당겨져 하나를 건너뛰거나 끝을 넘어갈 수 있었습니다. 이를 고쳤고 lit 5번 경우로 확인합니다(7절).

### 6-4. executable 고치기: 함수 `rebuildNamedOp`, `demoteFunc` (178–223행)

```cpp
178  static void rebuildNamedOp(IRRewriter &rewriter, linalg::LinalgOp op) {
179    Location loc = op.getLoc();
180    OperationState state(loc, op->getName());
181    state.addOperands(op->getOperands());
182    state.addTypes(op->getResultTypes());
183    state.addAttributes(op->getAttrs());
       ...
193      op.getRegionBuilder()(b, *body, state.attributes.getAttrs(),
194                            /*emitError=*/{});
       ...
197    Operation *newOp = rewriter.create(state);
198    rewriter.replaceOp(op, newOp->getResults());
```

named linalg op은 만들 때 body의 block 인자 타입이 입력 원소 타입으로 정해집니다. 그래서 입력만 bf16으로 바꾸면 body가 f32인 채로 남아 IR이 맞지 않습니다. 이 함수는 op의 region builder로 body를 다시 만듭니다. 그러면 기존 Preprocessing demote가 만든 것과 같은 모양(bf16 입력 → `extf`로 f32 → 곱하고 f32 누산)이 됩니다. `indexing_maps`(transpose_b 등)와 다른 속성은 그대로 옮깁니다. upstream demote는 op 종류마다 템플릿으로 만들지만, 이 방식은 named op이면 종류를 가리지 않습니다.

### 6-5. 규칙 1: 상수 (함수 `createDemotedConstant`, 307–337행)

```cpp
323    const llvm::fltSemantics &semantics = elemType.getFloatSemantics();
324    DenseElementsAttr demoted =
325        attr.mapValues(elemType, [&](const APFloat &value) {
326          APFloat narrow = value;
327          bool losesInfo;
328          narrow.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
329          return narrow.bitcastToAPInt();
330        });
331    Value load = createReplacementConstant(
332        rewriter, module, v.getLoc(),
333        "__demoted_constant_" + std::to_string(counter++), demoted, global,
334        replacedGlobals);
```

- 반올림은 `arith.truncf`와 같은 RNE입니다. VGG에서 NPU weight 16개가 기존 경로(const-eval이 실행한 truncf)와 바이트까지 같았습니다.
- 같은 상수를 여러 dispatch가 읽으면 global은 한 번만 만들고 load만 복사합니다(`cache`).
- f32 원본 global은 다른 곳에서 안 읽게 되면 마지막에 `eraseDeadGlobals`가 지웁니다. CPU dispatch가 같이 읽으면 남습니다.

### 6-6. 규칙 2: 앞 dispatch에 합치기 (함수 `fuseIntoProducer`, 491–552행)

```cpp
494    if (!canWriteElementType(getDispatchExecutableTarget(producer, module),
495                             request.type))
496      return false;
497    unsigned resultNo = root.getResultNumber();
498    if (producer.getTiedResultOperandIndex(resultNo)) return false;
       ...
503    collectLeafUses(root, leaves, reshapes);
       ...
506    if (!llvm::all_of(leaves, [&](OpOperand *use) { return wanted.count(use); }))
507      return false;
       ...
522    if (!getFullTensorStore(func.getArgument(argNo))) return false;
523    if (callers[func].size() > 1)
524      func = copyExecutableFor({producer}, module, callers);
       ...
529    Value narrow = fuseTruncFIntoGeneric(rewriter, store.getValue(), request.type);
530    if (!narrow) {
531      rewriter.setInsertionPoint(store);
532      narrow = createTruncF(rewriter, store.getLoc(), store.getValue(),
533                            request.type);
534    }
       ...
547    root.setType(withElementType(root.getType(), request.type));
548    for (Operation *reshape : reshapes)
549      reshape->getResult(0).setType(
550          withElementType(reshape->getResult(0).getType(), request.type));
```

조건은 다섯 가지입니다.
1. 앞 dispatch P의 장치가 bf16을 쓸 수 있음(지금은 host만, 73행).
2. 결과가 입력과 묶인(tied) in-place 결과가 아님.
3. 이 값을 읽는 곳이 모두 bf16을 원하는 dispatch임(f32로 읽는 곳이 있으면 규칙 4).
4. P 안에서 결과 binding이 전체 store 하나로 쓰임.
5. P의 executable을 다른 dispatch도 부르면 복사본을 고침.

합치는 방법은 두 가지입니다. 출력을 만드는 연산이 모든 축이 parallel인 generic이고 결과를 다른 곳에서 안 쓰면, 그 generic의 body에 truncf를 넣습니다(`fuseTruncFIntoGeneric`). 기존 경로에서 dispatch 생성 단계가 elementwise fusion으로 만들던 것과 같은 모양이라, VGG의 im2col 13개와 FC bias+ReLU generic이 기존과 똑같이 나왔습니다. 그렇지 않으면(예: pooling named op 뒤) 같은 dispatch 안에 truncf generic을 따로 붙이고, CPU codegen이 둘을 한 루프로 합칩니다.

### 6-7. 규칙 4: CPU cast dispatch (함수 `createCastDispatch`, 370–424행)

pad 패스가 쓰는 함수 `createPaddingDispatch`(`$P/Transforms/AMDAIEDispatchRewriteUtils.cpp:125`), `createCropDispatch`(`$P/Transforms/AMDAIEPadContractionDispatches.cpp:260`)와 같은 구조입니다. 변환만 하는 함수(`load → truncf generic → store`)를 담은 `cast_executable_N`을 만들고, host affinity를 붙여 값 정의 바로 뒤에 dispatch를 넣습니다. 같은 값을 여러 NPU dispatch가 읽으면 cast는 한 번만 만듭니다(`castCache`).

### 6-8. 결정 순서 (`runOnOperation` 안, 620–666행)

```cpp
624    for (auto &[key, request] : requests) {
625      Value root = key.first;
626
627      // a. Constant: convert at compile time.
628      if (isa_and_present<DenseFPElementsAttr>(
629              getConstantValue(root, /*global=*/nullptr))) {
           ...
641        continue;
642      }
643
644      // b. Result of a host dispatch read only here: fuse into it.
645      if (auto producer = root.getDefiningOp<IREE::Flow::DispatchOp>()) {
646        if (fuseIntoProducer(rewriter, module, producer, cast<OpResult>(root),
647                             request, callers))
648          continue;
649      }
650
651      // c. Otherwise a host cast dispatch per operand value, after its
652      // definition, shared by the consumers reading that value.
```

판단은 값마다 한 번 합니다. 값은 reshape를 거슬러 올라간 원래 값(root)으로 묶습니다. 그래서 같은 값을 여러 NPU dispatch가 읽어도 변환은 한 번입니다.

## 7. 상세: lit 테스트 (`$P/Transforms/test/demote_contraction_dispatch_inputs.mlir`)

| # | 입력 | 확인하는 것 |
|---|---|---|
| 1 | CPU im2col → reshape → NPU matmul, 상수 weight | im2col generic body에 truncf, 출력 bf16, reshape도 bf16. weight는 `__demoted_constant_0`이고 RNE 확인용 값(1.00390625 → 1.0, 1.01171875 → 1.015625). f32 global 제거. NPU matmul은 bf16 입력, f32 출력 |
| 2 | 모델 입력, f32로도 반환되는 relu 결과, 두 NPU dispatch가 같은 executable | 모델 입력 2개와 relu 결과에 cast dispatch 3개. relu f32 결과는 반환용으로 남음. 공유 executable은 한 번만 변환 |
| 3 | 같은 executable을 NPU와 CPU가 호출, 같은 weight를 둘 다 읽음 | NPU 쪽만 복사본 `@mm_0`(bf16), CPU 쪽은 `@mm`(f32) 그대로, f32 weight global 유지 |
| 4 | `contraction_input_type`이 없는 NPU | 아무것도 바뀌지 않음 |
| 5 | NPU dispatch 하나에 matmul 3개: A는 입력을 혼자 읽음, B의 입력 하나는 add도 읽음, C는 B와 입력 하나를 공유 | A만 bf16. B는 1회차에, C는 2회차에(마지막 후보) 빠져 f32 유지. cast dispatch는 A의 입력 2개에만 생김. 함수 `analyzeFunc`의 후보 제거 반복 확인용 |

---

## 8. 검증

### 8-1. lit 테스트

새 테스트와, 공용 helper 함수를 같이 쓰는 기존 Flow 패스의 lit 테스트를 돌렸습니다. 테스트 파일은 모두 `$P/Transforms/test/` 아래입니다.

| 테스트 | 결과 |
|---|---|
| `demote_contraction_dispatch_inputs.mlir` (새, 5개 경우. `analyzeFunc` 수정 후 5번째 추가해 다시 통과) | 통과 |
| `pad_contraction_dispatches.mlir` (패스 `AMDAIEPadContractionDispatches`) | 통과 |
| `fold_bias_into_contraction_dispatches.mlir` (패스 `AMDAIEFoldBiasIntoContractionDispatches`) | 통과 |
| `pad_pooling_dispatches.mlir` (패스 `AMDAIEPadPoolingDispatches`) | 통과 |

### 8-2. VGG16 Flow IR 비교 (기존 `--numeric bf16` 대 새 패스)

기존 경로(스크립트 옵션 `--numeric bf16`, 즉 기존 플래그)와 새 경로(`--numeric f32` + 새 플래그 `--iree-amdaie-contraction-input-type=bf16`)를 `iree-compile` 플래그 `--compile-to=flow`로 Flow 단계까지만 컴파일해, 실행 순서대로 dispatch를 맞추고 executable 본문(이름 제외)을 비교했습니다.

| 설정 | 기존 | 새 패스 | 비교 |
|---|---|---|---|
| A. epilogue 없는 설정 (IREE 본체 플래그 `--iree-global-opt-detach-elementwise-through-reshape`, `--iree-dispatch-creation-no-fuse-into-contraction-conv-roots`를 추가) | 76개 | 75개 | 74개가 본문까지 동일. 다른 하나는 pool 뒤 truncf가 pool dispatch 안으로 합쳐진 것(별도 dispatch 1개 감소) |
| B. 사용자 스크립트(`scripts/onnx/options.py`) 기본 옵션 (bias fold 포함) | 87개 | 89개 | 앞 76개 동일, pool은 truncf 위치만 다름(값 같음), FC1·FC2 뒤 CPU cast 2개 추가(3-7절) |

- NPU dispatch는 두 설정 모두 입력 타입과 모양이 기존과 같습니다.
- 설정 B에서 NPU weight 16개(bias fold로 붙은 bias 행 포함)를 생략 없는 Flow IR에서 꺼내 비교했고, **바이트까지 동일**했습니다.
- pool 쪽 차이는 max pool이 크기 순서를 유지하는 연산이라 truncf를 앞에서 하든 뒤에서 하든 값이 같습니다.

### 8-3. VGG16 e2e (NPU 실행, 설정 B)

| | 기존 `--numeric bf16` | 새 패스 |
|---|---|---|
| 컴파일 | 105.5 s | 110.2 s |
| 실행 (5회) | 2.506, 2.124, 2.307, 2.484, 3.425 s (중앙값 2.48) | 2.304, 2.332, 2.271, 3.023, 3.587 s (중앙값 2.33) |
| corr / maxabs / top-1 | 0.99996 / 0.0479 / 일치 | 0.99996 / 0.0424 / 일치 |
| 실행마다 같은 출력 | 예 | 예 |

- **출력은 두 경로가 비트 단위로 같지 않습니다**(최대 차이 0.024). 차이는 FC1·FC2 출력의 bf16 변환을 기존은 NPU가, 새 패스는 CPU가 해서 생깁니다. NPU의 변환이 RNE가 아니라 내림이라는 것을 따로 확인했습니다([2026-11-04_npu_truncf_rounding.md](2026-11-04_npu_truncf_rounding.md)). 새 경로가 f32 기준에 조금 더 가까운 것(maxabs 0.0424 대 0.0479)도 이와 맞습니다. 출력 타입 변화로 GEMV 타일링이 달라진 영향이 섞였는지는 분리해 확인하지 않았습니다.
- 실행 시간은 같은 빌드 안에서도 2.1~3.6 s로 흔들려, 차이가 있다고 볼 근거가 없습니다.
- 이 측정은 `analyzeFunc` 수정(6-3절) 전에 했습니다. VGG의 NPU dispatch는 후보가 빠지는 경우가 없어 수정 전후 결과가 같습니다.
- 컴파일 시간 증가(약 5 s)는 weight 약 1.38억 개를 APFloat로 하나씩 변환하는 비용입니다(Flow까지만 컴파일하면 기존 약 2 s, 새 패스 약 8 s).

### 8-4. 장치별 변환 확인

3-6절 표와 같습니다. 한 모델 안에서 NPU matmul 13개는 bf16, CPU에 배치한 FC matmul 3개는 f32 입력과 f32 weight로 남았고, cast dispatch는 생기지 않았습니다.

## 9. 알려진 문제와 제약

- **NPU producer에는 합치지 않음 (1단계 범위)**: 앞 dispatch가 NPU면 함수 `canWriteElementType`(새 패스 파일 73행)이 거절해 규칙 4(CPU cast)로 갑니다. VGG FC처럼 NPU 출력이 바로 다음 NPU 입력이면, f32로 한 번 왕복하고 cast dispatch가 하나 더 돕니다. 2단계 대상입니다.
- **NPU의 f32 → bf16 변환이 내림**: dev에서 반올림 모드를 바꾸는 곳은 matmul ukernel 소스 `$P/Target/uKernels/npu4/peano/matmul.cc`(47행 `aie::swap_rounding`) 하나이고, 호출이 끝나면 하드웨어 기본값(내림)으로 되돌립니다. 그래서 ukernel 밖의 epilogue truncf는 내림입니다. 코어 시작에 설정하는 커밋 `822f0fed`(최지훈, 원격 브랜치 `origin/3devices-f32-aie2p-bf16`, 이 커밋이 추가하는 플래그 `--iree-amdaie-core-rounding-mode`)는 있지만 dev에는 아직 없습니다. 이 작업과는 별도로 처리하기로 했습니다. 2단계에서 NPU epilogue에 변환을 합칠 때 이 반올림이 결과에 그대로 나타나므로, 그 전에 이 커밋이 dev에 들어오는지 확인해야 합니다.
- **GPU로는 직접 확인하지 못함**: 이 빌드에는 Vulkan이 꺼져 있습니다(CMake 옵션 `IREE_TARGET_BACKEND_VULKAN_SPIRV=OFF`). 대신 CPU 배치로 같은 동작을 확인했습니다.
- **대상 op 범위**: named contraction/convolution op만 바꿉니다(함수 `isDemotableContraction`, 88행). 이미 generic으로 풀린 contraction과 동적 shape 입력은 건너뜁니다(f32 그대로, 결과는 맞음).
- **f32로도 읽히는 값**: 규칙 3(두 개 출력)이 없어 cast dispatch로 처리합니다. residual이 많은 모델(ResNet 등)에서 비용을 봐야 합니다.
- **host 장치가 없으면** 패스가 아무것도 하지 않습니다(공용 함수 `getHostAffinity`가 null. pad 패스 `AMDAIEPadContractionDispatches`와 같은 동작). 스크립트는 local-host를 항상 첫 장치로 둡니다.
- **dispatch 이름**: 변환 뒤에도 이름 끝이 `_f32`로 남습니다. 표시만의 문제입니다.
- **컴파일 시간**: 상수 변환이 APFloat 기반이라 VGG에서 약 5 s 늘었습니다. 비트 연산으로 바꾸면 줄일 수 있습니다.
- **(해결됨) 함수 `analyzeFunc`의 후보 제거**: 처음에는 후보 집합(`SetVector`)을 순회하면서 원소를 바로 지워, 다음 원소를 건너뛰거나 마지막 원소를 지울 때 끝을 넘어갈 수 있었습니다. 뺄 후보를 모은 뒤 순회가 끝나고 지우도록 고쳤고(151–164행), lit 5번 경우를 추가했습니다(6-3절, 7절).
- **작업 머신의 Peano 버전 (이 작업과 무관)**: 호스트용 스크립트 `scripts/docker/run-onnx.sh`는 환경 변수 `PEANO_INSTALL_DIR=/workspace/llvm-aie`(= 저장소 루트의 `llvm-aie/`)를 씁니다. 설치 문서(`docs/2026-07-06_env_setup/DEV_CONTAINER.md`)대로라면 여기에 v22가 있어야 하지만, 이 머신의 `llvm-aie/`는 아직 v19(`19.0.0.2025052701+31d2aa6e`)입니다. v19로는 VGG GEMV 코어의 stack이 부족해(1472 > 1024 B) 기존 경로와 새 경로 모두 컴파일되지 않았습니다. 검증은 v22(`build/peano_v22/llvm-aie`, `22.0.0.2026101201+83ba9ea9`)로 했습니다. 설치 스크립트 `build_tools/download_peano.sh`로 `llvm-aie/`를 v22로 갱신하면 스크립트를 그대로 쓸 수 있습니다(기존 v19를 덮어씀). 버전은 `llvm-aie/bin/clang --version`으로 확인합니다.

## 10. 남은 과제

| 단계 | 내용 |
|---|---|
| 2단계 | NPU epilogue에 truncf 합치기: 함수 `canWriteElementType`(새 패스 파일 73행)을 넓혀, NPU dispatch가 "matmul + 단일 elementwise epilogue" 모양이면 그 epilogue에 합침. 기대 효과는 VGG FC 뒤 cast 2개 제거(89 → 87개). 반올림 커밋 `822f0fed`의 dev 반영 여부 확인 필요 |
| 3단계 | f32로도 읽히는 값: producer가 f32와 bf16 두 개를 출력 (2-3절 규칙 3) |
| 4단계 | GPU와 섞어서 확인(Vulkan 빌드 필요), NPU 입력 쪽 변환은 백엔드가 지원할 때만 |
| 다른 모델 | ResNet(residual), MLP·transformer(NPU → NPU 연결)로 일반성 확인 |
| 스크립트·문서 | `scripts/onnx/options.py`의 스크립트 옵션 `--numeric`에 새 방식 선택지 추가(그 선택지는 Python 함수 `check()`의 vulkan 금지 검사에서 제외), `docs/device_placement.md`에 장치별 입력 타입 절 추가. 두 파일 모두 공용이라 팀과 정한 뒤 수정 |
| 정리 | 상수 변환 비트 연산화 (함수 `createDemotedConstant`) |

## 11. 재현 방법

모두 dev 컨테이너 안, 저장소 루트(`/workspace`)에서 실행합니다.

```bash
# 빌드
cmake --build build --target iree-opt iree-compile FileCheck

# lit (새 테스트)
T=compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test
build/tools/iree-opt --split-input-file \
  --pass-pipeline='builtin.module(iree-amdaie-demote-contraction-dispatch-inputs)' \
  $T/demote_contraction_dispatch_inputs.mlir \
  | build/llvm-project/bin/FileCheck $T/demote_contraction_dispatch_inputs.mlir

# VGG16 e2e: 기존 경로와 새 패스 (Peano v22 사용)
# --numeric, --runs, --outdir, --extra-flag는 스크립트 옵션(scripts/onnx/options.py).
# --extra-flag=<플래그>는 그 iree-compile 플래그를 그대로 덧붙임.
# 기존 경로: --numeric bf16 → 기존 플래그 --iree-amdaie-demote-contraction-inputs-to-bf16
# 새 경로:   --numeric f32 + 새 플래그 --iree-amdaie-contraction-input-type=bf16
export PEANO_INSTALL_DIR=/workspace/build/peano_v22/llvm-aie
source /opt/venv/bin/activate && export PYTHONPATH=/workspace/build/compiler/bindings/python
python3 scripts/onnx/e2e.py models/vgg16/vgg16-12.onnx --numeric bf16 --runs 5 \
  --outdir build/demote_study/e2e/old_bf16
python3 scripts/onnx/e2e.py models/vgg16/vgg16-12.onnx --numeric f32 \
  --extra-flag=--iree-amdaie-contraction-input-type=bf16 --runs 5 \
  --outdir build/demote_study/e2e/new_bf16

# Flow IR만 보기: 위 outdir의 compile_cmd.sh에서 -o 대상을 바꾸고 --compile-to=flow를 붙임
# (weight를 생략하려면 --mlir-elide-elementsattrs-if-larger=16 --mlir-elide-resource-strings-if-larger=16)

# 장치별 변환 확인: FC matmul을 CPU에 배치 (실험용 플래그, 장치 배정 패스의 옵션)
#   위 새 패스 명령에 --extra-flag=--iree-amdaie-placement-map=~matmul_1x=cpu 추가
```

호스트에서는 스크립트 `scripts/docker/run-onnx.sh`로 같은 명령을 돌릴 수 있습니다. 이 스크립트는 `PEANO_INSTALL_DIR`을 `/workspace/llvm-aie`로 고정하므로, 거기에 Peano v22가 설치돼 있어야 합니다(9절).

---

## 부록: 쉽게 풀어서 설명

**1. 원래 문제**

NPU는 bf16(16비트 실수)으로 계산해야 빠르고, GPU는 bf16을 받지 못해 f32(32비트)로만 계산합니다. 그런데 컴파일러는 "어느 연산을 어느 장치에 맡길지" 정하기 **전에** 타입을 바꾸고 있었습니다. 그래서 bf16을 켜면 모든 연산이 bf16이 됐고, GPU를 같이 쓰려면 NPU까지 f32로 돌아야 했습니다.

**2. 바꾼 것**

순서를 바꿨습니다. 먼저 연산마다 장치를 정하고, 그다음에 "NPU에 간 연산만" 입력을 bf16으로 바꿉니다. 장치마다 원하는 타입을 장치 정보(target)에 적어 두고, 새 패스가 그걸 읽습니다. GPU나 CPU는 아무것도 적혀 있지 않으니 f32 그대로입니다.

**3. 변환은 어디서 하나**

NPU가 f32를 읽어서 안에서 바꾸면 메모리를 두 배로 읽어야 해서 느립니다. 그래서 NPU에 넘기기 전에 바꿔 둡니다.
- weight처럼 고정된 값은 컴파일할 때 미리 bf16으로 저장합니다.
- 앞 단계가 CPU 연산이면, 그 연산이 결과를 저장할 때 bf16으로 저장하게 합니다.
- 둘 다 안 되면 CPU에서 변환만 하는 작은 작업을 하나 끼워 넣습니다.

**4. 결과**

VGG16에서 NPU가 받는 데이터와 weight는 기존 bf16 방식과 같고, 정확도와 속도도 같은 수준입니다. FC 연산을 CPU로 보내 보니 그 연산은 f32로 남아, "장치마다 다른 타입"이 실제로 동작하는 것을 확인했습니다. 마지막 FC 두 곳만 아직 NPU가 직접 bf16으로 저장하지 못해 CPU 변환 작업이 하나씩 끼어 있고, 이건 다음 단계에서 없앨 예정입니다.
