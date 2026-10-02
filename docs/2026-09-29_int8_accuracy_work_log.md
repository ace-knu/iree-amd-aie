# 작업 기록 — BERT int8 정확도 개선 (인계 §4)

시작 2026-09-29 · 워크트리 `_local/wj` @ `1549c7f6` · 인계 문서 `docs/2026-09-29_handoff_int8_accuracy.md`

> 진행 중인 문서다. 각 단계는 **한 일 / 측정값 / 판정**을 그때그때 적는다.
> 예상과 측정을 섞지 않는다.

---

## 0. 현재 상태

측정은 두 집합이다 — **NPU 대조용 16 문장**(NPU 한 문장에 13 초라 그 이상은 무리)과
**ORT 평가 64 문장**(`quant/eval.py`). 둘을 섞어 비교하면 안 된다.

| | NPU (16) | ORT (16) | ORT (64) |
|---|---:|---:|---:|
| 기준선 | 0.6633 | 0.6627 | 0.6670 |
| ① GELU | 0.6918 | 0.7022 | 0.7044 |
| **①+② LN+GELU** | **0.7783** | 0.7914 | 0.7858 |
| ①+②+③ FC2 int16 | 0.8288 | 0.8434 (`cboth`) | 0.8420 |
| **+ LayerNorm ukernel bf16 제거 (§5)** | **0.8409** | 0.8434 (`cboth`) | |
| + 남은 4 개도 넓힘 (§5.6) | 0.8415 | 0.8434 (`cboth`) | 0.8414 (`cboth`), **NPU 64 문장 0.8400** (§5.7) |
| **+ FC2 가중치 per-channel (§7.5)** | | | ORT 0.9447, **NPU 64 문장 0.9448** |
| 인계 문서의 상한 (문제 텐서 fp32, 가중치 per-tensor) | | | 0.929 |

①②③ 모두 NPU 까지 끝났고, LayerNorm ukernel 2 단계에서 bf16 으로 내리던 과정을 없애 NPU - ORT
간극도 닫았다(§5) — NPU 기준 0.6633 -> **0.8409 (+0.178)**. ③의 플래그
`--iree-amdaie-widen-residual-requantization` 는 **기본 켜짐**이고, `=false` 로 끄면 ② 와
vmfb 바이트 동일이다.

LayerNorm ukernel 의 계통 오차(3.9)는 §5.4 의 커널 수정으로 함께 고쳐졌다. 그 뒤 가중치 per-channel(§7)로
FC2 12 개를 바꿔 **NPU 64 문장 0.8400 -> 0.9448** 이 됐다(디스패치 136 = NPU 131 / CPU 5 그대로).

**커밋했다 (2026-09-30), 푸시는 안 했다.** `_local/wj` 의 로컬 브랜치 `bert-int8-accuracy`
(`origin/bert-onnx` 의 `1549c7f6` 위):

| 커밋 | 내용 | 검증 |
|---|---|---|
| `986577fc` | ① 채널별 상수를 코어 메모리에서 읽기 | 그 상태의 NPU 출력이 ① 당시와 16/16 바이트 동일 |
| `9a94c4c0` | ② residual 채널별 배수 | 그 상태의 vmfb 가 ② vmfb 와 바이트 동일 |
| `e2dc08da` | ③ residual 가지 int16, **기본 켜짐** | 기본값 vmfb = ③ vmfb, 플래그 끔 = ② vmfb, 둘 다 바이트 동일 |
| `769e9f74` | LayerNorm ukernel 2 단계 f32 (§5.4) | 독립 모듈 99.996 %, 전체 모델 세 구성 ORT 와 비교(§5.5) |
| `a738f27c` | 넓힌 residual 배수의 정수 범위 검사 (§5.6) | 플래그 끔 IR 바이트 동일, 23 개 전부 넓힘, NPU 0.8415 |
| `fdbb6176` | peano 용 `input.ll` 에서 상수식 GEP 의 `nuw` 제거 (§7.3) | `sqall` NPU 16/16 바이트 동일 |
| `4da498bf` | per-channel 가중치 dequant 를 matmul 밖으로 (2D, §7.3) | 재현체 NPU vs ORT 2048/2048, `sqall` 16/16 바이트 동일 |
| `c5c58239` | `COMPLETED_FIXES.md` 에 이 작업 기록 | 문서 |
| `b0051a62` | 같은 패스를 unit-batch `batch_matmul` 로 확장 (§7.5) | BERT FC2 12 개 i8, f32 matmul 0 |
| `30bd07b0` | 새 패스 `AMDAIEPerChannelIntegerRequantization` (§7.5) | NPU 64 문장 0.9448, `sqall` 16/16 바이트 동일 |

바뀐 파일:

| 파일 | 어느 항목 |
|---|---|
| `Transforms/AMDAIELowerToUKernels.cpp` | ① 코어 지역 상수 읽기 패턴, ② LayerNorm 모양 게이트 |
| `Transforms/AMDAIECreateAIEWorkgroup.cpp` | ① 상수 체인 코어로 되밀기 + 유도 변수 갈아 끼우기 |
| `Transforms/AMDAIERaiseLayerNorm.cpp` | ② 채널별 Mb, ③ 재양자화 넓히기 (플래그) + producer 를 영역 안에, §5.6 넓힌 배수의 정수 범위 검사 |
| `Transforms/AMDAIEUnfoldLayerNormResidual.cpp` (새 파일) | ③ 코드젠 첫머리에서 producer 를 영역 밖으로 |
| `Transforms/Passes.{td,h,cpp}`, `PassDetail.h`, `CMakeLists.txt` | ③ 새 패스 등록, 파이프라인에 추가 |
| `Transforms/Utils/AMDAIELayerNormUtils.{h,cpp}` | ② residual 채널 스케일 벗겨내기, ③ `kLayerNormResidualInRegion` |
| `Transforms/KernelDispatch.cpp` | ② 뱅크 인식 타일 재현 |
| `Target/uKernels/npu4/peano/layernorm.cc` | ② `(4, 768)` 인스턴스 4 개, §5.4 2 단계 f32 |
| `Target/XCLBinGen.cpp` | §7.3 상수식 GEP 의 `nuw` 제거 |
| `Transforms/AMDAIEFoldPerChannelDequantIntoMatmul.cpp` (새 파일) | §7.3 2D matmul, §7.5 batch_matmul |
| `Transforms/AMDAIEPerChannelIntegerRequantization.cpp` (새 파일) | §7.5 채널별 정수 재양자화 |
| `PluginRegistration.cpp` | §7 두 패스를 전처리에 등록 |
| `docs/COMPLETED_FIXES.md` | 6 번 항목 |

모델 쪽 `fold_bias_into_k_general.py` 의 per-channel 판은 저장소가 아니라 스크래치 복사본으로만 있다(§7.5).

---

## 1. 사전 재현 — 완료 (2026-09-29)

인계 §1의 기대값을 그대로 확인했다. **공유 빌드 락 0회** — 새 커밋 `1549c7f6`이
`models/`와 `docs/`만 바꿔서 재빌드가 필요 없었다.

| 확인 항목 | 문서 | 측정 | |
|---|---|---|---|
| `bertx_int8.onnx` md5 | `a053ffb4…` | `a053ffb4e3d9cd3232e2ead0bd3e7ce8` | ✓ |
| 평가 64문장 corr (fp32 대비) | 0.667 | **0.6670** (최저 0.6041) | ✓ |
| 디스패치 | 136 = NPU 131 / CPU 5 | **136 = NPU 131 / CPU 5** | ✓ |
| 반복 실행 | 동일 | 3회 바이트 동일 | ✓ |
| NPU 대비 fp32 = ORT 대비 fp32 | 같아야 함 | **0.6633 vs 0.6627** (16문장) | ✓ |

CPU 5개 = `dispatch_0,1,2`(임베딩 + 임베딩 LN) + `dispatch_134,135`(마지막 LN).
raise된 `custom_op` 23개.

### 1.1 주의 — 한 문장으로 판정하면 안 된다

문장 0만 보면 `ORT 0.6644 / NPU 0.6409`로 NPU가 나빠 보인다. 16문장 평균으로는
차이가 **+0.0006**이다. 문장별 차이가 ±0.03으로 **양쪽으로** 흩어진다.

`NPU vs ORT` 자체는 **0.838**이다. 두 출력이 원소 단위로는 꽤 다른데 fp32에서 떨어진
거리는 같다 — 인계 §2의 "int8 모델끼리의 corr 은 지표로 쓰면 안 된다"가 이 형태로
나타난다.

### 1.2 재현 비용을 줄이는 사실

**`_local/int8_debug/bert_base_L12.onnx` 가 저쪽 `models/bert_base/bert_base.onnx` 와
동일한 export 다.** 그 파일을 소스로 파이프라인을 돌리면 md5가 문서값과 일치한다.
`export_bert_base.py`를 돌릴 필요가 없다 — `torch`·`transformers` 설치도 HuggingFace
다운로드도 불필요하다. `quant/calib_ids.npy`·`eval_ids.npy`도 커밋돼 있어 토크나이저가
필요 없다.

```bash
CALIB=real models/bert_base/prepare_bert_i8.sh \
  /workspace/_local/int8_debug/bert_base_L12.onnx /workspace/_local/wj_models/v3
BERT_FP32=/workspace/_local/int8_debug/bert_base_L12.onnx \
  python3 models/bert_base/quant/eval.py /workspace/_local/wj_models/v3/bertx_int8.onnx
```

---

## 2. ① FC1 디스패치에 채널별 상수 — GELU 쪽 SmoothQuant

**목표**: 0.667 → 0.704 (+0.04)

### 2.1 무엇을 하는 것인가

FC2 입력(= GELU 출력)을 채널별로 `÷s`(3072개 상수) 하고, 같은 `s`를 FC2 가중치에
`diag(s)·W`로 미리 곱해 둔다. 수학적으로 항등이지만 활성 쪽의 채널 간 동적 범위가
평탄해져서 int8 격자를 덜 낭비한다.

```
s_j = max|X_j|^α / max|W_j|^(1−α)        α = 0.6
X' = X / s     (런타임, FC1 꼬리에서)
W' = diag(s)·W (컴파일 타임, 모델에 구움)
```

가중치 쪽은 모델 변환(`quant/smooth.py`)이 처리한다. **백엔드가 해야 하는 것은
런타임 쪽 `÷s` 하나**다.

### 2.2 왜 컴파일이 안 되나 — 인계 문서의 진단은 틀렸다

인계 §4 는 "코어 입력 DMA 채널이 2 개(활성·가중치)뿐인데 상수가 세 번째로 L1 에 들어가야
해서" 라고 적었다. **실제로 실패한 연결은 DDR -> L2 이고, 채널이 바닥난 쪽은 shim 이다.**

```
'amdaie.connection' op no producer DMA channel available
  target: logicalobjectfifo<memref<64xf32, 1 : i32>, 2>     <- L2 (메모리 공간 1)
  source: logicalobjectfifo<memref<3072xf32>> @ %tile_0_0   <- shim (row 0)
```

에러는 **producer 쪽**, 즉 shim 타일이 낸다. npu4 shim 은 방향당 MM2S 가 2 개고 matmul 이
활성·가중치로 둘 다 쓴다.

**코어 L1 은 오히려 여유가 많다.** 코어가 한 번에 쓰는 것은 `memref<1x8x8xf32, 2>` = **256 B**
뿐이고, 기존 사용량은 64 KB 중 약 18.5 KB 다(가중치 6656x2 + 활성 832x2 + 출력 512x2 +
누산기 2048). 해법이 코어 지역 상수인 것은 어차피 맞지만 **L1 예산은 걱정할 필요가 없었다.**

인계의 괄호 "(stream 단계까지는 바인딩이 안 늘어난다)" 는 맞다 — 상수는 기존 read-only
버퍼의 다른 오프셋(`binding(1) offset(%c67310592)`)에 놓인다.

### 2.3 선례 — γ/β가 똑같은 문제였다

`AMDAIELowerToUKernels.cpp:332 materializeCoreLocalConstant`. 상수를 **연산자 operand가
아니라 코어 데이터 메모리의 `memref.global`** 로 만든다.

- `__amdaie_core_constant_<n>` 이라는 private `memref.global`
- 코어별 링커 스크립트가 스택·타일 버퍼 뒤의 남는 공간에 배치
- **DMA 채널도 전송도 필요 없다**
- 같은 상수는 하나를 공유 (`getInitialValueAttr()` 비교로 중복 제거)
- `memref.get_global` → `bufferization.to_tensor`(restrict, non-writable)로 텐서화

LayerNorm은 `:433`에서 `gammaBetaAttr`에 이걸 쓴다.

**할 일**: 같은 처리를 FC1+GELU 꼬리(ukernel이 아니라 생성된 elementwise 코드)의 작은
상수 operand에 적용한다.

### 2.4 진행

| 단계 | 상태 |
|---|---|
| (a) ORT 시뮬레이션으로 목표 수치 확인 (인계 §6-1) | **완료 — 0.7044** |
| (b) 모델 변환 경로 확정 (`smooth.py gelu α0.6` → 양자화 → 접기) | **완료** |
| (c) 컴파일 — 실패 메시지 재현 | **완료 + 원인 규명** |
| (d) 백엔드 수정 — 꼬리 상수를 코어 지역 상수로 | **완료 (단일 디스패치 확인)** |
| (e) 전체 모델 컴파일 + 디스패치 수 (136 = NPU 131 / CPU 5) | **완료** |
| (f) NPU 실행 — NPU 대비 fp32 = ORT 대비 fp32 인지 | **완료 — 0.692, ORT 0.702** |
| (g) 회귀 — 기준선 모델을 새 컴파일러로 다시 | **완료 — 바이트 동일** |

#### (a) ORT 시뮬레이션 — 통과 (2026-09-29)

```bash
python3 quant/smooth.py <fp32> bert_sq_gelu.onnx quant/calib_ids.npy 0.6 gelu   # 12 sites
CALIB=real models/bert_base/prepare_bert_i8.sh bert_sq_gelu.onnx <out>
python3 quant/eval.py <baseline>/bertx_int8.onnx <out>/bertx_int8.onnx
```

| | corr 평균 | 최저 |
|---|---:|---:|
| 기준선 | 0.6670 | 0.6041 |
| **SmoothQuant GELU α0.6** | **0.7044** | 0.6340 |

인계 §3.4의 **B' = 0.704** 와 일치. +0.037.

#### (c) 컴파일 실패 재현 — 완료, 그리고 문서보다 한 겹 더 있었다

실패는 그대로 재현된다.

```
'amdaie.connection' op no producer DMA channel available
  (!amdaie.logicalobjectfifo<memref<64xf32, 1 : i32>, 2>,            <- L2 에 64 f32
   !amdaie.logicalobjectfifo<memref<3072xf32, strided<[1], offset: 16852224>>>)   <- DDR 의 3072 f32
```

실패 지점은 **FC1 MatMul 디스패치**(`[1,32,832] x [832,3072]`)다. 그 꼬리에 GELU 와
`x (1/s)` 가 융합돼 있다.

```mlir
%2 = subspan binding(1) offset(%c67310592) : tensor<3072xf32>   // 1/s
%10 = linalg.batch_matmul(...) -> tensor<1x32x3072xi32>
%11 = linalg.generic ins(%10, %6) {  ...  %27 = arith.mulf %26, %in_8  ...  }
```

**인계 문서의 "꼬리의 작은 상수 operand 에 같은 처리를 하면 된다"는 그대로는 안 된다.**
`materializeCoreLocalConstant` 는 상수의 **바이트**(`DenseElementsAttr`)를 받는데, 이 시점의
상수는 바이트가 없고 **DDR 오프셋뿐**이다. γ/β 가 되는 이유는 Preprocessing 의 raise 가
바이트를 미리 읽어 `kLayerNormGammaBeta` 속성으로 실어 주기 때문이고, FC1 쪽에는 그런 단계가 없다.

**원인은 인라인 한계였다.** `iree-flow-inline-constants-max-byte-length` 의 기본값이 **256 B**
인데 이 상수는 3072 x f32 = **12,288 B** 라 디스패치 밖에 남아 바인딩이 됐다. 한계를 올리면
바이트가 디스패치 안으로 들어온다.

```bash
--iree-flow-inline-constants-max-byte-length=16384
```

```mlir
%cst_8 = arith.constant dense_resource<__elided__> : tensor<3072xf32>   // 바이트가 보인다
%9 = linalg.generic ins(%8, %cst_8 : tensor<1x32x3072xi32>, tensor<3072xf32>) ...
```

DMA 에러는 아직 남는다 — **아무도 이 `arith.constant` 를 코어 지역 상수로 바꾸지 않기 때문**이다.
전제 조건만 갖춰진 상태다.

⚠️ 이 플래그는 전역이므로 다른 디스패치의 상수도 함께 인라인된다. 회귀 확인이 필요하다.

#### (d) 백엔드 수정 — FC1 디스패치 컴파일 성공

**L1 예산은 문제가 아니었다.** 타일링 후 IR 을 보니 코어가 한 번에 쓰는 것은
`memref<1x8x8xf32, 2>` = **256 B** 뿐이고, 기존 L1 사용량은 가중치 6656x2 + 활성 832x2 +
출력 512x2 + 누산기 2048 ≈ **18.5 KB** 였다. 3072 x f32 = 12 KB 를 통째로 구워도 64 KB 안에
넉넉히 들어간다.

##### 상수가 실제로 흐르던 경로

```
arith.constant 3072xf32                      (DDR, 인라인 한계를 올린 뒤)
  -> extract_slice [%arg2] [512]             워크그룹 forall, N 을 512 씩 6 번
  -> pack into memref<8x64xf32, 1>           L2
  -> extract_slice [%arg5, 0] [1, 64]        코어 forall (4 x 8)
  -> pack into memref<1x8x8xf32, 2>          L1
```

DDR -> L2 구간이 **shim 의 MM2S 채널**을 하나 요구하는데 npu4 의 shim 은 방향당 2 개고
matmul 이 활성·가중치로 이미 둘 다 쓴다. `no producer DMA channel available` 은 코어가 아니라
**shim** 이 낸 에러였다.

##### 한 일 — 세 군데

체인의 모든 단계(단위 스트라이드 연속 `extract_slice`, 마지막 축 단일 타일 `linalg.pack`)가
**인덱스 이름만 바꾼다**. 그래서 코어가 읽는 값은 원래 상수의 연속 구간 하나이고, 체인 전체를
코어 지역 복사본의 슬라이스로 바꿔도 커널이 보는 값은 같다.

| 파일 | 한 일 |
|---|---|
| `AMDAIELowerToUKernels.cpp` | 새 패턴 `ReadStagedConstantFromCoreMemory`. L1 버퍼를 채우는 `linalg.pack` 을 뿌리로, 체인을 상수까지 거슬러 올라가며 선형 오프셋을 누적하고 `materializeCoreLocalConstant` 의 슬라이스로 치환한다. 16 KB 를 넘는 상수는 거른다 |
| `AMDAIECreateAIEWorkgroup.cpp` | (1) 코어 지역 상수를 읽는 순수 체인(`get_global` -> 뷰 -> 인덱스 연산)을 코어 안으로 되밀어 넣는다 — 앞선 패스들이 루프 불변이라고 밖으로 끌어내 놓기 때문이다. (2) 코어 본문을 새 루프로 splice 한 뒤 **유도 변수를 새 루프 것으로 갈아 끼운다** |
| — | 기존 `AMDAIEInsertCores` 의 `get_global` 처리는 건드리지 않았다 |

(2)가 필요한 이유: 이 패스는 코어마다 루프를 새로 만들어 본문을 옮겨 넣는데, 옮겨진 op 이
원래 루프의 유도 변수를 그대로 가리키고 있었다. 오브젝트 fifo 만 만지는 기존 코어 본문에는
그런 참조가 없어서 드러나지 않던 구멍이다.

##### 결과

```
$ iree-compile module_main_graph$async_dispatch_11.mlir --compile-mode=hal-executable
-rw-r--r-- 822148  /tmp/d11.pdi        # 4.1 초
```

`memref.global "private" constant @__amdaie_core_constant_0 : memref<3072xf32>` 하나가 생기고,
코어 안에서 `subview[%arg2 * 512 + %col * 64] [64]` 로 읽는다. **DDR 연결도 L2 버퍼도 사라졌다.**

#### (e) 전체 모델 — 디스패치 수 그대로

```
iree-compile /tmp/bert_sq.mlir ...  ->  208 MB vmfb, 4분 19초, 에러 0
```

| | 총 | NPU | CPU |
|---|---:|---:|---:|
| 기준선 | 136 | 131 | 5 |
| **SmoothQuant GELU** | **136** | **131** | **5** |

CPU 5 개 목록도 같다 (`dispatch_0,1,2` = 임베딩 + 임베딩 LN, `dispatch_134,135` = 마지막 LN).

#### (f) NPU 실행 — 0.692, 목표에 0.012 못 미침

16 문장, fp32 대비 corr. 문장당 13.4 초(인계 §2 의 열화 상태).

| | ORT | NPU | NPU - ORT |
|---|---:|---:|---:|
| 기준선 | 0.6627 | 0.6633 | **+0.0006** |
| **SmoothQuant GELU** | 0.7022 | **0.6918** | **-0.0104** |
| 개선폭 | +0.0394 | **+0.0284** | |

**정확도는 확실히 올랐다** — NPU 기준 0.663 -> 0.692 (+0.028).

**다만 기준선에 없던 NPU-ORT 간극 -0.0104 가 생겼다.** 16 문장 중 13 개가 음수라
분산으로는 설명되지 않는다(부호 검정 p ~ 0.02). 기준선에서는 NPU 와 ORT 가 fp32 로부터
같은 거리였는데, SmoothQuant 를 켜면 NPU 가 조금 더 멀다.

확인한 것:

- 상수 바이트 자체는 같다 — 디스패치 소스에 `dense<"0x...">` 로 박혀 있는 것을 확인했다.
- 꼬리 연산 순서도 같다 (`mulf %in_9` -> `divf` -> `roundeven` -> clamp -> `fptosi`).
- **배수의 동적 범위가 매우 넓다.** SQ 모델의 12 개 `sq_div_*` 상수:

  | | 최소 | 최대 | 비율 | 중앙값 |
  |---|---:|---:|---:|---:|
  | `sq_div_1` | 0.106 | **828.5** | **7785 x** | 0.264 |
  | `sq_div_2` | 0.141 | 64.6 | 459 x | 0.296 |
  | `sq_div_5` | 0.163 | 89.5 | 551 x | 0.333 |
  | 나머지 | ~0.12~0.17 | 1.7~38 | 14~347 x | 0.31~0.45 |

##### 간극의 출처 — FC1 꼬리는 아니다 (비트 단위로 확인)

FC1 디스패치와 같은 모양의 **독립 모듈**을 만들어(같은 batch_matmul, 같은 꼬리, 3072 짜리
채널별 상수를 모듈 안에 그대로) NPU 와 CPU 양쪽으로 컴파일하고 같은 입력으로 돌렸다.
98,304 원소, 배수는 0.106 ~ 828 전 범위.

| | numpy 대비 일치 | 최대 \|차이\| |
|---|---:|---:|
| IREE-CPU | 100.00 % | 0 |
| **NPU** | **100.00 %** | **0** |
| NPU vs CPU | 100.00 % | 0 |

배수 상위 10 % 채널에서도 차이 0 이다. 이것으로 세 가지가 동시에 배제된다.

- **이 재작성은 값을 보존한다.** 채널 매핑이 맞다.
- **코어의 f32 산술**은 이 꼬리에서 numpy float32 와 원소 단위로 같다.
- **erf 다항식 근사**는 98 K 원소에서 반올림을 하나도 뒤집지 않는다.
  (별도로 잰 최대 절대오차 2.787e-07 과 일치한다.)

##### 그러면 -0.0104 는 무엇인가 — §5 에서 LayerNorm ukernel 로 확정

IREE CPU 빌드를 사이에 끼운 이분:

| SmoothQuant 모델, 16 문장 | fp32 대비 corr | ORT 대비 |
|---|---:|---:|
| ORT | 0.7022 | — |
| IREE-CPU | 0.7036 | +0.0015 |
| NPU | 0.6918 | **-0.0104** |

그러나 **이것을 "NPU 코드젠 문제" 라고 부를 근거는 없다.** 격리해서 잴 수 있었던 곳,
즉 (1) 이 바꾼 FC1 꼬리는 비트 단위로 정확했다. AMD-AIE 경로에는 코드젠 말고도
LayerNorm raise(수치를 바꾸는 그래프 재작성), softmax·layernorm ukernel 이 들어 있고,
어느 것이 관여하는지 가리지 않았다.

관점을 바꿔야 한다. **NPU 와 ORT 는 원래 원소 단위로 크게 다르다** — 기준선에서도
원소 corr 0.838, SQ 에서 0.829 다. 기준선에서는 그 차이가 fp32 기준으로 **치우치지 않았을**
뿐이고, SQ 에서는 약간 치우친다.

| 16 문장, NPU - ORT | 평균 | 표준편차 | 표준오차 | | 음수 |
|---|---:|---:|---:|---:|---:|
| 기준선 | +0.0006 | 0.0195 | 0.0049 | 0.1 SE | 10/16 |
| SmoothQuant | **-0.0104** | 0.0185 | 0.0046 | **-2.3 SE** | 14/16 |

**2.3 표준오차다.** 실재할 가능성이 높지만 16 문장으로는 경계에 있다.
다음 단계는 코드 뒤지기가 아니라 **문장 수를 늘리는 것**이다(평가셋 64 문장).
NPU 한 문장이 13.4 초이므로 64 문장이면 락을 14 분 잡는다.

### 2.5 (1) 결론

| | |
|---|---|
| 컴파일 | 전체 모델 208 MB vmfb, 에러 0, 4분 19초 |
| 디스패치 | **136 = NPU 131 / CPU 5** — 기준선과 동일, CPU 목록도 동일 |
| 회귀 | 기준선 모델 출력 **16/16 문장 바이트 동일** |
| **NPU 정확도 (16 문장)** | **0.6633 -> 0.6918 (+0.0284)** |
| ORT 정확도 (같은 16 문장) | 0.6627 -> 0.7022 (+0.0394) |
| ORT 정확도 (평가 64 문장) | 0.6670 -> 0.7044 (+0.0374) |

인계의 "+0.04" 는 ORT 수치다. ORT 에서는 그대로 나왔고 NPU 에서는 +0.028 이다.

**코드 변경 2 개 파일, 299 줄 추가.** 커밋 `986577fc`.

NPU 가 ORT 보다 0.0104 낮던 것은 (1) 의 범위를 벗어난 별개 현상이었다 — 출처는 LayerNorm ukernel 의
bf16 연산이고 §5 에서 고쳤다.

---

## 3. ② LayerNorm residual 합의 배수를 채널별로 — LN 쪽 SmoothQuant

### 3.1 목표 수치 — 인계 §3.4 를 그대로 읽으면 안 된다

인계 §3.4 는 B(LN+GELU) 0.716, B'(GELU 만) 0.704 로 적었다. **B' 에만 `+ 접기` 표시가 있다.**
두 행은 같은 기준이 아니다 — 접기는 약 +0.07 을 준다(§3.2 에서 0.595 -> 0.667).

배포 경로(`prepare_bert_i8.sh`, 접기 포함)로 직접 쟀다. ORT, 평가 64 문장:

| | corr | 최저 |
|---|---:|---:|
| 기준선 | 0.6670 | 0.6041 |
| ① GELU 만 | 0.7044 | 0.6340 |
| **①+② LN+GELU** | **0.7858** | 0.7464 |

**(2) 는 +0.081 이다.** 0.716 + 0.07 = 0.786 으로 인계 수치와도 맞는다.

### 3.2 모델 쪽은 이미 다 되어 있다

`quant/smooth.py <fp32> <out> calib_ids.npy 0.6 all` 이 LN 24 곳 + GELU 12 곳 = 36 곳을 처리한다.
LN 사이트에서 하는 일:

- `gamma`, `beta` 를 `s` 로 나눈다 — **공짜**, 초기값만 바뀐다
- LN 출력의 MatMul 이 아닌 소비자(= residual 경로)마다 `Mul(t, s)` 를 끼운다 (`sq_res_{k}`)

### 3.3 배수 범위 — GELU 쪽과 전혀 다르다

24 개 `sq_res_*` 상수(각 768 개):

| | 최소 비율 | 중앙 | 최대 비율 |
|---|---:|---:|---:|
| LN 사이트 | 3.4 x | **8.1 x** | 17.3 x |
| (참고) GELU 사이트 | 14 x | ~100 x | **7785 x** |

int16 잔여 합의 여유를 몇 비트나 내주느냐가 관건인데 **LN 쪽은 3~4 비트면 된다.**

### 3.4 백엔드가 해야 하는 일

`AMDAIERaiseLayerNorm.cpp`:

- `computeResidualMultipliers(scaleA, scaleB)` -> 스칼라 `Ma`, `Mb` + `commonStep`.
  `Mb` 를 채널별 벡터로 만들고, int16 범위 검사를 `max_j (scaleB * s_j)` 로 해야 한다.
  `commonStep >= 127 * (scaleA + scaleB * max_j s_j) / 32767`
- `buildResidualProducer` -> `h = (a*Ma + b*Mb + 2^7) >> 8` 의 `cMb` 를 텐서 피연산자로.
- `Utils/AMDAIELayerNormUtils.cpp` 의 `matchDequantize` 가 지금은 본문이 정확히 2 개
  (`sitofp`, `mulf` 스칼라)여야 한다. 뒤에 붙는 채널별 `mulf` 를 받아들이고 그 벡터를
  돌려줘야 한다.

ukernel 은 그대로다 — int16 `h` 를 받는 `layer_norm_i16_*` 가 이미 있다.

### 3.5 진행

| 단계 | 상태 |
|---|---|
| (a) ORT 시뮬레이션 | **완료 — 0.7858** |
| (b) 컴파일 — 무엇이 걸리는지 | **완료 — 벽이 세 개** |
| (c) 백엔드 수정 | **완료 — 컴파일 통과** |
| (d) 디스패치 수 | **완료 — 136 = NPU 131 / CPU 5** |
| (e) NPU 실행 | **완료 — 0.7783** |
| (f) 회귀 | **완료 — 바이트 동일 16/16** |

### 3.6 벽 세 개 — 서로 다른 것들이었다

#### 벽 1 — 매처가 거부한다

`sq_res` 곱이 dequantize 와 residual add **사이에 별도 generic** 으로 앉는다. 양자화기가
그 뒤에 QDQ 를 넣지 않아 f32 상태로 남는다.

```
%92 = generic ins(%i8)          { sitofp ; mulf %cst_172 }   <- dequantize (스칼라)
%93 = generic ins(%92, %cst_56) { mulf }                     <- sq_res, 768 벡터
%94 = generic ins(%91, %93)     { addf }                     <- residual add
```

`matchDequantize` 는 본문이 정확히 2 개여야 해서 `%93` 에서 실패한다. 결과:
LayerNorm 24 개가 전부 일반 로워링으로 떨어져 **159 = NPU 108 / CPU 51**.

고침: `peelChannelScale` 이 `x * c[j]` 브로드캐스트 곱을 벗겨내고 그 아래 dequantize 로
내려간다. 양쪽 가지 모두 시도하되 residual 쪽만 허용한다(다른 쪽에 있으면 접을 곳이 없다).
raise 23 개 복구.

#### 벽 2 — 타일 크기 계산이 상수를 활성으로 센다

`KernelDispatch.cpp` 의 `setRootConfigForLayerNormCopyPipeline` 이 residual producer 의
피연산자를 전부 행마다 흐르는 것으로 센다.

```
bytesPerRow += features * (elementBitWidth / 8) * 2;
```

`tensor<768xi32>` 가 여기 걸려 6144 -> 12288, `rowsPerCore` 8 -> 4. 그런데 ukernel 은
`m in {8, 16}` 뿐이라 매칭 실패 -> `custom_op` 이 버퍼화까지 살아남아 터진다.

```
'iree_linalg_ext.custom_op' op operand #0 must be ... ranked tensor ...,
but got 'memref<4x768xi16, 2 : i32>'
```

숫자가 정확히 떨어진다: `768*(2+1)*2 + 768*1*2*2 - 768*2 = 6144`, `(65536-4224)/6144 = 9 -> 8`.
상수를 더하면 12288, `61312/12288 = 4`.

고침: 컴파일 타임 상수는 `bytesPerRow` 가 아니라 `flatBytes` 에 한 번만 더하고
`streamedInputs` 에서도 뺀다. **(2) 를 위한 우회가 아니라 원래 있던 계산 오류다** —
상수 피연산자가 처음 생겨서 드러났다.

#### 벽 3 — 코어 데이터 메모리가 2048 B 모자란다

타일 (0,2) 의 실제 배치:

```
stack        0 ..  4096
i16 중간  4096 .. 16384   (뱅크 0)
i8 x2    16384 .. 28672   (뱅크 1, 4096 낭비)
i8 x2    32768 .. 45056   (뱅크 2, 4096 낭비)
i8 x2    49152 .. 61440   (뱅크 3)
.data    61440 .. 65536   = 4096 B
```

버퍼가 16 KB 뱅크 단위로 배치되면서 뱅크마다 4096 씩 버려진다. `.data` 에 4096 B 만 남는데
γ/β 가 3072 을 쓰므로 배수 3072 가 안 들어간다. **i16 으로 줄여도(1536) 4608 > 4096 이라
m=8 로는 불가능하다.**

고침 — 행 타일을 4 로:

- `layernorm.cc` 에 `(4, 768)` 인스턴스 4 개 추가. `layernorm_impl(in, out, M, N, ...)` 에서
  M 은 런타임 인자라 커널 코드는 그대로다
- ukernel 게이트에 `m == 4` 허용
- 코어 지역 테이블이 있으면 `rowsPerCore` 를 절반으로 (측정한 배치를 주석에 남겼다)

#### 덤 — (1) 의 패턴을 일반화했다

특성 차원이 타일링되지 않으면 staging 이 `linalg.pack` 이 아니라 **`linalg.copy`** 다.

```
%cst_0 : tensor<768xi32>            DDR
%15 = linalg.copy %cst_0 -> L2
%28 = linalg.copy %15    -> L1
```

`ReadStagedConstantFromCoreMemory` 를 op 타입에 대해 템플릿화해 둘 다 뿌리로 받는다.

---

### 3.7 NPU 실행 — 0.7783

16 문장, fp32 대비 corr:

| | ORT | NPU | NPU - ORT | |
|---|---:|---:|---:|---|
| 기준선 | 0.6627 | 0.6633 | +0.0006 | 0.1 SE, 음수 10/16 |
| ① GELU | 0.7022 | 0.6918 | -0.0104 | -2.3 SE, 음수 14/16 |
| **①+② LN+GELU** | **0.7914** | **0.7783** | **-0.0132** | **-4.7 SE, 음수 16/16** |

**NPU 기준 0.6633 -> 0.7783 (+0.115).** (2) 만으로 +0.087 이다.

### 3.8 NPU - ORT 간극은 이제 의심의 여지가 없다

(1) 시점에 "16 문장으로는 경계에 있다(2.3 SE)" 고 적었는데, (2) 를 얹으니
**16 문장 전부 음수, 4.7 표준오차**다. 그리고 **SmoothQuant 를 더 걸수록 커진다**
(+0.0006 -> -0.0104 -> -0.0132).

알고 있는 것:

- FC1 꼬리(=(1) 이 바꾼 곳)는 NPU 가 numpy float32 와 **비트 단위로 같다**(98 K 원소).
- IREE CPU 빌드는 ORT 와 맞는다(+0.0015) -> AMD-AIE 경로 안이다.
- erf 다항식 근사는 아니다 — 최대 절대오차 2.787e-07, 자릿수가 안 맞는다.

모르는 것: 어느 연산이 치우침을 만드는가. AMD-AIE 경로에는 코드젠 말고도 LayerNorm raise
(스케일 흡수, γ/β 를 bf16 으로 패킹)와 softmax·layernorm ukernel(bf16 내부 연산)이 있다.

출처는 §5 에서 LayerNorm ukernel 의 bf16 연산으로 확정했고, 커널의 2 단계를 f32 로 바꿔 간극을 닫았다.

### 3.9 LayerNorm ukernel 의 계통 오차 — 원인 규명, 수정, 그리고 미채택 (2026-09-30)

(1) 에서 쓴 방법 그대로, LayerNorm 디스패치와 같은 모양의 독립 모듈을 만들어 NPU 를
돌렸다. CPU 백엔드는 `custom_op` 을 타일링하지 않아 벡터 8192 B 초과로 거부하므로
**numpy f32 를 기준**으로 삼았다(= raise 가 만든 `custom_op` 영역의 수식 그대로).

32 x 768 = 24,576 원소:

| | |
|---|---:|
| 일치 | **93.38 %** |
| 최대 \|차이\| | 1 |
| -1 | **1395** |
| +1 | **231** |

±1 LSB 인데 음수 쪽이 6 배다. 어디에 나는지 보면 **부호 편향이 아니라 크기 수축**이다:

| | ref 평균 | 반올림 전 소수부 |
|---|---:|---:|
| -1 자리 | **+11.57** | 0.578 |
| +1 자리 | **-15.40** | 0.462 |

`d = NPU - ref` 이므로 ref 가 양수인 곳은 NPU 가 더 작고, 음수인 곳은 덜 음수다 —
**NPU 출력이 일관되게 0 쪽으로 당겨진다.**

#### 두 가지 계통 오차 — 이득과 오프셋

`(NPU - 정답)` 을 정답값에 회귀하면 둘로 갈린다:

| | |
|---|---:|
| 기울기 -> **이득** | **0.99468** |
| 절편 -> **오프셋** | **-0.048 LSB** |

이득만으로는 -1/+1 이 673/662 로 균형이 맞아야 하는데 실측은 1395/231 이다. 치우침을
만드는 것은 이득이 아니라 오프셋이다. 둘을 같이 넣으면 1340/219 로 실측에 거의 닿는다.

#### 원인 — 하드웨어 bf16 변환이 -inf 로 반올림한다

`to_v32bfloat16` 은 반올림 피연산자를 받지 않는다. 다섯 narrowing 지점에 {정확, RNE, 0 방향 절삭, **-inf 방향**} 4^5 =
1024 조합을 돌리니 **일치 100.00 %** 인 조합이 나왔다:

```
입력 x     -inf
rstd       -inf   (rstd > 0 이라 0 방향과 같다)
centred    -inf
scale      -inf
biased     -inf 또는 무관
```

즉 **모든 bf16 narrowing 이 -inf 로 내림**한다. 양수는 크기가 줄고 음수는 커지므로 이득
손실이 나고, 값이 어느 부호든 아래로 밀리므로 오프셋이 생긴다.

**`invsqrt` 는 결백하다.** 앞서 "243 조합으로 재현 안 되니 남은 후보는 `invsqrt`" 라고
적었는데, 틀렸다. -inf 방향 절삭을 후보에 넣지 않았던 것이 전부다.

#### 모드 레지스터로는 못 고친다 — 확인했다

`set_fp2bf_mask(rnd_conv_even)` 를 커널 진입에 넣고 빌드해서 NPU 에서 돌렸다.
vmfb 는 커졌는데(31,989 -> 32,021 B, 명령이 실제로 들어갔다) **출력은 바이트 단위로 같다.**
`crF2FMask`/`crF2IMask`/`crFPMask` 는 반올림 모드가 아니라 **부동소수점 예외 마스크**다.
`crRnd`(`set_rnd`)는 SRS 계열에만 걸린다.

#### 고친 것 — 스칼라 두 지점을 미리 반올림한다 (2026-09-30)

하드웨어 모드를 못 바꾸므로 **narrowing 되기 전에 bf16 격자 위의 값으로 만들어** 둔다.
그러면 뒤따르는 변환이 무엇을 하든 정확하다. 정수 연산 세 개다:

```cpp
INTRINSIC(float) round_to_bfloat16(float v) {
  const int32_t bits = __builtin_bit_cast(int32_t, v);
  const int32_t rounded = bits + ((bits >> 16) & 1) + 0x7fff;
  return __builtin_bit_cast(float, rounded & ~0xFFFF);
}
```

적용한 곳은 **스칼라인 두 곳뿐**이다 — 행당 한 번인 `rstd`, 그리고 이미 스칼라 루프인
입력 widening. 나머지 둘(`centred`, `scale`)은 벡터라 반 ULP 를 더하려면 내부 루프에
벡터 연산이 하나씩 더 붙어서 두지 않았다.

| | 정답일치 | -1 | +1 | 이득 | 오프셋 |
|---|---:|---:|---:|---:|---:|
| 전 | 93.38 % | 1395 | 231 | 0.99468 | -0.048 |
| **후** | **96.27 %** | 767 | 150 | **0.99779** | **-0.025** |

NPU 실측이 모델 예측과 **비트 단위로 같다.**

시도했지만 안 되는 것들: `gamma` 에 (1+2^-9) 을 굽기 — `gamma` 가 커널에서 bf16 이라
f32 정밀도의 보정을 담지 못한다. 정수 중심화 — `|mean|` 82 가 `|h|` 2508 에 비해 작아
효과가 없다(95.91 %, 오히려 낮다).

#### 전체 모델에 얹은 결과 — 기준선이 나빠진다. 채택하지 않았다

16 문장, fp32 대비 corr:

| | ORT | NPU 전 | NPU 후 | 변화 | 중앙값 | 개선 문장 |
|---|---:|---:|---:|---:|---:|---:|
| 기준선 | 0.6627 | 0.6633 | **0.6564** | **-0.0069** | -0.0063 | 6/16 |
| ①+② | 0.7914 | 0.7783 | 0.7818 | +0.0035 | +0.0008 | 10/16 |

**기준선이 일관되게 나빠진다.** 이상치 하나가 끄는 게 아니라 중앙값이 -0.0063 이고
16 문장 중 10 개가 음수다. ①+② 는 중앙값 +0.0008 로 사실상 무효과이고, 평균 +0.0035 는
몇 문장이 끈 것이다. 둘 다 1.5 SE 안쪽이라 16 문장으로는 어느 쪽도 유의하지 않다.

**커널은 더 정확해졌는데 끝단은 나아지지 않았다.** 기준선의 NPU-ORT 는 +0.0006 에서
-0.0063 으로, ORT 에서 **멀어졌다**. int8 체인의 정확도가 이 오차에 어느 정도 기대고 있었다는
뜻이지만, 무엇과 상쇄되고 있었는지는 모른다.

그래서 **되돌렸다.** 트리는 ② 의 검증된 상태 그대로다(`layernorm.cc` 는 `(4, 768)` 인스턴스
4 줄만). 되살리려면:

```bash
cd _local/wj && git apply ../patches/layernorm_bf16_rne.patch
```

같은 폴더에 검증 스크립트도 뒀다 — `rank3.py`(1024 조합으로 NPU 를 비트 재현),
`doable.py`(구현 가능한 수정안별 예측), `verify.py`(NPU 실측 대 모델 대조).
셋 다 `/tmp/ln.mlir`, `/tmp/ln_a.npy`, `/tmp/ln_b.npy`, `/tmp/ln_out_npu.npy` 를 읽는다.

판단에 필요한 것: 16 문장으로는 ±0.007 을 가르지 못한다. 평가 64 문장을 NPU 로 돌리면
(문장당 13 초, 설정당 14 분) 표준오차가 절반이 된다. 공유 NPU 락을 설정 4 개 x 14 분
잡는 일이라 그냥 하지 않았다.

#### 남은 것

- 이 부분 수정은 §5.4 의 f32 커널로 대체됐다. 부분 수정에서 기준선이 -0.0069 (1.4 SE) 였던 것은
  규명하지 않았다 — f32 커널에서는 기준선도 ORT 와 같다(§5.5).

## 3.5 ② 결론

| | |
|---|---|
| 컴파일 | 전체 모델 208 MB vmfb, 에러 0 |
| 디스패치 | **136 = NPU 131 / CPU 5** — 기준선과 동일 |
| 회귀 | 기준선 출력 **16/16 문장 바이트 동일** |
| **NPU 정확도 (16 문장)** | **0.6918 -> 0.7783 (+0.087)** |
| ORT 정확도 (같은 16 문장) | 0.7022 -> 0.7914 (+0.089) |
| ORT 정확도 (평가 64 문장) | 0.7044 -> 0.7858 (+0.081) |

---

## 4. ③ FC2 출력 int16 (2026-09-30)

### 4.1 ORT 시뮬레이션 — 완료, 인계 수치를 넘는다

`quant/fc2_int16.py` 를 (1)+(2) 모델에 적용. ORT 평가 64 문장:

| | corr | 최저 |
|---|---:|---:|
| 기준선 | 0.6670 | 0.6041 |
| ① | 0.7044 | 0.6340 |
| ①+② | 0.7858 | 0.7464 |
| **①+②+③** | **0.8420** | 0.8179 |

인계 §3.4 의 C = 0.837(최저 0.812)보다 높다. **③ 단독으로 +0.056.**

### 4.2 인계가 요구한 선행 계산 — shift 를 올려야 한다

인계: "공통 step 을 `(amax_a + amax_b)/32767` 이상으로 잡아야 넘치지 않는다. 그 step 으로도
int8 보다 충분히 곱지를 먼저 계산해 볼 것."

step 자체는 **물리적으로 그대로다** — `amax` 가 같으므로 `(amax_a + amax_b)/32767` 은
int8 일 때와 같은 값이다. 문제는 step 이 아니라 **고정소수점 배수의 자릿수**다.

12 개 층의 실측 scale 로 계산:

| | `Ma` 범위 | 배수 자체의 상대오차 |
|---|---:|---:|
| shift=8 (현재 `kResidualShift`) | 20.2 ~ 238.9 | 0.21 % ~ **2.48 %** |
| shift=16 | 5164 ~ 61150 | 0.0008 % ~ 0.0097 % |
| *(비교) int8 입력 자체의 양자화 오차* | | *0.394 %* |

**현재 shift 로는 layer 7 의 배수 오차가 2.48 % 라 int16 으로 얻은 정밀도를 도로 까먹는다.**
shift 를 올리면 해결된다. 오버플로 한계는 `32767 * 2^shift + 2^(shift-1) <= 2^31 - 1`
에서 **shift <= 16**, 여유를 두면 15 다.

층별 값:

| 층 | sA(i16) | sB_eff | amaxA | amaxB | step | Ma@8 | Ma@16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.180e-03 | 4.793e-01 | 38.68 | 60.87 | 3.038e-03 | 99.5 | 25464 |
| 2 | 6.834e-03 | 3.720e-01 | 223.93 | 47.25 | 8.276e-03 | 211.4 | 54117 |
| **7** | 2.357e-04 | 7.109e-01 | 7.72 | 90.29 | 2.991e-03 | **20.2** | 5164 |
| 9 | 1.919e-02 | 3.552e-01 | 628.80 | 45.11 | 2.057e-02 | 238.9 | 61150 |
| 10 | 2.793e-02 | 5.834e-01 | 915.29 | 74.09 | 3.019e-02 | 236.8 | 60628 |

### 4.3 진행

| 단계 | 상태 |
|---|---|
| (a) ORT 시뮬레이션 | **완료 — 0.8420** |
| (b) shift 예산 확인 | **완료 — shift 8 -> 15 필요** |
| (c) 백엔드 설계 | **완료** |
| (d) 구현 | **완료 — 플래그 뒤, 이후 기본 켜짐 (`e2dc08da`)** |
| (e) 컴파일 | **완료** — 한 번 막혔다가 원인을 찾아 풀었다(4.5) |
| (f) NPU 실행 | **완료 — 0.8288** (4.6) |
| (g) 회귀 | **완료** — 플래그 끔: ② vmfb 바이트 동일, 기준선 NPU 출력 16/16 바이트 동일 |

### 4.4 설계 — 좁히기를 넓힌다

FC2 출력의 int8 좁히기는 **그래프가 그렇게 말해서 있을 뿐**이다. 그 유일한 소비자는 raise 가
어차피 정수 producer 로 접는 residual add 이고, 그 producer 는 int16 샘플을 받는다. 같은
캘리브레이션 범위를 int16 격자에 올리면 **258 배 정밀해지고 다른 무엇도 이를 알아채지 못한다.**

Preprocessing 의 FC2 출력 양자화는 이렇게 생겼다:

```mlir
%90 = linalg.generic ins(%89 : tensor<1x32x768xf32>) {
  divf %in, %cst_185 ; roundeven ; addf 0 ; maximumf -128 ; minimumf 127 ; fptosi -> i8
}
```

| 파일 | 한 일 |
|---|---|
| `AMDAIERaiseLayerNorm.cpp` | `matchWidenableRequantization` — 위 꼬리를 **단일 사용일 때만** 인식하고 IR 은 건드리지 않는다. `widenRequantization` — 같은 꼬리를 `s * 127/32767`, `clamp[-32768, 32767]`, `fptosi -> i16` 으로 다시 짓는다 |
| 같은 파일 | `computeResidualMultipliers` 가 가지별 최대 크기(127 / 32767)를 받는다. **넓어진 가지가 있을 때만** shift 를 8 -> 16 으로 올린다 — int8+int8 경로의 수식은 예전과 문자 그대로 같다 |
| `KernelDispatch.cpp` | ② 의 "절반" 임시 규칙을 **뱅크 인식 재현**으로 교체. 관측된 배치(m=8 에서 addr=61440)를 정확히 재현하고, 상수가 들어갈 자리가 날 때까지 타일을 내린다 |

거부는 전부 IR 을 만들기 전에 끝난다. 넓히기가 안 되면 그 가지는 i8 로 그대로 간다.

**이름으로 FC2 를 고를 필요가 없다.** attention output projection 까지 넓힌 ORT 시뮬레이션이
같은 0.8420 을 준다(최저만 0.8179 -> 0.8161). 그래서 구조로만 고른다.

Preprocessing 결과: custom_op 23 개(변동 없음), **19 개가 i16 으로 넓어짐**. 나머지 4 개는
조건을 못 맞춰 i8 로 남았다 — 설계상 안전한 후퇴다.

### 4.5 막혔던 지점 — residual producer 가 FC2 디스패치로 끌려갔다

수치도 구현도 됐는데 **컴파일이 안 된다.**

```
'amdaie.connection' op no producer DMA channel available
  (!amdaie.logicalobjectfifo<memref<1536xi8, 1 : i32>, 2>,            <- L2
   !amdaie.logicalobjectfifo<memref<1x1x32x832xi8, strided<...>>>)    <- 활성, DDR
```

실패하는 곳은 LayerNorm 디스패치가 아니라 **FC2 matmul 디스패치**
(`batch_matmul_1x32x768x3136`)다. 그 꼬리를 보면 residual producer 가 통째로 들어와 있다:

```mlir
^bb0(%in: i32, %in_4: i8, %in_5: i32, %out: i16):
  ... fptosi -> i16          // 넓힌 재양자화
  %21 = muli %19, %c25464    // Ma
  %22 = muli %20, %in_5      // 채널별 Mb
  %25 = shrui %24, %c16      // shift 16 — 의도대로다
  %26 = trunci %25 : i32 to i16
```

그래서 이 디스패치는 활성·가중치에 **residual 까지 세 번째 스트림**을 원하고, npu4 의 shim 은
열당 MM2S 가 2 개다.

**관측**: 같은 컴파일 안에서 갈린다.

| | producer 위치 | 결과 |
|---|---|---|
| i16 으로 넓힌 19 개 층 | **FC2 matmul 디스패치** | 채널 초과 |
| i8 로 남은 4 개 층 | LayerNorm 디스패치 | 통과 |

②(전부 i8)에서는 producer 가 LayerNorm 디스패치에 있었고 디스패치 수도 136 그대로였다.
23 개 모두 raise 되었으므로 4 개 층에도 producer 는 있다 — 위치만 다르다.

#### 원인 — IREE 가 ② 의 producer 를 dequant 로 알고 있었다

패스마다 IR 을 떠서(flow 단계까지 761 패스) 두 가지를 셌다. producer 가 matmul 영역에
들었는가, LN 영역에 들었는가:

| | `FormDispatchRegions` 직후 | 그 뒤 |
|---|---|---|
| off (②) | **어느 영역에도 없다** | `CloneProducersIntoDispatchRegions` 가 LN 영역에 복제 (23) |
| on (③) | **matmul 영역에 이미 있다** (19) | elementwise 융합이 requant 와 한 generic 으로 합침 |

갈림길은 그보다 앞의 `SinkReshapes` 다. `BubbleUpExpandShapes` 가 양쪽 producer 를 똑같이
`1x32x768` 로 펼친 뒤, off 에서는 `SinkReshapes` 가 producer 를 다시 `32x768` 로 접어
collapse 를 **requant 와 producer 사이**에 둔다. 그 reshape 가 끼어 있으니 consumer 융합이
안 된다. on 에서는 이 패스가 아무것도 안 한다.

`SinkReshapes` 가 collapse 를 generic 위로 올리는 조건은 하나, `isCloneableIntoDispatchOp`
이고, 그 안의 `LinalgExt::isBitExtendOp` 가 참이면 복제 가능이다. 판정은
`isBitExtendOrTruncateOp`(`LinalgExt/Utils/Utils.cpp`) 가 한다 — **최고 rank 입력들의 최대
비트폭이 출력보다 작으면 비트 확장 op(= dequant 류)**:

| | 최고 rank 입력 | 출력 | 판정 |
|---|---|---|---|
| ② | i8, i8 | i16 | **비트 확장** -> 복제 가능 -> LN 으로 복제 |
| ③ | **i16**, i8 | i16 | 해당 없음 -> 평범한 elementwise -> matmul 로 융합 |

(`[768]xi32` 채널 배수는 rank 1 이라 비교에서 빠진다.) **② 가 돌아간 것은 IREE 가 이
producer 를 dequant 로 잘못 알아본 덕이었다.** FC2 출력을 i16 으로 넓히는 순간 그 조건이
깨진다. `FormDispatchRegions` 의 `isFusableWithConsumer` 에 비트폭 조건이 없는 것은 맞지만,
비트폭은 그보다 앞에서 IR 모양을 바꾸는 방식으로 결정을 바꾸고 있었다.

#### 고친 것 — producer 를 영역 안에 넣었다가 코드젠이 다시 꺼낸다

IREE 의 휴리스틱을 건드리지 않고 우리 쪽에서만 해결했다.

1. **raise** (`AMDAIERaiseLayerNorm.cpp`): 넓힐 때만 producer 를 `custom_op` **영역 안**에서
   만든다. `custom_op` 은 a(i16), b(i8) 두 입력을 받고, 채널 배수 `Mb` 는 `gammaBeta` 처럼
   영역 안의 상수라 입력이 아니다 — 스트림 수가 ② 와 같다. 표시 속성
   `amdaie.layernorm_residual_in_region`. dispatch-creation 은 영역 안을 못 보니 producer 가
   어디로도 끌려가지 않는다.
2. **unfold** (새 패스 `AMDAIEUnfoldLayerNormResidual.cpp`, 코드젠 첫 블록의
   `TypePropagation` 바로 뒤): 디스패치 안에서 producer 를 영역 밖으로 다시 꺼내 **②와 똑같은
   모양**(producer generic -> `custom_op`(h))으로 되돌린다. 이후 타일 크기, L1 뱅크 배치, ukernel
   호출은 전부 ② 경로를 그대로 탄다.

unfold 가 신경 쓴 것 두 가지. 정규화가 영역에 `tensor.cast %arg0 : ?x? -> ?x768` 를 끼워
넣으므로 활성값은 cast 너머로 찾는다. 정규화 쪽이 활성값의 `tensor.dim` 을 쓰게 접히는 경우가
있어(CSE 와 dim 접기) 그런 dim 은 새 샘플 인자의 dim 으로 바꾼다 — 모든 피연산자가 같은 행·열을
덮으므로 같은 값이다. 그 밖에 정규화가 활성값을 읽으면 오류로 멈춘다.

`buildResidualProducer` 는 init 을 입력의 크기에서 만들도록 바꿨다(`getMixedSizes`). 정적
모양이면 이전과 같은 IR 이다.

#### 확인

| | 결과 |
|---|---|
| flow 단계 producer 위치 | on: **LN 23 / matmul 0** (전: LN 4 / matmul 19) |
| off 의 flow 출력 | 이전과 크기까지 같다 |
| 전체 컴파일 | 210,567,061 B vmfb, 에러 0 |
| 디스패치 | **136 = NPU 131 / CPU 5**, CPU 목록 ② 와 동일 |
| 회귀 — ② (플래그 끔) | **vmfb 바이트 동일** (208,693,077 B) |
| 회귀 — 기준선 | **NPU 출력 16/16 바이트 동일** |

넓혀지지 않은 4 개는 LN 3 (1층 FC2), LN 12·14·18 (6·7·9층 attention 출력) 이다. 원인과 수정은 §5.6.

### 4.6 NPU 실행 — 0.8288

16 문장, fp32 대비 corr:

| | ORT | NPU |
|---|---:|---:|
| ①+② | 0.7914 | 0.7783 |
| **①+②+③** | 0.8423 (`c`) / 0.8434 (`cboth`) | **0.8288** |

**③ 만으로 NPU +0.0505** (9.9 SE, 16 문장 전부 개선).

비교할 ORT 모델이 정확히 같지는 않다. `c` 는 FC2 12 개만, `cboth` 는 FC2 와 attention 출력
24 개를 모두 넓힌다. NPU 는 그중 19 개다. 가까운 쪽은 `cboth` 이고(24 개 중 4 개가 다름),
NPU - ORT 는 **-0.0146** (8.6 SE, 16/16 음수) 이다. ② 에서 이미 있던 -0.0132 와 같은 크기라
③ 이 새로 만든 간극은 아니다. 출처와 수정은 §5.

---

## 5. NPU - ORT 간극의 출처 — LayerNorm ukernel 의 bf16 연산 (2026-09-30)

### 5.1 종류별로 CPU 에 보내 가르기

배치 규칙은 matmul·conv, 그리고 ukernel 을 켠 softmax·LayerNorm 만 NPU 로 보낸다. 그래서
`--iree-amdaie-enable-ukernels` 목록만 바꿔도 종류별로 CPU 에 보낼 수 있다. 비교할 ORT 모델이
정확히 대응하도록 ③ 은 끄고(= ① + ② 구성) 16 문장:

| 변형 | NPU 에 남는 것 | NPU - ORT |
|---|---|---:|
| 전부 NPU | matmul + softmax + LayerNorm | -0.0132 (4.7 SE, 16/16 음수) |
| softmax -> CPU | matmul + LayerNorm | -0.0139 (4.3 SE) |
| **LayerNorm -> CPU** | matmul + softmax | **-0.0005 (0.2 SE)** |
| 둘 다 CPU | matmul | -0.0034 (1.0 SE) |

**간극은 LayerNorm 경로에 있다.** softmax 와 matmul(융합된 requant·GELU 꼬리 포함)은 아니다.
변형마다 출력이 ② 와 달라진 것(0/16 동일)을 확인해, 실제로 배치가 바뀐 것도 봤다.

### 5.2 LayerNorm 경로 안에서 가르기

CPU 로 보내면 세 가지가 같이 빠진다. 앞의 둘은 잴 수 있다:

| NPU 에만 있는 것 | 크기 |
|---|---|
| raise 의 정수 residual producer — 배수 `Mb` 반올림 | `Mb` 3639 ~ 25412, 최악 채널 **0.014 %** |
| γ/β 를 bf16 으로 패킹 | β 오프셋 평균 **0.002 LSB** (최대 0.11), γ 는 정규화 값 1 당 0.022 LSB |
| ukernel 의 bf16 연산 | 3.9 — 이득 0.9947, 오프셋 **-0.048 LSB**, 정답 일치 93.38 % |

남는 것은 ukernel 이다.

### 5.3 왜 SmoothQuant 를 걸수록 커지는가

같은 표에서 메커니즘이 보인다. 정규화 값 1 이 출력의 몇 LSB 를 차지하는가(`|γ/s_out|` 평균):

| | |
|---|---:|
| 기준선 | 5.1 LSB |
| ① + ② | **15.0 LSB** |

SmoothQuant 는 LayerNorm 출력 분포를 평평하게 만들어 출력 스케일을 줄인다. 그러면 커널의
**같은 상대오차가 LSB 로는 3 배**가 된다. 기준선에서는 반올림 아래에 묻혀 있던 bf16 의 오차와
-inf 방향 편향이 SQ 에서 드러난다. 간극이 +0.0006 -> -0.0104 -> -0.0132 로 커진 것과 맞는다.

3.9 에서 스칼라 두 지점만 고쳤을 때 ① + ② 간극이 -0.0132 -> -0.0097 로 준 것도 같은 방향이다.
남은 오차는 벡터 narrowing(`centred`, `scale`, `biased`)과 bf16 의 8 비트 가수 자체다.

### 5.4 고친 것 — 2 단계에서 bf16 을 없앴다 (`769e9f74`)

1 단계(행 통계)와 2 단계의 누산기는 원래부터 f32 였다. bf16 을 거친 것은 2 단계(원소별 정규화와
재양자화)의 곱셈 입력과 끝단 변환 — `centred`, `rstd`, `scale = rstd·γ` 를 bf16 으로 내리고,
`y + 0.5` 를 다시 bf16 으로 내려 `bfloat16_to_int` 로 내림했다. 네 번 모두 -inf 방향이다.

이 네 곳을 모두 f32 로 바꿨다. 샘플은 f32 로 넓히고,
`centred`, `scale = rstd·γ`, `y = centred·scale + β` 를 전부 `mul/mac_elem_16_accuracy_safe` 로
계산한다. `rstd` 에는 뉴턴 한 스텝을 더했다(NLF `invsqrt` 의 정확도와 무관하게 f32 수준).

막혀 있던 f32 -> int8 은 매직 넘버로 풀었다. `y + 1.5·2^23` 은 |y| < 2^22 에서 간격이 정확히 1 인
구간에 떨어지므로, 덧셈 자체가 y 를 정수로 반올림하고 결과의 비트가 `0x4B400000 + 정수` 가 된다.
그 비트를 정수로 읽어 `[-128, 127]` 로 자르고 `ssrs` 로 하위 바이트를 취한다(포화 모드가 꺼져 있어
wrap 한다 — 매직 넘버의 하위 바이트는 0). **bf16 을 전혀 거치지 않는다.** 누산기를 float 으로
꺼낼 때는 메모리를 거치는데(ISel 이 실패하는 변환), 행 통계 뒤에 비는 `lanes` 256 B 를 쓰므로 스택은
그대로다.

알아낸 것 하나: **aie2p 의 f32 누산기 덧셈은 최근접(짝수)으로 반올림한다** — bf16 변환의 -inf
방향과 다르다. 처음에 `+0.5` 를 더해 내림을 가정했더니 정확히 `frac(y+0.5) >= 0.5` 인 절반이 +1 로
틀렸다. `+0.5` 를 빼니 결과는 `round-half-even(y)` 로, ORT `QuantizeLinear` 와 같은 규칙이 됐다.

독립 모듈(24,576 원소, 정확한 기준값 대비):

| | 일치 | -1 | +1 |
|---|---:|---:|---:|
| bf16 커널 | 93.38 % | 1395 | 231 |
| **f32 커널** | **99.996 %** | 1 | 0 |

남은 1 개는 y 가 반올림 경계에서 3.4e-5 떨어진 자리 — f32 와 f64 의 차이다.

### 5.5 전체 모델 — 간극이 닫혔다

16 문장, fp32 대비 corr:

| | ORT | bf16 커널 NPU | **f32 커널 NPU** | 개선 |
|---|---:|---:|---:|---:|
| 기준선 (③ 끔) | 0.6627 | 0.6633 (+0.0006) | **0.6672 (+0.0045, 0.7 SE)** | |
| ① + ② (③ 끔) | 0.7914 | 0.7783 (-0.0132) | **0.7915 (+0.0001, 0.0 SE)** | +0.0132 (3.7 SE, 13/16) |
| ① + ② + ③ | 0.8434 (`cboth`) | 0.8288 (-0.0146) | **0.8409 (-0.0025, 1.6 SE)** | +0.0122 (5.5 SE, 15/16) |

**같은 조건끼리 비교한 두 구성에서 NPU 가 ORT 와 같아졌다.** ③ 의 -0.0025 는 비교 ORT 모델이 NPU 와
24 곳 중 4 곳에서 다른 만큼일 수 있다.

주의 — 처음 잰 기준선은 0.6951 (ORT +0.032) 이었다. ③ 의 넓히기가 **기본값으로 켜져 있어서** 기준선
모델에도 걸린 것이다. LayerNorm 을 CPU 로 보내면(raise 가 없으니 넓히기도 없다) 0.6582 로 ORT 와
같고, ③ 을 끄면 위의 0.6672 다. 부수적으로 **③ 은 SmoothQuant 없이도 +0.028** 이다(0.6672 -> 0.6951).

기준선 출력은 이제 바이트 동일이 아니다 — 커널을 일부러 바꿨다. 대신 ORT 와 같은지를 봤다. 속도는
재지 않았다(§6). 내부 루프가 16 레인에 accurate 곱셈 셋과 메모리 왕복 넷이라 느려졌을 것이다.

### 5.6 넓혀지지 않던 LayerNorm 4 개 (`a738f27c`)

**원인 — 부동소수점 비교.** `computeResidualMultipliers` 는 넓힌 가지에서 스텝을
`span / 32767` 로 잡고 곧바로 `span / step > 32767` 이면 거절한다. 식으로는 항상 32767 인데
double 로는 4 곳에서 `32767.000000000004` 가 나왔다 — 1층 FC2, 6·7·9층 attention 출력, 정확히
넓혀지지 않은 4 개다. 거절되면 int8 로 되돌아간다. `matchWidenableRequantization` 쪽은 24 곳 모두
통과한다(requant 모양, 한 번 사용, 스케일 일치 모두 같다).

**같이 드러난 것 — 음수 끝이 int16 을 넘는다.** 범위를 `127·sA + 127·sB` (넓히면 `32767·sA'`)
로만 잡아서 2 의 보수의 한 칸 더 아래(-128, -32768)를 빠뜨렸다. residual 가지의 int8 한 칸이 합의
약 200 칸이라, 두 입력이 모두 음수 끝이면 합이 -32,963 ~ -33,005 가 된다(-32768 을 0.5~0.7 % 넘음).
shift 16 에서는 그 전에 int32 합(-2^31)부터 넘는다. producer 는 `trunci` 라 클램프 없이 wrap
해서 부호가 뒤집힌다. 이미 넓혀진 19 곳 모두 해당했다 — 두 끝이 한 원소에서 겹칠 때만이라 16 문장에서는
드러나지 않았다.

**고친 것 (`AMDAIERaiseLayerNorm.cpp`, 넓힌 경로만).** 스텝을 음수 끝
`(32768·sA' + 128·sB·max c) / 32768` 에서 잡고, 배수를 반올림한 뒤 **실제 정수로** 두 끝의 최악을
계산해 int32 합과 int16 결과 안에 드는 가장 큰 shift(16 부터 내려감)를 고른다. 반올림 때문에 한 칸
넘으면 스텝을 1/4096 키워 다시 찾는다. 두 가지가 모두 int8 인 경로는 손대지 않았다.

| 확인 | 결과 |
|---|---|
| 24 곳 (Python 으로 같은 계산) | 모두 첫 시도에 shift 16, 스텝 +0.05~0.72 %, 배수 반올림 오차 ≤ 5e-6 |
| 플래그 끔 global-optimization IR | 이전과 바이트 동일 |
| 플래그 켬 | `residual_in_region` 19 -> **23** (raise 된 23 개 전부) |

raise 되지 않는 두 LayerNorm 은 의도대로다 — 임베딩 LN 과 마지막 LN(11층 FC2, 출력이 float 그대로
모델 출력이라 뒤에 양자화가 없다).

**NPU 16 문장, fp32 대비:**

| | corr | ORT `cboth` 대비 |
|---|---:|---:|
| 19 개 넓힘 (§5.5) | 0.8409 | -0.0025 (1.6 SE) |
| **23 개 넓힘** | **0.8415** | -0.0019 (1.4 SE) |
| 차이 | +0.0006 (0.3 SE, 9/16) | |

정확도 효과는 잡음 수준이다. 남은 -0.0019 는 이 4 개 때문이 아니다. 수정의 가치는 거절 버그와
음수 끝 wrap 을 없앤 데 있다.

### 5.7 64 문장 NPU — 0.8400

최종 상태(커밋 5 개, 기본 플래그)의 vmfb 로 평가 64 문장 전부를 NPU 에서 돌렸다. 16 문장씩 네 번 락을
잡았다. 오류 0, 0 번 문장 출력은 §5.6 의 16 문장 실행과 바이트 동일.

| 64 문장, fp32 대비 | 평균 | 최저 |
|---|---:|---:|
| ORT `cboth` | 0.8414 | 0.8171 |
| **NPU** | **0.8400** | 0.8188 |
| NPU - ORT | -0.0014 (2.0 SE, NPU 가 높은 문장 25/64) | |

인계 문서의 목표(ORT 시뮬레이션 C, 0.837)를 넘었다. 인계 시점 배포 경로 0.667 에서 **+0.173**.

ORT `cboth` 의 앞 16 문장이 여기서는 0.8425 로 §5.5 의 0.8434 와 조금 다르다 — 이번에는 ORT 스레드를
4 개로 제한했고, 스레드 수에 따라 float 합산 순서가 달라진다. NPU 쪽 숫자는 영향이 없다.

## 6. 규칙 (인계 §6·§7에서)

1. 수정마다 ORT 시뮬레이션 → 디스패치 수 → NPU 실행 → 회귀 순으로 확인한다.
2. **성능은 재지 않는다.** 장비가 디스패치당 ~95 ms로 느려진 상태다.
3. 정확도는 **항상 fp32 대비**. int8끼리 비교 금지.
4. 평가 문장(64)과 캘리브레이션 문장(90)은 겹치지 않는다.
5. NPU는 `with-npu-lock.sh` 경유, `timeout`은 lock **안쪽**에.
6. 모델 파일이 180~440 MB다. 중간 파일은 바로 지운다.

## 7. 다음 레버 — 가중치 per-channel (ORT 시뮬레이션, 2026-09-30)

NPU 는 쓰지 않았다. 전부 ORT, 평가 64 문장, fp32 BERT(`_local/int8_debug/bert_base_L12.onnx`) 대비.

### 7.1 남은 오차는 어디에 있나

지금 모델(`cboth`, 0.8414)에서 한 종류씩 양자화를 걷어 냈다. 배포용으로 접힌 모델이라 `loo.py` 는
그대로 쓸 수 없다 — Q 와 DQ 사이에 int8 인 채로 Pad·Slice·Reshape·Transpose 가 끼어 있고, Pad 의
채움 값(1, 2, 3, 4, 6)은 bias 열을 만드는 int8 값이라 float 로 바꿀 때 Q 의 스케일을 곱해야 한다.
그것을 반영한 변형으로 쟀다.

| fp32 로 남긴 것 | corr | |
|---|---:|---:|
| 없음 (`cboth`) | 0.8414 | |
| probs (softmax 출력) | 0.8500 | +0.009 |
| fc1_out | 0.8486 | +0.007 |
| gelu_out (FC2 입력) | 0.8460 | +0.005 |
| LN 출력 (FFN 뒤 / attention 뒤) | 0.8436 / 0.8433 | +0.002 |
| q/k/v, scores, context, int16 residual 둘 | 0.8408 ~ 0.8426 | 0 |
| LN 출력 전부 + gelu_out | 0.8508 | +0.009 |
| **activation 156 개 전부** | **0.8690** | +0.028 |

activation 을 전부 fp32 로 둬도 0.869 다 — **남은 0.13 은 가중치 쪽이다.** 가중치를 역양자화해 둔 것
(int8 값 그대로)은 0.8688 로 activation 만 걷은 것과 같다. 기준선 모델(`v3`)은 같은 조건에서 0.949 로,
인계 §3.3 의 "activation 전부 fp32 0.948" 과 맞는다. SmoothQuant 가 가중치 쪽 손실을 0.05 에서 0.13 으로
키운 셈이다 — 입력 채널별 `s_j`(3~37 배)가 가중치 행에 곱해지는데, 가중치가 텐서 하나에 스케일 하나
(per-tensor)라 작은 행이 뭉개진다. 접힌 bias 는 원인이 아니다(layer 0 query, 오차 rms 1e-4 / bias 0.28).

### 7.2 가중치 per-channel

인계 문서 README 의 절차 그대로(smooth α 0.6 -> `quantize_bert_base.py --calib-ids --method percentile`
-> `fc2_int16.py`), 양자화만 `per_channel=True` 로 바꿨다.

| 64 문장 | 가중치 per-tensor | **가중치 per-channel** |
|---|---:|---:|
| SmoothQuant (B) | 0.7159 | 0.7100 |
| **SmoothQuant + FC2 int16 (C)** | 0.8366 (인계 0.837 재현) | **0.9472** (최저 0.9245) |

**+0.11.** 인계 문서가 상한이라 부른 0.929 (문제 텐서 fp32, 가중치 per-tensor) 보다 높다. SmoothQuant 만으로는
per-channel 이 도움이 안 되는데(-0.006), FC2 출력이 int8 로 남아 있으면 그쪽이 병목이기 때문이다.
접힌 배포 모델이 아니라 접기 전 모델이라 NPU 에서 얻을 값은 조금 다를 수 있다(§5.7 의 `cboth` 는 C 보다 +0.005).

**막혀 있던 이유와 지금.** `docs/2026-09-10_per_channel_int8_blocked_in_npu_backend.md` — per-channel
가중치의 dequant generic 이 디스패치에 하나 더 남아 `AMDAIEBufferizeToAllocation` 이 "target op 하나"
요구에서 실패했다. 그 뒤로 재양자화가 정수 고정소수점으로 내려가고(`COMPLETED_FIXES` §5), 채널별 상수를
코어 메모리에서 읽는 경로(①)가 생겼다. per-channel 가중치는 matmul 뒤 재양자화의 배수가 **출력 열마다**
다를 뿐이라, 그 두 가지가 필요한 것을 대부분 갖고 있다.

모델: 컨테이너 `/tmp/pc/c_pc.onnx` (per-channel C), `c_pt.onnx` (per-tensor C), `s.onnx` (SmoothQuant fp32).

### 7.3 per-channel 가중치를 NPU 에서 — 단일 matmul 까지 (`fdbb6176`, `4da498bf`)

재현체: 단일 QDQ matmul (x `[32,64]`, W `[64,64]`, 가중치 스케일 `[64]`, zero point 0), per-tensor 판과 함께
(`scratchpad/pc/gen_pcmm.py`). 지금 컴파일러에서도 per-channel 은 `2026-09-10` 문서와 같은 자리에서 실패했다 —
가중치 dequant generic 과 활성 dequant generic, f32 matmul 이 한 디스패치에 남는다.

**먼저 목표 모양을 손으로 써서 확인했다.** `i8 x i8 -> i32` matmul + `sitofp(acc) * s[n]` 재양자화를
MLIR 로 직접 써서 컴파일하니 백엔드 로워링은 전부 통과하고, 마지막 peano `opt` 에서
`getelementptr inbounds nuw (i8, ptr @__amdaie_core_constant_0, i64 224)` 를 못 읽었다. 코어 지역 상수(①)를
0 이 아닌 오프셋에서 읽으면 LLVM 번역이 GEP 를 상수식으로 접는데, 호스트 LLVM 이 거기 붙이는 `nuw` 를
구버전 peano 가 모른다. 명령어 형태는 `createAMDAIERemoveWrapFlagFromGepPass` 가 이미 지우지만 상수식은
번역 뒤에 생긴다. `XCLBinGen.cpp` 에서 `input.ll` 텍스트의 그 플래그를 지운다 — 바로 위의
`nocreateundeforpoison` 제거와 같은 방식이다. 그 뒤 NPU 결과가 numpy 기준과 **2048/2048 일치**.

**그리고 전처리 패스.** `AMDAIEFoldPerChannelDequantIntoMatmul` (새 파일, `PluginRegistration` 의
전처리, `FoldBroadcastAddIntoDest` 바로 뒤):

```
%a = dequantize(%x : i8)            -- per tensor, sx
%b = dequantize(%w : i8)            -- per channel, sw[n], zero point 0
%y = linalg.matmul ins(%a, %b) outs(fill 0.0) -> f32
  =>
%acc = linalg.matmul ins(%x, %w) outs(fill 0) -> i32
%y   = sitofp(%acc) * (sx * sw[n])
```

K 에 대한 합에서 스케일이 빠지므로 정확히 같은 식이다. per-tensor 는 이미 `linalg.quantized_matmul` 로
들어와 같은 모양이 되고, per-channel 은 그 명명 op 이 zero point 를 스칼라로만 받아 float 로 떨어졌던 것이다.
2D `linalg.matmul` 만, zero point 가 상수 0 일 때만 발동한다.

| 재현체 | 컴파일 | 디스패치 | NPU vs ORT |
|---|---|---|---|
| per-tensor | ✓ | `matmul_32x64x64_i8xi8xi32` -> npu | 2047/2048 (1 LSB 하나) |
| **per-channel** | **✓** | **`matmul_32x64x64_i8xi8xi32` -> npu** | **2048/2048** |

재양자화는 스케일이 벡터라 정수 고정소수점 패스(`COMPLETED_FIXES` §5)가 발동하지 않고 float 로 남는다.
이 크기에서는 코어 프로그램 메모리에 들어갔다. BERT 의 FC2 에서는 넘쳤고, §7.5 의 새 패스로 정수로 바꿔 해결했다.

**BERT 까지 남은 것:**
1. BERT 의 matmul 은 배포 접기(`fold_bias_into_k*.py`, `head_split.py`, `hoist_kfold_pad.py`) 를 거쳐 batch_matmul
   과 `(12, 832, 64)` 같은 가중치가 된다. 그 스크립트들이 per-channel 스케일을 같이 쪼개고 채우게 해야 하고,
   bias 를 K 에 접는 방식(§7.1)도 채널마다 스케일이 다르면 그 행의 값이 달라진다.
2. 패스를 `batch_matmul` 과 SmoothQuant 쪽 채널별 상수(①)가 이미 붙은 FC1 꼬리에도 맞게.
3. 재양자화가 float 로 남는 것 — 프로그램 메모리를 넘으면 정수 패스를 채널별 배수로 넓혀야 한다.

회귀: 새 컴파일러로 최종 BERT 모델(`sqall`, per-tensor)을 다시 컴파일 — vmfb 크기 동일(210,968,981 B),
NPU 16 문장 출력이 §5.7 의 출력과 **16/16 바이트 동일**. 커밋 `fdbb6176` (GEP 플래그), `4da498bf` (패스),
`c5c58239` (`COMPLETED_FIXES`).

### 7.4 어느 matmul 에 per-channel 이 필요한가 — FC2 하나

§7.2 의 두 모델(`c_pt`, `c_pc`)은 같은 SmoothQuant 와 캘리브레이션에서 나와서 가중치만 다르다. `c_pt` 에
matmul 종류별로 `c_pc` 의 가중치(int8 값, 스케일 `[N]`, zero point, `axis=1`)를 끼워 넣어 쟀다(ORT, 64 문장,
코어 2 개).

| per-channel 로 바꾼 것 | corr (최저) | per-channel matmul |
|---|---:|---:|
| 없음 (`c_pt`) | 0.8365 (0.8122) | 0 |
| Q/K/V | 0.8359 (0.8137) | 36 |
| attention 출력 | 0.8360 (0.8160) | 12 |
| FC1 | 0.8420 (0.8189) | 12 |
| **FC2** | **0.9399** (0.9288) | **12** |
| attention 출력 + FC1 + FC2 (Q/K/V 는 per-tensor) | 0.9480 (0.9332) | 36 |
| 전부 (`c_pc`) | 0.9475 (0.9355) | 72 |

**이득의 거의 전부(+0.103)가 FC2 에서 나온다.** FC2 의 입력은 GELU 출력이고, 그쪽 SmoothQuant 계수는
네 자릿수에 걸쳐 퍼져 있다(§4.2 의 `computeResidualMultipliers` 주석). 그 계수가 FC2 가중치의 행에 곱해지니
per-tensor 스케일 하나로는 작은 쪽이 뭉개지는 것으로 본다(추정, 가중치 분포는 재지 않았다). Q/K/V 는 per-tensor 로 남겨도 손해가 없다 — head 쪼개기
(`head_split.py`)의 두 축 스케일 문제를 피할 수 있다.

다음: FC2 12 개만 per-channel 인 모델을 배포 경로(`prepare_bert_i8.sh` 의 접기)로 만들고, 컴파일러 패스가
BERT 의 FC2 모양에 걸리는지, ③ 의 int16 넓히기와 같이 동작하는지 본다.

### 7.5 FC2 per-channel — BERT-base 전체가 NPU 에서 0.9448 (`b0051a62`, `30bd07b0`)

**모델.** §7.4 의 결론대로 FC2 12 개만 per-channel, 나머지는 per-tensor. 배포 경로는 `prepare_bert_i8.sh` 의 단계를
그대로 따르되, bias 를 K 로 접는 `fold_bias_into_k_general.py` 만 per-channel 을 받게 한 복사본을 썼다
(`scratchpad/pcfold/fold_bias_into_k_general_pc.py`, 원본은 손대지 않음). 바꾼 것은 둘: 가중치 스케일이 `[N]`
(axis 1) 이면 받아서 bias 행을 열마다 `round(bias[n] / (sx * sw[n]))` 로 계산하고, 새 가중치 DQ 에 `axis` 를 옮긴다.

| 검증 | 결과 |
|---|---|
| s.onnx -> per-tensor 양자화 -> 원본 접기 | 지금 배포 모델 `sqall` 과 **md5 동일** (재현) |
| 같은 입력 -> 복사본 접기 | 원본과 md5 동일 (per-tensor 에서 바뀌는 것 없음) |
| attention 출력·FC2 int16 가짜 양자화를 `sqall` 에 | `cboth` 와 md5 동일 (ORT 비교 모델 절차 확인) |
| FC2 bias 접기 | 12 개 모두 접힘 |

ORT, 64 문장:

| | int8 residual | int16 residual (NPU 의 ③) |
|---|---:|---:|
| 지금 배포 (per-tensor) | 0.7863 | 0.8414 |
| **FC2 per-channel** | 0.7865 | **0.9447** |

**컴파일러.** 둘이 더 필요했다.

1. `AMDAIEFoldPerChannelDequantIntoMatmul` 에 `batch_matmul` 패턴. BERT 의 FC2 는 전처리 시점에
   `batch_matmul(1x32x3136, 1x3136x768)` 이고, lhs 는 3-D dequant, rhs 는 2-D per-channel dequant 를 batch 로 늘리는
   복사 generic (`(d0,d1,d2) -> (d1,d2)`) 이다. 같은 모양을 int8 로 다시 만들고 i32 batch_matmul 로 바꾼다
   (`expand_shape` 로 batch 를 붙이는 모양도 받는다). 적용 뒤 global-optimization 에서 f32 matmul 0 개, i8 FC2 12 개,
   ③ 의 넓히기 23 개 그대로.
2. 새 패스 `AMDAIEPerChannelIntegerRequantization` (전처리, RaiseLayerNorm 바로 뒤). 1 만으로는 FC2 코어가
   **program memory 를 넘었다** (`_XAie_LoadProgMemSection: Overflow of program memory`). 스케일이 벡터라 정수
   재양자화 패스(`COMPLETED_FIXES` §5 — 코어 코드의 스칼라 상수만 받는다)가 손대지 못하고 float 꼬리가 남아
   소프트 float 라이브러리가 붙은 것이다. 그래서 텐서 단계에서
   `sitofp(acc) * s[n]` + `roundeven(/S2) -> clamp -> iM` 두 generic 을
   `round_half_even(acc * m[n] >> shift) -> clamp -> iM` 하나로 바꾼다 (`m[n]` 은 i32 표, shift 는 모든 채널 공통 —
   가장 큰 채널의 배수가 i32 에 드는 만큼). 코어가 읽는 표가 정수가 되고 float 연산이 사라진다. ③ 의 넓히기가 float
   재양자화 모양을 보고 동작하므로 raise 뒤여야 한다. 정확성 정책은 스칼라 패스의 기본값과 같다 — 가장 가까운 정수
   배수를 쓰고 bit-exact 를 증명하지는 않는다.

| | 결과 |
|---|---|
| 재현체 (단일 matmul, per-channel) | 정수 꼬리로 NPU vs ORT **2048/2048** |
| BERT FC2 per-channel 컴파일 | vmfb 212,345,391 B, 오류 0 |
| **NPU 64 문장** | **0.9448** (최저 0.9315) — ORT 0.9447 과 차이 +0.0000 (0.1 SE) |
| 이전 NPU (per-tensor, §5.7) 대비 | 0.8400 -> 0.9448, **+0.1047, 64/64 개선** |
| 회귀 — `sqall` (per-tensor) | vmfb 크기 동일, NPU 16 문장 **16/16 바이트 동일** |

인계 시점 0.667 에서 **+0.278**. 인계 문서가 상한이라 부른 0.929 (문제 텐서 fp32) 보다 높다.

디스패치 수 (flow 단계): `sqall` **136 = NPU 131 / CPU 5**, FC2 per-channel **136 = NPU 131 / CPU 5** —
디스패치 이름(종류·모양)별 개수까지 같다. CPU 5 개도 그대로(임베딩 쪽 elementwise 1, reduction 4).

속도는 `_local/wj/docs/2026-10-01_bert_int8_speed.md` 에 따로 정리했다. 아직 안 본 것: 다른 모델에서의 두 패스. 커밋: 패스 확장(1) `b0051a62`, 새 패스(2) `30bd07b0`. 접기 스크립트의
per-channel 판은 스크래치 복사본으로만 있다(원본 미변경).

### 7.6 FC2 밖으로 넓히기 — attn_out 은 NPU 이득 없음, FC1 은 코어 메모리 초과 (2026-10-01)

§7.4 의 비교는 접기 전 모델(`c_pt`)에서 쟀다. 배포 모델에서 per-channel 을 FC2 밖으로 넓히면 얼마나 오르는지,
NPU 에서도 그만큼 오르는지 봤다. 모델은 §7.5 의 `q_hy.onnx`(FC2 per-channel) 에 `c_pc` 의 attn_out · FC1 가중치를
끼워 넣고(`scratchpad/pcfold/splice_mix.py`), §7.5 와 같은 접기와 ORT 비교 모델 절차(`both_int16.py`)를 거쳤다.

ORT, 64 문장 (배포 모델, attention 출력 · FC2 int16):

| per-channel 로 바꾼 것 | ORT | NPU |
|---|---:|---:|
| FC2 (§7.5) | 0.9447 | **0.9448** |
| attn_out + FC2 | 0.9462 | **0.9447** (최저 0.9319) |
| FC1 + FC2 | 0.9509 (최저 0.9420) | 컴파일 실패 |
| attn_out + FC1 + FC2 | 0.9514 (최저 0.9369) | 컴파일 실패 |
| Q/K/V 포함 전부 | 만들지 못함 | — |

**Q/K/V.** 원본 `fold_bias_into_k.py` 와 `head_split.py` 를 거친다. head 별로 나누면 채널 스케일이 (head, 채널) 두 축이
되는데 DequantizeLinear 의 per-axis 는 한 축뿐이라, 지금 접기 경로로는 모델을 만들 수 없다.

**attn_out + FC2.** 컴파일 성공, vmfb 213,621,569 B, 디스패치 136 개(§7.5 와 같다). NPU 0.9447 은 ORT 0.9462 보다
-0.0015 (-3.1 SE) 낮고, FC2 만 한 NPU 결과(0.9448) 와 비교하면 -0.0001 (문장별 개선 31/64) 이다. ORT 에서 오른 만큼이
NPU 에서는 나타나지 않았다.

**FC1 — 코어 데이터 메모리 초과.** FC1 을 포함하면 링크에서 실패한다:

```
ld.lld: error: section '.data' will not fit in region 'data': overflowed by 9024 bytes
```

`--iree-hal-dump-executable-intermediates-to` 로 중간 파일을 남겨 0 층 FC1(`dispatch_11_batch_matmul`, `core_0_2`) 을
봤다. 같은 디스패치를 §7.5 모델(FC1 per-tensor) 에서도 같은 방법으로 떴다.

| | FC1 per-tensor (§7.5) | FC1 per-channel |
|---|---:|---:|
| 링커 스크립트 `data` 영역 | `0x7C340`, 길이 0x3CC0 = 15,552 B | 같음 |
| `.rodata` | 0x3000 = 12,288 B | 0x6000 = 24,576 B |
| 결과 | 링크 성공 | 24,576 - 15,552 = **9,024 B 초과** |

`data` 영역 앞(`0x70000` ~ `0x7C340`)은 스택과 matmul 타일 버퍼(`buff_*`) 가 쓴다. `.rodata` 의 내용은 코어 상수 두 개다:

| 코어 상수 | 형태 | 크기 | `input.ll` 에서 | FC1 per-tensor 에도 |
|---|---|---:|---|---|
| `__amdaie_core_constant_0` | `[3072 x float]` | 12,288 B | GELU 결과에 채널마다 곱함 (`gelu * c[n]`, 이어서 출력 스케일로 나눠 양자화) | 있음 |
| `__amdaie_core_constant_1` | `[3072 x i32]` | 12,288 B | 정수 재양자화 배수 (`acc * m[n] >> 38`, round-half-even) | 없음 |

`constant_1` 은 `AMDAIEPerChannelIntegerRequantization` (§7.5) 이 만든 배수 표다. 두 표 모두 3,072 개 전체가 코어에
들어가고, 코어는 오프셋으로 자기 열을 골라 읽는다. 출력이 768 채널인 attn_out · FC2 는 배수 표가 3,072 B 라 들어갔다.

