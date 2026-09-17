# `bert-onnx` 브랜치 — 무엇이 어디서 바뀌었나

**저장소** `ace-knu/iree-amd-aie` · **브랜치** `bert-onnx` (최신 `faf1213`, 2026-09-11)
**베이스** `vgg16-onnx` + 10 커밋

상세 diff 는 커밋에 있습니다. 이 문서는 **어느 파일이 왜 바뀌었는지** 설명만 합니다.

```bash
git fetch origin && git checkout bert-onnx
git submodule update --init --recursive
```

`bert-onnx` 는 **검증 완료된 항목만** 담는 공유 브랜치입니다. 진행 중인 작업은 들어오지 않습니다.

---

## 커밋 목록

| 커밋 | 날짜 | 내용 |
|---|---|---|
| `ed4cee1` | 08-20 | **배치 matmul 타일 배수 패딩 누락 수정** (§1) |
| `13f9c4e` | 08-20 | BERT-tiny / BERT-base e2e 레시피 (`models/`) |
| `2e07978` | 08-22 | bert-base corr 재측정 |
| `98cb385` | 08-25 | iree 서브모듈 범프 (→ `5cea157` 에서 되돌림, §3) |
| `7b17209` | 08-26 | batch-0 delay 우회 (→ `1d60be9` 에서 제거, §2) |
| `c55b5cd` | 08-26 | ONNX → dispatch 프론트엔드 lowering 설명 문서 |
| `3fa570a` | 09-01 | 브랜치 목적 README |
| **`1d60be9`** | **09-11** | **batch-0 all-zero 근본 수정 + delay 제거** (§2) |
| `5cea157` | 09-11 | 서브모듈 포인터 정정 (§3) |
| `faf1213` | 09-11 | README 갱신 |

**실제로 고친 것은 2건**입니다 — `ed4cee1` 과 `1d60be9`. 나머지는 문서·레시피·되돌림입니다.

---

## 1. `ed4cee1` — 배치 matmul row overflow

### 바뀐 파일

| 파일 | 증감 | 무엇 |
|---|---|---|
| `Transforms/AMDAIEPadContractionDispatches.cpp` | +164 | 배치 matmul 용 패딩 경로 추가 |
| `Transforms/AMDAIEInsertCores.cpp` | ±30 | row 넘침 2차 방어선 |
| `docs/2026-08-20_batch_matmul_row_overflow_fix.md` | +222 | 원인 분석 |

### 왜

`AMDAIEPadContractionDispatches` 가 `linalg::MatmulOp` 만 순회해서 **`linalg.batch_matmul` 이
아예 안 보였습니다.** M/K 가 pack-peel 파이프라인의 타일 배수로 패딩되지 않으면
`KernelDispatch.cpp` 의 M-타일링이 M 을 타일링 안 된 단일 블록으로 취급하고, 그러면 넓은 N 의
타일 전부가 **코어 그리드의 row 축 하나에만** 몰립니다. N 이 row 개수보다 많은 타일을 요구하는
순간 `AMDAIEDeviceModel::getTileType` 에서 **진단 메시지 없이 abort** 합니다.

### 어떻게

- 배치 전용 대응물 추가 (`BatchMatmulInfo` / `getBatchMatmulInfo` / `growBatchMatmulInit` /
  `growBatchOutputStore`). 기존 2D 헬퍼들이 rank-2 형상과 루프 차원 위치를 하드코딩해 놔서
  제자리 일반화 대신 **나란히 추가**했고, 배치 경로는 선행 batch 차원을 패딩 없이 통과시킵니다.
- `AMDAIEInsertCores.cpp` 는 독립적인 2차 방어선: row 인덱스가 디바이스 row 수를 넘으면
  다음 열로 넘깁니다 (`col += row / numRows; row %= numRows`). 이미 범위 안이면 no-op.

### 효과

bert-base(batch=12 attention) corr **0.99252 → 0.99998** (max abs diff 1.37 → ~0.04).
원래 bf16 반올림 누적으로 오해했던 오차의 대부분이 실은 패딩 안 된 배치 matmul 때문이었습니다.

---

## 2. `1d60be9` — batch 0 출력이 0 (lock 이 producer 신원을 못 담음)

### 바뀐 파일

| 파일 | 증감 | 무엇 |
|---|---|---|
| `Transforms/AMDAIEObjFifoBufferization.cpp` | +82 / −54 | **lock 을 몇 개 만들지** |
| `Transforms/AMDAIELowerToAIE.cpp` | +253 | **BD 가 그 lock 을 어떻게 쓸지** |
| `Transforms/AMDAIELowerToAIE.h` | ±10 | `createDMABlocks` 시그니처 |
| `aie/AMDAIECoreToStandard.cpp` | **−88 (전부 삭제)** | `7b17209` 의 delay 우회 제거 |

### 왜

objectFifo 의 버퍼를 **독립적인 DMA producer 여럿이 각자 겹치지 않는 구역에 쓰고
consumer 하나가 전체를 꺼내가는** 구조에서, 기존 코드는 lock 쌍을 타일당 1개만 만들고
producer lock 을 `numProducers * depth` 로 미리 충전한 뒤 consumer 가
`AcquireGreaterEqual(numProducers)` 로 기다렸습니다.

AIE lock 은 카운팅 세마포어라 **`AcquireGreaterEqual(N)` 은 "릴리즈가 N번 있었다"만 보장하고
누가 했는지는 담지 못합니다.** `depth > 1` 이면 부풀린 크레딧 덕에 앞서 나간 producer 가
*다음* 버퍼 몫까지 미리 릴리즈할 수 있으므로, producer 일부가 낸 N번의 릴리즈가
"각자 1번씩" 과 똑같이 acquire 를 통과시킵니다.

```
의도한 상황   코어0×1  코어1×1  코어2×1  코어3×1  = 4   전체 준비됨
실제 허용됨   코어1×2  코어2×1  코어3×1  코어0×0  = 4   batch 0 미기록
```

consumer 는 크레딧이 찼으니 꺼내가고, 느린 코어의 구역은 **초기화되지 않은 메모리(=0)** 그대로
나갑니다. 그 코어가 batch 0 을 담당하면 **batch 0 이 통째로 0** 입니다.

초기값을 producer 수만큼 곱한 것은 **총 크레딧 개수만** 맞춘 것이고, 잃어버린 신원은
되찾아주지 않습니다.

### 어떻게

`AMDAIEObjFifoBufferization.cpp` — producer 가 여럿이고 consumer 가 하나면
**producer 마다 lock 쌍**(각 `depth` 크레딧):

```cpp
bool lockPairPerProducer = numProducers > 1 && numConsumers == 1;
size_t numLockPairs      = lockPairPerProducer ? numProducers : 1;
producerLockInitValue    = lockPairPerProducer ? depth : (기존 식);
```

`AMDAIELowerToAIE.{h,cpp}` — AIE2 의 DMA BD 하나는 **acquire lock 1개 + release lock 1개**만
들 수 있습니다. lock 을 N 쌍으로 쪼갠 순간 "N 쌍 전부 기다려라" 를 BD 하나로 표현할 방법이
없으므로, **consumer 전송을 producer 슬라이스별 BD 로 분할**하는 것이 필연적으로 따라옵니다.
`createDMABlocks` 가 그 분할을 하고, `verifyProducersTileBufferInOrder` 가 쪼갠 BD 들이
원본과 **같은 바이트를 같은 순서로** 옮기는지 컴파일 타임에 검사합니다 — 어긋나면 조용히
순서를 바꾸는 대신 **컴파일 에러**를 냅니다.

### 같이 제거한 것 — `7b17209` 의 delay 우회

`AMDAIECoreToStandard.cpp` 의 −88 줄은 코어마다 첫 lock release 앞에 넣었던 busy-wait 입니다.
**증상을 타이밍으로 가리기만 했고, 규모가 커지면 스스로 오답을 만들었습니다** — delay 가
켜져 있으면 12층 attention 의 배치 matmul 12개 **전부**가 0.12~0.45 오답이었고 손상이
`col % 8 == 0` 에 몰렸습니다.

즉 **delay ON 이면 배치 matmul 이 깨지고, delay 만 끄면 batch 0 이 0** 이라 두 상태 모두
틀렸습니다. 맞는 조합은 "이 수정 + delay 제거" 뿐입니다.

참고로 `98cb385` 부터 이 수정 직전까지 **모든 리비전에서** delay 를 끄면 batch 0 이
ALL-ZERO 로 재현됩니다. 중간에 우연히 고쳐진 적이 없었습니다.

### 검증

모델 27개 · 320회 실행. 기준은 **같은 양자화 ONNX 를 onnxruntime(CPU)** 로 돌린 값이고,
`maxerr ≤ 1e-6` **그리고** 실행간 출력 1종 **그리고** 전체 0 batch 0개 를 모두 만족해야 통과.

| 그룹 | 모델 | 실행 | 결과 |
|---|---|---|---|
| 원본 batch-0 재현체 (B=2, M=16, K=16, N=64) | 1 | 20 | maxerr 0 |
| 배치 matmul 단독, 배치 2~16 · 형상 스윕 | 12 | 120 | maxerr 0 |
| 24 디스패치 체인 (D=768, HEADS=12) | 1 | 20 | maxerr 0, corr 1.000000 |
| 과거 깨졌던 체인 L=10·11·12·13·14 | 5 | 75 | maxerr 0 |
| 같은 계열 L=16 | 1 | 15 | 결정론적, ±1 LSB 반올림 차 |
| 독립 시드 인스턴스 | 7 | 70 | maxerr 0 |

**26/27 이 CPU 기준과 비트 단위 일치**, 전부 실행간 출력 1종.
lit 테스트 `obj_fifo_bufferization.mlir`, `lower_to_aie.mlir` 통과.

과거 대비: 배치 수 9(정상)/10(손상) 경계가 사라졌고, 최악이던 L=13 (20회 → 6종,
원소 19.89% 오답) 이 15회 1종입니다.

---

## 3. `5cea157` — 서브모듈 포인터 정정 (⚠️ 받을 때 영향 있음)

`98cb385` 이 `third_party/iree` 를 `9c8b2b7` 로 올려뒀는데, **그 커밋이 `ace-knu/iree` 에
없습니다**(로컬 전용). 그래서 지금까지 이 브랜치를 recursive clone 하거나
`git submodule update` 하면 **받을 수 없는 ref 로 실패**했습니다. 밟으셨다면 이 건입니다.

`c73928f0`(= `origin/vgg16-onnx`, `origin/dev` 와 같은 값)으로 되돌려 고쳤습니다.
이제 그대로 체크아웃됩니다.

§1, §2 의 수정은 **플러그인 전용**이라 서브모듈 리비전과 무관하게 빌드됩니다.

---

## 4. int8 batched matmul 을 쓰려면 — 별도 패치 필요

torch-mlir 에 `aten.bmm` 의 int8 양자화 경로가 없습니다(2D `aten.mm` 은 이미 있음).
ONNX 의 배치 MatMul 은 `aten.matmul` 이 아니라 **`aten.bmm` 으로 임포트**되므로
`FuseQuantizedOpsPass` 에 매칭되는 패턴이 없어 **양자화가 조용히 통째로 건너뛰어집니다.**

커밋 1개(2개 파일, +66줄)로 해결되지만, 해당 서브모듈의 리모트가 우리 포크가 아니라
upstream `iree-org/torch-mlir` 이라 푸쉬할 수 없어 **패치 파일로 따로 공유**합니다.

| 파일 | 증감 |
|---|---|
| `lib/Dialect/Torch/Transforms/FuseQuantizedOps.cpp` | +2 |
| `lib/Conversion/TorchToLinalg/Linear.cpp` | +64 |

적용 위치는 `third_party/iree/third_party/torch-mlir` (3단 서브모듈). 베이스가 upstream
`main` 에 있는 커밋이라 깨끗이 붙고, 적용 후 IREE 재빌드가 필요합니다.
자세한 적용법은 패치와 같이 드리는 `README_int8_bmm_patch.md` 참고.

확인: 패치 후 int8 배치 matmul 모델을 임포트하면 `--compile-to=input` 단계에서
`linalg.quantized_matmul` 이 나와야 합니다. 패치 전에는 평범한 f32 `linalg.matmul` 이 나옵니다.

### ⚠️ 알려진 한계 — per-channel 양자화는 아직 안 됩니다

`linalg.quantized_matmul` 이 zero-point 를 **스칼라 `i32` 로만** 받아서 축별(per-channel)
스케일을 표현할 수 없습니다. per-axis 로 양자화하면 f32 로 폴백한 뒤 AMD-AIE 백엔드에서
컴파일 실패합니다 (`AMDAIEBufferizeToAllocation` — "expected only one target op, found 2").

**양자화는 per-tensor 로 하십시오** — `quantize_static(..., per_channel=False)`, 기본값입니다.
(현재 조사 중인 항목입니다.)

---

## 5. 포함되지 않은 것

- **matmul+bias 융합(3-input) 조사** — 미해결이라 이 브랜치에 없습니다. 코어 입력 채널이
  2개인데 bias 같은 3번째 텐서가 필요해지면 `packet-flow-strategy=inputs` 가 강제되고,
  그것이 fan-out + 공유 lock + ping-pong 패턴을 만들어 하드웨어에서 행(hang)합니다.
- **공유 호스트용 NPU/빌드 락 스크립트** (`scripts/lock/`) — 각자 관리.
