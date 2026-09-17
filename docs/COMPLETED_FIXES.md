# 완결된 수정사항 (누적 요약)

이 파일은 **끝난 것만** 모아둔다. 진행 중이거나 원인을 못 찾은 건은 날짜별 조사
문서(`docs/2026-*.md`)에 있고 여기 넣지 않는다. 새 항목은 아래 형식으로 추가.

브랜치 구조: `origin/vgg16-onnx` (베이스, 이종 CPU+NPU 배치) → `bert` (여기서 작업 중)

---

## 1. 이종 CPU+NPU 배치 — NPU에 안 되는 연산은 CPU로

**브랜치** `origin/vgg16-onnx` (푸쉬됨) · 주요 커밋 `53a63bd`, `c3a4f7e`, `eea2f1c`, `a2b3c7e`

npu4에서는 사실상 matmul 계열만 돌아간다. LayerNorm·Softmax·GELU·pooling 등은
NPU 경로가 없다. 이걸 **디스패치 단위로 나눠 matmul은 NPU, 나머지는 CPU(llvm-cpu)** 로
보내도록 만든 것이 이 베이스다. 덕분에 VGG16·BERT 같은 실제 모델의 e2e 실행이 가능해졌다.

같이 들어간 것: bf16 contraction demotion, N개 디바이스로의 일반화,
호스트 디바이스를 이름이 아니라 capability table로 해석, pooling을 호스트로 라우팅.

**쓰는 법** — 컴파일 플래그:
```
--iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local
--iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu
--iree-amdaie-target-device=npu4
```
실행은 디바이스를 둘 다 준다: `--device=amdxdna --device=local-task`

---

## 2. 배치 matmul이 패딩 패스에 안 걸리던 문제 (row overflow)

**커밋** `ba7bc57` (2026-08-20) · **문서** `docs/2026-08-20_batch_matmul_row_overflow_fix.md`

`AMDAIEPadContractionDispatches`가 `linalg::MatmulOp`만 순회해서
**`linalg.batch_matmul`이 아예 안 보였다.** M/K가 pack-peel 파이프라인의 타일 배수로
패딩되지 않으면, `KernelDispatch.cpp`의 M-타일링 삼항식이 M을 타일링 안 된 단일 블록으로
취급한다. 그러면 넓은 N의 타일 전부가 **물리 코어 그리드의 row 축 하나에만** 몰리고,
N이 row 개수보다 많은 타일을 요구하는 순간 `AMDAIEDeviceModel::getTileType`에서
**진단 메시지 없이 abort** 한다.

수정:
- 배치 전용 대응물 추가 (`BatchMatmulInfo` / `getBatchMatmulInfo` /
  `growBatchMatmulInit` / `growBatchOutputStore`). 2D 헬퍼들이 rank-2 형상과
  루프 차원 위치를 하드코딩해 놨어서 제자리 일반화 대신 나란히 추가했고,
  배치 경로는 선행 batch 차원을 패딩 없이 그대로 통과시킨다.
- `AMDAIEInsertCores.cpp`를 2차 방어선으로 강화: row 인덱스가 디바이스 row 수를
  넘으면 다음 열로 넘긴다 (`col += row / numRows; row %= numRows`).
  이미 범위 안이면 no-op이라 비용 0.

**효과** — bert-base(batch=12 attention) corr **0.99252 → 0.99998**
(max abs diff 1.37 → ~0.04). 원래 bf16 반올림 누적으로 오해했던 오차의 대부분이
실은 **패딩 안 된 배치 matmul** 때문이었다.

> 남은 의문(미조사): 수학적으로 정확한 matmul인데 M을 타일링하느냐 마느냐가
> 왜 결과를 바꾸는가. 배치 순서만으로는 30배 corr 개선을 설명하기 어려워,
> 패딩 안 된 M 경로에 또 다른 수치 문제가 있었을 가능성.

---

## 3. int8 양자화 지원 — `aten.bmm`

**커밋** `40654aa` (2026-08-25, 서브모듈 범프) · 3단 체인:

```
iree-amd-aie  40654aa  서브모듈 범프
  └─ iree      9c8b2b7  torch-mlir 범프
       └─ torch-mlir  6ce189ce  Add int8 quantization support for aten.bmm
```

torch-mlir에 `aten.bmm`의 int8 양자화(QDQ) 경로가 없어서 배치 matmul을 int8로
내릴 수 없었다. 2D `aten.mm`은 이미 있었다.

⚠️ **주의: 세 커밋 모두 원격에 없다(로컬 전용).** 특히 torch-mlir의 리모트는
포크가 아니라 업스트림 `iree-org/torch-mlir` 이므로 푸쉬 권한이 없을 수 있다.
팀원이 int8 배치 matmul을 컴파일하려면 이 패치가 필요하다.

---

## 4. int8 배치 matmul의 batch-0 출력이 0 되는 문제

**커밋** `65069ce` (2026-09-10) · **문서**
`docs/2026-09-10_per_producer_lock_patch_fixes_both_bugs.md`,
`docs/2026-09-10_verification_campaign_after_lock_fix.md`

objectFifo의 버퍼를 **독립적인 DMA producer 여럿이 각자 겹치지 않는 구역에 쓰고
consumer 하나가 전체를 꺼내가는** 구조에서, lock 쌍을 타일당 1개만 만들고
producer lock을 `numProducers * depth` 로 미리 충전한 뒤 consumer가
`AcquireGreaterEqual(numProducers)` 로 기다리고 있었다.

AIE lock은 카운팅 세마포어라 **`AcquireGreaterEqual(N)` 은 "릴리즈가 N번 있었다"만
보장하고 누가 했는지는 담지 못한다.** `depth > 1` 이면 부풀린 크레딧 덕에 앞서 나간
producer가 *다음* 버퍼 몫까지 미리 릴리즈할 수 있으므로, producer 일부가 낸 N번의
릴리즈가 "각자 1번씩"과 똑같이 acquire를 통과시킨다. consumer는 아직 아무도 안 쓴
구역(=초기화 안 된 메모리, 0)을 그대로 실어간다. 배치 matmul에서는 **배치 하나가 통째로 0.**

수정: producer마다 자기 lock 쌍(각 `depth` 크레딧). AIE2의 BD 하나는 acquire/release
lock을 각각 1개만 들 수 있으므로, consumer 전송을 **producer 슬라이스별 BD로 분할**하는
것이 함께 따라온다. 분할이 원본과 같은 바이트를 같은 순서로 옮기는지는
`verifyProducersTileBufferInOrder` 가 컴파일 타임에 검사하고, 아니면 컴파일 에러를 낸다.

**같이 제거한 것**: `1600078` 의 delay 우회(코어마다 첫 lock release 앞 busy-wait).
증상을 가리기만 했고, 규모가 커지면 스스로 오답을 만들었다
(delay ON에서 배치 matmul 12개 전부 0.12~0.45 오답).

**검증** — 모델 27개 · 320회, 양자화 ONNX를 onnxruntime(CPU)로 돌린 값 기준.
26개가 **maxerr 0 (비트 일치)** · 실행간 출력 1종 · 전체 0 batch 0개.
남은 1개(16층 체인)는 결정론적이며 차이가 전부 출력 양자화 1 LSB의 정수배 → 반올림 판정.

---

## 5. NPU 코어에서 float 재양자화가 불가능하던 문제 — 정수 고정소수점으로

⚠️ 범위: attention 의 **matmul 6 개와 그 재양자화**가 NPU 로 갔다는 것이지,
attention 블럭 전체가 NPU 에서 연속 실행된다는 뜻이 아니다. softmax·transpose 3 개는
아직 CPU 에 있다.

**브랜치** `bert` · 조사 문서 `docs/2026-09-16_attention_on_npu_integer_requantization.md`

int8 matmul 에 재양자화를 융합하면(`no-fuse` 플래그 제거) 백엔드가 깨졌다. 표면 증상은
세 가지였는데 원인은 하나다: **aie2p 에는 스칼라 float 산술 명령이 아예 없다.**
곱셈조차 `__mulsf3` 소프트 플로트 호출이라, 융합된 재양자화 꼬리 하나가 코어의 16 KB
program memory 중 5.7 KB 를 라이브러리로만 먹었다(같은 코어의 실제 벡터 matmul 은 208 B).

증상 셋과 대응:

| 증상 | 원인 | 수정 |
|---|---|---|
| `llc`: `unable to legalize G_FMINIMUM` | clamp 의 `arith.maximumf/minimumf` | `arith-expand` 를 파이프라인에 추가 (`cmpf`+`select` 로 풀림) |
| `llc`: `unable to legalize G_INTRINSIC_ROUNDEVEN` | `math.roundeven`. upstream 확장은 `copysign` 을 거치는데 그것도 없음 | 신규 패스 `iree-amdaie-expand-roundeven` |
| `Overflow of program memory` | 소프트 플로트 라이브러리 + 펼쳐진 호출 | 신규 패스 `iree-amdaie-integer-requantization` |

핵심은 세 번째다. scale 이 전부 컴파일 타임 상수이므로

```
sitofp(acc:i32) → [× S1] → [÷ S2] → roundeven → [+ zp] → clamp → fptosi → i8
```

를 `S = S1/S2 ≈ M / 2^n` (M ≈ 31 비트)로 바꿔 **정수 곱셈 + round-half-to-even 시프트 +
정수 clamp** 로 내린다. 소수 변환·곱셈·나눗셈이 전부 사라지고 남는 라이브러리는
`__muldi3` 112 B 하나다.

**근사가 아니다 — 다만 "31 비트면 항상 같다" 도 아니다.** 세 디스패치의 accumulator
전 범위(합계 27,870,915 개 값)를 전수 비교해 f32 경로와 **불일치 0, 최대 오차 0 LSB** 임을
확인했다. 하지만 이건 **이 scale 들과 이 accumulator 범위에서** 성립하는 것이고, 패스는
이를 가정하지 않는다: `M/2^n` 를 고른 뒤 **두 형태가 정말 같은지 검사하고, 아니면 rewrite 를
건너뛴다**. 두 형태 모두 accumulator 에 대해 단조라서 clamp 구간의 각 값마다 "처음 도달하는
accumulator" 만 이분 탐색으로 비교하면 전 구간 동치가 나온다(int8 이면 255 회).

결과 (layer 0 attention):

| | 이전 | 이후 |
|---|---|---|
| proj 코어 ELF | 21,680 B (16 KB 초과) | 8,964 B |
| QK^T 코어 ELF | 18,664 B (초과) | 5,760 B |
| PV 코어 ELF | 14,300 B | 5,388 B |
| NPU vs ORT 골든 | 컴파일 불가 | **24,576/24,576 비트 동일**, 10 회 단일 해시 |

⚠️ 회귀 없음 확인: 기존 배포본 `bb_kpadq.vmfb`(9/11 빌드)를 새 컴파일러로 다시 컴파일하면
**MD5 동일**하다. 재양자화가 CPU 에 남는 `no-fuse` 경로는 이 패스들이 아예 발동하지 않는다.

**한계**: `AMDAIEVectorization.cpp:82-91` 이 양자화 elementwise 를 의도적으로 벡터화 대상에서
빼기 때문에(upstream issue #594) 이 꼬리는 여전히 스칼라 루프다. 정수라서 작고 빠르지만
벡터는 아니다. 벡터 `fcmp` 도 aie2p 에서 합법화되지 않으므로 벡터화하려면 별도 작업이 필요하다.

---

## 실행 규칙 (공유 호스트)

`ace-amd01` 은 NPU 1개·Docker 1개를 여러 사용자가 공유한다. **NPU 실행과 전체 빌드는
반드시 락으로 감싼다.** 자세한 내용은 `CLAUDE.md` 와
`docs/2026-07-06_env_setup/DEV_CONTAINER.md` §5.

```bash
./scripts/lock/status.sh                    # 누가 쓰는지 (NPU/build)
./scripts/lock/host-load.sh                 # 타 사용자 CPU 부하 (빌드 중이면 NPU 결과가 흔들림)
./scripts/lock/check-containers.sh          # 락에 안 보이는 컨테이너 감사
./scripts/lock/with-npu-lock.sh <command>   # 실제 NPU 실행은 이걸로 감싼다
```

⚠️ 백그라운드 `docker run` 을 클라이언트째로 죽이면 컨테이너가 고아가 되어
락은 free로 보이는데 NPU를 계속 쓴다. 이어서 `docker stop` 하면 진행 중 디스패치가
끊겨 디바이스가 먹는다. 컨테이너를 직접 (`docker ps` → `docker stop <id>`) 멈출 것.
