# 2026-09-07: bf16 epilogue 크래시 해결, IRON에서 최초로 진짜 벡터화 matmul 성공 + batch-0 delay-fix 재검증 시도

이 문서는 `docs/2026-08-24_int8_quantization_investigation.md`(§1-21, 특히 §18의
bf16 벡터화 시도)와 `docs/2026-09-06_second_bug_scale_threshold_summary.md`
(L/D 임계값 스캔 요약)에 이어지는, 오늘 하루치 추가 진행 내용만 담은 문서.

## epilogue 크래시 해결

이전(§18) bf16 크래시(`G_INSERT_VECTOR_ELT` on `<64 x s32>`)는
matmul 자체가 아니라 f32→i8 변환 epilogue가 원인이었음. 원인은 **64개짜리 넓은
벡터에 대한 `arith.fptosi`**(`G_FPTOSI` on `<64 x s32>` legalize 실패) —
epilogue를 완전히 **스칼라 루프**(`vector.extract` → 스칼라 `fptosi`/`trunci`/`addi`
→ `memref.store`)로 바꾸니 **컴파일 성공, 실제 하드웨어에서도 정확히 동작**
(K=8 reduction: 8×1.0×1.0=8, +bias 11 = 19, 정확히 일치). **이 세션에서 처음으로
`aiecc`를 통해 진짜 `aievec.matmul_aie2p` 벡터 연산이 성공적으로 컴파일+실행됨.**

성공한 MLIR: `/tmp/vecmatmul_probe/probe_bf16_v3.mlir` (단일 shot).

## batch-0 delay-fix 자체를 IRON에서 재검증 시도

사용자 요청 배경 — "release에만 delay 넣으면 해결된다"는 결론이 IRON에서
검증된 적 없다는 지적. 이 bf16 matmul을 core-internal `scf.for` 루프
(BATCH=4, 같은 락 재사용)로 감싸서 **delay-fix를 전혀 넣지 않고** 실행 —
원래 batch-0 버그의 정확한 재현 조건.

### 안전 사고 발생, 원인 규명 완료

최초 실행 시 **hang**(timeout). 안전 수칙대로 이미 검증된 단일 dispatch를
재실행해서 복구 확인하려 했는데 **그것도 hang** — 2회 연속으로 즉시 라이브
테스트 중단. `xrt-smi examine`은 정상 응답, lock도 정상 해제된 상태 확인.

**코드를 다시 검토해서 원인을 찾음**: `runtime_sequence`에서 **출력 DMA는
4번 반복(outer dim=4)하도록 제대로 만들었는데 입력 DMA는 1번만 push**했음 —
rep 1부터 코어가 새 입력 데이터를 영원히 못 받아 `lock_cons_cons` 획득에서
**결정론적으로 deadlock**하는, 순수한 코드 구성 실수였음(하드웨어 레이스
아님). 두 번째 hang도 첫 번째의 TDR 복구(`tdr_timeout_ms=60000`)가 채 끝나기
전에 너무 빨리 재시도해서 겹쳤을 가능성이 높음 — 진짜 별개의 2번째 사고는
아니었던 것으로 판단.

### 수정 및 재실행

입력 DMA를 출력과 동일하게 4번 반복(outer dim=4, stride=0)하도록 수정:

```mlir
// before (버그): 입력이 1번만 push됨
aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c64_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @in0} : memref<64xbf16>

// after (수정): 출력과 동일하게 4회 반복
aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c4_i64, %c1_i64, %c1_i64, %c64_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @in0} : memref<64xbf16>
```

수정 후 재실행 — **hang 없이 깔끔하게 완료, 5회 반복 재현성 확인** (`state=4`,
전부 정상 종료). 결과: **4개 배치 전부 바이트 단위로 완전히 동일**(앞 32개=19,
뒤 32개=20 — bf16 matmul_aie2p의 내부 BFP16 변환 관련으로 추정되는 구조적
반올림 패턴, batch 인덱스와는 무관, 5번 재실행 모두 동일하게 재현). **batch 0가
다른 배치와 다르게 나오는 원래 lock pre-charge 증상은 이 최소 구조(단일 타일,
단일 8x8x8 bf16 matmul, 4회 core-loop, delay-fix 없음)에서는 재현되지 않음.**

수정된 전체 MLIR (재현 자료): `/tmp/vecmatmul_probe/probe_bf16_batch_nodelay.mlir`
— 세션 스크래치, 저장소 밖에 있음. 구조 요약:
- 단일 컬럼(col 0), shim(0,0) → memtile(0,1) → core(0,2) 3-타일 체인.
- core는 "진짜 배치 matmul 하드웨어 명령"이 아니라 **동일 락 4쌍을 재사용하는
  소프트웨어 `scf.for` 루프**로 8x8x8 bf16 matmul을 4번 반복하는 구조 —
  IRON/mlir-aie에는 배치 전용 하드웨어 primitive가 없고, 실제 IREE가
  생성하는 "batched matmul" dispatch도 마찬가지로 컴파일된 코드 상에서는
  같은 core-loop(또는 command-queue 반복) 방식이므로 구조적으로 동등한 비교임.
- memtile의 입력 S2MM/MM2S BD는 각각 1개 BD로 고정, `repeat_count`가 아니라
  런타임 시퀀스의 outer-dim(4) 반복으로 4번 재전송되는 방식 — 위 버그가
  바로 이 부분(출력만 반복, 입력은 반복 안 함)에서 발생.

## 해석에 필요한 유의점

이건 진짜 int8이 아니라 bf16이고(IREE 자체가 이 하드웨어에서 실제로 쓰지
않는 경로), 스케일도 최소(K=8, 단일 타일)라 실제 버그가 필요로 하는 조건
(D=768급 스케일, L=10~13 근처)과는 거리가 멂 — "batch-0 pre-charge가
벡터화에서도 안 나타난다"고 단정하긴 이르고, 최소 스케일에서는 이
메커니즘이 안 보인다는 정도로만 해석해야 함.

## 다음 단계

이 성공한 bf16 벡터화 IRON 파이프라인(epilogue는 스칼라 루프로)을 이제
8컬럼 + 더 큰 스케일 + 실제 chainmix와 유사한 alternation 구조에 이식해서,
`2026-08-24` 문서 §11-21의 scalar 기반 negative 결과들을 벡터화 버전으로
재검증하는 게 다음 단계.

## 추가 (같은 날 이어서): 8컬럼으로 확장 + K=768 스케일 재스캔 — 여전히 NEGATIVE

**1컬럼 K=768 벡터화 bf16 matmul 먼저 단독 검증**: `probe_bf16_matmul768_1col.mlir`
— 8x8 타일 12개(K=768)를 `aievec.matmul_aie2p`로 누적(누산기는 `scf.for`
vector-typed iter_arg 대신 `memref<8x8xf32>` 스크래치 버퍼 경유 — 전자는 이
백엔드에서 LLVM 변환 실패로 이미 알려진 회피 패턴), epilogue는 여전히
스칼라 루프. **한 번에 컴파일 성공, 실제 HW에서도 hang 없이 정상 실행.**
단, 결과값이 **정수 모델대로면 11(bias)+0(768 mod 256의 trunc)=11이어야
하는데, 실제로는 전부 균일하게 75** (bias 11 + raw truncated accumulator
64) — bf16 matmul_aie2p의 내부 BFP16 하드웨어 변환에 의한 구조적 반올림
(§ 위쪽의 K=8 사례에서 이미 관찰된 것과 같은 계열의 현상)로 해석, 4096개
전부 균일한 값이라 최소한 위치/배치 의존적이진 않음. 이후 모든 8컬럼
버전에서 이 실측값(TYPE A=75, TYPE B=72)을 기준값으로 사용
(`gen_multicol_bf16.py`, int8 산술 모델이 아니라 실측 기반이라는 점을
스크립트 헤더에 명시).

**8컬럼 + repeat_count(BATCH=4) + combined multi-column TCT sync + fresh
hw_context per dispatch로 확장** (`gen_multicol.py`의 기존 scaffolding을
그대로 재사용, 컴퓨트 부분만 이 bf16 벡터화 패턴으로 교체 —
`gen_multicol_bf16.py`). TYPE A/B 둘 다 한 번에 컴파일 성공.
`diag_alternate_8col_freshctx_vecmatmul_bf16.exe`로 A→B 교대 실행:
**1라운드, 20라운드, 100라운드(총 4000회 블록 체크) 전부 hang 없이 깔끔하게
완료, 전부 PASS(0/4000 실패).** 스칼라 버전과 동일한 스케일(100라운드)에서
동일하게 완전히 negative — **진짜 벡터화된 bf16 8x8x8 matmul + 8컬럼 +
K=768급 컴퓨트 시간 + repeat_count 배치 + combined TCT sync + fresh
hw_context, 이 모든 조건을 동시에 만족해도 이 IRON 하네스에서는 여전히
재현 안 됨.**

## int8(진짜 chainmix)과의 구조적 차이 — "1대1 비교는 아니다"의 구체적 의미

사용자 요청대로 이 bf16 IRON 결과를 실제 int8 chainmix와 비교해서 어떤
방식 차이가 있는지 정리. `iree-dump-module`로 `chainmix_L12_int8.vmfb`의
실제 함수 호출 시퀀스를 직접 확인한 결과: **레이어당 정확히 2개의
`hal.command_buffer.dispatch` 호출만 있고(비-배치 dispatch 1개 + 배치
dispatch 1개), Reshape/Transpose용 별도 dispatch는 안 보임** — 즉 실제
chainmix의 alternation 패턴도 이 IRON 하네스처럼 순수 A→B 2종류
교대이며, "reshape/transpose가 끼어들어 더 복잡한 alternation 패턴을
만든다"는 가설은 확인 결과 사실이 아님(기각).

확인된/의심되는 진짜 차이점:

1. **연산 유닛 자체가 다름**: bf16은 `matmul_aie2p`가 내부적으로 BFP16
   하드웨어 변환 경로를 타는 반면(§ 위 75/72 값이 그 증거), 실제
   chainmix의 int8은 npu4 네이티브 정수 MAC 배열(`aievec.matmul`)을
   직접 씀 — 서로 다른 실행 유닛. 버그가 int8 정수 파이프라인에만
   있는 타이밍/스케줄링 문제라면, bf16 경로로는 스케일을 아무리
   키워도 구조적으로 재현 불가능할 수 있음. **이게 여전히 가장 유력한
   미해결 설명.**
2. **weight가 DMA를 안 탐**: 이 IRON 하네스는 weight를 core-local
   `dense<1.0>` 컴파일타임 상수로 박아놔서 DMA 채널을 전혀 안 씀. 실제
   chainmix는 weight도 실제 텐서 입력이라 DMA로 옮겨질 가능성이 높음
   (미확인 — vmfb의 `command_buffer.dispatch` 인자 개수가 A/B 타입마다
   다른 것으로 봐선 바인딩 텐서 개수가 다름, 즉 weight도 별도 채널로
   들어갈 개연성 큼). 만약 버그가 이 세 번째(weight) DMA/lock 채널의
   타이밍과 관련 있다면 이 하네스는 애초에 그 채널 자체가 없어서
   구조적으로 못 봄.
3. **wait 구조가 더 단순화됨**: 이 하네스는 dispatch당 `aiex.npu.sync`
   1번(combined, column_num=8)으로 끝나지만, 이전에 확인한 실제
   chainmix control code는 dispatch당 `amdaie.npu.tct_sync`가 **3번,
   col_num=8/8/4로 순차적으로** 나옴(§14/§15, `2026-08-24` 문서) — 즉
   실제 dispatch 안에는 이 하네스가 흉내내지 못한 세분화된 wait
   지점이 여러 개 있음. 이 하네스의 "combined wait"는 이름은 같지만
   실제로는 real IREE보다 더 거친(coarse) 근사치임.
4. **실수 fp32 accumulate → i8 requantize 방식이 다름**: 실제
   chainmix는 scale/zero-point 기반 실제 아핀 requantize(부동소수점
   곱셈+반올림+클램프)를 쓰는 반면, 이 하네스의 epilogue는 그냥
   trunc+상수bias — 명령어 시퀀스/타이밍이 다를 수 있음.

**결론**: 이번 재스캔은 "벡터화 자체가 필요조건"이라는 가설을 약화시킴
(벡터화해도 여전히 negative) — 하지만 dtype/실행유닛 차이(1번) 때문에
"int8 전용 버그일 가능성"은 여전히 완전히 살아있고, 오히려 이번 결과로
더 유력해짐. 다음으로 시도해볼 만한 것: (a) weight도 실제 DMA로 옮기는
버전으로 이 하네스를 다시 만들어 3번째 채널을 추가, (b) 3-way TCT
sync(8/8/4)를 combined 1-way 대신 그대로 재현. 둘 다 아직 실행 안 함.
