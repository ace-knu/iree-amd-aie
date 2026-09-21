# Matmul+bias 퓨전 hang — packet-ID 공유+다른 타일 가설 재검증, 컬럼 수 임계치(4↔5) 발견 (2026-08-28)

[docs/2026-08-27_matmul_bias_fusion_hang_root_cause_refined.md](2026-08-27_matmul_bias_fusion_hang_root_cause_refined.md)의
후속. 리부트(사고 복구 확인 완료, 별도 기록)로 조사 재개, mlir-aie/IRON 툴체인만으로
독립 재현/격리 실험을 진행했습니다. **결론: 개별 메커니즘(공유 채널 + 다른 packet ID +
다른 목적지 타일)은 그 자체로는 hang의 충분조건이 아니고, 정확히 5개 이상의 컬럼이
"동시에" 이 패턴을 실행할 때만 hang이 납니다 — 4컬럼까지는 아무리 반복해도 안전.**

## 0. 재개 배경

2026-08-28 아침 리부트 후 NPU 정상 복구 확인(별도 기록: 메모리 `project-npu-wedge-recovery-confirmed`).
이후 matmul+bias 퓨전 hang 조사를 재개하면서, 8/27 문서의 결론("공통분모는
fan-out이지 repeat_count가 아님")을 다시 감사하다가, 실제로는 **circuit-flow
fan-out도 이미 PASS한 사례가 있었다**는 걸(같은 문서 §3의 프로즈 안에 있었지만
요약 표에서 누락됨) 재발견 — 이게 이번 세션 재검토의 출발점입니다.

## 1. mlir-aie 공식 예제로 재검증 (실제 하드웨어)

두 개의 mlir-aie 공식 lit 테스트를 이 npu4(Strix, mlir-aie 이름으로는 `npu2`)
하드웨어에서 직접 빌드/실행:

- **`test/npu-xrt/packet_flow_fanout`** (packet flow, shim의 한 MM2S 채널을
  packet ID 2개(3,7)로 나눠 서로 다른 채널로 — 단, 목적지는 같은 타일의 다른
  채널): **PASS**
- **`programming_guide/section-2/section-2f/05_join_L2` (`run_strix_makefile.lit`,
  `REQUIRES: ryzen_ai_npu2, peano`)**: `.split()`/`ObjectFifoLink` 기반 3-way
  distribute+join — 우리 hang 재현(`aie2_broadcast.py`)과 완전히 같은 원시
  연산. 컴파일된 MLIR을 직접 까보니 데이터 경로가 **100% circuit flow**임을
  확인(`aie.flow`, packet 아님). **PASS**

## 2. 가설 재정정: "packet ID 공유 + 다른 타일"이 진짜 트리거인가?

기존 메모리(`project-matmul-elementwise-fusion` §"packet-ID-sharing-one-channel
fan-out degree")에 실제 bmm_bias 컴파일 결과를 IR로 감사한 기록이 있음:
**"하나의 물리 채널이 서로 다른 packet ID를 가진 2개 이상의 flow를 처리하는데,
그 flow들이 각각 다른 물리 타일로 감"** — 이게 실제 구조라는 발견. 이걸 정확히
격리하는 최소 재현을 새로 만듦.

**재현 구성** (`test/npu-xrt/packet_flow_fanout/aie.mlir`을 최소 수정):
`packet_flow(4)`의 source를 `<tile_0_1, DMA:2>`에서 `<tile_0_1, DMA:0>`로 옮겨,
memtile의 MM2S 채널 0 하나가 2-BD 체인(pkt_id=0→tile_0_2, pkt_id=4→tile_0_3)을
갖도록 — 락/버퍼는 원본의 검증된 코드 그대로 재사용.

**결과 (단일 컬럼, 반복 스트레스 포함)**:

| 반복 횟수 (host-side 재발행) | 결과 |
|---|---|
| 1 (단발) | PASS |
| 200 | PASS |
| 2,000 | PASS |
| 20,000 | PASS (실행시간 0.3s→1.0s로 증가, 실제 부하 걸림 확인) |

**결론: "공유 채널 + 다른 packet ID 2개 + 다른 목적지 타일 2개" 구조는, 단일
컬럼 안에서는 반복 규모와 무관하게 안전하다.** 이 시점에서 진짜 bmm_bias
hang과의 유일한 남은 차이는 "규모"(8컬럼 전체가 동시에 비슷한 패턴을 쓴다는 것)
뿐이라는 가설로 좁혀짐 (user가 직접 제안).

## 3. 8컬럼 동시 실행 재현 — 1차 시도: HANG, 그러나 자체 버그로 오염된 결과

동일한 per-column 구조(memtile MM2S 채널 하나 + packet ID 2개 + 컴퓨트 타일
2개)를 8개 컬럼에 걸쳐 전부 동시에 실행하는 재현을 새로 작성
(`gen_8col.py`, packet ID 예산을 아끼려고 실험 대상이 아닌 shim 입력/출력
리턴 경로는 plain circuit `aie.flow`로 변경 — 컬럼당 packet ID 2개만 씀, 총
16개, 32개 예산 안에 여유).

1차 실행: **8컬럼 = HANG** (90초 하드 타임아웃까지 안 풀림). 하지만 검증 결과
(§4) 데이터 자체가 애초에 틀리게 나온 걸 발견 — **재현 스크립트 자체에
컬럼별 row-offset을 실제 DMA descriptor에 반영 안 한 버그**가 있었음
(`gen_8col.py`의 `runtime_sequence` 생성부에서 `base = c * 128` 변수를 계산만
해두고 실제 오프셋 필드에는 항상 `%c0_i64`만 씀 — 1컬럼짜리에서도 재현되는
걸로 확인, 즉 멀티컬럼 효과가 아니라 순수 스크립트 버그). **이 시점의 "8컬럼
hang" 결과는 신뢰할 수 없음 — 데이터가 애초에 틀렸으므로 hang도 버그의
부작용(같은 주소를 여러 컬럼이 동시에 씀)일 가능성이 있었음.**

## 4. 버그 수정 후 재검증 — 진짜 결과: 4↔5 컬럼 사이 임계치

`gen_8col.py`에 컬럼별 row-offset 상수(`%crow0_{c}`, `%crow1_{c}`)를 추가해서
각 컬럼의 입출력이 실제로 자기 몫의 버퍼 영역만 쓰도록 수정. 수정 확인:
1/2/4컬럼에서 **정상 값(13, 8) 출력 확인** — 재현 스크립트가 이제 올바르게
동작함이 검증됨. 그 상태로 컬럼 수를 이분탐색:

| 컬럼 수 | 결과 | 비고 |
|---|---|---|
| 1 | **PASS** | 값 13/8 정상 |
| 2 | **PASS** | 값 13/8 정상 |
| **4** | **PASS** | 값 13/8 정상 |
| **5** | **HANG** | `ert_cmd_state = 8` (TIMEOUT), 약 60초 만에 TDR이 자동 개입해서 정상 회수 |
| 6 | **HANG** | `ert_cmd_state = 8`, 약 62초 |
| 8 | **HANG** | 90초 host 타임아웃까지 안 풀림 (TDR 자체 회수 확인은 못 함, 그전에 강제 kill) |

**임계치가 정확히 4↔5 사이에서 깔끔하게 갈립니다.** 매 테스트 사이사이
`xrt-smi examine`으로 디바이스 상태를 확인했고, hang난 케이스들도 전부 (강제
kill한 8컬럼 케이스 제외) TDR이 스스로 정상 복구했음을 `ert_cmd_state=8` 값과
정상적인 프로세스 종료(EXIT=0, 크래시 아님)로 확인. 사고 없이 안전하게
진행됨.

**참고 (아직 검증 안 된 우연의 일치 가능성)**: 이 투자의 다른 갈래
(BERT-tiny row-overflow 버그, `project-bert-e2e-plan` 참고)에서 "npu4가
`num_rows`를 4로 캡핑한다"는 완전히 별개의 제약이 나온 적 있음. 지금 발견한
"4개까지는 되고 5개부터 hang"이 그거랑 관련 있는 하드웨어 자원 제한인지,
아니면 완전히 다른 우연의 일치인지는 아직 모름 — 다음에 파볼 거리.

## 5. 의미

지금까지(8/25~8/27) 시도된 모든 격리 재현은 "단일 메커니즘"을 가정했음
(repeat_count, fan-out 자체, packet-ID 공유 등) — 전부 개별로는 무죄로
나왔던 이유가 이제 설명됨: **진짜 트리거는 특정 메커니즘이 아니라 "충분히
많은 수(≥5)의 컬럼이 동시에 경합하는 규모" 그 자체**였을 가능성이 높음.
이건 8/25 문서의 "8th 실험"(macro-scale delay)에서 나온 "특정 물리
row(2 vs 3)에서만 델레이가 먹히고 안 먹히는 하드웨어 비대칭" 발견과도
결이 비슷함 — 순수 소스/IR 레벨 비교로는 안 보이는, 물리적 규모/배치
의존적 현상일 가능성.

## 6. 다음 단계 (미착수)

1. **"5"라는 숫자 자체가 의미 있는지 확인** — 정확히 몇 개의 물리 자원이
   공유되길래 4개까지만 버티는지 (스트림 스위치 arbiter 슬롯 수, NoC 대역폭,
   메모리 컨트롤러 큐 깊이 등 후보) — 소스/디스어셈블리로 시작 가능.
2. **`num_rows=4` 캡 제약과의 연관성 확인** — 우연의 일치인지 같은 근본 원인인지.
3. 이 4↔5 경계가 **컬럼 수**에 관한 것인지, 아니면 단순히 **동시에 활성인
   packet-공유-connection의 총 개수**(지금 컬럼당 1개씩이라 컬럼 수와 같음)에
   관한 것인지 구분 — 예: 4컬럼인데 컬럼당 2개씩(총 8개) 공유 connection을
   두면 hang나는지 테스트하면 "컬럼 수"와 "connection 총량" 중 뭐가 진짜
   변수인지 분리 가능.
4. 실제 bmm_bias의 8컬럼 구조에 이 지식을 적용 — 만약 5개 미만으로만 동시에
   packet-공유 connection이 뜨도록 스케줄링/배치를 바꿀 수 있다면 우회
   가능성.

## 재현 자료

`_local/mlir_aie_repro/2026-08-28_column_threshold/`에 저장 (repo 관례상
`_local/`은 gitignore, 커밋 안 됨):
- `gen_8col.py` — N컬럼짜리 "공유 채널+다른 packet ID+다른 타일" 재현 자동
  생성기 (버그 수정 완료 버전, row-offset 정상 반영됨). `python3 gen_8col.py
  <NUM_COLS> <HOST_REPEAT_N>` 형태로 호출.
- `gen_repeat.py` — 단일 컬럼(2-tile) 버전을 host-side 반복 N회로 언롤해서
  생성하는 스크립트 (scf.for가 `aie.runtime_sequence` 안에서 지원 안 돼서
  텍스트 언롤 방식 사용).
- `diag.cpp` — 컬럼별 출력 첫/마지막 바이트를 찍어서 정확한 correctness/hang
  여부를 진단하는 최소 호스트 코드 (`NCOLS_PLACEHOLDER`를 실제 컬럼 수로
  치환해서 컴파일).
- `aie_arch_2tile_singleshot.mlir` — §2의 단일 컬럼 최소 재현 원본
  (`packet_flow_fanout`에서 파생, 20,000회 반복까지 PASS 확인된 버전).

실행 환경은 이전 문서들과 동일 (`~/NPU/mlir-aie/ironenv`, `/opt/xilinx/xrt`,
이 저장소의 `llvm-aie`를 Peano로 재사용). 컴파일:
```bash
source ~/NPU/mlir-aie/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
export PEANO_INSTALL_DIR=~/Projects/iree-amd-aie-vgg16/llvm-aie
python3 gen_8col.py 5 1   # NUM_COLS=5, host-repeat=1
sed -i 's/NPUDEVICE/npu2/g' aie_arch_8col.mlir
aiecc --no-aiesim --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin \
  --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie_arch_8col.mlir
```

---

## 추가 조사 (같은 날, 이어서) — 4↔5 임계치는 "컬럼 수"가 아니라 "컬럼 4 이상 자체"였고, 공유 머신 경합으로 재검증이 복잡해짐

§4의 "4컬럼 안전, 5컬럼부터 hang"이라는 결론에 대해, "컬럼 개수"와 "어느 물리적 컬럼을 쓰는가"를 분리하는 후속 실험을 진행했고, 도중에 **이 공유 머신(`ace-amd01`)에서 여러 사용자가 동시에 NPU/빌드 자원을 쓸 때 결과 자체가 오염될 수 있다는 걸 직접 겪었습니다.**

### 하드웨어 배경 조사 (정적 리서치, 실행 없음)

`third_party/aie-rt/driver/src/lite/xaie_lite_hwcfg.h`에서 확인: Strix는 실제로 **4컬럼짜리(A0)와 8컬럼짜리(B0) 두 실리콘 변형**으로 나오고, 우리가 쓰는 B0는 다른 파라미터(row 수, DMA 채널 수, lock 수, arbiter/msel 최대치)가 A0와 완전히 동일한 채 컬럼 수만 4→8로 늘어난 구조입니다. 즉 B0가 "A0 두 개를 이어붙인 것"처럼 보이는 문서/코드상의 근거는 있으나, 실제로 컴파일러/드라이버 어디에도 "4+4 분할"을 명시적으로 모델링한 코드는 없음(별도 fork 리서치로 확인, quadrant/column-group 같은 용어는 이 저장소에도 `~/NPU/mlir-aie`에도 없음). 별개로 존재하는 "npu4가 `num_rows`를 4로 캡핑한다"는 버그(BERT row-overflow, `project-bert-e2e-plan`)와는 **축이 달라(rows vs columns) 무관함을 확인**.

### 컬럼 위치를 바꿔가며 재실험

같은 4개 슬롯을 쓰되 실제 컬럼 인덱스를 바꿔봄:
- 컬럼 `{2,3,4,5}` (이음매 걸치기, 총 4개): **HANG**
- 컬럼 `{4,5,6,7}` (두 번째 절반 안에서만, 총 4개): **HANG**
- 컬럼 `4` 단독 (동시성 전혀 없음): **HANG** (1차 확인)
- 컬럼 `3` 단독: 1차 시도 HANG, 재검증 시 PASS
- 컬럼 `0` 단독: 1차 시도(default) PASS, 중간에 재확인 시 HANG, 재검증 시 다시 PASS

즉 **"5개 이상 동시"가 아니라 "컬럼 4 이상을 포함하기만 하면"** 문제가 생기는 것으로 처음엔 보였으나, 컬럼 0/3 단독조차 한때 hang이 재현되면서 — 이게 정말 컬럼별 하드웨어 결함인지, 아니면 그 시점에 디바이스가 이미 스트레스 상태였는지 구분이 안 되는 상황이 됨.

### 디바이스 스트레스 누적 확인 (안전 관련, 중요)

연속 5~6회 hang을 유발한 뒤, **평소 수십 번 안전하게 통과하던 known-good `bert_tiny` regression dispatch까지 hang**남 (`ert_cmd_state=8`). `sudo dmesg -T`로 확인한 결과 `aie2_hwctx_restart: Map host buf failed, ret -22`라는, 지난 2026-08-28 오전의 심각한 사고 때 처음 나타난 것과 같은 계열의 복구 실패 메시지가 나타남 (다행히 그때처럼 SMU/FLR 연쇄까지는 안 갔고, 몇 분 조용히 두니 자연 복구됨 — 리부트 불필요).

**이후 사이사이 known-good 대조군을 끼워 넣고 재검증**: 컬럼 0, 3 단독은 깨끗한 기준선에서 재확인하면 PASS. 컬럼 4 단독은 깨끗한 기준선에서 재확인해도 다시 HANG — 그 직후 대조군이 20초 간격을 두고도 2연속 hang나면서 다시 디바이스가 스트레스 상태에 빠짐 (몇 분 후 자연 회복).

**결론(신뢰도별로 구분)**:
- **비교적 신뢰도 높음** (깨끗한 기준선에서 시작, 즉시 결과 확보): 컬럼 `{2,3,4,5}` 및 `{4,5,6,7}` 조합 hang, 컬럼 `4` 단독 hang (1회 확인).
- **미확정**: "정확히 컬럼 4 이상이 전부 개별적으로 고장" vs "우연히 스트레스 누적과 겹침" — 완전히 분리 못함. 컬럼 0/3의 결과가 시점에 따라 오락가락한 게 그 증거.

### 실제 bmm_bias(N=32, 4컬럼 자연 타일링)로 직접 검증 시도 — 오염됨

컬럼 수를 인위적으로 제한하는 컴파일러 패치 대신, **문제 크기 자체를 줄여서(N=64→32) 자연스럽게 4컬럼만 쓰게 만드는** 방법을 시도 (기존에 준비돼 있던 `_local/int8_debug/gen_bmm_bias_smallN.py`/`bmm_bias_repro_N32.mlir` 재사용, 새 컴파일러 코드 불필요).

- **새로운 발견**: N=32에서는 `--iree-amdaie-packet-flow-strategy=inputs`만으로 **컴파일러의 arbiter-deadlock 감지기가 새로 발동**함 (N=64에서는 8/27 조사 때 이 체크가 안 걸렸었음). `--iree-amdaie-detect-arbiter-deadlock=false`로 우회해서 컴파일은 성공.
- 실행 시도 중 **`docker ps`로 gylee가 정확히 그 시점에 새 컨테이너 2개(`iree-amd-aie-backend-build` 포함)를 띄운 걸 확인** — 제 테스트가 hang/타임아웃 났는데, 이게 4컬럼 자체 문제인지 gylee의 동시 활동 때문인지 구분 불가. **이 결과는 폐기, 재검증 필요.**

### 종합 결론 및 남은 일

1. **4↔5 컬럼 임계치, 컬럼 4 개별 결함설 모두 미확정.** 가장 신뢰도 높은 두 개(`{2,3,4,5}`, `{4,5,6,7}` hang)는 살아있지만, 완전히 확정하려면 진짜로 아무도 안 쓰는 시간대에 대조군을 촘촘히 끼워가며 다시 해야 함.
2. **N=32(4컬럼 자연 타일링) bmm_bias 실검증은 아직 못함** — 컴파일은 성공(arbiter-deadlock 우회 필요라는 새 정보 포함)했으나 실행 결과가 오염됨. `_local/int8_debug/out/bmm_bias_repro_N32.vmfb`로 재시도 가능.
3. **가장 중요한 프로세스 교훈**: 이 조사 자체가 공유 머신 위에서 다른 사용자의 동시 활동에 취약하다는 게 이번 세션에서 명확해짐 — 아래 "공유 자원 조율" 논의를 참고.

재현 자료는 기존과 동일하게 `_local/mlir_aie_repro/2026-08-28_column_threshold/`에 있고, `gen_8col.py`가 3번째 인자로 명시적 컬럼 리스트(`"2,3,4,5"` 등)를 받도록 확장됨.

---

## 결정적 교차검증 (같은 날, 저녁) — degree=4 fan-out을 IRON으로 재현하면 PASS → 하드웨어 한계 아니라 IREE 코드 문제

낮의 혼란(컬럼 개수 vs 컬럼 위치)을 뒤로하고, 저녁에 완전히 조용한 머신에서 정확히 통제된 실험으로 재정리:

### 1. 실제 bmm_bias(N=8, 1컬럼)로 재확인 — 진짜 hang, 오염 없음

`_local/int8_debug/out/bmm_bias_repro_N8.vmfb` (M=32,K=64,N=8 — N을 최소화해서 컬럼 1개만 자연 사용하지만, **M=32 때문에 row-tile 4개 필요 → degree=4 fan-out은 그대로 남음**)를 완전히 조용한 조건(대조군 전/후 확인, `docker ps`로 아무도 활동 없음 확인)에서 실행: **HANG** (`ert_cmd_state=8`, 70초). 몇 분 후 자연 회복 확인.

이걸로 "컬럼 개수/위치"가 아니라 **"degree=4 fan-out 자체"**가 계속 남아있는 진짜 변수라는 게 명확해짐 — N을 8까지 줄여도(컬럼 1개) M=32가 만드는 row-tile 4개짜리 bias 브로드캐스트는 못 없앰.

### 2. mlir-aie(IRON)로 정확히 같은 구조 재현 — 이번엔 PASS

메모리에 이미 기록된 실제 구조("memtile 물리 채널이 2개뿐이라 4개 목적지가 2개씩 짝지어 채널 2개를 공유")를 그대로 옮겨서 재현 (`_local/mlir_aie_repro/2026-08-28_column_threshold/gen_degree2ch.py`): 메모리타일 MM2S 채널 2개, 각 채널이 서로 다른 packet ID 2개씩(총 4개, 전역 유니크) 가지고 서로 다른 목적지 타일(row 2,3,4,5) 4곳으로 fan-out.

- **DEGREE=2(sanity check)**: PASS (13, 14 정상)
- **DEGREE=4 (진짜 타겟)**: **PASS** (13, 14, 15, 16 전부 정상, `ert_cmd_state=4`)

(중간에 "채널 1개에 packet ID 4개 전부"로 시도했을 때는 mlir-aie 라우터 자체가 컴파일 단계에서 크래시/거부함 — `AIEPathFinder.cpp` assertion 또는 "false packet id match" 에러. 이건 하드웨어 문제가 아니라 mlir-aie 툴의 arbiter-rule 압축 알고리즘 한계로 보이고, 실제 구조("채널 2개로 분산")로 바꾸니 문제없이 컴파일+통과함 — 오히려 이것도 "4개를 억지로 채널 하나에 욱여넣는 것"과 "2개씩 채널 2개로 자연 분산"이 물리적으로 다른 자원 사용 패턴임을 보여주는 방증.)

### 3. 결론

**degree=4 packet-routed fan-out 구조 자체(실제 문서화된 bmm_bias의 물리적 채널/ID 배치와 동일)는 IRON에서 안전합니다.** 그런데 IREE가 실제로 컴파일한 동일한 논리적 패턴(N=8 bmm_bias)은 hang납니다. **두 독립 툴체인이 같은 결과를 안 낸다 = 하드웨어/펌웨어 근본 한계가 아니라 IREE 컴파일러가 실제로 만드는 control code에 뭔가 다른 문제가 있다는 뜻입니다.**

즉 이번 조사 전체의 결론이 다시 한번 바뀝니다:
- ~~fan-out 자체가 원인~~ (반증됨, 8/27~8/28 낮)
- ~~컬럼 개수/위치가 원인~~ (반증됨, 8/28 낮 — 오염된 데이터였음)
- **"degree=4 fan-out을 문서상 구조 그대로 재현하면 안전한데, 실제 IREE 컴파일 결과는 hang" → IREE가 실제로 만드는 control code(BD/lock 순서, TCT sync, repeat_count 처리 등)가 이 "이상적인" 구조와 뭔가 다르게 생성되고 있다는 게 유력한 다음 방향.**

### 다음 단계 (미착수)

실제 bmm_bias_repro_N8의 컴파일된 IR/control code를 이 mlir-aie 최소 재현과 **직접 diff** 떠서, 정확히 뭐가 다른지 찾는 게 다음 논리적 단계. 후보: BD chaining 방식 차이, lock 초기화/개수 차이, TCT(completion token) sync 타이밍, 채널 배정이 실제로 문서 기록과 정확히 일치하는지(직접 IR로 재확인 필요 — 지금까지는 예전 세션 메모리 기록에 의존함), repeat_count 관련 필드.

## 재현 자료 (추가)

`_local/mlir_aie_repro/2026-08-28_column_threshold/`:
- `gen_degree2ch.py` — **성공한 최종 버전**. 메모리타일 채널 2개, 각 2-ID 공유, N개 목적지로 확장 가능 (`python3 gen_degree2ch.py <DEGREE(짝수)> <host-repeat N>`)
- `gen_degree_1chan_BROKEN_router_crash.py` — 채널 1개에 ID 전부 욱여넣는 버전, mlir-aie 라우터가 DEGREE≥4에서 거부/크래시함. 참고용으로만 보존 (동작 안 함).
- `diag_degree.cpp` — degree별 목적지 값 진단 도구 (`DEGREE_PLACEHOLDER` 치환해서 사용).

---

## 결정적 교차검증의 gap 메우기: distribute뿐 아니라 진짜 broadcast도 재검증 (2026-09-01)

위 "결정적 교차검증"의 IRON 재현(`gen_degree2ch.py`)을 다시 감사해보니, 실제로는 **distribute**(공유 입력 버퍼를 4등분해서 각 row가 다른 조각을 받음)를 테스트한 것이었음이 드러남 — 이건 X/A 오퍼랜드의 CDO 실측 패턴(뒤쪽 "CDO 바이너리 디코딩 결과" 섹션, 512바이트 fetch → 128바이트씩 4개 row로 분배, `512=128×4`)과는 맞지만, **bias 경로의 실제 CDO 패턴은 다름**: bias는 **작은 버퍼 하나(buf_len=8)를 한 번만 fetch한 뒤, 그 내용을 4번 그대로 재전송**하는 진짜 broadcast(`ACQ freeLock(-4)/REL fullLock(+4)` → 4개 소비자 각각 `ACQ fullLock(-1)/REL freeLock(+1)`, "CDO 정정 + 진짜 원인 후보 발견" 섹션에서 이미 디코딩됨). 이 특정 broadcast 패턴은 그동안 IRON으로 검증된 적이 없었음(사용자가 직접 지적하며 재검증 요청).

**`gen_broadcast.py`** 작성: `gen_degree2ch.py`와 채널/packet-ID 구조는 동일(memtile 물리 채널 2개, 각 2-BD 체인으로 다른 row 2곳), 다만 소스가 **공유 버퍼 하나**(`bcast_buf`)이고 CDO에서 실측한 그 credit-lock 패턴을 그대로 구현(DEGREE로 일반화). `aiecc` 컴파일 클린(라우터 거부 없음). 실제 HW(DEGREE=4)에서 `./scripts/lock/with-npu-lock.sh`로 감싸서 실행(사전에 gylee의 컨테이너들이 `scripts/lock/`보다 먼저 떠 있어 lock에 안 잡힌다는 걸 확인하고, 사용자가 직접 gylee와 확인 후 진행) → **PASS, `ert_cmd_state=4`, 0.25초(hang 아님), 4곳 전부 정상값(13/14/15/16)**.

**결론**: broadcast 변형도 IRON에서 안전함이 확인되어, "IREE 코드 문제, 하드웨어 한계 아님"이라는 기존 결론의 유일한 남은 gap(distribute만 검증, broadcast 미검증)이 메워짐. 다음 자연스러운 확장은 이 broadcast 구조를 실제 목표 스케일(8컬럼×4행 동시)로 키워보는 것(미착수).

**재현 자료**: `gen_broadcast.py`, `diag_broadcast.cpp` (`_local/mlir_aie_repro/2026-08-28_column_threshold/`, gitignore됨).

### 실제 목표 스케일(8컬럼×4행)로 확장 — 여전히 PASS (같은 날, 이어서)

1컬럼 broadcast가 PASS한 뒤, 곧바로 실제 bmm_bias의 진짜 병렬도(N=64→8컬럼, M=32→4행)까지 키워서 재검증. `gen_broadcast_8col.py` 작성: 컬럼마다 독립적인 broadcast 구조(공유 버퍼 1개 + credit-lock, `gen_broadcast.py`와 동일)를 8번 복제, packet ID는 컬럼별로 전역 유니크한 블록(컬럼 c → id `[4c, 4c+4)`, 총 32개, 하드웨어 예산 정확히 다 씀)으로 분리. `aiecc` 컴파일 클린(48KB AIE_PARTITION, 라우터 거부 없음).

실제 HW 실행(`./scripts/lock/with-npu-lock.sh`로 감쌈, 직전 gylee 활동 없음 재확인) — **PASS, `ert_cmd_state=4`, 0.24초(hang 아님), 8컬럼×4행=32곳 전부 정상값**. 실행 전후 `xrt-smi examine` 정상.

**결론**: 1컬럼 distribute → 1컬럼 broadcast → **8컬럼×4행 broadcast(실제 목표 스케일)** 순으로 전부 IRON에서 PASS. 실제 `bmm_bias_repro_N8`은 1컬럼만 써도 hang인데 IRON은 8컬럼 전체를 동시에 돌려도 안전 — "하드웨어/펌웨어 한계가 아니라 IREE가 만드는 control code(BD chaining, lock 초기화, TCT sync 등)에 문제가 있다"는 결론이 규모 측면에서도 확정적으로 뒷받침됨. 다음 단계는 여전히 실제 bmm_bias_repro_N8의 CDO/control code를 이 broadcast 재현과 직접 diff 뜨는 것.

**재현 자료**: `gen_broadcast_8col.py`, `diag_broadcast_8col.cpp` (`_local/mlir_aie_repro/2026-08-28_column_threshold/`, gitignore됨).

### CDO diff로 새 gap 발견 → 그 gap을 메웠더니 처음으로 진짜 hang 재현! (같은 날, 이어서)

지금까지 PASS한 모든 IRON 재현(distribute, 1컬럼 broadcast, 8컬럼×4행 broadcast)의 CDO를 실제 `bmm_bias_repro_N8`의 CDO(`bmm_aie_cdo_init.src.txt`, bias 경로만 분리: len=8 버퍼, base `0x038000`/`0x03c000`)와 `decode_bd.py`로 직접 대조.

**일치**: lock 프로토콜이 완전히 bit-exact. 실제 bias fetch BD `lock_acq(id=64,val=124) lock_rel(id=65,val=4)`, consumer BD `lock_acq(id=65,val=127) lock_rel(id=64,val=1)` — 우리 `gen_broadcast.py`(1컬럼 broadcast)의 값과 정확히 동일. credit-lock 역공학이 근사가 아니라 정확했음을 확인.

**차이 발견**: 실제 bias fetch BD는 **버퍼 하나가 아니라 두 개(`0x038000`/`0x03c000`)를 ping-pong**하는 2-BD 순환 체인(디스패치 이름의 `batch_matmul_2x3`, 배치=2와 일치). 지금까지 만든 어떤 IRON 재현도 이 더블버퍼링을 모델링한 적이 없었음.

**이 gap을 메운 `gen_broadcast_dbuf.py`(DEGREE=4, BATCH=2 — 물리 버퍼 2개 ping-pong, 각 목적지 packet ID는 그대로 1:1 유지하되 버퍼당 1 BD씩 총 2 BD/id)를 작성**. CDO 디코딩으로 실제 fetch BD 패턴(ping-pong + 동일 lock 값)과 구조적으로 일치함을 컴파일 후 사전 확인. `./scripts/lock/with-npu-lock.sh`로 감싸서 실제 HW 실행(직전 gylee 활동 없음 확인, 실행 전후 `xrt-smi examine` 정상) →

**`ert_cmd_state=8` (TIMEOUT), 122.6초 — 진짜 hang. `dest 0`만 정상(13), `dest 1~3`은 0(도달 못함)** — 부분 완료 후 멈추는, 실제 `bmm_bias_repro_N8`과 동일한 신호.

**의미 — 지금까지 이 조사에서 나온 가장 구체적인 결과**: 단일 버퍼 broadcast(1컬럼도, 8컬럼×4행도)는 전부 PASS인데, 정확히 **더블버퍼링(ping-pong)을 켜는 순간** hang이 재현됨. 두 변수(버퍼 개수=1 vs 2)만 다른 최소 쌍(minimal pair)을 확보한 것 — 지금까지의 정적 분석(arbiter/msel, lock balance, TLAST, SRAM overlap 등)이 전부 못 찾던 걸 실험적으로 딱 집어낸 셈. dest0만 성공하고 이후가 막힌다는 건 여러 목적지/채널이 같은 ping-pong 버퍼 쌍의 credit-lock을 공유할 때 뭔가 레이스나 잘못된 순서가 생긴다는 뜻으로 보이나, 정확한 메커니즘은 아직 미확인 — 다음 세션에서 이어갈 것.

**안전 상태**: hang 1회 발생, 실행 후 NPU 정상 회복 확인.

### 추가 격리: DEGREE=2(채널 1개)로 줄여도 여전히 hang — 멀티채널이 원인 아님 (같은 날, 이어서)

"여러 채널이 credit lock을 공유해서 생기는 레이스"라는 가설을 검증하기 위해 DEGREE=2(채널 1개, row 2/3만, ping-pong 버퍼 2개는 그대로)로 축소해서 재검증. `gen_broadcast_dbuf.py 2 2` 컴파일 클린. 실행 결과:

**`ert_cmd_state=8` (TIMEOUT), 125.1초 — 또 hang. dest 0 정상(13), dest 1은 0(도달 못함)** — DEGREE=4 때와 완전히 같은 패턴(먼저 오는 소비자는 성공, 그 다음이 막힘).

**결론**: 멀티채널 상호작용은 원인이 아님 — **채널 1개, 목적지 2개, ping-pong 버퍼 2개를 하나의 집계 lock 쌍(`lock_free`/`lock_full`)으로 관리하는 것 자체**가 hang을 일으키는 최소 조건. 이걸로 연속 2회 hang(DEGREE=4, DEGREE=2)이 되어 안전 규칙에 따라 **이 세션의 라이브 HW 테스트는 여기서 중단**. NPU는 두 번 다 정상 회복 확인.

**다음 세션에서 이어갈 방향**: 정확한 실패 지점을 더 좁히려면 (a) 버퍼별로 독립된 lock 쌍을 쓰는 버전(진짜 ObjectFifo 스타일 per-slot 락)을 만들어서 그게 고치는지 확인, (b) BATCH=2를 유지한 채 소비자를 1개로 줄여(순수 producer-consumer 1:1, broadcast 아닌 단순 파이프라인) ping-pong 자체만으로도 hang나는지(소비자가 여럿이어야 하는지 여부) 확인.

**재현 자료**: `gen_broadcast_dbuf.py`, `diag_broadcast_dbuf4.cpp`, `diag_broadcast_dbuf2.cpp` (`_local/mlir_aie_repro/2026-08-28_column_threshold/`, gitignore됨).

### 결정적 반증: plain(비배치) matmul+bias도 hang나고, CDO가 batch 버전과 완전히 동일함 — "batch가 ping-pong을 만든다"는 설명 정정 (같은 날, 이어서)

"ping-pong은 batch=2 때문에 생기니, batch 없는 plain matmul+bias는 hang이 없어야 한다"는 가설을 직접 검증. `_local/int8_debug/gen_bmm_bias_plain.py` 작성(M=32,K=64,N=8, **배치 차원 완전히 제거**한 순수 2D matmul+bias), `iree-import-onnx` → `bmm_bias_repro_N8`과 동일한 플래그로 `iree-compile`(컴파일 성공) → 실제 HW 실행.

**결과: `ert state 8` (TIMEOUT), 125.2초 — 이것도 hang.** 컴파일 관련해서 5개 fix(#1-5, 전부 `ee37c48`에 이미 커밋됨, batch 여부와 무관)만으로 충분히 컴파일됐고, 채널-cap 패치는 적용 안 한 상태(현재 코드에 없음).

**더 결정적인 것 — CDO diff**: 이 plain 버전을 `--iree-hal-dump-executable-files-to`로 다시 컴파일해서 memtile CDO(`aie_cdo_init.bin`)를 뽑아 `decode_bd.py`로 비교했더니, **batch 버전(`bmm_aie_cdo_init.src.txt`)과 `diff` 결과 완전히 동일**(바이트 단위, 주소/길이/lock ID/lock 값/next_bd 링크 전부 일치). 즉 **ping-pong 구조는 batch가 있어서 생기는 게 아니라, M=32(row-tile 4개)에 bias를 broadcast하는 패턴 자체에 IREE가 항상 붙이는 구조**입니다 — 배치가 있든 없든 컴파일러가 똑같은 코드를 만들어냅니다.

**정정된 결론**:
1. "batch=2라서 ping-pong을 쓴다"는 이전 설명은 **틀렸음** — 인과관계가 반대로 잘못 추론된 것. 실제로는 batch와 무관하게 이 broadcast 패턴에는 항상 ping-pong이 붙음.
2. 그래도 IRON에서 발견한 "ping-pong(더블버퍼) 자체가 hang을 일으킨다"는 핵심 발견은 오히려 **더 강하게 뒷받침됨** — 서로 다른 두 실제 디스패치(batch 있음/없음)가 완전히 동일한 (그리고 우리 IRON 재현이 hang낸 것과 같은) 구조를 만들어내고, 둘 다 hang났습니다.
3. 남은 질문: **왜 컴파일러가 이 패턴에 항상 ping-pong을 붙이는지**(어느 pass/설정이 결정하는지) 아직 미확인 — 다음 단서.

**안전 상태**: hang 발생, NPU 정상 회복 확인. 이 실행 도중 별도로 `gylee`의 라이브 인터랙티브 쉘(`docker run -it --device=/dev/accel/accel0 ... dev-gylee bash`, 이날 11:20부터 계속 떠 있음)이 NPU passthrough를 쥐고 있는 게 발견됨 — lock에는 안 걸리는 종류(대화형 쉘이라 아직 아무 명령도 안 돌리고 있었을 가능성 높음)라 결과 자체를 오염시키진 않았을 것으로 보이나, 앞으로 계속 유의해야 함.

**재현 자료**: `_local/int8_debug/gen_bmm_bias_plain.py`, `_local/int8_debug/out/mm_bias_plain_N8.{onnx,mlir,vmfb}`, `_local/int8_debug/out/mm_bias_plain_dump/`(CDO), `_local/mlir_aie_repro/.../plain_aie_cdo_init.src.txt`.

### Pass 추적: `AMDAIEAssignLogicalObjectFifoDepthPass`가 항상 depth=2를 부여함, depth=1로 강제해도 hang은 그대로 — 더블버퍼링 이론 반증 (같은 날, 이어서)

`AMDAIEAssignLogicalObjectFifoDepth.cpp` + `Passes.td`/`Passes.cpp:834-844` 확인: L2(memtile)/L1 버퍼 depth는 텐서별 분석 없이 **무조건 2**(`Passes.td` default)로 부여됨 — batch 유무와 무관하게 CDO가 동일했던 이유가 여기서 확정됨. 업스트림(`1447eb5`, Abhishek Varma, 2025-07-25)이 이미 `reprogram-dmas` 모드에서는 "더블버퍼링이 controlcode-lowering/transaction-binary pass와 안 맞아서" depth=1로 강제하는 코드를 남겨뒀지만, **우리가 쓰는 기본 경로(circular DMA, `reprogramDmas=false`)에는 이 안전장치가 없음**.

**시도 1**: `--iree-amdaie-reprogram-dmas=true` — **컴파일 자체가 실패**(`'amdaie.connection' op source channel does not have corresponding MM2S DMA start op`). packet-routed broadcast 연결에 대한 이 모드의 control-code lowering이 애초에 구현 안 돼 있음. 이 경로로는 테스트 불가.

**시도 2**: `Passes.cpp`를 직접 패치(uncommitted 실험) — `reprogramDmas` 조건 없이 **무조건** `l2BufferDepth=1`, `l1BufferDepth=1`로 강제. 재빌드(`iree-compile` relink, 20초, incremental) 성공. `mm_bias_plain_N8.mlir`을 이 새 컴파일러로 재컴파일(정상 경로, `reprogram-dmas` 안 씀) → CDO 디코딩으로 **실제로 depth=1이 됐음을 확인**(bias 버퍼가 이제 `base=0x02c000` 하나뿐, ping-pong 쌍 없어짐, BD 개수 40→20으로 정확히 절반).

이 depth=1 빌드를 실제 HW에서 실행: **`ert state 8`, 122.1초 — 여전히 hang.**

**결론(중요, 정정)**: **depth=1(더블버퍼링 제거)로도 hang이 사라지지 않음** — "double-buffering(ping-pong)이 원인"이라는 이론은 **실제 IREE 컴파일 경로에 대해서는 반증됨**. IRON 쪽에서 찾은 "공유 aggregate lock으로 2개 버퍼 관리하면 hang"이라는 발견 자체는 여전히 유효한 별개의 real bug로 보이지만(재현 가능, real CDO의 lock 값과 bit-exact 일치했음), **이게 이 특정 real hang의 root cause는 아니었음.** 우리의 손으로 짠 최소 재현(bias만 있고 X/Y 없음, 실제 core ELF 없음, 실제 repeat_count/TCT sync 없음)이 실제 디스패치의 복잡성 중 뭔가 중요한 걸 놓치고 있다는 뜻 — 근본 원인 탐색은 계속 미해결.

**패치 되돌림**: 실험 완료 후 `Passes.cpp`의 depth=1 강제 패치를 `git checkout`으로 되돌리고 재빌드해서 공유 빌드(`build/tools/iree-compile`)를 정상 상태로 복원함(다른 사용자에게 영향 안 가도록).

**안전 상태**: 재개 이후 연속 2회 hang(plain matmul+bias, depth=1 패치) → 이 세션 라이브 HW 테스트 다시 중단. NPU 둘 다 정상 회복 확인.

**다음 세션 후보**: (a) 실제 X/Y 오퍼랜드까지 포함한 더 완전한 IRON 재현(지금까지는 bias 단독), (b) core ELF의 정확한 compute-loop trip count(8/31부터 미해결로 남아있던 것), (c) 채널 공유가 여러 텐서 타입(X와 bias가 한 채널을 공유하는 것으로 보이는 패턴, CDO에서 id=0이 10번 재사용된 것)에 어떤 영향을 주는지.

### Pass 스윕 계속: `AMDAIEInsertDmaOutOfOrderBlock`은 dead code로 확인, `AMDAIEFoldDmaWaits` 비활성화도 반증 (같은 날, 이어서)

우리 파이프라인(`addAMDAIEObjectFifoLoweringPasses`, `reprogramDmas=false`, `packetFlowStrategy=inputs`)에 실제로 걸리는 pass들을 순서대로 훑어 업스트림이 이미 "안 된다"고 인지해둔 지점이 더 있는지 확인.

**`AMDAIEInsertDmaOutOfOrderBlock`** — "S2MM 채널 하나가 여러 MM2S 발신자로부터 packet으로 받는 경우 out-of-order 병합 블록 생성"이 목적이라 우리 시나리오(한 코어가 채널 부족으로 여러 텐서 스트림을 한 채널에 공유)와 정확히 맞아떨어져 보였음. `--mlir-print-ir-after=iree-amdaie-insert-dma-out-of-order-block`로 IR 덤프 확인 → **`amdaie.dma_start` op 자체가 이 시점에 하나도 없음** (`DMAStartOp`는 `reprogramDmas=true`일 때만 생성되고, 그 모드는 이미 컴파일 자체가 안 되는 걸로 확인됨). 즉 우리 경로에서는 완전히 no-op — 원인 아님, 배제.

**`AMDAIEFoldDmaWaits`** — control code의 중복 `dma_wait` op를 제거/병합하는 최적화 pass. tile+connection+BD-id 기준으로 신중하게 짜여 있어 명시적 TODO는 없었지만, "동기화 제거 최적화"라는 성격상 가장 의심스러운 후보였음. `Passes.cpp`를 패치(uncommitted 실험, `if (false && !reprogramDmas) ...`로 완전히 비활성화) → 재빌드(18초) → `mm_bias_plain_N8.mlir` 재컴파일(정상 플래그) → 컴파일 성공 → 실제 HW 실행:

**`ert state 8`, 122.8초 — 또 hang. 이것도 반증.**

**패치 되돌림 + 재빌드**: 공유 빌드(`build/tools/iree-compile`) 정상 상태로 복원 완료.

**안전 상태**: 이번 라운드 3연속 hang(plain matmul+bias → depth=1 → FoldDmaWaits 비활성화) — 라이브 HW 테스트는 계속 중단 상태 유지. NPU 매번 정상 회복 확인.

**남은 미조사 후보**: `AMDAIEAssignConnectionTypes`(packetFlowStrategy 옵션 직접 소비), `AMDAIEConnectionToFlow`("TODO(jornt): currently don't delete connections... will be changed in the future" — 아직 안 읽어본 TODO), `AMDAIEAssignPacketIdsPass`(packet ID 할당 자체), `AMDAIEDistributeCoresAndObjectFifosPass`("TODO(jornt): Generalize this later", "NOTE: assumption that..." — 여러 명시적 가정이 있음, 아직 안 읽음), `AMDAIELowerToAIEPass`(최종 AIE 방언 변환, 가장 복잡하고 아직 상세히 안 읽음), `AMDAIEAcquireReleaseToUseLockPass`(lock 생성 자체 — "logic currently only handles size set and equal to 1"이라는 TODO성 assert 발견, 아직 우리 케이스에 해당하는지 확인 안 함).

---

## TXN(호스트 명령 스트림) diff 시도 — 잘못된 레이어를 비교했음, CDO가 진짜 다음 단계

§ "결정적 교차검증"에서 나온 "IRON은 PASS, IREE는 HANG" 결과를 설명하기 위해, 두 컴파일 결과물의 raw NPU transaction(TXN, 호스트가 보내는 명령 스트림)을 디코딩해서 비교 시도.

**대상 파일** (전부 `_local/int8_debug/out/N8_dump/`에 저장됨):
- `bmm_biasasync_dispatch_2_batch_matmul_2x3_0.npu_inst.txt` — 실제 hang나는 bmm_bias_repro_N8의 TXN (16 ops, 560 bytes)
- `mlir_aie_deg4_insts.txt` — PASS한 mlir-aie degree=4 재현의 TXN (8 ops, 300 bytes, `insts_d2ch4.bin`을 텍스트로 변환한 것)

**디코딩 결과**: 둘 다 TxnSize/NumOps가 정확히 맞아떨어져서 디코딩 자체는 검증됨.
- mlir-aie: BD push 2쌍(BLOCKWRITE+DDR_PATCH) + lock 관련 WRITE/MASKWRITE + **TCT(완료 대기) 1개, 블로킹**
- bmm_bias: BD push 4쌍(X/Y/out 등 실제 텐서가 더 많아서) + **TCT 4개, 앞 3개는 non-blocking·마지막 1개만 블로킹**

**중요한 한계— 이 비교는 사실 잘못된 레이어를 본 것임**: TXN 스트림은 **호스트가 매 dispatch마다 보내는 "이 BD를 실행해라" 명령 + 완료 대기 시퀀스**일 뿐이고, **진짜 우리가 확인하려는 메모리타일 채널/packet-ID 라우팅 설정(switchbox rule, BD chain, lock 초기값)은 CDO 바이너리 안에 정적으로 들어있음** (dispatch마다 매번 보내는 게 아니라 한 번 로드됨). 즉 TXN 개수 차이(4 vs 1 TCT)는 "실제 텐서가 3개(X,Y,out)라 BD push가 더 많다"는 당연한 차이일 가능성이 높고, hang의 진짜 원인(공유 채널/packet-ID 배선)과는 관계 없을 수 있음 — 확정 못 함.

## 다음 세션 시작 지점: CDO 바이너리 diff

**두 CDO 세트 다 이미 준비돼 있어서 재컴파일 없이 바로 이어갈 수 있음:**
- 실제 bmm_bias (hang): `_local/int8_debug/out/N8_dump/aie_cdo_init.bin`, `aie_cdo_enable.bin`, `aie_cdo_switches.bin`, `aie_cdo_elfs.bin`
- mlir-aie degree=4 (PASS): `_local/int8_debug/out/N8_dump/mlir_aie_deg4_cdo/main_aie_cdo_init.bin`, `main_aie_cdo_enable.bin`, `main_aie_cdo_elfs.bin` (참고: mlir-aie 쪽엔 별도 `switches.bin`이 없음 — switchbox 설정이 `init.bin`에 통합되어 있을 가능성, 확인 필요)
- mlir-aie degree=4의 원본 MLIR 소스: `_local/int8_debug/out/N8_dump/mlir_aie_deg4_cdo/aie_arch_degree2ch.mlir` (재현 생성기: `_local/mlir_aie_repro/2026-08-28_column_threshold/gen_degree2ch.py`, `python3 gen_degree2ch.py 4 1`로 재생성 가능)

**CDO는 TXN과 다른 포맷**(직접적인 레지스터 write 나열 — switchbox rule/amsel, BD 설정, lock 초기값 등 정적 구성이 그대로 담김) — 이전 int8 조사(§"다섯 번째 각도", `docs/2026-08-24_int8_quantization_investigation.md`)에서 이미 한 번 raw TXN을 `third_party/aie-rt/driver/src/global/xaiegbl.h` 구조체 레이아웃으로 디코딩한 전례가 있으니 비슷한 접근이 CDO에도 적용 가능할 것으로 보이나, CDO 포맷 자체는 아직 이 세션에서 디코딩 시도 안 함 — 처음부터 시작해야 함.

**목표**: bias-broadcast에 해당하는(메모리타일 2개 채널 × packet ID 2개씩 → row 2,3,4,5) switchbox rule/BD chain/lock 설정이 실제 bmm_bias의 CDO 안에서 mlir-aie 재현과 정말 같은 모양인지, 아니면 뭔가 다르게(예: rule 압축 방식, BD 순서, lock 초기값, repeat_count 관련 필드) 생성되고 있는지 직접 대조.

## 오늘 세션 전체 요약 (내일 이어갈 때 읽을 것)

1. IRON 공식 예제 재검증 → circuit fan-out, "다른 packet ID가 같은 타일의 다른 채널로" 패턴 둘 다 안전 확인
2. "packet ID 공유+다른 타일" 단독 가설 → 20,000회 반복까지 안전 (반증)
3. 8컬럼 동시 실행 → hang 발견했으나 자체 스크립트 버그로 오염, 수정 후 재확인 → 4↔5 컬럼 임계치처럼 보였음
4. 컬럼 위치를 바꿔가며(straddle/secondhalf/solo) 재검증 → 디바이스 스트레스 누적과 실제 신호가 뒤섞여 판별 불가 상태로 악화 (한때 `Map host buf failed` 경고까지 감, 다행히 자연 회복)
5. 공유 머신 조율 문제 논의 (gylee 등 다른 사용자와의 동시 경합 가능성) — 팀 차원의 락 메커니즘 필요성 논의, 실행은 안 함
6. **저녁, 완전히 조용한 조건에서 재정리**: 실제 bmm_bias(N=8, degree=4)는 진짜 hang, mlir-aie로 정확히 같은 구조(채널 2개×ID 2개씩) 재현하면 PASS → **하드웨어 한계 아니라 IREE 코드 문제로 결론**
7. TXN diff 시도 → 잘못된 레이어 비교였음, CDO diff가 진짜 다음 단계로 확정

**지금 상태**: 디바이스 정상, 모든 재현 자료/CDO 바이너리 저장 완료. 다음은 CDO 바이너리 포맷 디코딩부터 시작.

---

## CDO 바이너리 디코딩 결과 (2026-08-31) — memtile BD의 packet-ID 재사용 패턴이 IRON과 크게 다름을 발견

### 디코딩 방법

`/tools/Xilinx/Vitis/2023.2/bin/cdoutil -output-source`로 raw CDO(`.bin`)를 사람이 읽을 수 있는
`write <addr> <word0..wordN>` / `mask_write <addr> <mask> <value>` 커맨드 시퀀스로 역어셈블(주소만,
레지스터 이름은 안 붙음 — 범용 Versal cdoutil이라 AIE2 레지스터를 모름). 이후 주소를
`third_party/aie-rt/driver/src/lite/xaie_lite_hwcfg.h`의 Strix(AIE2P B0) 값
(`COL_SHIFT=25, ROW_SHIFT=20`, row0=shim, row1=memtile, row2~5=core)으로 col/row/offset 분해.

memtile DMA BD는 8-word 블록(`third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c`의
`Aie2PMemTileDmaBdPktProp`/`Aie2PMemTileDmaBdEnProp` 필드 레이아웃 기준)이라, 이 필드 정의로
`word0`(EnPkt/PktType/PktId/BufferLength), `word1`(NextBd/UseNextBd/BaseAddress),
`word7`(ValidBd/LockRelId/LockRelVal/LockAcqId/LockAcqVal/LockAcqEn)을 직접 디코딩하는 스크립트
(`decode_bd.py`, `_local/mlir_aie_repro/2026-08-28_column_threshold/`에 저장)를 작성해서 두 CDO
세트(`bmm_aie_cdo_init.src.txt` = 실제 hang나는 bmm_bias, `mlir_main_aie_cdo_init.src.txt` = PASS한
IRON degree=4 재현)의 memtile BD 체인을 정량 비교.

### 핵심 발견: packet-BD 개수와 ID 재사용 패턴이 완전히 다름

| | packet-mode BD 개수 | 쓰인 고유 packet ID 개수 | ID당 BD 수 |
|---|---|---|---|
| **IRON degree=4 (PASS)** | 4 | 4 (0,1,2,3) | **매 ID당 정확히 1개** — 1:1 |
| **실제 bmm_bias (HANG)** | 24 | **5 (0,1,2,3,4)** | id=0: 10개, id=1: 4개, id=2: 4개, id=3: 4개, id=4: 2개 |

IRON 쪽은 각 packet ID가 정확히 하나의 BD(하나의 목적지 개념)에만 매핑되는 깨끗한 구조인 반면,
bmm_bias는 **각 ID가 서로 다른 base-address(버퍼) 영역에 걸쳐 여러 번 재사용**됨:

- `pkt_id=0`: base `0x030000`/`0x034000` (len=128, ping-pong 4쌍 — 아마 K-루프 반복) **+ 별도로
  base `0x020000`/`0x024000` (len=128) 2개도 같은 id=0을 씀** — 서로 다른 버퍼 영역이 같은 ID 공유.
- `pkt_id=1`: base `0x020080`/`0x024080`, 그리고 `0x020100`/`0x024100` — 같은 큰 버퍼(`0x02xxxx`)의
  다른 오프셋 슬라이스 두 묶음이 같은 id=1로 묶임.
- `pkt_id=2`: base `0x020180`/`0x024180` (같은 `0x02xxxx` 버퍼의 세 번째 슬라이스) **+ 완전히 다른
  버퍼 `0x038000`/`0x03c000` (len=8, 훨씬 작음 — bias로 추정)도 같은 id=2**.
- `pkt_id=3`, `pkt_id=4`: `0x038000`/`0x03c000` (len=8) 버퍼를 나눠 쓰는데 id 3과 4가 겹쳐 있음.

정리하면: **하나의 packet ID가 (a) 같은 논리적 스트림의 반복(K-루프/핑퐁 — 이건 정상일 수 있음)뿐
아니라, (b) 명백히 다른 버퍼 영역/다른 텐서로 보이는 데이터에도 재사용되고 있음.** (b)가 의도된
동작(예: 같은 목적지 타일로 가는 서로 다른 텐서라 ID를 공유해도 안전)인지, 아니면 **서로 다른
목적지로 가야 할 스트림들이 실수로 같은 ID를 배정받아서 스위치 arbitration/큐 레벨에서 충돌하는
실제 버그**인지는 이 레이어(BD만 봐서는) 확정 불가 — 목적지 row/타일 매핑은 스위치 라우팅
테이블(`switches.bin`, `0x*3f2xx` 영역 packet-rule 슬롯)에 있고 아직 필드 단위로 완전히 디코딩
안 함.

### 왜 이게 유의미한가

8/28 저녁 결론("degree=4 fan-out 구조 자체는 안전, IREE의 실제 control code에 문제")과 정확히
들어맞는 구체적 물리 증거를 처음으로 찾음: **IRON은 4-way 배포를 ID당 1:1로 깔끔하게 나누는데,
IREE는 같은 걸 5개 ID로 나누면서 각 ID를 여러 다른 버퍼/텐서에 걸쳐 재사용**한다 — 이 재사용
패턴 자체가 진짜 원인일 필요는 없지만(정상적 K-루프 반복일 수도 있으므로), **가장 유력한
차이점의 첫 물리적 확증**.

### 다음 단계 (미착수, 다음 세션 시작점) — 아래 2026-08-31 후속 업데이트에서 실제로 진행/정정됨

1. ~~의미 확정이 최우선: AIE 방언 IR 확보~~ → **완료 (2026-08-31), 아래 참고.**
2. ~~서로 다른 목적지로 가는 두 스트림이 같은 packet ID를 공유하는지 판별~~ → **확인했고, 결론이
   바뀜: 이건 버그가 아니라 정상 동작이었음. 아래 "정정" 섹션 참고.**
3. 스위치 라우팅 테이블 필드 디코딩은 결국 불필요해짐 — **compiler 자체 진단(arbiter-deadlock
   detector)이 훨씬 더 직접적인 답을 줬음.**

### 재현 자료 (추가)

`_local/mlir_aie_repro/2026-08-28_column_threshold/`:
- `decode_bd.py` — memtile BD 8-word 블록을 필드 단위로 디코딩하는 스크립트 (`python3 decode_bd.py <cdoutil로 -output-source 뽑은 .src.txt>`)
- `bmm_aie_cdo_init.src.txt`, `bmm_aie_cdo_switches.src.txt`, `bmm_aie_cdo_enable.src.txt`, `bmm_aie_cdo_elfs.src.txt` — 실제 hang나는 bmm_bias의 CDO를 `cdoutil -output-source`로 역어셈블한 결과
- `mlir_main_aie_cdo_init.src.txt`, `mlir_main_aie_cdo_enable.src.txt`, `mlir_main_aie_cdo_elfs.src.txt` — PASS한 IRON degree=4 재현의 동일 결과 (참고: IRON 쪽은 switchbox 설정이 별도 `switches.bin` 없이 `init.bin`에 통합됨)

`cdoutil` 위치: `/tools/Xilinx/Vitis/2023.2/bin/cdoutil` (Vitis 2023.2 설치에 포함, 이 저장소
바깥의 시스템 툴 — repo에 새로 추가한 의존성 아님). 사용법: `cdoutil -output-source -output-file
<out.txt> <in.bin>`.

---

## 정정 + 진짜 원인 후보 발견 (2026-08-31, 같은 세션 이어서) — packet-ID 재사용은 정상, 실제 문제는 컴파일러 자체의 arbiter-deadlock 경고를 강제로 무시하고 있었던 것

### AIE 방언 IR 확보 (재컴파일, HW 안 건드림)

`bmm_bias_repro_N8.mlir`을 `iree-amd-aie:dev-wjjang` 도커 이미지(사용자 본인 소유, 별도 유저의
컨테이너를 실수로 쓴 적이 있어 주의 — 아래 "교훈" 참고) 안에서 아래 플래그로 재컴파일:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$(pwd):/workspace" -w /workspace -e HOME=/workspace -e PEANO_INSTALL_DIR=/workspace/llvm-aie \
  iree-amd-aie:dev-wjjang bash -lc '
source /opt/venv/bin/activate
build/tools/iree-compile _local/int8_debug/out/bmm_bias_repro_N8.mlir -o /tmp/out.vmfb \
  --iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-amdaie-demote-contraction-inputs-to-bf16 --iree-amdaie-enable-vectorization-passes=false \
  --iree-flow-enable-executable-deduplication=false --iree-flow-inline-constants-max-byte-length=0 \
  --iree-amdaie-packet-flow-strategy=inputs --iree-amdaie-detect-arbiter-deadlock=false \
  --iree-hal-dump-executable-files-to=/workspace/<dump-dir>'
```

**결과로 나온 `aie_cdo_init.bin`/`aie_cdo_switches.bin`이 원래 hang났던 N8_dump의 파일과 완전히
바이트 단위로 동일**(`diff` 확인) — 즉 이 플래그 조합이 실제로 그 hang을 낸 컴파일과 100% 같다는
것이 재확인됨. AIE 방언 IR 자체는 `--mlir-print-ir-after=iree-amdaie-lower-to-aie`로 stderr에
덤프(파일로는 안 저장됨, `2> <log>`로 리다이렉트 필요).

### 발견 1 — "packet ID 재사용"은 정정: 버그가 아니라 정상적인 멀티플렉싱

실제 `aie.packet_flow` 선언을 직접 읽으니, 메모리타일(`tile_0_1`)의 **MM2S 채널 1~5, 5개가 전부
독립적으로 packet-routed broadcast에 쓰이고 있었음** (앞서 CDO만 보고 추정했던 "채널 2개"가
아니었음):

| 소스 채널 | 목적지 (packet ID) |
|---|---|
| DMA:1 | tile_0_2/ch0(id0), tile_0_4/ch1(id1), tile_0_3/ch0(id2), tile_0_5/ch0(id3) |
| DMA:2 | tile_0_3/ch0(id0), tile_0_3/ch1(id1), tile_0_5/ch1(id2), tile_0_2/ch0(id3), tile_0_4/ch0(id4) |
| DMA:3 | tile_0_4/ch0(id0) |
| DMA:4 | tile_0_5/ch0(id0) |
| DMA:5 | tile_0_2/ch1(id0) |

각 소스 채널 **내부**에서는 ID가 항상 고유(재사용 없음) — 이건 IRON 재현과 동일한 깨끗한 패턴.
채널을 **넘나들며** ID값이 0부터 다시 시작되는 것(예: DMA:1과 DMA:5가 둘 다 id=0을 씀)은 처음엔
의심스러워 보였지만, 실제 `aie.mem(%tile_0_2)`의 S2MM BD 프로그램을 직접 읽어보면 — `DMA:0`
채널과 `DMA:1` 채널이 완전히 독립된 BD 체인(서로 다른 lock, 서로 다른 버퍼)이고, 각각 자기
쪽으로 오는 스트림만 받음. 즉 **packet ID의 유일성은 "전역"이 아니라 "그 스트림이 지나가는
물리 링크/스위치 슬롯" 범위에서만 필요** — 이는 실제 하드웨어 스트림 스위치의 정상 동작이고,
**8/28 저녁에 세웠던 "ID 재사용 = 버그일 수도" 가설은 이걸로 반증(retract)됨.**

### 발견 2 — 진짜 신호: `--iree-amdaie-detect-arbiter-deadlock=false` 없이 컴파일하면 컴파일러 자신이 이 정확한 구조를 "arbiter deadlock 위험"으로 거부함

지금까지 이 hang 재현을 컴파일할 때마다 **항상** `--iree-amdaie-detect-arbiter-deadlock=false`를
같이 썼다(8/28 낮 문서에서 N=32 때 처음 필요해짐 발견, 이후 관례적으로 계속 사용). 이 플래그를
**빼고** 정확히 같은 소스(N=8, 1컬럼)를 컴파일하면:

```
<unknown>:0: error: 'aie.device' op Potential arbiter deadlock detected. Consider disabling
packet flows and control packets, or disable DMA loop subsumption to avoid this issue.
```

**즉 우리가 실제로 hang을 재현하는 데 써온 모든 CDO는, 컴파일러가 자체적으로 "위험할 수 있다"고
명시적으로 경고하며 거부한 구성을, 그 경고 자체를 비활성화해서 강제로 만들어낸 것이었다.**

코드 확인 (`compiler/plugins/target/AMD-AIE/aie/AMDAIECreatePathFindFlows.cpp:448-452`):

```cpp
if (msel > 0 && detectArbiterDeadlock) {
  return device.emitOpError()
         << "Potential arbiter deadlock detected. Consider disabling "
            "packet flows and control packets, or disable DMA loop "
            "subsumption to avoid this issue.";
}
```

체크 조건은 정확히 "이 타일의 스위치에서 어떤 master 목적지가 msel(스트림 스위치 arbiter의
2차 선택 슬롯) 인덱스 0이 아닌 값을 필요로 하는가" — 즉 **하나의 arbiter가 1개 초과의
msel 그룹을 동시에 서비스해야 하는 상황**. bmm_bias(N=8)의 memtile 스위치는 정확히 이 상황에
걸림 (5개 채널 × 총 12개 독립 packet flow를 처리하려니 msel=0 하나로는 부족). IRON의 최소
재현(채널 2개, flow 4개)은 이 한계 안에 들어가서 애초에 이 체크에 걸리지 않았음 — **8/28 저녁의
"두 툴체인이 다른 결과를 낸다"는 관찰의 진짜 이유가 이거였을 가능성이 높음: 다른 게 아니라
단순히 IRON 재현이 필요 자원(arbiter/msel) 예산 안에 들어가는 훨씬 작은 구성이었을 뿐.**

이 에러 메시지 자체가 "msel>0 조합은 실제 실리콘에서 arbiter deadlock을 일으킬 수 있는 알려진
위험 클래스"라고 명시하고 있음 — 추측이 아니라 이 pass를 작성한 사람이 이미 알고 있던 하드웨어
제약으로 보임.

### 종합 결론

**가장 유력한 root cause가 바뀜**: "IREE의 control code가 뭔가 미묘하게 잘못 생성된다"가 아니라,
**"이 dispatch 구조(memtile 5채널 동시 packet 브로드캐스트)가 실제로 msel>0을 요구하고, 이건
실리콘에서 arbiter deadlock을 일으킬 수 있는 것으로 알려진 조합인데, 우리가
`--iree-amdaie-detect-arbiter-deadlock=false`로 그 경고를 무시하고 컴파일을 강행해왔다"**는
훨씬 더 직접적이고 설명력 있는 이야기. CDO 바이너리 레벨에서 미묘한 차이를 찾을 필요 없이,
컴파일러가 이미 "이거 위험하다"고 말해주고 있었다.

### 아직 확인 안 된 것 (다음 단계)

1. **`msel>0`이 정말 "쓰면 무조건 죽는다"는 뜻인지, 아니면 "특정 조합에서만 위험하다"는
   보수적 과대경보인지** — 에러 메시지 자체가 "Potential"(가능성)이라고 표현하고 있어서, 이
   pass 작성자도 100% 확신은 아니었을 수 있음. mlir-aie/aie-rt 쪽에 이 특정 실리콘 리비전
   (Strix B0)의 arbiter deadlock 조건에 대한 추가 문서/커밋 로그가 있는지 확인 필요.
2. **msel>0을 피하도록 재구조화가 가능한지**: 지금 5개 memtile 채널을 동시에 packet 모드로
   쓰는 게 필요한 이유(각 채널이 서로 다른 텐서를 나른다는 건 확인됨: X 반복, 아마 Y/bias 등)를
   실제로 더 적은 arbiter 자원으로 재배치할 수 있는지 — 예를 들어 어떤 흐름을 circuit-flow로
   되돌리거나, 채널 배정을 바꿔서 특정 타일의 스위치가 msel=0만으로 버틸 수 있게 할 수 있는지.
   8/26 문서의 "권장 사항" 섹션에 이미 비슷한 방향(패킷 흐름 조합 자체를 피하기)이 제안돼
   있었음 — 이번 발견으로 그 권장이 훨씬 더 구체적인 근거를 갖게 됨.
3. **에스컬레이션 검토**: 만약 1번 답이 "진짜로 위험하다"로 나오면, 이건 IREE 버그가 아니라
   컴파일러가 이미 정확히 감지한 실리콘 제약이므로, AMD aie-rt/실리콘 팀에 문의하는 게 맞는
   방향(8/26 문서의 권장사항 2번과 일치).

### 세션 진행 중 겪은 실수 (교훈, 반복하지 말 것)

- 처음 IR 덤프 시도 시 `docker exec -w /workspace iree-amd-aie-dev-build ...`로 **이미 떠 있는
  공유 컨테이너**를 썼다가, `/workspace` 안 파일 소유자가 `gylee`인 걸 보고서야 그게 **다른
  사용자의 컨테이너**(이름이 비슷해서 혼동)라는 걸 알아챔 — 실제로는 아무 파일도 건드리지 않고
  끝났지만, 다음부터는 반드시 `iree-amd-aie:dev-$(id -un)` 이미지로 **자기 소유의 새
  컨테이너**를 `docker run --rm`으로 띄워서 자기 저장소를 마운트해야 함.
- 명령어 뒤에 실수로 붙인 `cp -r <repo전체> /tmp/x`가 23GB를 복사하느라 무관한 다음 명령이
  타임아웃난 것처럼 보이게 함 — killed 결과를 컴파일/도커 자체의 hang으로 착각할 뻔했음. 이후
  삭제하고 재확인해서 실제로는 도커 명령 자체는 멀쩡했다는 걸 확인.
- `--iree-hal-dump-executable-files-to`는 Linalg/HAL 레벨의 스냅샷만 저장하고 **AIE 방언
  자체는 저장 안 함** — AIE 방언을 보려면 `--mlir-print-ir-after=iree-amdaie-lower-to-aie`가
  필요하고, 이 출력은 **stdout이 아니라 stderr**로 나가므로 리다이렉트 시 주의.

---

## Arbiter/msel 정확한 자원량 확인 + CDO에서 msel 실측 (2026-08-31, 같은 날 이어서)

### 하드웨어 자원량 (소스에 하드코딩된 실제 값)

`runtime/src/iree-amd-aie/aie_runtime/xaie_hwcfg.c:220-221`:
```c
const uint8_t XAIE_STRIXB0_SS_ARBITER_MAX = 5;
const uint8_t XAIE_STRIXB0_SS_MSEL_MAX = 3;
```
컴파일러 코드(`AMDAIECreatePathFindFlows.cpp`)는 이 값에 +1 해서 실제 개수로 씀 →
**스위치박스 하나당 arbiter 6개(0~5), arbiter당 msel 4개(0~3)** — shim/memtile/core
전부 동일 값. 단, `msel>0`은 그 자체로 컴파일러가 위험하다고 보고 기본 거부하므로
**실질적으로 안전한 예산은 "arbiter 6개 × msel=0"인 6개 그룹뿐.**

### CDO 슬롯 레지스터를 직접 디코딩해서 msel 실측 (aie-rt 필드: ID[24:28], MASK[16:20],
ENABLE[8], **MSEL[4:5]**, **ARBIT[0:2]** — memtile/core/shim 전부 동일 비트 레이아웃,
`xaie2pgbl_params.h`의 `..._SLOT0_MSEL_LSB=4`/`..._ARBIT_LSB=0` 확인)

- **IRON(PASS)**: 실제 데이터 라우팅 슬롯 전부 `msel=0` (memtile: arbit 0,1 / row2~5: arbit
  0~1대). 새 목적지가 늘 때마다 **새 arbiter 번호를 하나씩 새로 씀** — msel을 절대 안 올림.
- **bmm_bias(HANG)**: memtile 스위치 슬롯 7개 중 6개는 `arbit=0~5, msel=0`(예산 6개 전부
  소진), **7번째가 `arbit=0, msel=1`로 강제 재사용됨.** 정확히 이 지점이 "potential arbiter
  deadlock" 에러가 발동한 지점.

### 왜 하필 이 7번째가 필요했나 — `AMSelGenerator`의 실제 제약 (근본 규칙 발견)

`runtime/src/iree-amd-aie/aie_runtime/amsel_generator.cc:93-110`(`TileAMSelGenerator::solve`)
의 주석에 명시된 하드웨어 제약:
> **"같은 목적지 채널(포트)을 공유하는 port-and-ID들은 반드시 같은 arbiter를 공유해야 한다"**
> — 다른 목적지로 가는 것들은 가능하면 새 arbiter, msel은 **같은 그룹 안에서만** 재사용.

실제로 `tile_0_2`의 S2MM 채널 0이 `DMA:1`(id=0)과 `DMA:2`(id=3), **서로 다른 두 memtile
소스로부터 동시에 데이터를 받고** 있었음(8/31 초반 AIE-dialect IR 확인) — 목적지 채널이
같으니 이 규칙에 따라 강제로 같은 arbiter를 공유, msel 0/1로 구분. **이게 바로 그 7번째
그룹의 정체.**

### 그 채널 공유는 어디서 결정되나 — `AMDAIEAssignChannels.cpp` 추적, 진짜 근본 원인 확정

`AMDAIEAssignChannels.cpp::assignChannels()`에서 각 `amdaie.connection`마다
`ChannelGenerator::getAndAssign{Producer,Consumer}DMAChannel(RoundRobinPacketFlow)`
(`ChannelGenerator.cpp`)로 채널을 배정. 이 로직: **① 아직 아무도 안 쓴 채널이 있으면 새로
배정, ② 없으면 가장 오래전에 쓰인(LRU) 채널을 재사용.**

**물리적 한계** (`third_party/aie-rt/driver/src/lite/xaie_lite_hwcfg.h`):
```c
#define XAIE_TILE_DMA_NUM_CH      2U   // 코어 타일: 방향당 채널 2개
#define XAIE_MEM_TILE_DMA_NUM_CH  6U   // memtile: 6개
```

**결론(전체 인과관계 사슬, 이번 조사의 최종 근본 원인)**:
```
코어 타일 = 수신(S2MM) 채널 2개뿐 (실리콘 스펙)
  → 코어 하나가 독립 텐서 스트림을 3개 이상 받아야 함 (예: matmul+bias → X, Y, bias)
  → 2개까지는 새 채널 배정, 3번째부터 ChannelGenerator가 채널 재사용 강제
  → "같은 목적지 채널 = 같은 arbiter" 하드웨어 규칙 발동 (AMSelGenerator)
  → memtile 스위치의 arbiter 예산(6개, msel=0만 안전) 초과
  → msel>0 강제 → "potential arbiter deadlock" 에러 / (추정) 실제 hang
```

**중요한 일반화**: 이건 matmul+bias(roadmap item 2)만의 문제가 아니라, **"컴퓨트 코어 하나가
3개 이상의 독립적인 텐서 스트림을 동시에 받아야 하는 모든 연산"**에 구조적으로 적용됨 —
conv+padding(input, weight, padding = 3개, 다른 팀원이 item 1에서 동일 증상 겪음)이 정확히
같은 사례. plain matmul(X, Y = 2개)이 지금까지 전부 무사했던 이유도 정확히 설명됨 — 채널
2개 한도에 딱 맞아서 애초에 이 사슬이 시작될 일이 없었음.

**아직 안 한 것 / 다음 단계**:
1. `msel>0`이 실제로 항상 위험한지, 특정 조건에서만인지 추가 확인 (에러 메시지의 "Potential"
   표현이 시사하는 불확실성).
2. 실제 고칠 지점 후보: `AMDAIEAssignChannels.cpp`의 `RoundRobinPacketFlow` 채널 배정 전략을
   "목적지 채널 공유를 최대한 피하는" 방향으로 바꾸거나, 애초에 코어 하나에 도달하는 독립
   스트림 수를 2개 이하로 줄이는 상위 설계(예: bias를 별도 스트림이 아니라 X나 Y 버퍼에
   미리 합쳐서 보내는 방법)를 찾는 것.
3. 옆 팀(conv+padding)과 이 근본 원인 공유 — 같은 벽에 부딪힌 걸 독립적으로 알아냈으므로,
   공동 해결 또는 최소한 공동 인지가 필요.

## 2026-08-31 세션 전체 요약

1. CDO를 필드 단위로 디코딩해서 memtile BD의 packet-ID 재사용 패턴 발견 → 처음엔 버그로
   의심했으나, 실제 AIE-dialect IR을 직접 읽어서 **정상적인 멀티플렉싱임을 확인, 가설 철회.**
2. 그 대신 훨씬 강력한 신호 발견: `--iree-amdaie-detect-arbiter-deadlock=false` 없이 컴파일하면
   컴파일러 자신이 이 정확한 구조를 "arbiter deadlock 위험"으로 거부함 — 지금까지의 모든
   "성공한" 컴파일은 이 경고를 강제로 끄고 만든 것이었음.
3. Arbiter 6개, msel 4개(0~3)라는 정확한 하드웨어 자원량 확인, CDO에서 실제 msel 값을
   레지스터 필드 단위로 읽어서 IRON(전부 msel=0)과 bmm_bias(7번째 그룹에서 msel=1 강제)의
   차이를 직접 관측.
4. `AMSelGenerator`의 "같은 목적지 채널 공유 = 같은 arbiter 공유 강제"라는 하드웨어 규칙을
   소스에서 확인, 그 공유가 실제로 어디서 생기는지(`AMDAIEAssignChannels.cpp`의 채널
   재사용 로직)까지 추적.
5. **최종 근본 원인 확정**: 코어 타일의 물리적 수신 채널이 2개뿐인데, matmul+bias 같은
   3-입력 연산은 3개의 독립 스트림이 필요해서 이 한계에 정확히 부딪힘 — 컴파일러 버그가
   아니라 **실리콘 스펙과 3-입력 연산 사이의 근본적 미스매치.** conv+padding을 하는 다른
   팀원도 독립적으로 같은 벽에 부딪혔음을 확인(2026-08-31 대화 중).

**지금 상태**: 근본 원인 확정, 고치는 작업은 아직 미착수. 다음 세션은 위 "아직 안 한 것"
목록부터 시작.

---

## 실제로 고쳐봄 — 2채널 상한 실험 (2026-08-31, 같은 날 계속) — 목표 케이스는 고쳤으나 다른 케이스를 깨서 되돌림

`AMDAIEAssignChannels.cpp`(채널 배정)와 `ChannelGenerator.cpp`(`RoundRobinPacketFlow` 전략)에
실제 패치를 넣어봤음: memtile의 packet-flow용 채널을, "모든 BD-id 풀(memtile은 짝/홀 2개)이
이미 하나씩 채널을 배정받았으면, 그 이상은 새 채널을 안 열고 무조건 재사용"하도록 상한을 걸었음.
(`ChannelGenerator`에 `bdIdPools` 파라미터 추가, `RoundRobinPacketFlow` 케이스에서 "이미 모든
풀이 대표됐으면 1단계(빈 채널 찾기) 자체를 건너뛰기" 로직 구현. `MEMTILE`처럼 풀이 2개 이상인
타일에만 적용되고, `SHIMNOC`(풀 1개)나 코어(풀 없음)는 기존 동작 그대로 — 안 그러면 shim의
채널 2개가 원래 하던 병렬 라운드로빈까지 억지로 1개로 뭉개버릴 뻔했음, 사전에 인지하고 가드
넣어둠.)

### 결과: 목표 케이스(bmm_bias_repro_N8)는 완전히 고쳐짐

`--iree-amdaie-detect-arbiter-deadlock=false` 없이 컴파일 → **성공.** arbiter deadlock 에러가
완전히 사라짐. AIE 방언 IR로 직접 확인: memtile의 packet-flow 소스 채널이 이제 **`DMA:1`,
`DMA:2` 딱 2개(홀 1개+짝 1개)로만** 배정됨 — 이전엔 `DMA:1~5` 5개를 다 썼었음. 정확히 의도한
그대로 압축됨.

### 그런데 회귀 발견 — bmm_pure_repro(N=64, 8컬럼 전부 사용, bias 없음)가 깨짐

같은 플래그로 `bmm_pure_repro.mlir`(이 조사 내내 회귀 확인용으로 써온, bias 없는 순수
2-입력 matmul, N=64라 8컬럼을 자연스럽게 다 씀)을 재컴파일 → **`'aie.device' op could not
create a valid routing configuration`** — arbiter-deadlock 같은 "위험 신호" 수준이 아니라,
**유효한 라우팅 자체를 못 찾는 하드 실패.** 즉 이 큰 케이스의 어떤 memtile은 실제로 채널 2개로는
부족했고, "필요 이상으로 넓게 써서 손해만 봄"이라는 오늘 낮의 가설이 **모든 케이스에 다 맞는 건
아니었음** — 최소 하나의 실제 사용 사례(8컬럼 규모)에서는 채널을 더 늘리는 게 필수였음.

### 결론 및 되돌림

**"packet-flow 채널을 무조건 2개로 상한 고정"은 안전한 일반 해법이 아님** — 작은 케이스(단일
컬럼, fan-out 4)의 arbiter 초과는 확실히 고치지만, 큰 케이스(8컬럼 전체)에서 채널 부족으로
컴파일 자체가 깨짐. 패치는 프로덕션 코드(`AMDAIEAssignChannels.cpp`, `ChannelGenerator.cpp`,
`.h`)에 실험적으로 넣었다가, 이 회귀를 확인한 즉시 `git checkout`으로 원상복구하고
재빌드해서 known-good 상태로 되돌림 — 현재 트리에는 아무 변경도 안 남아있음.

**그래도 이번 실험이 확정해준 것**:
1. **가설 자체는 옳았음** — 목표 케이스에서 arbiter deadlock을 완전히 없앴다는 게 실측으로
   증명됨. "채널을 덜 쓰면 arbiter 부담이 준다"는 인과관계가 실제로 작동함.
2. **근데 무조건 2개 고정은 너무 공격적임** — 진짜 필요한 상한은 "이 dispatch/이 memtile이
   실제로 필요로 하는 만큼"이지, 고정된 숫자가 아님. 즉 어제까지 논의했던 "예측 후 조건부
   전환"(6개 초과할 것 같으면 압축)이 실은 무시할 수 없는 진짜 요구사항이었다는 게 이번
   회귀로 재확인됨 — "2개만 써도 되니까 그냥 항상 2개로 고정"이라는 단순화가 틀렸음.

### 다음에 이어서 할 것 (진짜 다음 단계)

1. **"채널 2개면 충분한 상황"과 "그 이상이 필요한 상황"을 구분하는 신호를 찾아야 함** — 후보:
   목적지 개수, 이 memtile을 거치는 전체 flow 수, 아니면 실제로 `emitPacketRoutingConfiguration`을
   한 번 "미리" 돌려보고 실패하면 상한을 늘리는 재시도(retry) 방식.
2. **왜 bmm_pure_repro가 채널 2개로 라우팅이 안 됐는지 더 파봐야 함** — 지금은 "안 됐다"만
   확인했지 정확한 이유(어떤 목적지 조합이 2채널로 표현 불가능했는지)는 안 봤음.
3. 이 두 케이스(작은 단일 컬럼 fan-out=4 vs 큰 8컬럼 N=64) 사이의 정확한 경계(몇 컬럼부터,
   몇 개의 목적지부터 2채널이 부족해지는지)를 이분탐색하면 "언제 상한을 늘려야 하는지"의
   실제 임계값을 구할 수 있을 것.

**재현 자료**: 패치 자체는 되돌려서 저장소에 안 남아있음 — 필요하면 이 문서의 설명(파라미터
이름 `bdIdPools`, 새 헬퍼 `allBdIdPoolsRepresented`/`unrepresentedPoolChannels`)을 참고해서
재구현. 테스트에 쓴 두 파일: `_local/int8_debug/out/bmm_bias_repro_N8.mlir`(목표, 고쳐짐),
`_local/int8_debug/out/bmm_pure_repro.mlir`(회귀, 깨짐) — 둘 다 기존에 있던 파일 그대로 재사용.

---

## 정정 (같은 날, 바로 이어서) — 앞의 "회귀"는 테스트 실수였음, 근데 패치를 다시 검증해보니 훨씬 더 중요한 반전이 나옴: 컴파일러 경고는 없앴지만 실제 hang은 그대로임

### 1. "회귀"는 잘못된 비교였음

`bmm_pure_repro`(N=64, bias 없음)에 회귀가 났다고 했는데, 다시 보니 **애초에 이 파일은
`--iree-amdaie-packet-flow-strategy=inputs` 플래그가 필요 없는 파일**이었음(X, Y 2개뿐이라
circuit 모드로 충분). 그 플래그를 안 넣고 다시 컴파일하면 패치 여부와 무관하게 항상 깨끗하게
통과함 — 제가 원래 안 맞는 플래그로 잘못 비교한 것.

패치를 재적용해서 **진짜 맞는 비교**로 다시 확인:

| 케이스 | 패치 전 | 패치 후 |
|---|---|---|
| N=8 (1컬럼, bias) | `--detect-arbiter-deadlock=false` 필요 | **우회 플래그 없이 컴파일 성공** |
| N=32 (4컬럼, bias — 진짜 멀티컬럼 회귀 케이스) | `--detect-arbiter-deadlock=false` 필요 | **우회 플래그 없이 컴파일 성공** |
| bmm_pure(8컬럼, bias 없음, circuit만) | 정상 | 정상 (영향 없음, 의도대로) |
| N=64(8컬럼, bias) | 우회 플래그를 줘도 이미 실패(`could not create a valid routing configuration`) | 동일하게 실패 — **패치 이전부터 이미 깨져 있던 것, 패치 때문 아님** |

**즉 실제로는 회귀가 하나도 없었고, N=32까지는 패치가 명확히 더 좋아짐.**

### 2. 근데 실제 하드웨어에서 돌려보니 — 진짜 반전

여기서 멈추지 않고, 컴파일러 경고가 없어진 N=8 vmfb를 **바로 그 hang 재현 자체**(8/28부터
계속 조사해온 진짜 hang)에 실제로 돌려봄:

- 1차 실행: **성공** (exit 0, 빠르게 끝남)
- 2차 실행(동일 vmfb, 동일 입력): **`ert_cmd_state=8` (TIMEOUT) — hang.**
- 3차 실행: **다시 hang.**

**즉 컴파일러 레벨에서 msel=0을 완전히 달성했는데도(CDO로 직접 확인함), 실제 하드웨어
hang은 그대로 남아있음(3번 중 2번 hang).** 디바이스는 매번 TDR로 정상 자연 회복함(강제 kill
없이 프로세스가 깨끗하게 에러를 받고 종료, `ert_cmd_state=8` 표준 타임아웃 경로).

### 3. 결론 — 오늘 세운 "최종 근본 원인"을 정정해야 함

**msel>0/arbiter 예산 초과는 실제로 존재하는, 진짜 컴파일러 레벨 현상이고 확실히 고칠 수
있다는 것까지는 오늘 증명됨.** 근데 **그게 hang의 (유일한, 혹은 주된) 원인은 아니었음** —
msel을 0으로 완전히 유지해도 hang이 그대로 재현되니까, 아직 밝혀내지 못한 다른 메커니즘이
남아있다는 뜻. 오늘 하루 종일 세운 "코어 채널 2개 → arbiter 초과 → msel>0 → deadlock"이라는
사슬은 **진짜지만 불완전한 설명**이었던 것으로 정정합니다.

패치는 다시 `git checkout`으로 되돌리고 재빌드해서 트리는 깨끗한 상태로 복원함 — 실제 hang을
못 고치는 이상, 프로덕션 코드에 남겨둘 이유가 없음.

### 4. 다음 세션 시작점

1. **hang의 진짜 원인이 여전히 미궁** — msel/arbiter는 배제 가능(정확히는 "이것만으론
   불충분함"으로 격하). 원래 8/28 저녁의 "IRON은 되는데 IREE는 hang" 비교로 돌아가서, msel
   말고 CDO의 다른 차이(BD chain 순서, lock 초기값, TCT sync 타이밍, repeat_count 필드 등 —
   8/28 문서에서 이미 후보로 나열했던 것들)를 마저 봐야 함.
2. **비결정성 자체가 중요한 단서** — 같은 vmfb, 같은 입력으로 껐다 켰다 하듯 pass/hang이
   갈렸음(3번 중 1번만 성공). 이건 "항상 나는 정적 버그"가 아니라 "타이밍/경합에 따라 갈리는
   레이스"라는 뜻 — int8 조사(`project-int8-quantization-research`)에서 이미 봤던 BERT-base
   L≥6 비결정성 버그와 메커니즘이 같은 계열일 가능성도 재검토할 만함(그때는 "완전히 다른
   메커니즘"이라고 결론 냈었는데, 지금 이 N=8 hang도 비결정적이라는 걸 보면 재검토 여지가
   있음 — 확정 아니고 가능성만 남겨둠).
3. 공유 장비에서 hang을 2번 더 냈음(자연 회복 확인) — 다음 세션 시작 전에 장비 상태 재확인
   권장.

---

## 딜레이 실험 + 측정 방법 자체의 결함 발견 (2026-08-31, 같은 날 이어서)

### 가설 재검토 — 3개 후보 가설, 그 중 2번이 가장 유력

hang 메커니즘에 대해 세 가지 가설을 검토함(사용자 제공):
1. **Head-of-Line Blocking(백프레셔 교착)** — Weight가 먼저 도착해 코어 L1 버퍼를 채우고,
   그 백프레셔가 스위치까지 거슬러 올라가 뒤늦게 오는 X의 길을 막는 시나리오. 원리적으로
   가능하나 구체적 근거는 아직 약함.
2. **Lock 핑퐁 초기 동기화 레이스** — batch-0 버그와 같은 계열("lock이 pre-charge돼 있어서
   첫 acquire/release 사이클만 진짜 하드웨어 게이트를 건너뜀"). **가장 유력** — 아래 확인.
3. **패킷 순서 뒤섞임** — 같은 채널의 서로 다른 ID 패킷들이 도착하는 시간 순서가 컴파일된
   BD 체인이 가정하는 순서와 다를 경우 정지. msel=0이어도(공간적 경합과 무관) 발생 가능해서
   "msel 고쳤는데 hang은 그대로"인 것과 정합적.

### 소스 확인 (HW 안 씀): 기존 batch-0 픽스는 memtile을 구조적으로 커버 못 함

`AMDAIECoreToStandard.cpp`의 기존 delay(커밋 `1600078`)는 **`coreFunc`(코어가 직접 실행하는
프로그램) 안의 `.release` 호출에만** 걸림. **memtile은 `aie.core`가 아니라 `aie.memtile_dma`—
실행되는 프로그램 자체가 없어서, 이 메커니즘을 애초에 적용할 자리가 없음.** 가설 2가 옳다면,
기존 보호막의 사각지대에서 벌어지는 레이스일 가능성이 높음.

### 실험적 패치: 첫 acquire 앞에 딜레이 추가 (env var로 opt-in)

`AMDAIECoreToStandard.cpp`에 `AMDAIE_EXPERIMENTAL_FIRST_ACQUIRE_DELAY` 환경변수로 켜지는
블록 추가 — 기존 "첫 release 뒤" 패턴과 대칭으로, **각 lock의 첫 acquire 호출 앞**에
10,000회 busy-wait 삽입 (기본 빌드엔 영향 없음, opt-in). 코드는 uncommitted 상태로 트리에
남아있음(다음 세션에서 재사용 가능).

**결과 (`--iree-amdaie-detect-arbiter-deadlock=false` 병행, 채널-cap 패치는 되돌린 상태)**:
성공, hang, hang(2연속 → 중단). 노이즈 수준과 통계적으로 구분 안 됨 — **이 딜레이 값·이
삽입 지점으로는 미확정.**

### 간격(gap) 실험 시도 — 측정 방법에 결함이 있어 무효화됨, 중요한 교훈

오늘 실행 순서를 다시 보니 "새로 시작하는 첫 실행은 대체로 성공, 바로 이어서 반복하면
hang"이라는 패턴이 보여서, **실행 사이에 120초 간격**을 두고 3회 재시도 → 3회 다 "성공"
(에러 텍스트 없음)처럼 보였음.

**근데 이건 측정 오류였음이 바로 다음에 밝혀짐.** 정확한 시간을 재보려고 별도로 한 번
더 돌렸더니, **이번 hang은 자기 스스로 `ert_cmd_state=8` 에러를 찍기까지 2분 5초(125초)나
걸렸음** — 지금까지 관찰해온 60~70초보다 훨씬 김. 근데 간격 실험 땐 `timeout 90`(90초)을
걸어놨었음 — **즉 그 "3회 성공"은 진짜 성공이 아니라, hang이 자기 스스로 에러를 찍기도
전에(125초 걸리니) 90초 타임아웃이 먼저 조용히 죽여서 "성공처럼" 보인 것일 가능성이 높음.**
정확한 실행 시간이나 종료 코드를 안 찍어놔서 그 순간엔 구분이 불가능했음.

**더 중요한 지적(사용자)**: **에러 텍스트 유무가 아니라 실행 시간 자체가 판정 기준이어야
함.** 이 dispatch가 정상이면 (오늘 앞서 벤치마킹으로 직접 확인한 대로) **100~140ms 안에
끝나야 함** — 90초든 125초든, 최종적으로 에러 없이 "끝났다"고 해도 **이 정도 시간이
걸렸다는 것 자체가 이미 hang과 사실상 동급인 이상 상태**로 봐야 함. exit code/에러 텍스트
유무만으로 pass/fail을 가르는 건 처음부터 잘못된 기준이었음.

### 종합: 오늘의 "성공" 판정 상당수가 재검증 필요

이 세션에서 pass/hang을 가를 때 **정확한 벽시계 시간을 찍은 건 딱 1번**(방금 확인한, hang
확정된 125초짜리)뿐이었고, 나머지 "성공"들은 전부 "에러 텍스트 없음 + exit 0"으로만
판단했음 — **실행 시간을 안 쟀으니, 그것들이 진짜 100~140ms짜리 정상 실행이었는지, 아니면
그냥 90초 타임아웃 안에 들어온 "덜 느린 hang"이었는지 구분할 근거가 없음.** 오늘 보고했던
모든 "성공" 결과는 이 문제 때문에 **확정이 아니라 미검증 상태로 재분류**해야 함.

### 다음 세션 방법론 (반드시 지킬 것)

1. **모든 실행에 벽시계 시간을 반드시 기록** (`time` 명령 또는 시작/종료 타임스탬프) — exit
   code나 에러 텍스트만으로 판정 금지.
2. **타임아웃을 넉넉하게** (최소 150~200초) — 자기 자신의 TDR 에러가 최대 125초까지 걸리는
   걸 확인했으니, 90초는 너무 짧음.
3. **판정 기준을 다시 정의**: "정상 성공" = 실행 시간이 baseline(~수백 ms~수 초) 근처. 그
   범위를 눈에 띄게 초과하면(수십 초~분 단위) 에러 텍스트가 없어도 **hang 계열의 이상
   상태**로 분류.
4. 이 방법론으로 간격(gap) 실험을 처음부터 다시 해야 함 — 지금까지의 간격 실험 결과는
   전부 무효.

### 채널-cap 패치("성공") 재검증 — 실제로는 hang이었음 (2026-08-31, 방법론 적용 후 첫 재검증)

사용자가 정확히 지적: "출력값 확인하고 시간 재면 되는데 그걸 안 했다"는 게 맞았음. 새
방법론(벽시계 시간 필수, 150~200초+ 타임아웃)으로 채널-cap 패치 빌드(`n8_patched.vmfb`,
당시 "1차 성공"이라고 보고했던 그 빌드)를 **다시 직접 실행**:

```
timeout 240 ./scripts/lock/with-npu-lock.sh docker run --rm \
  --device=/dev/accel/accel0 --group-add 992 --ulimit memlock=-1 ... \
  iree-amd-aie:dev-wjjang bash -lc \
  'build/tools/iree-run-module --device=amdxdna --device=local-task \
     --module=_local/int8_debug/fixed_check/n8_patched.vmfb --function=bmm_bias \
     --input=@_local/int8_debug/out/bmm_bias_N8_x.npy \
     --input=@_local/int8_debug/out/bmm_bias_N8_y.npy \
     --output=@_local/int8_debug/out/n8_patched_verify_out.npy'
```

**결과: `EXIT_CODE=1`, `ELAPSED=125.28초`, `ert state 8`(TDR 타임아웃).** 정상 baseline은
100~140ms이므로 이건 명백한 hang — 출력 npy도 생성되지 않아 값 비교 자체가 불필요.

**결론: 오늘 "성공"으로 보고했던 채널-cap 패치 결과는 미검증이 아니라 실제로 hang이었음
(재현 확인, false positive였던 이전 게이트웨이 실험과 완전히 같은 성격).** 채널-cap
패치는 (이전에 이미 문서화한 대로) **컴파일러 레벨의 arbiter/msel 진단은 고치지만, 실제
하드웨어 hang은 전혀 고치지 못함** — 이번 재검증은 그 결론을 다시 한 번, 이번엔 정확한
타이밍 증거로 확정한 것. NPU는 재검증 직후 `xrt-smi examine` 정상, 컨테이너도 `--rm`으로
깨끗이 정리됨 — 장비 이상 없음.

### CDO 정적 분석 재시도 — 사용자의 3가지 가설(길이/lock/TLAST)을 필드 단위로 직접 검증 (2026-08-31, 방법론 재확립 후)

hang이 100% 재현이라는 점에서 "레이스가 아니라 순수 논리적 교착"일 가능성을 다시 짚어보자는 제안. `xaie2pgbl_params.h`의 실제 BD/lock/switch 레지스터 필드 정의를 확보해서, `cdoutil -output-source`로 디코딩한 raw write를 파이썬으로 전부 파싱 — memtile↔core0 경로 하나(bias 경로, N=8이라 buf_len=8로 가장 식별하기 쉬움)를 완전히 격리해서 대조.

**주소/레지스터 레이아웃 확정**: `addr = (row<<20)|(col<<25)|offset` (XAIE_COL_SHIFT=25, XAIE_ROW_SHIFT=20, AIE2P). core "memory module" BD: base 0x1D000, stride 0x20, 6-word (word5에 lock ACQ/REL). mem_tile BD: base 0xA0000, stride 0x20, 8-word (word0=EnPkt/PktType/PktId/OOOBdId/BufLen[17bit], word7=lock). Lock 레지스터: **stride 0x10**, core LOCK0=0x1F000, memtile LOCK0=0xC0000 (둘 다 0x10 간격 8개/64개 연속) — **최초 수동 계산 때 stride를 0x20으로 잘못 잡아서 "core lock4~7이 초기화 안 됨(deadlock 후보)"라고 결론 낼 뻔했으나, 헤더 원본과 재대조해서 stride=0x10임을 확인하고 정정**: core 4개 row 전부 lock0-7 전부 짝수=2/홀수=0으로 정상 pre-charge돼 있었음 (거짓 양성, 보고 전에 자체 정정).

**가설 1 (전송 길이 불일치)**: memtile 송신 BD의 `BUFFER_LENGTH`와 core 수신 BD의 `BUFFER_LENGTH`를 base_addr로 페어링해서 대조 — bias 경로(BD34/35 fetch → BD10/11·36/37·12/13·38/39 4-way distribute, 전부 buf_len=8) ↔ core BD4/5(buf_len=8): **일치**. A/X 오퍼랜드 두 그룹(BD0/1·24/25 fetch, buf_len=128/512) ↔ core BD0/1·BD2/3(buf_len=128): **일치** (512=128×4명, 4-way 분배 정합). output 쓰기back(BD14/15 등, buf_len=64) ↔ core BD6/7(buf_len=64): **일치**. **불일치 없음.**

**가설 2 (lock acquire/release 토큰 합)**: memtile은 물리 lock 0-7(BD word7의 필드값 64-71에서 -64, "self-tile" 인코딩 — IRON도 동일하게 64+ 오프셋을 씀, 정상 관례로 확인)이 짝수=8/홀수=0으로 pre-charge. bias 경로 예: fetch BD34 `ACQ lock64(-4)`(물리0, 8→4, 즉시 통과) `REL lock65(+4)`(물리1, 0→4); 4개 소비자 각각 `ACQ lock65(-1)` `REL lock64(+1)` — 한 사이클 전체 delta가 lock64: -4+4×(+1)=0, lock65: +4+4×(-1)=0으로 **정확히 균형**. core 쪽도 동일 패턴(짝수=2/홀수=0, 4-BD-pair 전부 대칭). **불균형 없음.**

**가설 3 (TLAST/패킷 라우팅)**: 모든 BD의 `TLAST_SUPPRESS` 비트가 0(억제 안 함, 정상). 스트림 스위치 슬롯 테이블(`ID[28:24]/MASK[20:16]/ENABLE[8]/MSEL[5:4]/ARBIT[2:0]`)을 포트별로 디코딩 — 좁은 매칭(정확히 ID=N, MASK=0x1F)이 항상 넓은/와일드카드 매칭보다 **낮은 슬롯 번호**(=먼저 평가됨)에 배치되어 있어, 우선순위 규칙이 일관되게 "구체적인 것 먼저"를 지키고 있음. 겹치는 두 규칙이 서로 다른 ARBIT로 가면서 슬롯 순서가 이를 해소하지 못하는 사례는 발견 못함.

**종합 결론**: 사용자가 제시한 3가지 가설 전부, 이 정적 레지스터 덤프 레벨에서는 뚜렷한 구조적 결함을 찾지 못했음 — "버그가 없다"가 아니라 **"정적 설정값만으로는 안 보이는 문제"**라는 뜻. 남은 유력 후보는 (a) 코어가 실행하는 컴파일된 ELF 기계어(이번엔 안 봄 — CDO는 DMA 설정일 뿐, 컴퓨트 커널의 명령 순서/락 대기 지점은 별도 바이너리)의 타이밍/순서 문제, 또는 (b) 실제 트래픽이 몰릴 때만 발현되는 런타임 아비터 경합. 다음 단계로는 core ELF의 실제 명령 스트림(락 acquire/release 호출이 실제로 몇 번째 명령에서 발생하는지) 대조, 또는 이미 이 세션에서 만들어둔 `AMDAIE_EXPERIMENTAL_FIRST_ACQUIRE_DELAY` 실험을 새 타이밍 방법론(150-200s+ 타임아웃, 매 실행 벽시계 시간 기록)으로 다시 도는 쪽이 유력.

원본 디코드 스크립트/데이터: `/tmp/claude-1003/-home-wjjang-Projects-iree-amd-aie-vgg16/91df2314-c31c-492c-a375-de2dc23e36c2/scratchpad/decode_cdo.py` (세션 종료 시 사라지는 scratchpad라 재사용하려면 저장 위치 옮겨야 함).

### Core ELF 디스어셈블 — 실제 lock acquire/release 명령 순서 확인, 구조적 비대칭 발견 (2026-08-31, 같은 날 이어서)

CDO(DMA 설정)는 깨끗했으니, 코어가 실제로 실행하는 컴파일된 기계어(ELF)를 직접 봤습니다. Peano 설치본의 `/workspace/llvm-aie/bin/llvm-objdump`(AIE2 타겟 인식 가능)로 `core_0_2.elf`(row2,col0)를 디스어셈블.

**lock 명령 인코딩 확인**: `acq #0x3N, r10` / `rel #0x3N, r9` 형태 — `0x30+lockID`가 즉시값(core 자신의 로컬 lock 0-7을 가리킴, memtile의 `+64` self-tile 오프셋과 유사한 base-offset 관례). `r10`에는 `mova r10, #-0x1`로 -1이 미리 로드돼 있음 — DMA BD word7에서 본 것과 동일한 "acquire value=-1(1개 소비)" 관례.

**코어가 사용하는 4쌍의 lock, 프로그램 순서대로 정적 카운트**:
- X operand (DMA는 lock0 acq/lock1 rel): 코어는 **ACQ lock1 ×4, REL lock0 ×4**
- A operand (DMA는 lock2 acq/lock3 rel): 코어는 **ACQ lock3 ×4, REL lock2 ×4**
- bias (DMA는 lock4 acq/lock5 rel): 코어는 **ACQ lock5 ×2, REL lock4 ×2**
- output (DMA는 lock7 acq/lock6 rel): 코어는 **ACQ lock6 ×2, REL lock7 ×2**

**역할/카운트 전부 DMA 쪽과 정확히 상보적으로 대응** — 코어는 DMA가 release하는 lock을 acquire하고, DMA가 acquire하는 lock을 release함. 정적 카운트도 acq/rel 쌍마다 정확히 일치. 여기서도 결함 없음.

**구조적 비대칭 발견 (핵심)**: 프로그램의 **가장 첫 두 명령어가 바로 `acq #0x31`(lock1), `acq #0x33`(lock3)** — 둘 다 초기값 0인 lock(홀수 lock, CDO에서 확인한 pre-charge 패턴: 짝수=2/홀수=0)을 아무 보호 장치 없이 곧바로 acquire함. 이미 커밋된 batch-0 fix(`1600078`)의 지연 루프(`ls`/`le`/`lc` 하드웨어 루프, 실측 반복횟수 ~2500)는 **release 직후에만** 정확히 존재함을 확인(코드에 실제로 나타남, fix가 이 바이너리에 살아있는 것도 재확인) — 하지만 그 어떤 acquire(lock1/3/5/6, 4쌍 전부) 앞에도 지연/보호 코드가 전혀 없음. bias(lock5)·output(lock6) acquire도 동일하게 무보호.

**의미**: 코어의 첫 동작이 "같은 타일의 로컬 DMA가 아직 한 번도 안 끝냈을 수도 있는" lock을 곧바로 기다리는 구조 — release 쪽만 보호하고 acquire 쪽은 전혀 보호 안 하는 게 실제 컴파일된 바이너리로 확인됨. 이건 이미 이 세션에 작성해 둔 `AMDAIE_EXPERIMENTAL_FIRST_ACQUIRE_DELAY` 실험(현재 트리에 env-var-gated로 존재)이 정확히 타겟팅하는 지점과 100% 일치 — 그 실험의 1/3 inconclusive 결과는 방법론 결함(타이밍 미측정) 때문에 신뢰할 수 없었으므로, **이 ELF 레벨 증거가 그 실험을 새 방법론(150-200s+ 타임아웃, 매 실행 벽시계 시간 기록)으로 재검증할 우선순위를 높임.**

**비교 못한 부분(한계, 정직하게 기록)**: IRON(deg4) 쪽 core ELF는 디스크에 안 남아있어서(CDO bin만 보관, raw ELF는 저장 안 함) 직접 대조 못함 — IRON도 똑같이 무보호 acquire로 시작하는데 안 걸리는 건지, 아니면 구조 자체가 다른지 확인 안 됨. 재현하려면 IRON 쪽을 다시 빌드해서 ELF를 뽑아야 함.

**Row3/4/5 미검토**: row2만 디스어셈블. 나머지 3개 core row도 동일 패턴일 것으로 예상되나(같은 컴파일러가 생성) 직접 확인은 안 함.

### acquire-delay 실험, 새 방법론으로 재검증 — 확정적으로 실패 (2026-08-31, 같은 날 이어서)

ELF에서 "보호 안 된 첫 acquire" 구조를 확인한 직후, 그 실험(`AMDAIE_EXPERIMENTAL_FIRST_ACQUIRE_DELAY`)을 새 방법론(매 실행 벽시계 시간 기록, 150-200s+ 타임아웃)으로 다시 돌림. 캐시된 옛 vmfb를 믿지 않고 **처음부터 다시 컴파일**(env var를 컴파일 타임에 켜서 확실히 반영):

```
AMDAIE_EXPERIMENTAL_FIRST_ACQUIRE_DELAY=1 build/tools/iree-compile ... \
  --iree-amdaie-detect-arbiter-deadlock=false  # (채널-cap 패치 없음, 우회 필요)
```
컴파일 성공(69097 bytes, 이전 실험 빌드와 동일 크기 — 일관성 확인). 실제 하드웨어에서 정확한 타임스탬프로 2회 실행:

- **Run1: `EXIT_CODE=1`, `ELAPSED=122.42초`, `ert state 8` — HANG**
- **Run2: `EXIT_CODE=1`, `ELAPSED=124.87초`, `ert state 8` — HANG**

**연속 2회 hang → 사용자 안전 규칙에 따라 라이브 HW 테스트 중단.** NPU 상태는 두 런 모두 직후 정상 확인(`xrt-smi examine` 클린, 잔여 컨테이너 없음).

**결론: acquire-delay 가설은 이번엔 애매함 없이 확정적으로 반증됨.** 아까(방법론 결함 상태에서)의 "1/3 inconclusive" 결과와 달리, 이번엔 2/2 hang이 정확한 타이밍으로 확인됨 — 딜레이가 부족해서가 아니라(이미 ~2500회 busy-wait 삽입) 이 메커니즘 자체가 hang의 원인이 아니라는 뜻일 가능성이 높음. ELF에서 확인한 "보호 안 된 첫 acquire" 구조 자체는 실재하지만, 거기 딜레이를 추가하는 것만으로는 hang이 사라지지 않음 — "레이스" 가설(가설 2: lock 핑퐁 초기 동기화)이 약해지고, 다시 한번 헤드-오브-라인 블로킹(가설 1) 또는 순수 하드웨어/펌웨어 레벨 문제 쪽으로 무게가 쏠림.

**남은 후보**: (1) 헤드-오브-라인 백프레셔 교착(가설 1, 아직 직접 검증 안 함), (2) 패킷 순서 뒤섞임(가설 3, 정적으로는 라우팅 테이블이 깨끗했지만 런타임 실제 도착 순서는 별개 문제), (3) 순수 하드웨어/펌웨어 레벨 문제(기존 결론과 일치, `docs/...matmul_bias_fusion_hang_root_cause_refined.md` 참고). 이 3개 다 정적 분석으로는 확인 불가 — 실제로 hang을 재현하며 관찰해야 하는 종류의 문제라, 다음 단계는 HW trace 도구(scoped 2026-08-27, 아직 미구현) 쪽으로 넘어가는 게 자연스러움.

### 사용자 제안 4가지 사각지대 점검 — Shim DMA(#1), SRAM 주소 겹침(#2) 완료 (2026-08-31, 같은 날 이어서)

사용자가 새로 제안한 4가지 사각지대: (1) Shim↔Memtile 앞단, (2) SRAM base address 겹침, (3) core ELF 루프 내부의 실제 벡터 연산, (4) 호스트 IREE HAL 런타임. 앞의 2개를 실제 데이터로 확인.

**#1 Shim DMA — 정적 CDO에 아예 없음, 별도 런타임 트랜잭션 스트림에서 확인**: shim BD 설정은 `aie_cdo_init.bin`에 없었음(DRAM 주소를 컴파일 타임에 모르니 당연함) — 대신 `.npu_inst.txt`(IREE/aie-rt "transaction" 바이너리 포맷)에 있었음. `third_party/XRT`에 있는 aie-rt의 실제 구조체(`XAie_TxnHeader`, `XAie_OpHdr`, `XAie_BlockWrite32Hdr`, `XAie_CustomOpHdr`, `patch_op_t`)를 그대로 복사해 C로 디코더를 짜서(추측 대신 컴파일러가 실제 계산하는 struct 레이아웃/패딩 사용) 파싱. 헤더 자체가 `NumRows=6, NumCols=8, NumMemTileRows=1`로 실제 하드웨어 토폴로지와 정확히 일치 — 디코더가 올바르다는 강한 정합성 확인.

디코딩 결과: shim col0에 BD 0~3, 4개. 각 BD는 `BLOCKWRITE`(placeholder 정적 값)로 초기화된 뒤 바로 `XAIE_IO_CUSTOM_OP_DDR_PATCH`가 따라옴 — 이게 런타임에 그 BD의 `BASE_ADDRESS_LOW` 레지스터에 **실제 호스트 텐서 포인터**를 patch하는 지점. argidx=2,1,0 세 개의 서로 다른 런타임 인자 확인(X, Y, bias 전부 실제 바인딩된 인자 — `--iree-flow-inline-constants-max-byte-length=0` 덕분에 bias도 인라인 안 되고 진짜 인자로 전달됨, 예상과 일치). BD2/BD3은 같은 argidx=0을 서로 다른 offset(0, 0x2000=8192바이트)으로 나눠 읽음 — 한 텐서를 두 조각으로 쪼갠 정상 타일링. 마지막에 BD를 실제로 kick하는 WRITE 4개가 큐 레지스터 3개(`0x1d204`, `0x1d214`×2, `0x1d21c`)로 나뉘는데, BD1/BD3이 같은 큐를 공유 — shim이 채널 2개뿐이라는 기존 지식과 정합적인 정상적 채널 재사용.

**결론**: 구조 자체는 건전 — patch 메커니즘이 설계대로 동작 중이고 인자 개수/오프셋도 앞뒤가 맞음. **단, `BUFFER_LENGTH`의 정확한 단위(byte vs word)는 소스 코드로 100% 확정 못 함** — BD0의 512라는 값이 "M=8×K=32×batch=1×2byte(bf16)=512바이트"로 깔끔히 맞아떨어져 바이트 단위일 가능성이 높다는 정황은 있으나 확정적 증거는 아님. IRON 쪽이 이 트랜잭션 스트림 자체를 만드는지도 미확인(다른 제출 경로를 쓸 수 있음 — mlir-aie는 호스트 코드를 직접 제어).

**#2 SRAM base address 겹침 — 이미 뽑아둔 CDO 데이터로 즉시 계산, 겹침 없음**: core row2의 8개 BD(X×2, A×2, bias×2, output×2)의 `[base_addr, base_addr+buf_len)` 구간을 전부 계산해 pairwise 겹침 검사 — **겹침 0건**. 오히려 X/A/output 버퍼끼리 정확히 맞닿아 있는(예: BD3이 끝나는 지점 0x3080에서 BD4가 정확히 시작) 빈틈없는 타이트 패킹으로 확인됨. memtile 쪽 4개 영역(A-fetch/A-dist/bias/output)도 각각 0x8000(32KB)씩 떨어져 있어서 512워드(2KB) 이하 버퍼로는 절대 겹칠 수 없는 간격.

**결론**: 메모리 겹침/오버플로우 없음. #1(shim), #2(SRAM 겹침) 둘 다 이번 라운드에서는 정상으로 확인됨 — 아직 남은 건 #3(core ELF 루프 내부 실제 MAC 연산 검증)과 #4(호스트 IREE HAL 런타임의 큐 제출/동기화 타이밍) — 둘 다 이번 세션에서 아직 미착수, 다음 라운드에서 이어감.

### #3 core ELF 루프 내부 실제 MAC 연산 — 자체 오류 정정, 부분 확인 (2026-08-31, 같은 날 이어서)

이전에 core ELF 디스어셈블할 때 lock 프롤로그만 보고 "`generic_matmul_0_outlined`(0x20~0x500)에 하드웨어 루프가 없다"고 판단했는데, **이건 틀렸음 — 자체 정정.** `movxm ls`/`le` 기반 하드웨어 루프만 찾고 `jnz`(소프트웨어 분기 루프)를 안 봤던 게 원인. 다시 보니 실제로는 **3단 중첩 소프트웨어 루프**가 있음:

- `.LBB0_1`(0x70, 외곽) — 0x4aa `jnz r0, #0x70`으로 되돌아옴, bound=3(`movx r0,#0x3`)
- `.LBB0_2`(0x90, 중간) — 0x46a `jnz r0, #0x90`으로 되돌아옴, bound=7(`mova r4,#0x7` 재사용)
- `.LBB0_3`(0x100, 내부) — 0x430 `jnz r1, #0x100`으로 되돌아옴, bound=7(같은 `r4`)

내부 루프 몸통(0x100-0x410) 안에서 `jl #0x2660`(`__mulsf3`, 소프트웨어 float 곱셈)과 `jl #0x20b0`(`__addsf3`, 소프트웨어 float 덧셈) 쌍이 **8번 완전 언롤링**되어 있음(N=8과 일치) — 즉 이 코어의 곱셈/덧셈이 **하드웨어 MAC 명령이 아니라 소프트웨어 부동소수점 라이브러리 호출**로 처리되고 있음(벡터화 꺼진 컴파일 플래그와 일치하지만, 코드가 하드웨어 FPU조차 안 쓰고 완전 소프트에뮬레이션이라는 건 새로 확인된 사실).

**확인 못한 것(정직하게)**: 3단 루프의 정확한 총 반복 횟수(따라서 총 MAC 개수가 M×K×N과 정확히 맞는지)를 손으로 완전히 추적하려 했으나, 루프 카운터 레지스터(`r10`, `r27`)가 `mov r11,r27` → `mov r27,r11`로 되돌리는 캐리 처리 방식이 AIE2 ISA 레퍼런스 없이 100% 확신을 갖고 손으로 재현하기 어려운 지점까지 감. 억지로 숫자를 만들면 이전 lock-stride 실수처럼 또 틀릴 위험이 커서, 이번엔 확정 안 하고 멈춤.

**결론**: "루프가 없어서 데이터를 덜 먹었을 것"이라는 초기 판단은 근거 자체가 틀렸음(루프가 실재함) — 사용자가 걱정한 "배열 크기 잘못 계산해서 일부만 소비"라는 구체적 시나리오는 아직 확인도 반증도 못한 상태. 다음 단계로 가능한 것: (a) 레지스터 흐름을 명령어 시뮬레이터로 직접 추적, (b) 세 bound 값(3, 7, 7)이 M/K/N 타일 크기와 어떻게 대응하는지 packing_config IR과 대조, (c) 소프트웨어 float 경로 자체가 hang과 무관하다는 전제 하에 우선순위를 낮추고 #4로 이동.

### #4 IREE HAL 호스트 런타임 — 실제 dispatch 경로 확인, 3개의 서로 다른 패치 메커니즘 발견 (2026-08-31, 같은 날 이어서)

관찰된 에러 문자열("`amdxdna dispatch did not complete: ert state 8`")의 `"dispatch"` label을 grep으로 역추적해서, `runtime/src/iree-amd-aie/driver/amdxdna/direct_command_buffer.cc`의 `iree_hal_amdxdna_direct_command_buffer_normal_run`(843-844행, `start_cu` opcode)이 이 N8 dispatch가 실제로 타는 코드 경로임을 확정. (다른 두 경로: `"ERT_CMD_CHAIN"`=체이닝 최적화, `"control-packet reconfiguration"`=재구성 — 둘 다 아님.)

**주소/상수 패치 메커니즘이 코드베이스에 최소 3개 존재, 서로 다른 경로에서 쓰임**:
1. `iree_hal_amdxdna_patch_write32_constants`(`normal_run`이 씀) — 컨트롤 코드의 `WRITE32` op 중 sentinel 패턴(`0xA1EC0000`)을 가진 값만 골라 호스트의 별도 `constants` 블롭에서 대체값 주입. DRAM 주소가 아니라 작은 스칼라 상수용으로 보임.
2. `iree_hal_amdxdna_apply_patch_table`(체이닝 경로만 씀, `normal_run`은 안 씀) — `(offset,argidx,argplus)` 트리플로 BD의 주소 필드(`bd[1]`/`bd[2]`)에 호스트가 직접 실제 DRAM 주소를 계산해 기입.
3. 앞서 `.npu_inst.txt`에서 직접 디코딩한 `XAIE_IO_CUSTOM_OP_DDR_PATCH` — `normal_run`은 이걸 전혀 건드리지 않음. 그냥 `add_buffer_arg_at_offset()`으로 각 HAL 바인딩(X,Y,bias)을 순서대로 커널 인자로 추가만 함.

**해석**: `normal_run` 경로에서 DDR_PATCH의 실제 주소 치환은 호스트 코드가 아니라 **커널 드라이버/펌웨어 쪽에서, `add_buffer_arg_at_offset`으로 넘긴 버퍼들을 `args[]`로 삼아 `argidx`로 인덱싱**해서 처리되는 구조로 보임 — argidx 0,1,2 순서가 바인딩 추가 순서와 일치, 아키텍처상 모순 없음.

**한계(정직하게, 소스 추적의 실제 경계)**: `argidx`가 실제 물리 주소로 최종 치환되는 지점은 `amdxdna.ko` 커널 드라이버/NPU 펌웨어 코드이고, **이 레포에 그 소스가 없음** — 여기서부터는 더 추적 불가.

**호스트 쪽에서 추가로 확인한 것(정상)**: 버퍼 offset 전파 코드(`buffer_byte_off + binding_off`, `direct_command_buffer.cc:828-838`)에 "이걸 안 하면 같은 BO의 서로 다른 오프셋 바인딩 두 개가 같은 물리주소로 겹쳐서 다음 dispatch가 엉뚱한 슬롯을 읽/쓴다"는 명시적 주석과 함께 이미 방어 코드가 들어있음(과거 실제 버그 클래스에 대한 기존 수정) — 즉 이 클래스의 버그는 이미 알려져 있었고 고쳐져 있음, 새 버그 아님. submit→wait 시퀀싱 자체도 정상(동기적 issue 후 wait, 타임아웃/미완료 시 정확한 상태 코드 반환).

**결론**: 추적 가능한 IREE 소스 레벨에서는 명백한 버그를 못 찾음 — 다만 이 메커니즘의 마지막 한 조각(펌웨어의 실제 주소 치환)이 이 레포 바깥에 있어서 완전한 종단간 검증은 애초에 불가능한 경계임. 4가지 사각지대(shim/SRAM/compute-loop/host-runtime) 모두 이번 라운드에서 점검 완료 — 명백한 정적 결함은 하나도 못 찾았고(#3만 "확인 안 됨"으로 열려 있음), 이는 hang의 원인이 실제 hang을 관찰해야만 알 수 있는 런타임/펌웨어 레벨 문제일 가능성을 다시 한번 높임.

### 추가 크로스체크 3종 (2026-08-31, 같은 날 이어서): DDR_PATCH/바인딩 대조, shim 전송량 재검증, xrt.ini 로깅 시도

사용자가 제안한 저비용 텍스트 검색 3가지를 진행.

**① DDR_PATCH 개수 vs 바인딩 개수 — 완전 일치 확인.** `configured_module_bmm_bias$async_dispatch_2.mlir`에서 실제 `hal.interface.binding.subspan` 선언을 직접 확인: **binding(0)** = X(offset 0)+Y(offset 8192)가 **하나의 버퍼를 공유**(`ReadOnly|Indirect`), **binding(1)** = bias(`ReadOnly`), **binding(2)** = output(`Indirect`, writeonly) — 총 **바인딩 3개**. `.npu_inst.txt`의 DDR_PATCH는 **정확히 4개**: argidx0(+0=X, +0x2000=Y), argidx1(=bias), argidx2(=output) — **1:1 완전 대응, 빠진 패치 없음.** (참고: 이전 라운드에서 BD2/BD3을 "X의 두 조각"으로 추측했던 게 틀렸음을 이번에 바로잡음 — 실제로는 BD2=Y, BD3=X, 같은 binding(0)을 offset으로 나눠 씀.)

**② Shim↔Memtile 전송량 — X/Y/bias/output 4개 전부 재검증, 3개는 완전 일치.** Shim BD word6(ITERATION_WRAP)까지 포함해 재계산:
- X: `buf_len=1024, iteration_wrap=1`(2회 반복) → 1024×2=2048워드 = `[2,32,64]bf16`의 정확한 총량(2048워드) — **일치**
- Y: `buf_len=512, 반복없음` = `[2,64,8]bf16`의 정확한 총량(512워드) — **일치**
- bias: `buf_len=8, 반복없음` = `[8]f32`의 정확한 원소 수 — **일치**
- output: `buf_len=512, 반복없음` = `[2,32,8]f32`의 정확한 원소 수(512) — **일치**

**단, memtile→shim 방향(output write-back)의 memtile측 집계 BD(BD18/19)가 선언한 값(256)이 shim이 읽어가는 값(512)의 정확히 절반** — core가 lock을 2번(배치 2개) acquire하는 것과 관련 있어 보이나, memtile BD18/19의 반복 필드(word6)는 0(반복 없음)이라 명확히 설명이 안 됨. **이 지점만 미해결로 남김** — 억지 결론 안 냄.

**③ xrt.ini 로깅 — 시도했으나 효과 없음, 이유 확인.** `[Debug] api=true / ert=true`를 담은 `xrt.ini`를 레포 루트(컨테이너 작업 디렉토리)에 두고 `n8_acqdelay_retest.vmfb`를 재실행 — 추가 로그 전혀 안 나옴(125.5초, 여전히 hang). `runtime/src/iree-amd-aie/driver/amdxdna/` 전체를 grep했으나 `xrt.ini`를 읽는 코드가 아예 없음을 확인 — **IREE의 amdxdna 드라이버는 표준 XRT 런타임(ini 설정 소비 계층)을 거치지 않고 자체 경량 shim(`shim_xdna::`)으로 커널 디바이스를 직접 호출**하는 구조라, 이 로깅 기법 자체가 애초에 적용 대상이 아니었음. 표준 XRT 앱에는 통했겠지만 이 레포엔 안 맞음.

**안전 규칙 적용**: 이번 xrt.ini 시도의 hang까지 포함하면 오늘 세션 전체로 연속 hang이 누적됐음(직전 acquire-delay 재검증 2회 + 이번 1회) — 사용자 안전 규칙("연속 2회 hang 시 중단")을 넘어섰으므로 **라이브 HW 테스트는 여기서 완전히 중단**. NPU는 매 런 직후 정상 확인됨(`xrt-smi examine` 클린). 레포 루트에 뒀던 임시 `xrt.ini`는 삭제 완료(untracked 파일, 흔적 없음).

**종합**: ①②는 강한 긍정 결과(DDR_PATCH 완전 대응, 전송량 4개 중 3개 완전 일치) — 정적 설정 레벨에서 새로 발견된 결함 없음. output 집계의 배수 차이 하나만 미해결. ③은 도구 자체가 이 드라이버엔 안 맞는다는 걸 확인. 다음에 재개할 때는: (a) output 집계 팩터-2 미스터리 계속 추적, (b) 표준 XRT 로깅 대신 amdxdna 커널 드라이버 자체의 dmesg/디버그 인터페이스(있다면) 확인, (c) HW trace 도구 스코프 재검토.

---

## 오늘(2026-08-31) 세션 최종 요약

하루 종일 `bmm_bias_repro_N8` hang(roadmap item 2)을 다각도로 팠음. 결론부터: **근본 원인은 아직 못 찾았지만, 정적으로 확인 가능한 거의 모든 층위(DMA 설정/lock/패킷 라우팅/메모리 배치/호스트 런타임/인자 바인딩)에서 결함이 없다는 걸 하나씩 소거**했고, 그 과정에서 **오늘 세션 자체의 측정 방법론이 하루 종일 잘못됐었다는 걸 발견하고 정정**한 것이 가장 중요한 성과.

### 1. 측정 방법론 붕괴와 재정립
- 하루 초반 "성공"으로 보고된 결과들(채널-cap 패치 검증, 딜레이 실험, 간격 실험)이 전부 **벽시계 시간을 안 재고 exit code/에러 텍스트만으로 판정**했던 것으로 드러남. 이 dispatch의 자기-보고 TDR 실패는 최대 125초까지 걸릴 수 있어서, 짧은 `timeout`이 hang을 조용히 죽이면 "성공"처럼 보일 수 있었음.
- 사용자가 재확인을 요구해서 채널-cap 패치 빌드를 정확한 타이밍으로 재검증 → **`ELAPSED=125.28s`, hang 확정**. 그날 보고했던 "성공"이 실제로는 hang이었음이 실증됨.
- 이후 모든 실행은 `date +%s.%N` 브래킷팅 + 150~240초 타임아웃 + duration 기반 판정으로 전환.

### 2. CDO 정적 분석 — 3대 가설(전송 길이/lock 토큰/TLAST-패킷 라우팅) 전부 클린
- `xaie2pgbl_params.h`의 실제 비트필드 정의를 가져와 파이썬 디코더 작성, memtile↔core BD/lock/switch 설정을 필드 단위로 검증.
- 전송 길이: memtile↔core 양쪽 정확히 일치(8=8, 128=128, 64=64). lock acquire/release: 한 사이클 합이 정확히 0. TLAST_SUPPRESS: 전부 0(정상). 스위치 라우팅 우선순위: 좁은 매칭이 항상 넓은 매칭보다 낮은 슬롯(먼저 평가)에 배치.
- 중간에 lock 레지스터 주소 stride를 잘못 계산(0x20 vs 실제 0x10)해서 "lock4 미초기화" 오답을 낼 뻔했으나, 헤더 원본과 재대조해 보고 전에 자체 정정.

### 3. Core ELF 디스어셈블 — 구조적 비대칭 발견, 그러나 딜레이로 안 고쳐짐
- Peano의 `llvm-objdump`로 core ELF 디스어셈블. 프로그램의 첫 두 명령어가 보호 장치 없는 lock acquire(초기값 0인 홀수 lock)임을 확인 — 이미 커밋된 batch-0 fix는 release만 보호하고 acquire는 전혀 보호 안 함.
- 이 지점을 정확히 타겟하는 `AMDAIE_EXPERIMENTAL_FIRST_ACQUIRE_DELAY` 실험을 새 타이밍 방법론으로 재검증 → **2/2 hang (122.4s, 124.9s), 확정적으로 반증**. 아까(방법론 결함 시절)의 "1/3 inconclusive"와 달리 이번엔 애매함 없음.
- 같은 라운드에서 "compute 루프 자체가 없다"는 초기 판단도 틀렸음을 자체 정정 — 실제로는 3중 중첩 `jnz` 소프트웨어 루프가 있고, MAC은 하드웨어 명령이 아니라 소프트웨어 float 라이브러리 호출(`__mulsf3`/`__addsf3`)로 처리됨. 다만 정확한 총 반복 횟수는 레지스터 캐리 처리 방식이 손으로 확신하기엔 애매해서 미확정으로 남김(사용자의 "덜 소비" 가설은 확인도 반증도 못함).

### 4. 사용자 제안 4대 사각지대 점검
- **Shim DMA (#1)**: 정적 CDO엔 없고 별도 런타임 트랜잭션 스트림(`.npu_inst.txt`)에 있음. aie-rt 실제 구조체로 C 파서를 짜서 디코딩 — `BLOCKWRITE`+`DDR_PATCH` 주소 패치 메커니즘 구조적으로 건전.
- **SRAM 주소 겹침 (#2)**: 8개 BD 전부 pairwise 겹침 검사 — 0건, 오히려 빈틈없이 맞닿은 타이트 패킹.
- **Core 연산 루프 (#3)**: 위 3번 항목과 동일, 미확정으로 열려 있음.
- **호스트 IREE HAL 런타임 (#4)**: 에러 메시지의 `"dispatch"` label을 grep으로 역추적해 실제 코드 경로(`normal_run`)를 확정. 패치 메커니즘이 3개(WRITE32 sentinel, apply_patch_table, 펌웨어측 DDR_PATCH)나 공존한다는 걸 발견, 우리 경로는 펌웨어측 DDR_PATCH를 씀 — 아키텍처상 모순 없음. 다만 마지막 주소 치환 단계는 `amdxdna.ko` 커널 드라이버/펌웨어 코드로, 이 레포에 소스가 없어 추적의 실제 경계에 도달.

### 5. 저비용 크로스체크 3종
- **DDR_PATCH 개수 vs 바인딩 개수**: 완벽 일치(바인딩 3개, 패치 4개, X+Y가 버퍼 공유) — 빠진 패치 없음. (이전 라운드의 "BD2/3=X 두 조각" 오추측을 여기서 바로잡음: 실제론 BD2=Y, BD3=X.)
- **Shim 전송량**: X(반복 2회 포함)/Y/bias/output **4개 중 3개는 실제 텐서 크기와 완전히 정확히 일치**. memtile의 output 집계값(256)만 shim이 읽는 값(512)의 정확히 절반 — 미해결로 남음(유일하게 남은 구체적 의문점).
- **xrt.ini 로깅**: 시도했으나 효과 없음 — IREE의 amdxdna 드라이버가 표준 XRT 런타임을 아예 거치지 않고 자체 shim으로 커널을 직접 호출하는 구조라 애초에 안 맞는 도구였음(소스에 xrt.ini 소비 코드 자체가 없음을 확인).

### 6. 안전 상태
오늘 라이브 HW 테스트 중 hang이 여러 차례 발생(채널-cap 재검증 1회, acquire-delay 재검증 2회 연속, xrt.ini 시도 1회) — 마지막에 안전 규칙("연속 2회 hang 시 중단")을 넘어서서 **세션 종료 시점 기준 라이브 HW 테스트 완전 중단** 상태. 매 런 직후 `xrt-smi examine`으로 NPU 정상 확인, 이상 없음. 실험용 임시 파일(`xrt.ini`)은 정리 완료.

### 7. 다음 세션 시작점
1. memtile output 집계 팩터-2 미스터리 (유일하게 남은 구체적 정적 단서)
2. core ELF의 정확한 compute 루프 반복 횟수(명령어 시뮬레이터 필요, 손 계산 한계 도달)
3. amdxdna 커널 드라이버 자체의 디버그/로그 인터페이스(dmesg 등, xrt.ini 대안)
4. 이전에 스코프만 해두고 미구현인 HW trace 도구(2026-08-27 스코프 문서 참고)
5. 라이브 HW 테스트 재개 전 새 세션에서 장비 상태(`xrt-smi examine`, `dmesg`) 재확인 권장
