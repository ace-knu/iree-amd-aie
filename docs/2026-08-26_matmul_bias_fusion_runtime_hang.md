# Matmul+bias dispatch 퓨전 (roadmap item 2): 컴파일은 성공, 실제 하드웨어에서 hang — IREE 버그 아닐 가능성 높음 (2026-08-26)

## 요약 (TL;DR)

roadmap item 2 (`matmul(x,y) + bias`를 두 개의 dispatch가 아니라 하나로 퓨전)는 이제
**컴파일 레벨에서는 완전히 끝났습니다** — 여기 필요했던 컴파일러 버그 8개를 전부 고쳐서
`bert` 브랜치에 커밋 완료(`1600078`, `ee37c48`, `1a1ec51`, `552c8c2`).

근데 컴파일된 dispatch가 **실제 npu4 하드웨어에서 hang**이 나요 (`ert state 8` =
`ERT_CMD_STATE_TIMEOUT`, 드라이버 TDR이 60초 후 강제 종료). 컴파일 타임 이슈들과는
별개의 새로운 버그입니다.

실제 하드웨어로 그럴듯한 원인 6개를 하나씩 테스트해서 전부 반증했어요. 결정적이었던 실험:
문제가 되는 정확한 패턴(**packet-switched flow + 하나의 BD가 쓰는 하드웨어 `repeat_count`
필드**)을 AMD 자체의 독립적인 `mlir-aie`/IRON 툴체인으로 — IREE와 완전히 무관하게 —
바닥부터 재현했더니, 같은 실리콘에서 똑같이 hang이 났습니다. circuit flow + repeat_count,
packet flow + repeat_count=1은 두 툴체인 모두 정상 작동하고, **오직 이 조합에서만** 두
툴체인 모두 hang이 납니다.

**결론: 이건 IREE 컴파일러 쪽 버그일 가능성이 낮습니다.** 서로 완전히 독립적으로 개발된
두 컴파일러가 정확히 같은 하드웨어 기능 조합에서 똑같이 막힌다는 건, 진짜 npu4
실리콘/펌웨어 레벨의 한계이거나, IREE와 mlir-aie가 둘 다 그대로 가져다 쓰는 `aie-rt`
(저수준 BD/stream-switch 레지스터 프로그래밍 라이브러리)의 버그일 가능성을 강하게
시사합니다. 권장: `iree-amd-aie` 소스를 더 파는 건 지금부터는 비효율적일 것 같고, 우리
dispatch 구성에서 이 조합 자체를 피하거나 AMD 쪽에 에스컬레이션하는 게 맞아 보입니다.

---

## 배경

BERT류 워크로드는 `matmul(x, y) + bias`를 하나의 dispatch로 퓨전해야 합니다 (지금은
`DetachElementwiseFromNamedOps`가 둘로 쪼개놔서, `FormDispatchRegions`의 no-fuse
옵션을 끄지 못하게 막고 있음 — 이게 roadmap item 2, "3-input dispatch backend").
여기까지 오는 데 컴파일러 버그 8개를 찾아 고쳤고 (`AMDAIEBufferizeToAllocation`,
`AMDAIEDistributeL1Allocations`, `AMDAIEConvertToDma`,
`AMDAIELogicalObjFifoSplittingUtils`의 버그 2개, `AMDAIEAssignTiles`의 타일 배치
쏠림, `AMDAIENpuDmaToHalfDmaCpyNd`의 phantom half-DMA, `AMDAIEControlCodeLowering`의
하드웨어 BD-length/packet-flag spec 위반), 거기에 아키텍처 변경도 하나 있었어요 (bias를
별도 elementwise dispatch가 아니라 matmul 자신의 accumulator 초기값으로 fold —
`AMDAIEFoldBroadcastAddIntoDest`, 신규 pass). bug #1~8의 세부 내용은 git 히스토리 /
이전 세션 기록에 있고 여기선 반복하지 않습니다.

마지막 남았던 블로커는 `'aie.device' op expects region #0 to have 0 or 1 blocks`
검증 실패였어요 — fuse 안 된 원래 producer의 alloc이 `aie.device` 영역에 감싸지지
않은 채 남아있다가, 나중에 `scf.for`→CFG lowering을 거치면서 여분의 block을
만들어버리는 문제였습니다. 새 pass `AMDAIEEraseDeadAllocAndStores`
(`compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEEraseDeadAllocAndStores.cpp`)로
고쳤고, vectorization 이후 파이프라인에서 `SCFToControlFlowPass` 바로 앞에 두 번
(사이에 canonicalization 끼워서) 삽입했습니다. **검증 완료: `bmm_bias_repro.mlir`가
이제 아래 표준 플래그로 깨끗하게 컴파일되고 실제 vmfb를 만듭니다:**

```
--iree-amdaie-packet-flow-strategy=inputs \
--iree-flow-inline-constants-max-byte-length=0 \
--iree-amdaie-demote-contraction-inputs-to-bf16 \
--iree-amdaie-enable-vectorization-passes=false \
--iree-flow-enable-executable-deduplication=false
```

(`packet-flow-strategy`는 `auto`가 아니라 `inputs` — 이유는 아래 arbiter-deadlock
항목 참고. `--iree-amdaie-detect-arbiter-deadlock=false` 우회도 필요 없습니다.)

이걸로 bug #8은 컴파일 레벨에서 완전히 닫혔습니다. **위 4개 커밋 전부 `bert`
브랜치에만 있고, 아직 어떤 원격 저장소에도 push 안 했습니다.**

## 새로운 문제: 런타임 hang

컴파일된 vmfb를 실제 npu4에서 돌리면 (`iree-run-module` / `scripts/debug/pipeline_dump.py`),
컴파일/검증 에러가 아니라 **dispatch 자체가 완료가 안 됩니다**:

```
runtime/src/iree-amd-aie/driver/amdxdna/native_linux_kmq.cc:607: INTERNAL;
amdxdna dispatch did not complete: ert state 8
```

`ert state 8` = `ERT_CMD_STATE_TIMEOUT` (`third_party/XRT/src/runtime_src/core/include/ert.h:598`).
드라이버 TDR(`tdr_timeout_ms`, 이 호스트에선 60000ms)이 완료 신호를 60초 동안 못 받으면
명령을 강제로 죽입니다. 이건 진짜 hang이지 잘못된 결과가 아니에요 — 이 프로젝트에서
지금까지 봤던 다른 모든 버그(예: batch-0 lock-precharge race,
[docs/2026-08-24 (한국어)](2026-08-24_int8_quantization_investigation.md))와는
카테고리가 다릅니다. 그것들은 dispatch가 제때 완료는 되는데 **결과가 틀린** 경우였어요.

재현: `_local/int8_debug/gen_bmm_bias.py` → `bmm_bias_repro.onnx`
(`B=2,M=32,K=64,N=64` batched matmul + row-bias-add), 플래그는 위와 동일. 대조군인
`bmm_pure_repro.onnx`(같은 shape, bias 없음)는 컴파일도 되고 **실행도 빠르고
정상**입니다 (numpy 레퍼런스 대비 corr 0.9999973), packet flow 자체가 필요 없는 케이스라서요.

## 조사: 6개 가설, 전부 실제 하드웨어 테스트로 반증

| # | 가설 | 테스트 | 결과 |
|---|---|---|---|
| 1 | arbiter-deadlock 컴파일 체크가 false positive라서 우회해도 안전함 | `--iree-amdaie-detect-arbiter-deadlock=false`로 우회 | 컴파일은 되지만 hang — **그리고 이 우회 플래그를 빼면 정확히 같은 "Potential arbiter deadlock detected" 에러가 재현됨**, 즉 이 체크는 이 케이스에선 진짜였음 (예전에 "confirmed false positive"라고 판단했던 다른 케이스와는 다름) |
| 2 | packet flow를 **core output** 쪽에 쓰면 deadlock 난다는 게 업스트림에 이미 문서화돼있음 ([nod-ai/iree-amd-aie#1148](https://github.com/nod-ai/iree-amd-aie/issues/1148), [PR #1170](https://github.com/nod-ai/iree-amd-aie/pull/1170)) — 우리가 쓰던 `--packet-flow-strategy=auto`는 input/output 구분을 안 함 | `--iree-amdaie-packet-flow-strategy=inputs`로 전환 | 컴파일 깨끗하게 통과, **arbiter-deadlock 우회 플래그가 아예 필요 없어짐** (원리적으로 맞는 fix) — **근데 런타임 hang은 바이트 단위로 똑같이 남음** |
| 3 | 코어 레벨 lock-release 레이스, 이미 고친 int8 batch-0 버그(`AMDAIECoreToStandard.cpp`, 10,000회 delay)와 같은 종류 | delay를 10,000 → 500,000 (50배)으로 증가 | 효과 없음 — 돌이켜보면 당연함: `bmm_pure_repro`가 이미 똑같은 코어 레벨 lock 메커니즘을 쓰면서 정상 작동하므로, 코어 자신의 lock은 애초에 의심 대상이 아니었음 |
| 4 | `AMDAIEInsertDmaBdChain.cpp`에 진짜 빈틈이 있음: packet flow를 "sequence point로 취급해서 열린 체인들을 다 끊어야 한다"는 로직이 `if (repeatCount > 1) return`이라는 이른 리턴 *뒤에* 있어서, `repeat_count>1`인 packet-flow DMA는 이 보호 로직을 완전히 건너뜀 | 픽스 구현 (체크를 앞으로 옮기고, repeat>1 op는 애초에 체인에 안 들어가니 무조건 다 끊도록) | 진짜 버그 맞지만 **이 dispatch의 컴파일된 IR에는 아무 영향 없음** (픽스 전후 `push_to_queue`/`tct_sync`/`write_bd`를 diff 떠보니 완전히 동일) — 이 dispatch는 그 시점에 끊어야 할 다른 열린 체인이 애초에 없었음. 되돌림 (효과 검증 안 된 상태라 upstream 안 함) |
| 5 | packet-flow DMA에 대해서만 `repeat_count=1`을 강제해서 깔끔한 A/B 테스트 (`AMDAIEDmaLoopSubsumption.cpp`의 `SubsumeLoopIntoDMA` 패턴을 packet-flow 커넥션에 대해 막기) | 시도함 | **이 파이프라인 단계에서는 구조적으로 불가능**: subsumption이 도는 `iree-amdaie-dma-composition`(`amdaie.npu.circular_dma_cpy_nd` op에서 작동)이 packet/circuit 여부를 결정하는 `AMDAIEAssignConnectionTypes`보다 *먼저* 실행됨 — 디버그 프린트로 확인: 이 시점엔 `connectionOp.getFlowOp()`가 항상 `nullopt`. pass 순서를 재구성하지 않고는 테스트 자체가 불가능 (범위 밖) |
| 6 | 이번 세션에서 반복된 TDR 타임아웃 ~6번이 amdxdna 드라이버/하드�웨어에 잔여 상태를 남김 | 사용자가 커널 모듈 리로드 (`sudo modprobe -r amdxdna && sudo modprobe amdxdna`) — 첫 시도는 조용히 실패함 (sudo 비밀번호 오류 + 팀원 컨테이너가 디바이스를 잡고 있었음 — `stat /dev/accel/accel0` 타임스탬프로 검증 필요, 이 공유 호스트에서는 "조용히 성공"을 절대 믿으면 안 됨), 두 번째 시도는 실제로 새로워진 디바이스 노드 타임스탬프로 확인됨 | **진짜로 새로워진 드라이버 상태에서도 동일하게 hang** — 세션 누적 오염 가설 배제 |

정적 분석/디스어셈블리로도 확인 가능한 건 다 봤는데 전부 깨끗했어요:
`10.executable-targets.mlir` phase 덤프엔 에러/TODO 마커 없음; 문제로 지목됐던 코어
타일의 실제 컴파일된 ELF를 디스어셈블(Peano 자체 툴체인의 `llvm-objdump`)해보니
`acq`/`rel` lock 명령어 카운트가 그 코어 자신의 프로그램 안에서 다 맞아떨어짐 (runaway
lock 없음); control code의 `npu.tct_sync`(완료 대기) 개수를 `npu.push_to_queue`(DMA
제출) 개수와 channel/column/direction 그룹별로 전부 대조 — **전부 정확히 일치**,
빠지거나 초과된 대기 없음; 이 dispatch는 `use_next_bd`/BD chaining을 아예 안 써서
깨진 체인이 원인일 수도 없음.

## 결정적 실험: AMD 자체 mlir-aie/IRON 툴체인으로 독립 재현

이 호스트에 이미 `mlir-aie` 저장소가 클론돼 있었어요 (`~/NPU/mlir-aie`, upstream
`Xilinx/mlir-aie` — IREE의 AIE dialect와 `aie-rt` 기반이 파생된 원조 프로젝트).
`ironenv`라는 Python venv에 실제 작동하는 툴체인(`aiecc`, `aie-opt`, `aie-translate`)이
이미 빌드/설치돼 있었고, `iree-amd-aie`의 컴파일러와는 완전히 독립적입니다. Peano는
이 저장소 것을(`llvm-aie/`) 재사용, XRT는 `/opt/xilinx/xrt` 것을 사용.

먼저 이 정확한 하드웨어에서 툴체인 자체가 작동하는지 sanity check — 둘 다 통과:
- `programming_examples/basic/passthrough_dmas` (순수 DMA passthrough): `PASS!`
- `programming_examples/basic/packet_switch` (packet flow, Strix 전용 lit 테스트로
  검증된 예제): `PASS!`

그 다음 `test/npu-xrt/objectfifo_repeat/simple_repeat/aie2.py` (기존에 있던,
XFAIL 아닌, 실제 테스트 — shim/mem tile 사이 plain `object_fifo`에
`.set_repeat_count(2)` 설정)를 `npu2`(mlir-aie에서 Strix/npu4를 가리키는 디바이스명)
타겟으로 수정해서 컴파일/실행. 같은 MLIR에서 packet flow 토글만 바꿔봤어요:

- **circuit flow** (`aiecc` 기본값): `PASS!`
- **packet flow** (`aiecc --packet-sw-objFifos`, repeat_count=2는 그대로, 나머지
  MLIR도 동일): 컴파일은 깨끗한데 **hang** — 90초 타임아웃 wrapper한테 강제 종료당함,
  커널 이름 출력 이후 아무 반응 없음. IREE 케이스와 완전히 같은 종류의 실패(TDR
  강제종료, `ert state 8`류)

이건 IREE 코드가 단 한 줄도 안 들어간, 처음부터 새로 짠 40줄짜리 최소 재현입니다.
재현 파일 전체는 `/tmp/claude-1003/.../scratchpad/mlir_aie_repro/`에 있어요
(세션 로컬이라 커밋 안 됨 — 재생성 방법은 맨 아래 "재현 방법" 참고).

(참고로 조사 중에 발견한 것: `test/npu-xrt/objectfifo_repeat/distribute_repeat/aie2.py`가
이 mlir-aie 체크아웃에서 `XFAIL: *`로 돼있는데, `repeat_count` + distribute 조합
관련이긴 하지만 이건 **컴파일 타임 호스트 사이드 segfault** 버그였어요
([Xilinx/mlir-aie#2470](https://github.com/Xilinx/mlir-aie/issues/2470), 나중에
PR #3028로 고쳐졌고 이 체크아웃엔 아직 반영 안 됨) — 우리가 보는 런타임 hang이랑은
다른 종류의 실패입니다. 같은 동네긴 한데 같은 버그는 아니니 헷갈리지 말 것.)

## 결론

독립적으로 개발된 두 개의 AIE 컴파일러(IREE 자체 AIE-dialect lowering, AMD 자체
mlir-aie/IRON lowering)가 정확히 같은 조건 — **packet-switched DMA 커넥션의 buffer
descriptor가 하드웨어 `repeat_count` 필드를 같이 쓰는 경우** — 에서 실제 npu4
하드웨어를 hang나게 하는 control code를 만듭니다. circuit flow + repeat, packet
flow (repeat 없이) 둘 다 두 툴체인 모두에서 정상 작동해요. 서로 무관한 두 코드베이스가
우연히 같은 버그를 만들었다고 보기보단, 훨씬 아래 레이어의 공유 원인일 가능성이 높습니다:

- 가장 유력하게는 진짜 npu4 실리콘/펌웨어 레벨의 한계 (stream-switch arbiter / TCT
  완료 토큰 메커니즘이 packet-routed BD의 하드웨어 auto-repeat를 제대로 못 다룸), 아니면
- `aie-rt` (IREE와 mlir-aie가 둘 다 그대로 가져다 쓰는, 저수준 BD/stream-switch
  레지스터 프로그래밍 라이브러리) 자체의 버그.

어느 쪽이든 **IREE 컴파일러 버그는 아닙니다**, 그리고 앞으로 `iree-amd-aie` 소스를
더 파는 건 생산적이지 않을 것 같아요. 이 과정에서 발견하고 고친 진짜, 독립적으로
가치 있는 것 두 개는 이 결론과 별개로 유지할 가치가 있어요: `packet-flow-strategy=inputs`
fix (원리적으로 맞는 fix, 문서화된 output-packet-flow 위험을 실제로 회피함)랑,
`AMDAIEInsertDmaBdChain.cpp`의 `repeat_count>1` packet flow에 대한 chain-ordering
빈틈 (진짜 버그, 이 dispatch엔 효과 없었지만 아직 upstream 안 함).

## 권장 사항

roadmap item 2 한정으로는: 하드웨어/펌웨어 fix를 쫓기보다는 **이 조합 자체를
피하는** 게 맞아 보여요. 컴파일 타임 작업으로 이미 bias 브로드캐스트를 실제 operand
레이아웃이랑 똑같이 컬럼당 하나의 커넥션으로 만들어놨으니, 남은 선택지는:
1. bias 전달은 `packet-flow-strategy=inputs`로 유지하되, bias 커넥션에 한해서만
   `repeat_count>1`을 피하는 방법 찾기 (예: packet-routed 커넥션에 한해 batch 차원을
   하드웨어 repeat로 접지 않고 별도 BD push로 유지) — 위 가설 #5가 구조적으로
   시도하려던 게 이거고, `AMDAIEDmaLoopSubsumption`에서는 못 닿았음; guard를
   `AMDAIEAssignConnectionTypes` 이후로 옮기거나 flow 타입 결정을 더 앞당겨야 함.
2. 위의 mlir-aie 최소 재현을 들고 AMD 쪽(aie-rt/실리콘 관계 담당자)에 에스컬레이션 —
   IREE와 완전히 무관한, 이 이상 깔끔할 수 없는 이슈 리포트임.

## 재현 방법

**IREE 쪽** (dev container 안, `./scripts/docker/run-debug.sh`):
```
build/tools/iree-compile _local/int8_debug/out/bmm_bias_repro.mlir \
  -o /tmp/bmm_bias_repro.vmfb \
  --iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-amdaie-demote-contraction-inputs-to-bf16 --iree-amdaie-enable-vectorization-passes=false \
  --iree-flow-enable-executable-deduplication=false --iree-flow-inline-constants-max-byte-length=0 \
  --iree-amdaie-packet-flow-strategy=inputs
# 이후 iree-run-module (또는 scripts/debug/pipeline_dump.py) --device=amdxdna --device=local-task
```

**mlir-aie 쪽** (호스트, dev container 아님):
```bash
source ~/NPU/mlir-aie/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
export PEANO_INSTALL_DIR=~/Projects/iree-amd-aie-vgg16/llvm-aie
# aie2.py: test/npu-xrt/objectfifo_repeat/simple_repeat/aie2.py를 복사해서
# dev = AIEDevice.npu2, memtile_repeat_count = 2 로 수정
python3 aie2.py 4096 > aie2.mlir
aiecc --no-aiesim --no-xchesscc --no-xbridge --packet-sw-objFifos \
  --aie-generate-npu-insts --aie-generate-xclbin --no-compile-host \
  --xclbin-name=final.xclbin --npu-insts-name=insts.bin aie2.mlir
# test.cpp는 같은 simple_repeat 디렉토리 것을 XRT 링크해서 빌드, 그 다음:
timeout 90 ./test.exe -x final.xclbin -i insts.bin -k MLIR_AIE -l 4096 -r 2
# hang (exit 124). --packet-sw-objFifos 빼면 PASS로 통과함.
```

## 파일별 변경사항 (이번 컴파일 타임 fix 라운드, 전부 커밋 완료)

| 파일 | 변경 내용 | 커밋 |
|---|---|---|
| `compiler/plugins/target/AMD-AIE/aie/AMDAIECoreToStandard.cpp` | int8 batch matmul의 batch-0 lock-precharge race 픽스 (10,000회 delay, 이 문서 주제와는 별개 작업) | `1600078` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEFoldBroadcastAddIntoDest.cpp` | **신규 파일**. bias를 별도 elementwise dispatch가 아니라 matmul의 accumulator 초기값으로 fold하는 새 pass (bug #1~7의 전제였던 "별도 consumer" 구조 자체를 제거) | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEEraseDeadAllocAndStores.cpp` | **신규 파일**. 마지막 컴파일 블로커(`aie.device` multi-block) 픽스 — fuse 안 된 producer의 dead alloc을 제거 | `ee37c48`, 매크로 누락 수정은 `552c8c2` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/PluginRegistration.cpp` | `AMDAIEFoldBroadcastAddIntoDest`를 preprocessing 파이프라인에 wiring | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEBufferizeToAllocation.cpp` | bug #2: `tensor.empty()` 초기화에 dummy `linalg.fill` 삽입 (upstream bufferizeToAllocation이 요구하는 단일 aliasing operand 조건 충족용) | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEDistributeL1Allocations.cpp` | bug #3: `TypeSwitch` 케이스를 `linalg::GenericOp`에서 `linalg::LinalgOp` 인터페이스로 일반화 | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEConvertToDma.cpp` | bug #4: `bufferization::ToBufferOp`를 "offset 없는 전체 범위" 입력으로 인식하도록 추가 | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/Utils/AMDAIELogicalObjFifoSplittingUtils.cpp` | bug #5: rank-4 하드코딩된 `transposedL2Dims`를 `getTransposedL2Dims(rank)`로 일반화 + source/target rank 불일치 시 우측 정렬 인덱스 변환. 이후 bug #8 관련해서 `splitLogicalObjectFifoForElementwiseOp`를 offset별 그룹핑으로 재작성 (8-column 브로드캐스트를 실제 operand처럼 컬럼당 버퍼 하나로) | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEAssignTiles.cpp` | bug #6: `uniqueL3L2Pair` truncation을 `priorityCol != -1`일 때만 적용하도록 gate + 같은 memref를 공유하는 `LogicalObjectFifoFromMemrefOp` clone들 사이에 priority column을 전파하는 pre-pass 추가 | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIENpuDmaToHalfDmaCpyNd.cpp` | bug #7: 원본 op에서 실제로 채워진 쪽(source/target)에 대해서만 half-DMA op를 생성 (phantom half-DMA 방지) | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEControlCodeLowering.cpp` | bug #8b 연장선: `insertWriteBdOps`에서 `Buffer_Length=0`이면 `Enable_Packet`도 강제로 0으로 설정 (arch spec 3.7.8 요구사항) | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIELowerToAIE.cpp` | 별개의 latent 버그: `workgroupToAIE`/`lowerToAIE`의 PreOrder walk가 `rewriter.clone` 이후 `WalkResult::advance()`를 반환해서 중첩 자식을 중복 clone하던 것을, `WalkResult::skip()`으로 수정 (2곳) | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Target/AIETarget.h`, `AIETarget.cpp` | `--iree-amdaie-detect-arbiter-deadlock` 플래그 추가 (기본값 true) | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/Passes.h`, `Passes.td`, `PassDetail.h`, `CMakeLists.txt` | 신규 pass 2개(`AMDAIEFoldBroadcastAddIntoDest`, `AMDAIEEraseDeadAllocAndStores`) 등록/wiring. `PassDetail.h`의 `GEN_PASS_DEF_AMDAIEERASEDEADALLOCANDSTORES` 매크로 누락은 별도 커밋에서 수정 | `ee37c48` (누락 매크로는 `552c8c2`) |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/Passes.cpp` | 위 신규 pass 2개를 실제 파이프라인에 삽입 (vectorization 이후, `SCFToControlFlowPass` 직전 등) | `ee37c48` |
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIETileAndFuse.cpp` | `linalg.broadcast`를 dest-producer로 허용하는 fusion 확장 (테스트 결과 이 fix에 필수는 아니었던 걸로 확인됐지만, 무해해서 그대로 유지) | `ee37c48` |
| `docs/2026-08-16_frontend_lowering_passes.md`, `docs/2026-08-17_pass_examples.md` | 이번 세션 주제와 무관한 프론트엔드 로워링 패스 조사 문서 | `1a1ec51` |

이번 hang 조사 자체는 **코드 변경이 없습니다** (시도했던 실험적 픽스 2개는 효과 검증
안 돼서 전부 되돌림 — `AMDAIEInsertDmaBdChain.cpp`, `AMDAIEDmaLoopSubsumption.cpp`).
`AMDAIECreatePathFindFlows.cpp`에 디버그용 `llvm::errs()` 프린트가 몇 줄
uncommitted로 남아있는데(stderr 출력만 하는 무해한 코드), 다음에 이 조사를 재개할 게
아니라면 정리하고 커밋하는 걸 권장합니다.
