# 네이티브 conv 파이프라인 구현 기록

> 작성 2026-11-08 · 브랜치 `conv-npu`(= origin/dev `786d1d19`, 미커밋) · 타겟 npu4 · 계획 [1103_conv_pipeline_tasks.md](1103_conv_pipeline_tasks.md)
>
> 요청: "계획성을 바탕으로 구현을 진행해봐."
>
> 검증 정책: 컴파일 · 정적 IR · lit/ctest 만. NPU 에서는 아무것도 실행하지 않았다. 수치 정확도는 검증하지 않았다.

origin/dev 는 11-08 기준 `f51ef857` 로 13 커밋 앞서 있지만 모두 Vulkan·GPU 쪽(IREE submodule bump, GPU→CPU link)이라 conv 경로와 무관해 기준은 `786d1d19` 로 유지했다.

---

## 1. 진행 요약

(작업이 끝날 때마다 갱신)

| 단계 | 작업 | 상태 | 근거 |
|---|---|---|---|
| S0 | 반증 스파이크 T0.4·T0.6(a)·T0.7, T3.0 사전 확인, dev-conv 이식성 | 완료 | §3 |
| S1 | 도구: `--conv` 스위치(T1.3 일부), 계산기(T0.3), 정적 감사(T0.2), 원소 단위 매핑 검사 | 완료 | §2, §5.4, §7.2 |
| S2–S4 | conv 인식(T2.2)·`conv` 파이프라인 override(T2.1)·`chooseConvTiling`(T2.3), 수직 슬라이스(T3.1)와 결함 수정(T3.2) | 완료(컴파일 전용) | §4, §5 |
| S5 | conv bias fold(T5.1), 코어 내 ReLU epilogue(T5.2 일부) | 완료(컴파일 전용) | §6 |
| S5 | Flow 단계 halo 청크·FHWC 필터(T2.4 일부) → VGG conv 13 개 전 배열(T3.3·T3.5 의 컴파일 측) | 완료(컴파일 전용) | §7.1–§7.3 |
| S5 | split 패스 IV 가드(T2.5 일부) | 완료 | §7.4 |
| — | 독립 리뷰 2 차례(지적 16 건 수정), 검사 도구 강화(다중 블록 매핑 검사, audit 의 BD 한계·lock 검사) | 완료 | §7.6 |
| S5 | 출력 join·필터 fan-out 의 공유 카운팅 lock 제거(T3.3 의 lock 항목) → VGG conv 13 개 감사·매핑 PASS | 완료(컴파일 전용) | §7.7 |
| — | 세 번째 독립 리뷰(지적 11 건)와 그 수정의 반증 검토 3 회(지적 14 건): 셔임 4 B 단위 위반을 컴파일 오류로(word 재접기 포함), conv epilogue 진단, 검사 도구 강화 | 완료 | §7.8 |
| S5 | OH 패딩(T2.4 일부): 청크당 최소 행 수로 청크를 고르고, 남는 행은 패딩 후 잘라냄. CPU 수치 비교로 Flow 재작성 확인(VGG 포함) | 완료(NPU 미실행) | §7.9 |
| — | T2.4 나머지(OH 패딩, 청크 우선 레이아웃), T2.5 나머지(fail-fast 배치), 커널 경로(T4.x) | 남음 | §8 |
| 11-09 | 첫 NPU 실측(스칼라 코어): VGG-16 corr 0.99999·top-1 일치, run 5.83 s(im2col 0.388 s), 프로브 4 개 corr ≥ 0.99999 | 완료 | [1109_native_conv_npu_run.md](1109_native_conv_npu_run.md) |

---

## 2. 도구 (T0.2, T0.3, T1.3)

### 2.1 `--conv {im2col,native}` (T1.3 일부)

`scripts/onnx/options.py` 의 graph 그룹에서 `--iree-global-opt-use-im2col-for-convs` 를 고정 플래그에서 사용자 선택으로 옮겼다. 기본은 `im2col` 이라 기본 컴파일 플래그는 그대로다(순서만 바뀜). 출력 디렉터리 이름에 `conv_<선택>` 이 붙는다. `native` 는 처음에 Flow 까지만 봤고, §6.3 부터 끝까지 컴파일된다.

변경 전:

```python
    "graph": {
        "flags": [
            "--iree-preprocessing-pass-pipeline="
            "builtin.module(iree-preprocessing-convert-conv-to-channels-last)",
            "--iree-global-opt-use-im2col-for-convs",
            # NHWC max pooling, like the convs, so the transposes around each
            # pool cancel (an NCHW pool on the CPU reads through one).
            "--iree-amdaie-pooling-channels-last",
        ],
    },
```

변경 후:

```python
    "graph": {
        "flags": [
            "--iree-preprocessing-pass-pipeline="
            "builtin.module(iree-preprocessing-convert-conv-to-channels-last)",
            # NHWC max pooling, like the convs, so the transposes around each
            # pool cancel (an NCHW pool on the CPU reads through one).
            "--iree-amdaie-pooling-channels-last",
        ],
        # User choice: how a convolution reaches the accelerators. im2col turns
        # it into a CPU gather and a matmul; native keeps it one conv dispatch
        # for the AIE conv pipeline (in development: compile with
        # --extra-flag=--compile-to=flow until that pipeline lands).
        "conv": {
            "im2col": [
                "--iree-global-opt-use-im2col-for-convs",
            ],
            "native": [],
        },
    },
```

`DEFAULTS` 에 `"conv": "im2col"` 을 더했다. `scripts/onnx/README.md` 와 [EXPERIMENT_ENV.md](../docs/EXPERIMENT_ENV.md) 의 옵션 표도 고쳤다.

### 2.2 `scripts/debug/conv_tile_calc.py` (T0.3, 신규)

변경 전 — 없음(신규 추가). §5.0 의 L1 식(버퍼별 64 B 정렬, depth, 코어 내 epilogue 항), MAC/B, BD 차원, BD 전송 4 B 배수, IC 청크 반복 ≤ 63, L3 연속 구간, memtile L2, kw 시프트 64 B 정렬을 층·격자(g2 / g1 / single)별로 계산한다. `--one-level` 은 IC 청크를 한 단계(L2 청크 = L1 청크)로 묶는다(GEMV 구조 파이프라인의 경우).

`--selftest` 는 실측 세 값(52,480 / 105,984 / 105,728 B), 가중치 슬랩 73,728 B, 오늘 타일의 2.8 MAC/B, 계획서 §5.0 표의 다섯 층(A 안) 값을 단언한다. 모두 통과했다.

| 층 | 계획서 표(A 안) | 계산기 g2 |
|---|---|---|
| conv1_x | 28·16·8·8, 48,256 B, 26.4 | 같음 |
| conv2_x | 14·16·16·16, 56,320 B, 37.3 | 같음 |
| conv3_x | 14·8·32·16, 57,344 B, 36.0 | 같음 |
| conv4_x | 7·32·16·16, 57,472 B, 35.8 | 7·16·32·16, 같은 L1·MAC/B(동률) |
| conv5_x | 4·16·64·8, 54,656 B, 26.9 | 같음. 단 IC 청크 64 개로 반복 한도 63 초과 → 두 단계 청크 필요 |

`--one-level` 이면 conv5_x 는 4·16·32·16(41,728 B, 23.3 MAC/B, 청크 32 개)로 바뀐다. 모든 층에서 kw 시프트 읽기는 64 B 정렬이 아니다(§3.2 와 같은 결론).

### 2.3 `scripts/debug/aie_audit.py` (T0.2, 신규)

변경 전 — 없음(신규 추가). `--print-flags` 가 내는 다섯 덤프(assign-channels 뒤, assign-npu-dma-bd-ids 뒤, lower-to-aie 뒤, amdaie-assign-bd-ids 앞, amdaie-assign-buffer-addresses 뒤)를 읽어 dispatch 별로 타일당 DMA 채널(shim 2/2·memtile 6/6·코어 2/2), connection 의 타겟 수(broadcast), shim 전송의 크기 1 이 아닌 차원 수(≤ 4)와 바깥 반복(≤ 63, stride 0 이면 ≤ 256), memtile BD 수(≤ 48)와 BD 차원(코어 3, memtile 4)·크기 < 1023·stride 0 금지·innermost stride·전송 길이 4 B 배수, 코어 L1 = max(address + 크기) ≤ 65,536, 64 B 이상 버퍼의 64 B 정렬을 검사한다. 알려진 실패 진단도 위반으로 센다. 덤프 헤더가 없으면 실패한다.

골든(npu4, 컴파일 전용):

| 입력 | 결과 |
|---|---|
| bf16 matmul 512³(pack-peel, ukernel all, 벡터화 off) | PASS: 코어 32·memtile 8·열 8, memtile S2MM 6/6·MM2S 3/6, connection 타겟 8 개 4 건·4 개 8 건(행·열 broadcast), L1 57,344 B |
| conv-decompose, 2×18×18×32 → 2×16×16×64 bf16 | PASS: 코어 4·열 1, memtile S2MM·MM2S 6/6, L1 14,592 B |
| conv-decompose, 2×34×34×256 → 2×32×32×256 bf16 | FAIL(기대대로): `allocated buffers exceeded available memory (105984>65536)`, memtile BD 60 > 48 |

처음 판에서 shim 전송의 크기 1 차원까지 세어 정상 컴파일을 위반으로 잡았다. 크기 1 차원은 접히므로 그 뒤로는 크기가 1보다 큰 차원만 센다.

부수 관찰(컴파일 전용):

- batch 1 conv(1×30×30×256)는 conv-decompose 를 지정해도 `Unhandled pass pipeline in setRootConfig` 로 실패한다. 단위 batch 가 접힌 6-D generic 이 되어 7 반복자를 요구하는 인식에서 빠지기 때문이다(G04). CI conv 테스트가 batch 2 를 쓰는 이유다.
- CI 형태 `2×14×14×32` 는 npu4 에서 OW=12 가 OW 타일 8 로 나뉘지 않아 `memref.subview ... dynamic shape` 로 실패한다.
- 벡터화를 켠 bf16 matmul(ukernel none)은 npu4 에서 `vector.contract` 를 합법화하지 못해 실패한다(C6).

11-08 리뷰 뒤 바뀐 점(§7.6 #13·#15):
- 덤프가 빠진 dispatch 는 위반입니다. 위의 "덤프 헤더가 없으면 실패" 는 처음에는 로그 전체에만 적용됐습니다.
- BD 한계는 "크기 < 1023" 이 아니라 타일별 wrap·step 의 32-bit word 값(aie-rt AIE2P 표)입니다.
- shim 의 stride-0 반복 규칙을 고쳤습니다.
- 공유 카운팅 lock(`AcquireGreaterEqual N > 1`)을 위반으로 셉니다.

세 번째 리뷰와 그 수정 검토 뒤 바뀐 점(§7.8):
- 모든 error 진단이 위반이다. 덤프 전에 실패한 dispatch 를 놓치지 않기 위해서다. 다만 bank-aware 배치가 순차 배치로 복구한 오류는 따로 센다. 판단 기준은 오류 문구가 아니라, note 줄 바로 뒤의 순차 배치 경고다.
- IR 이 없는 덤프와 코어가 0 인 dispatch 도 위반이다.
- 셔임 전송은 하강이 접는 차원 그대로 본다. 크기 1 차원을 빼고, 이어진 차원을 한계 안에서 합친다. 자리별 한계는 다음과 같다.
  - D0·D1 의 wrap ≤ 1023, D2 ≤ 1024(필드 없음, aie-rt 한계)
  - iteration ≤ 64, 반복 횟수 ≤ 256
- 원소가 4 B 보다 좁은 셔임 전송은 시작·바깥 stride·안쪽 길이가 word 여야 하고, 안쪽 stride 는 1 이어야 한다. 하강처럼, 안 맞으면 안쪽 차원을 word 단위로 다시 접어 본다.
- 코어·memtile BD 는 바깥 stride 와 안쪽 크기가 4 B 배수여야 한다(런타임 `configureDMABD` 규칙).
- lock 의 init 이 그 lock 을 쓰는 DMA BD 가 지키는 버퍼 수보다 크면 위반이다. 다른 쪽이 코어인 공유 카운팅 쌍을 잡는다.
- compile 종료 코드는 이 도구가 보지 못하므로 따로 확인한다.

### 2.4 ctest 기준선

`ctest -R amd-aie` 는 232 개 중 229 개 통과였고, 실패 3 개(`ctrlpkt_gen`, `elf_pm_size`, `convert_device_to_control_packets`)는 `PEANO_INSTALL_DIR` 가 없어서였다. `-e PEANO_INSTALL_DIR=/workspace/llvm-aie` 를 주면 3 개도 통과해 **232/232** 다. 이후 ctest 는 이 변수를 넣어 돌린다.

## 3. S0 반증 스파이크

재빌드 없이 현재 바이너리(11-03 빌드)로 돌렸다. 스파이크 4 개는 각각 독립 검증자가 다시 실행해 반박을 시도했다(결과는 §3.6). 산출물은 `log_shjo/s0/<스파이크>/` 에 있다.

### 3.1 T0.4 — origin/dev 레시피에서 im2col 만 뺀 VGG 의 Flow

```bash
./scripts/docker/run-onnx.sh compile models/vgg16/vgg16-12.onnx --numeric bf16 --conv native \
  --outdir build/onnx/vgg16-12-conv_native-flow --extra-flag=--compile-to=flow \
  --extra-flag=--mlir-elide-elementsattrs-if-larger=8 --extra-flag=--mlir-elide-resource-strings-if-larger=8
```

| 항목 | `--conv native` | `--conv im2col` |
|---|---|---|
| flow.dispatch | 42 | 87 |
| NPU / CPU | 16 / 26 | 16 / 71 |
| NPU 의 내용 | conv 13 + FC 3 | conv matmul 13 + FC 3 |

- conv 13 개가 모두 NPU dispatch 하나씩이고, root 는 batch 가 접힌 6-loop generic `[P,P,P,R,R,R]` 이다. 맵은 입력 `(d0+d3, d1+d4, d5)`, 가중치 `(d3, d4, d5, d2)`(HWCF), 출력 `(d0, d1, d2)` 이다. 입력은 이미 `(OH+2)×(OW+2)×IC` bf16 로 패딩돼 있고 가중치는 const-eval global 의 bf16, 누산은 `fill(0)` 의 f32 다.
- bias 덧셈과 ReLU 는 epilogue generic 하나로 **같은 dispatch 에 융합**된다. truncf 는 융합되지 않아 출력은 f32 다.
- bias 는 13 개 중 11 개에서 세 번째 읽기 전용 binding(`tensor<OC×f32>`)이다. conv1_x 두 층은 256 B 이하라 dispatch 안의 `arith.constant` 로 들어온다.
- 13 개 중 8 개는 출력이 다음 층의 패딩 버퍼에 묶인(tied) readwrite binding 이고 `[1, 1, 0]` 오프셋에 쓴다. pool 앞의 5 개는 writeonly 다.
- bias fold 패스와 pad 패스는 conv dispatch 를 건드리지 않았다(둘 다 `linalg.matmul` 하나만 찾는다). 진단 0 건.

검증자는 덤프를 다시 파싱하고 소스 인용을 대조해 실질 주장을 하나도 반박하지 못했다(VGG 재컴파일 없이). 덧붙인 것: `--iree-flow-inline-constants-max-byte-length=128` 이면 conv1_x 의 bias 도 binding 이 되어 13 층이 같은 형태가 된다(2-conv 프로브로 확인). fold 패스의 bias 맵 판정(`isPerColumnMap`)은 2-D 맵만 받아 conv epilogue 의 3-D bias 맵 `(d0,d1,d2)->(d2)` 와 맞지 않는다.

→ 네이티브 conv 를 VGG 에 쓰려면 conv bias fold(T5.1)가 tied 결과와 inline bias 를 다뤄야 하고, ReLU 는 코어 내 epilogue(T5.2)로 가야 한다. 첫 수직 슬라이스는 epilogue 없는 conv 로 한다.

### 3.2 T0.6 (a) — aievec kw 시프트 정렬: **no-go**

- 레이아웃을 바꾸지 않는 aievec 경로는 쓸 수 없다. 이 빌드에서 bf16 `vector.contract`(lhs/rhs `vector<8x8xbf16>`)는 npu4 에서 aievec 로 내려가지 않는다(지원 형은 i8 8×8×8 뿐). OW=16 으로 바꾼 conv-decompose 컴파일이 `test-lower-vector-to-aievec` 에서 멈춘다.
- 그 전의 `align-transfer-reads` 는 "can't be aligned with a read twice as large" 경고를 8 번 냈다. 입력 1024-bit 읽기 오프셋은 `640·kh + 160·ic_o + 16·kw` B 로, 출력 타일당 36 개 중 12 개만 64 B 정렬이다(kw=1 은 0/12). 가중치와 누산기 읽기는 모두 64 B 정렬이다.
- 이 패스는 32 B 정렬만 보므로 i8 대조군에서 "맞춘" 읽기도 18/36 이 32 mod 64 에 남는다.
- → D5: 커널 경로는 kw 복제 64 B 레이아웃, conv ukernel(정렬 로드 + 레지스터 시프트), IGEMM 중에서 고른다.

### 3.3 T0.7 — G2 objectFifo·채널: **go**(IC 청크 1 개 기준)

- 손으로 쓴 G2 IR 이 distribute → split → assign-channels 를 rc 0 으로 통과했다. 가중치 split factor 8, 열 c 의 가중치 objFifo 는 조각 c(L3 오프셋 8c)를 읽는다. 활성값 청크는 타겟 8 개 connection 하나(행마다 1 개, 4 개), 가중치 슬랩은 타겟 4 개 connection 하나(8 개)다. memtile 최대 S2MM 6/6·MM2S 3/6, shim 열당 MM2S 2 다. 대조군 pack-peel bf16 512³ matmul 과 타일별 채널 분포가 같다.
- 조건 1: split 패스는 전역 파이프라인이 pack-peel·pack-peel-4-level·general-copy 일 때만 돈다(`Passes.cpp:1011-1014`). dispatch 별 override 로 conv 를 넣으면 전역은 pack-peel 이라 돈다.
- 조건 2: L2→L1 가중치 접근에서 OC 블록 오프셋이 stride `KH·KW·IC_c` 인 차원에 있어야 한다(FHWC). 같은 원소를 읽되 오프셋을 stride 576 의 앞 차원에 둔 변형은 모든 열이 조각 0 을 가리키고 `no producer DMA channel available` 로 실패했다. → HWCF 가중치를 열로 나누면 같은 실패가 날 수 있다(T2.4 의 FHWC 사전 pack 이 필요한 이유).
- 조건 3: 여유가 없다(memtile 0–3 S2MM 6/6, shim 0–3 MM2S 2/2, 코어 S2MM 2/2). control-packet 오버레이를 켜면 matmul 과 같이 실패한다.

### 3.4 dev-conv 지식 이식 가능성

- 코드 커밋 11 개 중 그대로 적용되는 것은 `855c3168`(BD 수정), `780c4acd`(depth 1), `99a90f92`(뒤 커밋이 상쇄)뿐이다. 6-D 인식(`0812bb51`)과 라우팅(`93d66148`)은 서로 얽혀 있고 origin/dev 의 override 구조와 충돌한다. Flow 패딩 3 커밋은 구조가 바뀌어 이식해야 한다.
- T3.0 의 패스 수정 4 건(DistributeL1Allocations, InsertCores, PeelForLoop only-outermost, FuseConsumerIntoLoop affine.min)은 대상 파일이 그대로라 모두 적용된다(`log_shjo/s0/devconv/t30_pass_fixes.patch`). 다만 InsertCores 수정은 matmul lit 2 곳의 기대값을 바꾸고, FuseConsumerIntoLoop 수정은 lit 4 곳을 바꾼다.
- `855c3168` 은 텍스트로는 적용되지만 gtest `AMDAIEDmaUtilsTest.FoldRepetitionCountFail` 의 단언 8 개 중 4 개와 어긋날 것으로 보인다(파이썬 모델 추정, 미실행).
- dev-conv 의 실패 사슬 대부분은 pack/unpack 과, fill 과 코어 forall 사이에 끼인 KH·KW 루프에서 나왔다. 이번 설계(pack 없음, IC 루프 하나)에서는 일부가 생기지 않을 수 있다.

### 3.5 T3.0 사전 확인 — GEMV 패스 순서를 conv 에 그대로 적용

- GEMV 파이프라인의 패스 순서(48 개)는 손으로 쓴 6-D conv generic 과 7-D 청크 변형에서 rc 0 으로 돈다. 블록 forall, L3→L2 복사, 첫·마지막을 떼어 낸 IC 청크 `scf.for`, 코어 forall, L2→L1 복사, 첫 청크의 fill, 마지막 청크의 write-out 이 계획대로 나온다.
- 첫 실패는 모두 뒤의 objectFifo 단계에 있고, dev-conv 의 T3.0 수정 4 건 중 어느 것도 아니다. (F1) 6-D 형태를 2-D 코어 격자에 두면 halo 가 겹치는 입력이 1 행 조각 6 개로, HWCF 가중치가 KH 조각 3 개로 잘못 나뉘고 `no producer DMA channel available` 로 실패한다. (F2) split 을 끄면 출력 join 이 16 producer 가 된다. (F3) GEMV 의 버퍼화 콜백은 4-D 누산기를 L2 에 둔다. (F4) 7-D 에서 N 을 행에 두면 출력 join 이 8 producer 가 되어 `no consumer DMA channel available` 이다.
- 입력만 바꿔(열 1 개, 또는 N 청크를 열에) 32 코어·8 memtile 까지 Peano 직렬화를 통과했다.
- 검증자는 결과를 모두 재현했다. 정정: 7-D 변형에서 누산기가 L1 에 남는다는 판정 문구는 틀렸다(F3 그대로 L2). halo 입력의 잘못된 분할은 HWCF/FHWC 와 무관한 창 겹침 문제(G08)다. split factor 의 열 수는 target 의 8 이 아니라 코어가 쓰는 열 수다(`AMDAIESplitLogicalObjFifos.cpp:399-401`). 그래서 열 1 개면 아무것도 나뉘지 않는다.

### 3.6 독립 검증 결과

| 스파이크 | 재현 | 판정 유지 | 정정·추가 |
|---|---|---|---|
| T0.4 | 예 | 예 | inline 상수 한도 플래그로 conv1_x bias 를 binding 으로 만들 수 있다. fold 의 bias 맵 판정은 2-D 전용 |
| T0.6 (a) | 예 | 예 | no-go 의 근거는 정렬 결과(3)(4)이고, (1) OW=12 실패와 (2) bf16 contract 미지원은 레이아웃과 무관하다. kw=2 의 어긋남은 짝수 ic_o 에서 생긴다 |
| T0.7 | 예 | 예(좁게) | go 는 split 되는 모든 L2 버퍼의 첫 비단위 차원이 열 인덱스일 때만 성립한다(FHWC 가중치, `[c][r][…]` 출력, 호스트에서 미리 나눈 입력). NHWC 출력이나 겹치는 halo 창은 split 을 깨고, 열 4 개에서는 오류 없이 틀릴 수 있다. 컴파일 성공만 믿지 말고 열→조각 매핑을 확인하라고 권했다 |
| T3.0 사전 | 예 | 예 | 위 §3.5 정정 |

## 4. 구현 — 네이티브 conv 타일 파이프라인 v1 (T2.1·T2.2·T2.3 일부, T3.1)

### 4.1 설계

GEMV 파이프라인의 구조를 그대로 쓰는 복사 기반 파이프라인이다. pack 은 없고 KH·KW 는 코어 op 안에 남는다.

```
block forall (출력 타일)                         ← L3 → L2: 입력 청크(halo 포함), 필터 청크
  scf.for (입력 채널 청크, 첫·마지막은 peel)
    core forall (청크 → 열, 출력 채널 → 행)       ← L2 → L1: 코어의 청크, 코어의 OC 타일
      fill(첫 청크) ; conv generic ; write-out(마지막 청크, L1 → L2)
  L2 → L3: 블록 출력
```

- **경로**: GEMV·pool 과 같은 dispatch 별 override 다. 루트 설정이 `translation_info` 에 `amdaie.tile_pipeline = "conv"` 를 붙이고, `AMDAIELowerExecutableTarget` 이 `addConvPassPipeline` 을 고른다. 전역 파이프라인은 pack-peel 그대로라 objectFifo 단계의 split 패스도 돈다.
- **켜는 법**: `--iree-amdaie-enable-native-conv`(기본 off). target config 에 `native_conv = true` 를 넣고, 꺼져 있으면 키 자체가 없어 기존 target attr 문자열이 바뀌지 않는다. 전역 파이프라인이 conv-decompose 면 conv 는 그대로 conv-decompose 를 탄다(D11 공존 규칙).
- **인식(`getConvDims`)**: 풀링 판정(`getMaxPoolingDims`)처럼 맵으로 본다. 출력 `(n?, oh, ow, oc)` 항등, 입력 `(n?, oh + kh, ow + kw, ic)`(stride·dilation 1), 필터 `(kh, kw, ic, oc)` 또는 `(oc, kh, kw, ic)`, 본문은 곱셈-누적. 명명 conv 와 generic, batch 가 접힌 6-D 와 청크가 있는 7-D 를 모두 받는다.
- **타일(`chooseConvTiling`)**: 코어당 버퍼 세 개(halo 포함 입력 청크, 필터 청크, f32 누산기)를 depth 2·64 B 정렬로 더해 `L1 − 스택` 이하인 것 중에서, 코어 수 최대 → IC 청크 반복 ≤ 63 → MAC/B 최대 → IC 청크 큰 것 순으로 고른다. IC 청크는 8 의 배수(IC < 8 이면 전체)이고 전송은 4 B 배수, 코어당 출력은 2 개 이상이다.
- **격자**: 청크(n)를 **열**에, 출력 채널을 **행**에 둔다. 계획서의 G2(OC 를 열에)는 아래 §4.3 의 이유로 G1 방향으로 바꿨다. 청크 차원이 없으면(batch 가 접힌 6-D) 출력 채널만 한 열의 행 4 개에 나눈다.
- **HWCF 필터 규칙**: 청크가 여러 열에 걸치면 HWCF 필터는 행으로 나누지 않는다(모든 코어가 블록의 필터 타일을 함께 읽는다). §4.3 참고.

### 4.2 코드 변경

| 파일 | 변경 | 분류 |
|---|---|---|
| `Transforms/KernelDispatch.h` | `kConvPipelineName = "conv"` | 추가 |
| `Transforms/Utils/AMDAIEUtils.{h,cpp}` | `ConvDims`·`getConvDims`, `ConvTiling`·`chooseConvTiling`, `getConfigNativeConv`, `alignedCoreBufferBytes`, `<tuple>` include | 추가 |
| `Transforms/KernelDispatch.cpp` | `setRootConfigForConvPipeline`, `takesConvPipeline`; generic·명명 conv 루트 설정에서 이 경로를 먼저 본다; `setConvRootConfig` 가 numRows·numCols·stackSize 를 받는다 | 추가·변경 |
| `Transforms/Passes.{h,cpp}` | `addConvPassPipeline` | 추가 |
| `Transforms/AMDAIELowerExecutableTarget.cpp` | override `"conv"` 분기 | 추가 |
| `Target/AIETarget.{h,cpp}` | `enableNativeConv` 옵션과 `--iree-amdaie-enable-native-conv`, 켜졌을 때만 `native_conv` config | 추가 |
| `Transforms/test/lowering_strategy_conv_native.mlir` | 새 lit(7-D FHWC, 명명 HWCF, 6-D HWCF 의 타일과 override) | 추가 |
| `Transforms/test/conv_pipeline.mlir` | 새 lit(파이프라인 구조) | 추가 |
| `Transforms/test/lowering_strategy_conv.mlir` | `native_conv` 가 있어도 전역 conv-decompose 가 conv 를 가져가는 사례 | 변경(사례 추가) |
| `Test/samples/conv_native_objectfifo_e2e.mlir` | 새 샘플(8 청크 conv → 코어 32 개) | 추가 |
| 두 `CMakeLists.txt` | 새 lit 등록 | 변경 |
| 그 밖의 패스 | — | 유지 |

계획서 T3.0 의 dev-conv 패스 수정 4 건은 이 설계에서 필요하지 않아 넣지 않았다(pack 이 없고 IC 루프가 하나뿐이다). 코어 안에 KH·KW `scf.for` 를 두는 벡터화 단계에서는 PeelForLoop only-outermost 와 InsertCores 수정이 필요해진다(§3.5).

변경 전 코드(override 분기, `setConvRootConfig` 머리, 전역 conv 판정)는 다음과 같다.

```cpp
// AMDAIELowerExecutableTarget.cpp (변경 전)
        if (pipelineOverride.getValue() == kGemvPipelineName) {
          addGemvPassPipeline(executableLoweringPipeline);
        } else if (pipelineOverride.getValue() == kPoolPipelineName) {
          addPoolPassPipeline(executableLoweringPipeline);
        } else {
```

```cpp
// KernelDispatch.cpp (변경 전)
static LogicalResult setConvRootConfig(mlir::FunctionOpInterface entryPointFn,
                                       linalg::ConvolutionOpInterface convOp,
                                       TilePassPipeline passPipeline,
                                       AMDAIEDevice targetDevice) {
  assert(!getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(convOp) &&
         "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(convOp.getOperation());

  // Current tiling strategy is based on llvm-cpu ConvTileAndDecomposeExpert.
  if (passPipeline == TilePassPipeline::ConvDecomposePipeline)
```

변경 후:

```cpp
// AMDAIELowerExecutableTarget.cpp (변경 후)
        if (pipelineOverride.getValue() == kGemvPipelineName) {
          addGemvPassPipeline(executableLoweringPipeline);
        } else if (pipelineOverride.getValue() == kPoolPipelineName) {
          addPoolPassPipeline(executableLoweringPipeline);
        } else if (pipelineOverride.getValue() == kConvPipelineName) {
          addConvPassPipeline(executableLoweringPipeline);
        } else {
```

```cpp
// KernelDispatch.cpp (변경 후)
static LogicalResult setConvRootConfig(mlir::FunctionOpInterface entryPointFn,
                                       linalg::ConvolutionOpInterface convOp,
                                       TilePassPipeline passPipeline,
                                       AMDAIEDevice targetDevice,
                                       uint32_t numRows, uint32_t numCols,
                                       uint32_t stackSize) {
  assert(!getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(convOp) &&
         "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(convOp.getOperation());

  if (FailureOr<ConvDims> conv =
          takesConvPipeline(entryPointFn, linalgOp, passPipeline);
      succeeded(conv)) {
    return setRootConfigForConvPipeline(entryPointFn, linalgOp, *conv,
                                        getDeviceModel(targetDevice), numRows,
                                        numCols, stackSize);
  }

  // Current tiling strategy is based on llvm-cpu ConvTileAndDecomposeExpert.
  if (passPipeline == TilePassPipeline::ConvDecomposePipeline)
```

새 함수 전체(`getConvDims`, `chooseConvTiling`, `setRootConfigForConvPipeline`, `addConvPassPipeline`)는 변경 전 — 없음(신규 추가)이고, 본문은 각 파일에 있다(`git diff` 로 확인).

### 4.3 수직 슬라이스에서 드러난 결함과 수정(T3.2)

| 결함 | 증거 | 수정 |
|---|---|---|
| G2(출력 채널을 열에)에서 NHWC 출력의 L2 버퍼가 첫 차원 n(행)으로 나뉘어 memtile 하나가 한 행의 코어 8 개 출력을 받는다 | `no consumer DMA channel available`; split 뒤 출력 조각 4 개(`memref<1x4x16x64xf32>`)가 memtile 0–3 에, 각 8 producer | 격자를 G1 방향(청크를 열에, OC 를 행에)으로 바꿨다: 코어 forall 에 `swapThreadXY`, `chooseConvTiling` 의 열·행 몫 교환. 입력·출력의 첫 차원이 청크라 열마다 나뉜다 |
| HWCF 필터를 여러 열·행으로 나누면 split 이 KH(3)로 자른다 | 8 열 HWCF: 필터 조각 3 개 `memref<1x3x64x64xbf16>`, `aie.dma_bd ... out of bounds`; 원소 검사 FAIL(필터 fill) | 청크가 여러 열이면 HWCF 필터는 행으로 나누지 않는다(`maxOcCores = 1`). 열 1 개는 split 이 없어 HWCF 도 맞다(원소 검사 PASS) |

출력 채널을 열에 두는 G2 는 L2 출력을 `[열][행][…]` 순서로 두는 레이아웃(pack-peel 의 C 처럼)이 생기면 다시 고를 수 있다.

## 5. 검증 (컴파일·정적 IR·lit/ctest 만, NPU 미사용)

### 5.1 lit·ctest

- 새 lit 3 개와 고친 lit 1 개를 각 파일의 RUN 줄로 실행해 통과했다. `lowering_strategy_conv_native.mlir` 의 타일 기대값은 C++ 와 따로 짠 파이썬 미러로 계산했고 세 사례 모두 일치했다.
- `ctest -R amd-aie`(PEANO 설정): **235/235**(기준선 232 + 새 3).

### 5.2 기본 경로 불변

VGG 를 기본 레시피(`--conv im2col`)로 다시 컴파일했다. NPU 실행 파일(PDI)은 11-03 vmfb 와 **바이트 단위로 같다**(md5 `2781c8d4…`). CPU ELF 의 16 B 차이는 디버그 정보에 든 출력 디렉터리 경로(`conv_im2col-` 추가) 때문이다.

### 5.3 VGG 크기 슬라이스(npu4, iree-compile 전 과정 + Peano)

| 입력 | 형태 | 코어 / 열 | memtile S2MM/MM2S | L1 끝 주소 B | 감사 |
|---|---|---|---|---|---|
| conv5_1, 청크 4 | 7-D FHWC | 16 / 4 | 6/3 | 57,344 | PASS |
| conv5_1, 청크 8 | 7-D FHWC | 32 / 8 | 6/3 | 61,696 | PASS |
| conv1_2, 청크 4 | 7-D FHWC | 16 / 4 | 6/3 | 64,768 | PASS |
| conv1_2, 청크 8 | 7-D FHWC | 32 / 8 | 6/3 | 62,976 | PASS |
| conv5_1, batch 접힘 | 6-D HWCF | 4 / 1 | 6/6 | 61,952 | PASS |

청크 8 개면 입력 청크 connection 8 개가 각각 열의 코어 4 개로, 필터 조각 connection 4 개가 각각 행의 코어 8 개로 broadcast 된다(pack-peel matmul 과 같은 패턴). L1 의 버퍼 합은 `chooseConvTiling` 예측과 같다. 끝 주소가 더 큰 것은 할당기가 16 KB 뱅크 4 개에 나눠 놓기 때문이다.

### 5.4 원소 단위 DMA 매핑 검사 — `scripts/debug/conv_dma_check.py`(신규)

변경 전 — 없음(신규 추가). T0.7 검증자의 `e2e_check.py` 를 일반화했다. assign-channels 뒤 IR 에서 코어마다 L3→L2→L1(입력·필터)과 L1→L2→L3(출력) 접근 패턴을 원소 단위로 합성하고, 각 코어의 입력이 한 halo 청크 전체이고 필터가 연속된 OC 구간이며 출력이 그 청크·OC 위치에 떨어지는지, 모든 (청크, OC) 쌍이 정확히 한 번 덮이는지 본다. IC 청크마다 memtile 조각이 다시 채워지는 순서도 따른다. 블록이 하나인 dispatch 만 검사할 수 있다.

| 입력 | 결과 |
|---|---|
| 청크 8·OC 64·IC 64 FHWC(코어 32) | PASS |
| 같은 모양, 입력 DMA 오프셋 하나를 8 원소 옮긴 사본 | FAIL(그 열의 코어 4 개를 잡음) — 검사기가 헛돌지 않음 |
| 청크 2·4·8, OC 32–128, IC 16–64, FHWC·HWCF, 6-D·7-D 의 9 가지 | 모두 PASS(코어 4–32) |
| 8 열 HWCF(수정 전) | FAIL(필터) — §4.3 의 결함 |
| 청크당 출력 1 행 | 인식 실패(IREE 가 단위 OH 를 접어 `oh + kh` 꼴이 사라진다). VGG 청크는 2 행 이상이라 해당 없음 |

11-08 리뷰에서 이 판은 (청크, oc0) 격자만 세어, 채널이 빠진 매핑도 PASS 할 수 있음이 드러났다(§7.6 #12). 그래서 다중 블록 검사기로 교체했다. 이 검사기는 제어 코드를 해석하고, 출력 원소 단위로 정확히 한 번 덮이는지 센다. 위 표의 PASS 는 새 검사기로 모두 다시 확인했다. "블록이 하나인 dispatch 만" 이라는 제약은 없어졌다(VGG conv 13 개 PASS).

세 번째 리뷰와 그 수정 검토 뒤(§7.8) 다음 검사를 더했다.
- 역할은 모양이 맞는 쌍으로 정한다. 커널 크기에 상관없다.
- 원점은 항상 출력한다. 출력이 들어갈 자리가 있어야 하고, 버퍼가 출력과 같은 모양이면 0 이어야 한다. `--origin` 으로 기대값을 줄 수 있다.
- L3 전송은 자기 memref 안에 있어야 한다. binding 조건은 다음과 같다.
  - 역할마다 binding 이 하나이고, 셋은 서로 다르다.
  - 입력·필터는 ReadOnly, 출력은 아니다.
  - 전송 기준(reinterpret_cast offset)이 subspan offset 과 같고, 역할마다 기준이 하나다.
- memtile 조각의 모든 쪽에서 반복 횟수가 같아야 한다. L1 쪽은 하강이 따로 접으므로 비교하지 않는다. IC 는 채널 청크의 정수배여야 한다.
- 드레인이 코어가 쓴 것만, 출력 상자를 정확히 한 번 L3 에 써야 한다.
- 경로 밖 이동이 없어야 한다. 코어마다 출력 1·입력 2 연결이고, DMA 연결은 모두 경로 위에 있으며, L3 전송은 모두 역할의 것이다.

### 5.5 수치

NPU 에서 실행하지 않았으므로 수치 정확도는 검증하지 않았다. 코어 코드는 스칼라 bf16 MAC(벡터화 off)이고, 성능은 다음 단계(커널 경로 D5)의 몫이다.

## 6. VGG 통합 1 단계 — epilogue(T5.2 일부)와 conv bias fold(T5.1)

### 6.1 코어 내 elementwise epilogue

origin/dev 레시피에서 im2col 만 빼면 conv dispatch 마다 bias 덧셈과 ReLU 가 epilogue 로 붙는다(§3.1). bias 를 접고 나면 ReLU 만 남는데, 이 단항 epilogue 를 conv 파이프라인이 처리하게 했다.

- 처음에는 `AMDAIEDistributeL1Allocations` 가 "'linalg.generic' op needs logic implemented for handling" 으로 실패했다. GEMV 는 `AMDAIEInsertCopyOps` 가 epilogue 를 contraction 결과 버퍼에서 바로 읽게 하고 epilogue 결과만 단계별로 옮기는데(`33a7112e`), 그 판정(`isContractionEpilogue`)이 producer 가 contraction 인터페이스일 때만 맞았다.
- 수정: producer 판정을 `isEpilogueSource`(contraction 또는 `getConvDims` 가 받는 conv)로 넓혔다. conv 파이프라인에 GEMV 와 같은 'epilogue 출력의 L2 승격' 단계를 넣었다. `chooseConvTiling` 에 epilogue 출력 타일 항(depth 2)을 더했고, 루트 설정이 dispatch 안의 elementwise 소비자에서 그 바이트를 센다(GEMV 와 같은 방식).
- 결과: 청크 8·IC 64 conv + ReLU 와 IC 256(청크 4 개) + ReLU 가 컴파일·감사·원소 검사를 통과했다. 코어 본문은 첫 청크에서 fill + conv, 마지막 청크에서 conv → ReLU(epilogue 타일) → epilogue 타일만 L1→L2 다. L1 끝 주소 64,512 B 는 epilogue 항을 넣은 예측(63,488 B + 스택)과 같다.

변경 전(`AMDAIEInsertCopyOps.cpp`):

```cpp
bool isContractionEpilogue(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || !isElementwise(genericOp)) return false;
  return llvm::any_of(genericOp.getDpsInputs(), [](Value input) {
    return input.getDefiningOp<linalg::ContractionOpInterface>();
  });
}
```

변경 후:

```cpp
bool isEpilogueSource(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def) return false;
  if (isa<linalg::ContractionOpInterface>(def)) return true;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(def);
  return linalgOp && succeeded(getConvDims(linalgOp));
}
...
bool isContractionEpilogue(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || !isElementwise(genericOp)) return false;
  return llvm::any_of(genericOp.getDpsInputs(), isEpilogueSource);
}
```

`promoteEpilogueOperands` 와 결과 승격 생략 조건도 같은 판정을 쓴다. 'option 으로 막지 않는 이유' 주석(`f938dd78`)은 conv 파이프라인을 포함하도록 고쳤다.

### 6.2 conv bias fold

`AMDAIEFoldBiasIntoContractionDispatches` 에 conv 형태를 더했다(`native_conv` 일 때만). 계획서 D7 의 (a) 다.

- 매칭: dispatch 안의 conv 하나(`getConvDims`), 입력·필터가 binding 의 전체 load, init 이 `fill(0)`, 결과를 쓰는 elementwise 소비자 하나에서 conv 값은 항등 맵, bias 는 마지막 출력 차원(OC) 맵 `(d0, …, dk) -> (dk)` 으로 읽혀 `addf` 하나에 들어간다. bias 는 binding 또는 dispatch 안 `arith.constant`(IREE 가 256 B 이하 bias 를 inline 하는 conv1_x)다.
- 재작성: 입력 채널을 IC+1 의 8 배수로 늘린다(3→8, 64→72, 128→136, 256→264, 512→520). 입력은 호스트 pad dispatch 로 새 채널을 1.0 으로 채운다. 필터(HWCF 또는 FHWC 상수)는 새 채널 IC 의 탭 (KH/2, KW/2) 에 bias(필터 형으로 반올림)를, 다른 새 채널·탭에 0 을 넣은 새 global 이 된다. 소비자에서 bias `addf` 를 빼고, ReLU 만 남거나 아무것도 안 남으면 소비자를 지운다.
- tied 결과: matmul 경로는 tied 결과를 거부하지만, conv 는 다음 층 패딩 버퍼에 쓰는 tied 출력이 13 개 중 8 개라 허용했다. 입력·필터·bias 에 묶인 결과만 거부하고, bias 인자를 지운 뒤 그보다 뒤의 tied 인덱스를 하나씩 당긴다.
- 새 lit 4 건(`fold_bias_into_contraction_dispatches.mlir`): 1×1 HWCF(접힌 값 전체 확인), 1×1 FHWC + inline bias + tied 출력, 2×2 탭 위치·tied 인덱스 이동, `native_conv` 없음(그대로).

변경 전 — conv 경로는 없음(신규 추가: `ConvBiasFoldInfo`, `getConvBiasInfo`, `appendBiasChannel`, `removeConvBias`, `foldConvBias`). `runOnOperation` 의 분기만 바뀌었다.

```cpp
// 변경 전
    std::optional<BiasFoldInfo> info = getPostAddBiasInfo(func);
    if (!info) info = getAccumulatorBiasInfo(func);
    if (!info) continue;
```

```cpp
// 변경 후
    std::optional<BiasFoldInfo> info = getPostAddBiasInfo(func);
    if (!info) info = getAccumulatorBiasInfo(func);
    if (!info) {
      if (getConfigNativeConv(target)) {
        if (std::optional<ConvBiasFoldInfo> convInfo = getConvBiasInfo(func)) {
          if (failed(foldConvBias(rewriter, module, dispatch, func, *convInfo,
                                  numCallers[func.getOperation()],
                                  hostAffinity, counter, replacedGlobals)))
            return signalPassFailure();
        }
      }
      continue;
    }
```

### 6.3 `--conv native` 레시피와 VGG 컴파일

`options.py` 의 `native` 선택이 `--iree-amdaie-enable-native-conv` 를 넘기게 했다.

```bash
./scripts/docker/run-onnx.sh compile models/vgg16/vgg16-12.onnx --numeric bf16 --conv native
```

rc 0, compile 32.4 s(import 4.7 s), vmfb 283,669,269 B, commit `786d1d19`(+ 미커밋 변경). 출력 `build/onnx/vgg16-12-conv_native-ukernel_all-vectorize_off-numeric_bf16-pool_cpu/`.

| 항목 | `--conv native` | `--conv im2col`(기준) |
|---|---|---|
| NPU entry point | 16(conv 13 + FC 3) | 16(conv matmul 13 + FC 3) |
| CPU entry point | 39(transpose 1, max pool 5, elementwise 13, pad 19, crop 1) | 68 |
| NPU 실행 파일 묶음(PDI) | 1.39 MB | 3.34 MB |
| compile | 32.4 s | 110–118 s |

감사 덤프를 넣어 다시 컴파일한 결과 NPU dispatch 16 개 모두 위반 0 이다. conv 13 개는 batch 가 접힌 6-D 라 한 열의 코어 4 개(OC 를 행에)로 내려간다(memtile S2MM·MM2S 6/6, memtile BD 24, L1 끝 주소 58,048–64,000 B). FC 3 개는 이전처럼 32 코어다.

같은 바이너리로 기본 레시피(im2col)의 NPU PDI 는 여전히 11-03 과 바이트 단위로 같다. ctest 235/235.

**의미와 한계**: VGG-16 의 conv 13 개가 im2col 없이 NPU 에서 conv 로 컴파일된다(구조 증명, M3·M6 의 컴파일 측). 이 시점에는 아직 전 배열이 아니었고(코어 4 개, §7 에서 28–32 코어로 바꿨다), 코어 코드는 스칼라 bf16 MAC 이다. NPU 에서 실행하지 않았으므로 수치·시간은 모른다. 원소 단위 매핑은 같은 구조의 작은 모양에서 확인했지만, VGG 층 자체는 블록이 여러 개라 검사기로 직접 확인하지 못했다.

## 7. 전 배열 conv — Flow 단계 halo 청크(T2.4 일부)와 split 가드(T2.5 일부)

### 7.1 설계

batch 가 접힌 6-D conv 는 한 열(코어 4 개)로만 내려갔다(§6.3). 청크 차원이 없으면 split 패스가 자를 첫 차원은 입력 행이다. 그런데 이웃한 출력 행은 입력 행(창의 halo)을 함께 읽으므로 입력을 제자리에서 행으로 자를 수 없다. 그래서 bias fold 바로 뒤의 Flow 단계에서 청크를 명시적으로 만든다. 새 패스 `AMDAIEChunkConvDispatches` 이고 `native_conv` 일 때만 돈다. 전처리 단계의 `PoolingNhwcMaxHaloSplit` 가 겹치는 풀링에 하는 일을 dispatch 형성 뒤의 conv 에 한다.

VGG conv1_1 의 Flow(변경 전 → 변경 후):

```
변경 전
  %in  = pad_dispatch (host)  : [226,226,3] -> ones splat [226,226,8]
  %out = conv dispatch (NPU)  : 6-D, HWCF 상수 [3,3,8,64], readwrite [226,226,64] f32 의 [1,1,0] 에 저장 (tied)
변경 후
  %in  = halo_dispatch (host) : [226,226,3] -> ones splat [8,30,226,8]    (pad 를 흡수)
  %c   = conv dispatch (NPU)  : 7-D, FHWC 상수 [64,3,3,8], writeonly [8,28,224,64] f32
  %r   = flow.tensor.reshape  : [8,28,224,64] -> [224,224,64]
  %out = insert_dispatch (host): %r 를 [226,226,64] f32 의 [1,1,0] 에 복사 (tied)
```

- **입력**: host dispatch 가 청크 N 개를 모은다. 청크 n 은 입력 행 n·S 부터 S + KH − 1 행(halo 포함)이다. gather 는 맵이 `(n, r, w, c) -> (n·S + r, w, c)` 인 `linalg.generic` 이다. 입력이 채널만 늘린 host pad dispatch(fold 의 ones 채널)의 결과면 그 pad 의 원본을 읽어 pad 의 splat 에 바로 쓰고, pad dispatch 는 지운다. 그래서 층마다 입력 쪽 host dispatch 는 여전히 1 개다.
- **필터**: HWCF 상수를 컴파일 때 FHWC 로 바꾼 새 global(`__conv_fhwc_weight_N`)로 대체한다. 청크가 여러 열에 걸칠 때 split 이 출력 채널로 자르는 레이아웃이라 출력 채널을 행 4 개에 나눌 수 있다. HWCF 로 두면 §4.3 규칙에 따라 행 1 개(코어 8 개)다. 상수가 아니면 HWCF 그대로 둔다.
- **실행 파일**: conv 를 청크 루프가 앞에 붙은 7-D generic 으로 다시 쓴다. 출력은 새 writeonly `[N, S, OW, OC]` binding 에 쓰고, fold 가 남긴 단항 epilogue(ReLU)도 4-D 로 옮긴다.
- **출력**: host 는 `flow.tensor.reshape` 로 `[OH, OW, OC]` 로 읽는다. 청크 우선 행 순서가 곧 행 우선 순서라 복사가 없다. 원래 결과가 더 큰 텐서의 내부에 쓰던 tied 결과(다음 층의 패딩 버퍼)였으면, 그 텐서에 tied 된 host insert dispatch 가 원래 offset(`[1,1,0]`)에 복사한다.
- **청크 수 N**: OH 의 약수 중 열 수 이하이고 청크당 출력 행이 2 이상인 가장 큰 값이다. VGG 는 OH 224·112·56 이 8, 28·14 가 7 이다. 한 행짜리 청크는 IREE 가 단위 행 차원을 접어 conv 로 인식되지 않는다(§5.4).
- **NPU 가 패딩 버퍼에 직접 쓰지 않는 이유**: 실행 파일 안에서 청크를 행으로 다시 접어 strided 내부에 저장하면(`tensor.collapse_shape` + offset 저장) objectFifo 단계가 `no consumer DMA channel available` 로 실패한다(`log_shjo/impl/halo/exe_collapse.log`). IREE 도 함수 수준의 청크 conv + 패딩 삽입을 같은 모양으로 만든다. 결과는 writeonly 로 두고 reshape 뒤 host `slow_memcpy` 로 삽입한다(`log_shjo/impl/halo/collapse_tied.mlir`).

매칭 조건(`getConvChunkInfo`)은 다음과 같고, 하나라도 어긋나면 dispatch 를 그대로 둔다(1 열).

- 실행 파일 안에 conv 하나(청크 차원 없음)가 있다. 입력·필터는 binding 의 전체 load 이고 init 은 fill 이다.
- 결과는 단항·항등 맵 elementwise 사슬을 거쳐 store 하나로 저장된다. 저장 위치는 binding 전체이거나 내부(static offset, stride 1)다.
- 함수 인자는 3 개이고, 그 밖의 op 는 상수·`tensor.empty` 뿐이다.
- dispatch 는 호출자가 1 개이고 workload 가 없다. 피연산자는 입력·필터와, tied 결과면 그 대상뿐이다.

### 7.2 코드 변경

| 파일 | 변경 | 분류 |
|---|---|---|
| `Transforms/AMDAIEChunkConvDispatches.cpp` | 새 패스(`getConvChunkInfo`, `chooseConvChunks`, `transposeFilterToFHWC`, `getChannelPadDispatch`, `createHaloSplitDispatch`, `chunkExecutable`) | 추가 |
| `Transforms/Passes.td`, `Passes.h`, `PassDetail.h`, `Transforms/CMakeLists.txt` | 패스 선언·정의 매크로·소스 등록 | 추가 |
| `PluginRegistration.cpp` | Flow 단계에서 fold 바로 뒤에 실행 | 추가 |
| `Transforms/AMDAIEDispatchRewriteUtils.{h,cpp}` | `createExecutable`·`createInsertDispatch` 추가, `createPaddingDispatch` 를 공통 `createCopyIntoDispatch` 로 재구성(만드는 IR 은 같음), `firstFreeHelperIndex` 접두어 3 개 | 추가·변경 |
| `Transforms/Utils/AMDAIELogicalObjFifoSplittingUtils.cpp` | 맨 loop IV offset 가드(§7.4) | 추가·변경 |
| `Transforms/test/chunk_conv_dispatches.mlir`, `test/CMakeLists.txt` | 새 lit(3 사례) | 추가 |
| `Transforms/test/split_logicalobjfifos.mlir` | 부정 사례 1 개 | 추가 |
| `scripts/debug/conv_dma_check.py` | strided 상수 binding 타입, bf16 출력 식별 | 변경 |
| `scripts/onnx/options.py`(주석), `docs/EXPERIMENT_ENV.md` | `native` 설명 갱신 | 변경 |
| 타일 파이프라인·`chooseConvTiling`·KernelDispatch | 없음(7-D 청크 conv 는 §4 의 경로 그대로) | 유지 |

새 패스 파일 전체와 lit 는 변경 전 — 없음(신규 추가)이다. 기존 코드의 변경은 아래와 같다.

**`PluginRegistration.cpp`** — 변경 전:

```cpp
    passManager.addPass(
        AMDAIE::createAMDAIEFoldBiasIntoContractionDispatchesPass());
    // With affinity known per-dispatch, pad NPU contraction operands up to the
```

변경 후:

```cpp
    passManager.addPass(
        AMDAIE::createAMDAIEFoldBiasIntoContractionDispatchesPass());
    // Spread native conv dispatches (`native_conv`) over the array columns:
    // their output rows are split into chunks with a host gather of the input
    // rows each chunk reads (merged with the fold's ones-channel pad), the
    // filter constant becomes FHWC, and the chunked output is read back as
    // rows. No-op without such a dispatch.
    passManager.addPass(AMDAIE::createAMDAIEChunkConvDispatchesPass());
    // With affinity known per-dispatch, pad NPU contraction operands up to the
```

**`AMDAIEDispatchRewriteUtils.cpp`** — `firstFreeHelperIndex` 접두어. 변경 전:

```cpp
         {"pad_executable_", "crop_executable_", "concat_executable_",
          "__padded_constant_", "__bias_folded_weight_"}) {
```

변경 후:

```cpp
         {"pad_executable_", "crop_executable_", "concat_executable_",
          "halo_executable_", "insert_executable_", "__padded_constant_",
          "__bias_folded_weight_", "__conv_fhwc_weight_"}) {
```

**`AMDAIEDispatchRewriteUtils.cpp`** — `createPaddingDispatch`. 실행 파일 래퍼와 "v 를 dest 의 offset 에 복사" 부분을 떼어 insert dispatch 와 같이 쓴다. pad 의 symbol 이름·op 순서·tied 형태는 그대로다(기존 pad lit 전부 통과). 새 코드의 builder 호출은 deprecated 경고가 없는 `Op::create(builder, …)` 꼴로 썼다. 변경 전:

```cpp
Value createPaddingDispatch(IRRewriter &rewriter, ModuleOp module, Value v,
                            ArrayRef<int64_t> dstShape,
                            IREE::HAL::DeviceAffinityAttr hostAffinity,
                            int &counter, TypedAttr padValue) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  if (dstShape == srcType.getShape()) return v;
  auto dstType = RankedTensorType::get(dstShape, srcType.getElementType());
  auto inBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, srcType);
  auto outBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadWrite, dstType);
  std::string idx = std::to_string(counter++);

  // Worker func: load the full input and store it into the low corner of the
  // pad-value-filled output.
  auto funcOp = func::FuncOp::create(
      loc, "pad_dispatch_" + idx,
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value loaded = fb.create<IREE::TensorExt::DispatchTensorLoadOp>(
      loc, srcType, entry->getArgument(0), /*sourceDynamicDims=*/ValueRange{});
  SmallVector<OpFoldResult> offsets(dstShape.size(), fb.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(dstShape.size(), fb.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes;
  for (int64_t d : srcType.getShape()) sizes.push_back(fb.getIndexAttr(d));
  fb.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, loaded, entry->getArgument(1), /*targetDynamicDims=*/ValueRange{},
      offsets, sizes, strides);
  fb.create<func::ReturnOp>(loc);

  // Executable wrapping the func, with a default workgroup-count region.
  OpBuilder mb(&module.getBody()->back());
  auto exeOp =
      IREE::Flow::ExecutableOp::create(mb, loc, "pad_executable_" + idx);
  exeOp.setPrivate();
  OpBuilder exeBuilder = OpBuilder::atBlockBegin(&exeOp.getBlock());
  auto innerModule = mlir::ModuleOp::create(exeBuilder, loc);
  innerModule.push_back(funcOp);
  OpBuilder exportBuilder(exeOp.getBody());
  auto exportOp = IREE::Flow::ExecutableExportOp::create(
      exportBuilder, loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *wcBlock = &exportOp.getWorkgroupCount().emplaceBlock();
  OpBuilder wb = OpBuilder::atBlockBegin(wcBlock);
  Type indexTy = wb.getIndexType();
  auto countOp = wb.create<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
      loc, TypeRange{indexTy, indexTy, indexTy},
      /*ordinal_operands=*/ValueRange{});
  wb.create<IREE::Flow::ReturnOp>(loc, countOp.getResults());

  // Host splat of the pad value and the dispatch writing into it in place,
  // both pinned to the host device.
  Type elemType = srcType.getElementType();
  TypedAttr fillAttr = padValue ? padValue : rewriter.getZeroAttr(elemType);
  Value padScalar = rewriter.create<arith::ConstantOp>(loc, fillAttr);
  auto splatOp = rewriter.create<IREE::Flow::TensorSplatOp>(
      loc, dstType, padScalar, /*result_dims=*/ValueRange{});
  splatOp->setAttr("stream.affinity", hostAffinity);
  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{},
      /*arguments=*/ValueRange{v, splatOp.getResult()},
      /*argument_dims=*/ValueRange{},
      /*tied_operands=*/rewriter.getIndexArrayAttr({1}));
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}
```

변경 후:

```cpp
IREE::Flow::ExecutableExportOp createExecutable(ModuleOp module,
                                                func::FuncOp funcOp,
                                                StringRef name) {
  Location loc = funcOp.getLoc();
  OpBuilder mb(&module.getBody()->back());
  auto exeOp = IREE::Flow::ExecutableOp::create(mb, loc, name);
  exeOp.setPrivate();
  OpBuilder exeBuilder = OpBuilder::atBlockBegin(&exeOp.getBlock());
  auto innerModule = mlir::ModuleOp::create(exeBuilder, loc);
  innerModule.push_back(funcOp);
  OpBuilder exportBuilder(exeOp.getBody());
  auto exportOp = IREE::Flow::ExecutableExportOp::create(
      exportBuilder, loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *wcBlock = &exportOp.getWorkgroupCount().emplaceBlock();
  OpBuilder wb = OpBuilder::atBlockBegin(wcBlock);
  Type indexTy = wb.getIndexType();
  auto countOp = IREE::TensorExt::DispatchWorkgroupCountFromSliceOp::create(
      wb, loc, TypeRange{indexTy, indexTy, indexTy},
      /*ordinal_operands=*/ValueRange{});
  IREE::Flow::ReturnOp::create(wb, loc, countOp.getResults());
  return exportOp;
}

/// Creates a host dispatch, `<prefix>_executable_<i>::<prefix>_dispatch_<i>`,
/// that copies `v` into `dest` at `offsets`, tied to `dest`.
static Value createCopyIntoDispatch(IRRewriter &rewriter, ModuleOp module,
                                    Value v, Value dest,
                                    ArrayRef<int64_t> offsets,
                                    IREE::HAL::DeviceAffinityAttr hostAffinity,
                                    StringRef prefix, int &counter) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  auto dstType = cast<RankedTensorType>(dest.getType());
  auto inBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, srcType);
  auto outBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadWrite, dstType);
  std::string idx = std::to_string(counter++);

  // Worker func: load the full input and store it into the output at
  // `offsets`.
  auto funcOp = func::FuncOp::create(
      loc, (prefix + "_dispatch_" + idx).str(),
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value loaded = IREE::TensorExt::DispatchTensorLoadOp::create(
      fb, loc, srcType, entry->getArgument(0),
      /*sourceDynamicDims=*/ValueRange{});
  SmallVector<OpFoldResult> offsetAttrs, sizes;
  SmallVector<OpFoldResult> strides(dstType.getRank(), fb.getIndexAttr(1));
  for (int64_t o : offsets) offsetAttrs.push_back(fb.getIndexAttr(o));
  for (int64_t d : srcType.getShape()) sizes.push_back(fb.getIndexAttr(d));
  IREE::TensorExt::DispatchTensorStoreOp::create(
      fb, loc, loaded, entry->getArgument(1),
      /*targetDynamicDims=*/ValueRange{}, offsetAttrs, sizes, strides);
  func::ReturnOp::create(fb, loc);
  IREE::Flow::ExecutableExportOp exportOp =
      createExecutable(module, funcOp, (prefix + "_executable_" + idx).str());

  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{},
      /*arguments=*/ValueRange{v, dest},
      /*argument_dims=*/ValueRange{},
      /*tied_operands=*/rewriter.getIndexArrayAttr({1}));
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}

Value createPaddingDispatch(IRRewriter &rewriter, ModuleOp module, Value v,
                            ArrayRef<int64_t> dstShape,
                            IREE::HAL::DeviceAffinityAttr hostAffinity,
                            int &counter, TypedAttr padValue) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  if (dstShape == srcType.getShape()) return v;
  auto dstType = RankedTensorType::get(dstShape, srcType.getElementType());

  // Host splat of the pad value and the dispatch writing into its low corner
  // in place, both pinned to the host device.
  Type elemType = srcType.getElementType();
  TypedAttr fillAttr = padValue ? padValue : rewriter.getZeroAttr(elemType);
  Value padScalar = arith::ConstantOp::create(rewriter, loc, fillAttr);
  auto splatOp = IREE::Flow::TensorSplatOp::create(
      rewriter, loc, dstType, padScalar, /*result_dims=*/ValueRange{});
  splatOp->setAttr("stream.affinity", hostAffinity);
  SmallVector<int64_t> offsets(dstShape.size(), 0);
  return createCopyIntoDispatch(rewriter, module, v, splatOp.getResult(),
                                offsets, hostAffinity, "pad", counter);
}

Value createInsertDispatch(IRRewriter &rewriter, ModuleOp module, Value v,
                           Value dest, ArrayRef<int64_t> offsets,
                           IREE::HAL::DeviceAffinityAttr hostAffinity,
                           int &counter) {
  return createCopyIntoDispatch(rewriter, module, v, dest, offsets,
                                hostAffinity, "insert", counter);
}
```

**`AMDAIEDispatchRewriteUtils.h`** — 헤더 주석에 새 패스 이름을 더했고, `firstFreeHelperIndex` 주석에 새 접두어 3 개를, `createExecutable`·`createInsertDispatch` 선언을 추가했다.

**`scripts/debug/conv_dma_check.py`** — 작은 2 층 모델의 첫 conv 에서 필터가 상수 풀 안의 offset 으로 바인딩되어 타입이 `memref<16x3x3x16xbf16, strided<[144, 48, 16, 1], offset: 3456>>` 였다. 출력은 truncf 가 합쳐져 bf16 이었다. 변경 전에는 둘 다 읽지 못했다("could not identify input/filter/output L3 buffers"). 변경 전:

```python
        mm = re.match(r"memref<([0-9x]+)x(\w+)>$", ty)
    ...
    out_ty = [t for t, (s, e) in shapes.items() if e == "f32"]
    bf = [t for t, (s, e) in shapes.items() if e == "bf16"]
    flt_ty = [t for t in bf if shapes[t][0].count(3) >= 2]
    in_ty = [t for t in bf if t not in flt_ty]
```

변경 후(그리고 objectFifo 정규식 3 곳의 `memref<[^>]*>` 를 중첩 `<…>` 를 받는 `memref<(?:[^<>]|<[^<>]*>)*>` 로):

```python
        # A constant bound at an offset into a packed buffer has a strided type.
        mm = re.match(r"memref<([0-9x]+)x(\w+)(?:, strided<\[[^\]]*\], offset: \d+>)?>$", ty)
    ...
    # The output is the L3 buffer a connection writes (f32, or bf16 with a
    # fused truncf); of the buffers read, the filter has two 3-sized dims.
    written = {ty for ty, names in l3.items() if any(c["tgt"] in names for c in conns.values())}
    out_ty = [t for t in shapes if t in written]
    read = [t for t in shapes if t not in written]
    flt_ty = [t for t in read if shapes[t][0].count(3) >= 2]
    in_ty = [t for t in read if t not in flt_ty]
```

고친 검사기로 이전 사례(`sample_c8`, `c8_relu`, `c8_relu_ic256`)를 다시 돌려 모두 PASS 였다.

### 7.3 검증 (컴파일·정적 IR·lit/ctest 만, NPU 미사용)

**lit** — `chunk_conv_dispatches.mlir`(RUN 줄 그대로):

| 사례 | 확인 |
|---|---|
| 1×1, HWCF 상수, writeonly 출력, OH 4 → 청크 2 | FHWC 상수 값 전체(전치 확인), 원래 global 삭제, 7-D 맵·iterator, gather 맵 `d0 * 2 + d1`, writeonly gather(splat 없음), reshape |
| 3×3 + ReLU, 앞에 ones-채널 pad dispatch, tied 출력 `[1,1,0]` | pad dispatch·실행 파일 삭제, gather 가 pad 원본 `[6,4,7]` 을 ones splat `[2,4,4,8]` 에 저장, ReLU 4-D, insert dispatch 가 `[1,1,0]` 에 tied 복사 |
| OH 2(청크당 2 행 불가), `native_conv` 없는 target | 그대로 |

lit 하네스는 FileCheck 를 `--enable-var-scope` 로 돌린다. 그래서 맵 별칭은 전역 변수(`#[[$IN:…]]`)로 잡았다. 처음에는 손으로 돌린 FileCheck 만 통과하고 ctest 에서 실패했다.

**작은 2 층 모델**(`log_shjo/impl/chunk/gen_pair.py`, `cc_het.sh`: host + npu4, `--iree-amdaie-enable-native-conv`). pad → conv 3×3 → bias + ReLU 를 두 번 하고, 두 번째 출력도 패딩한다. Flow 는 설계대로 나온다. halo 2 개(ones 채널 8→16, 16→24 흡수), 7-D conv 2 개, reshape 2 개, insert 2 개(bf16·f32 패딩 버퍼)다.

| 모델 | dispatch | 청크 × 행 | 코어 / 열 | 감사 | 원소 단위 매핑 |
|---|---|---|---|---|---|
| H 16(`pair16`) | conv 1(IC 16, 출력 bf16) | 8 × 2 | 32 / 8 | PASS | PASS(쌍 32, 중복·누락 0) |
| H 16 | conv 2(IC 24, 출력 f32) | 8 × 2 | 32 / 8 | PASS | PASS |
| H 14(`pair14`) | conv 1 | 7 × 2 | 28 / 7 | PASS | PASS(쌍 28) |
| H 14 | conv 2 | 7 × 2 | 28 / 7 | PASS | PASS |

처음 컴파일은 Peano `llc` 에서 `unable to legalize instruction: G_FMAXIMUM` 으로 실패했다. 테스트 모델이 ReLU 를 `arith.maximumf` 로 써서다. VGG 와 같은 `cmpf ugt + select` 로 바꾸자 통과했다. 이 패스와는 무관하다.

**VGG-16 `--conv native`**:

```bash
./scripts/docker/run-onnx.sh compile models/vgg16/vgg16-12.onnx --numeric bf16 --conv native
```

rc 0, compile 90.8 s(import 5.1 s), vmfb 286,668,609 B, commit `786d1d19`(+ 미커밋 변경). 출력 `build/onnx/vgg16-12-conv_native-ukernel_all-vectorize_off-numeric_bf16-pool_cpu/`. NPU PDI 묶음은 4.38 MB 다(1 열 1.39 MB, im2col 3.34 MB). compile 이 1 열(32.4 s)보다 긴 것은 코어가 7–8 배라 objectFifo·AIE lowering 과 Peano 가 할 일이 늘어서다. 감사 덤프를 넣은 재컴파일(`build/onnx/vgg16-12-conv_native-audit/`)에서 NPU dispatch 16 개 모두 위반 0 이다.

| 층 | OH | 청크 × 행 | 코어 / 열(전 → 후) | 블록 타일 (oh, ow, oc) | IC 청크 × 반복 | memtile BD | L1 끝 B |
|---|---|---|---|---|---|---|---|
| conv1_1 | 224 | 8 × 28 | 4/1 → 32/8 | 14, 14, 64 | 8 × 1 | 24 → 18 | 64,000 |
| conv1_2 | 224 | 8 × 28 | 4/1 → 32/8 | 14, 14, 64 | 8 × 9 | 24 → 18 | 64,000 |
| conv2_1, 2_2 | 112 | 8 × 14 | 4/1 → 32/8 | 14, 14, 64 | 8 × 9, 17 | 24 → 18 | 64,000 |
| conv3_1–3_3 | 56 | 8 × 7 | 4/1 → 32/8 | 7, 28, 64 | 8 × 17, 33 | 24 → 18 | 64,512 |
| conv4_1 | 28 | 7 × 4 | 4/1 → 28/7 | 4, 28, 64 | 24 × 11 | 24 → 30 | 60,800 |
| conv4_2, 4_3 | 28 | 7 × 4 | 4/1 → 28/7 | 4, 14, 64 | 40 × 13 | 24 → 30 | 60,416 |
| conv5_1–5_3 | 14 | 7 × 2 | 4/1 → 28/7 | 2, 14, 64 | 40 × 13 | 24 → 30 | 56,064 |

- 타일은 Flow 덤프의 청크 실행 파일 13 개에 lowering strategy 를 직접 돌려 읽었다(`log_shjo/impl/chunk/vgg_tiling_probe.out`). 코어 타일은 모두 청크 1 × OC 16 이다. 파이썬 미러가 13 개 모두 같은 타일을 낸다.
- L1 끝 주소가 버퍼 합보다 큰 층(conv4_2 이후)은 할당기가 16 KB 뱅크 4 개에 나눠 놓은 결과다(§5.3 과 같음). 합은 모두 예산(65,536 − 스택 1,024) 안이다.
- Flow(`build/onnx/vgg16-12-conv_native-flow3/`): dispatch 63 개(host 47, NPU 16)다. halo 13 개가 conv 의 ones-채널 pad 13 개를 모두 흡수했고, insert 8 개는 tied 출력 층에만 있다. 남은 pad 6 개는 FC 층(K+1 ones 열·K 패딩) 것이다. 변경 전 55 개(host 39)에서 insert 8 개만 늘었다.
- 경고: 뱅크 배치(bank-aware) 실패 후 순차 배치로 넘어간 코어가 252 개다(변경 전 32 개). 경고이고 컴파일은 성공한다. L1 을 64 KB 가까이 쓰는 타일이라 버퍼가 뱅크에 맞지 않는 것이다. 성능에는 영향이 있을 수 있다(§8).

**기본 경로 불변**: 같은 바이너리로 `--conv im2col` VGG 를 다시 컴파일했다. NPU PDI md5 는 `2781c8d4…` 로 11-03 과 같다. 가드를 넣은 뒤 한 번 더 확인했다.

**ctest**: `ctest -R amd-aie`(PEANO 설정) **236/236**(새 `chunk_conv_dispatches` 1 개, split 부정 사례는 기존 파일 안). 가드 추가 뒤에도 236/236 이다.

### 7.4 split 패스 가드(T2.5 일부)

`getNewOffsetConfig` 는 split 차원의 offset 이 맨 loop IV 이면 조각 번호를 0 으로, offset 을 그대로 둔다. IV 가 첫 조각 밖을 가리키면 다른 조각에 갈 접근이 첫 조각으로 가고, 컴파일 오류 없이 하드웨어 결과가 틀린다(dev-conv 의 corr 0.111778 사례, 계획서 §2.3). split 이 실제로 일어날 때(splitFactor > 1) IV 범위가 static 이고 첫 조각 안에 머무는 경우만 허용하고, 나머지는 오류로 바꿨다.

변경 전:

```cpp
    } else if (auto blockArg = dyn_cast<BlockArgument>(offsetValue);
               blockArg &&
               isa<LoopLikeOpInterface>(blockArg.getOwner()->getParentOp())) {
      objFifoIndex = 0;
      newOffsetValue = offsetValue;
    } else {
```

변경 후(`ivStaysInFirstSplit` 는 신규 추가: loop 의 static 하한·상한·step 으로 IV 최댓값을 구해, 단위 split stride 면 조각 크기 미만, 아니면 0 인지 본다):

```cpp
    } else if (auto blockArg = dyn_cast<BlockArgument>(offsetValue);
               blockArg &&
               isa<LoopLikeOpInterface>(blockArg.getOwner()->getParentOp())) {
      // The access keeps its offset and goes to the first new objectFifo. An
      // induction variable that reaches past it would silently read or write
      // the wrong objectFifo (wrong results on hardware, no compile error).
      if (splitFactor > 1 &&
          !ivStaysInFirstSplit(blockArg, splitDimSize, splitStride)) {
        return emitError() << "the offset on index " << offsetIdx
                           << " is a loop induction variable that reaches "
                              "past the first of "
                           << splitFactor << " split objectFifos";
      }
      objFifoIndex = 0;
      newOffsetValue = offsetValue;
    } else {
```

- 부정 lit: 2 조각으로 나뉘는 L2 입력을 소비자 하나는 `scf.forall` IV(0..1)로, 다른 하나는 상수 1 로 읽는다. 가드 메시지와 호출자의 "failed to find an offset index and offset" 이 함께 나온다. 가드가 없으면 이 사례는 조용히 통과한다.
- 정상 경로 영향 없음(`log_shjo/impl/chunk/guard_regression.log`): ctest 236/236. VGG `--conv native` 는 91.8 s 에 같은 크기 vmfb 이고 뱅크 경고 밖의 오류는 0 이다. `--conv im2col` 의 PDI md5 는 `2781c8d4…` 그대로다. 작은 2 층 모델 2 개는 감사 PASS 이고, dispatch 4 개 모두 원소 단위 매핑 PASS 다.

### 7.5 의미와 한계

- VGG-16 의 conv 13 개가 전 배열로 컴파일된다. 7 개는 32 코어 8 열, 6 개는 28 코어 7 열이다(T3.3·T3.5 의 컴파일 측. 계획서의 G2 대신 G1 방향).
- NPU 에서 실행하지 않았으므로 수치·시간은 모른다. host 쪽 gather·insert 의 의미는 lit 로 구조만 확인했다. NPU 쪽 데이터 이동은 VGG conv 13 개 모두 원소 단위로 확인했다(§7.6 의 다중 블록 검사기). lock 은 §7.7 에서 생산자·소비자별 쌍으로 바꿨고 정적 감사로 확인했다. 실제 동기화 동작은 실행해야 안다.
- host 비용이 늘었다. tied 출력 8 개 층마다 f32 출력 전체를 한 번 더 복사한다(conv1_1 이면 12.8 MB). halo gather 는 입력 행을 (S + 2)/S 배로 늘린다(conv1 30/28, conv5 4/2).
- L3 연속 구간이 짧다. IC 청크가 8(bf16 16 B)인 층은 한 픽셀에서 이어서 읽는 구간이 16 B 뿐이다(감사 `l3_contiguous_run_bytes` 16). 계획서 G16 의 우려 그대로이고, 청크 우선 레이아웃은 아직 없다.
- OH 28·14 층은 7 열이다(OH 패딩 없음).
- 코어 코드는 여전히 스칼라 bf16 MAC 이다.
- 상수가 아닌 HWCF 필터는 행 1 개만 쓴다(§7.7). 상수 필터는 Flow 에서 FHWC 가 되므로 VGG 에는 해당하지 않는다.

### 7.6 독립 리뷰 두 차례와 수정

T2.4 를 마친 뒤 두 차례 리뷰했다. 리뷰어는 코드를 읽고, 필요하면 이미 빌드된 도구로 작은 IR 을 돌려 재현했다(빌드·NPU 없음).

1. 1 차는 리뷰 에이전트 하나로, 청크 패스·dispatch 재작성 유틸·split 가드를 봤다.
2. 2 차는 워크플로(리뷰 4 차원 → 차원별 반증 검증)다. 차원은 Flow 재작성, conv bias fold, codegen, 검증 근거(도구·lit·문서 주장)다.

2 차 지적은 검증 에이전트가 모두 재현해 확인(confirmed)했다. 재현 파일은 `log_shjo/review/` 에 있다. 확인 결과는 `log_shjo/impl/fixcheck/` 에 있다.

| # | 지적(출처) | 심각도 | 수정 | 확인 |
|---|---|---|---|---|
| 1 | `eraseDeadGlobals` 가 두 번 들어온 global 을 지운 뒤 다시 읽음(필터 global 을 공유하는 두 conv, 기존 matmul fold 경로도 해당) | 크래시 | 방문 집합으로 한 번만 처리 | lit(공유 global) |
| 2 | 청크 패스 epilogue 매처가 conv 값을 init 으로 쓰는 op 도 받음 | 조용한 오답 | 입력이 conv 값이고 init 을 읽지 않을 때만 | lit |
| 3 | `getFullTensorLoad` 가 부분·rank 축소 load 를 받음(fold·pad 패스 공통) | 오답/verifier | `isLoadOfWholeSource()` + 타입 일치 | lit |
| 4 | fold 가 전치 epilogue 의 init 맵을 버림(NCHW 출력으로 가는 conv) | 치명(조용한 오답) | init 맵 보존, 항등일 때만 소비자 제거(matmul 도) | lit, 리뷰 재현 t1·t2·m2·m4 |
| 5 | bias 값이 모두 같으면(bf16 반올림 포함) bias 바이트를 범위 밖에서 읽음(matmul·conv) | 조용한 오답 | `getConvertedBiasBytes` 로 값마다 펼침 | lit 2 건, 재현 t6·t7·f2·m6 |
| 6 | 연산 결과가 소비자의 init 이어도 bias 소비자로 매칭(matmul·conv) | 조용한 오답 | 입력인지 확인 | lit, 재현 t3 |
| 7 | 다른 op 도 읽는 입력·필터·bias binding 을 고쳐 씀 | verifier/assert | 이유를 밝힌 fold 오류 | lit, 재현 t4·t5 |
| 8 | 명명 conv 가 conv 파이프라인에서 크래시 | 크래시 | strategy 에서 "generic 형태가 필요" 오류(iree-compile 은 conv 를 항상 generic 으로 넘김) | 오류 lit, 재현 |
| 9 | OC 방향 DMA 의 4 B 정렬을 `chooseConvTiling` 이 보지 않음(OC 3·12·100) | 컴파일 실패 | 모든 절단 지점의 4 B 정렬 조건 | lit 2 건, 재현 5 개 rc 0 |
| 10 | 이항 epilogue·비-fill init 의 conv 가 늦게, 이유 없이 실패 | 진단 | strategy 에서 이유를 밝힌 오류 | 오류 lit 2 건, 재현 4 개 |
| 11 | `--iree-amdaie-enable-native-conv` 도움말의 행·열이 반대 | 문서 | 고침 | — |
| 12 | `conv_dma_check.py` 가 (청크, oc0) 격자만 세어 채널이 빠져도 PASS | 근거 결함 | 다중 블록 검사기로 교체(아래) | 합성 누락 사례 FAIL |
| 13 | `aie_audit.py` 가 덤프가 빠져도 PASS, BD 한계 1023 하나로 판정 | 근거 결함 | 덤프 누락 = 위반, 타일별 wrap/step 한계(32-bit word) | 합성 위반 3 종 FAIL |
| 14 | 청크 lit 가 1×1 필터만 전치 확인 | 근거 결함 | non-splat 3×3 사례 | lit |
| 15 | 출력 join 이 공유 카운팅 lock(경쟁 상태) | 주요(타이밍 의존 오답) | §7.7 | §7.7 |
| 16 | (§7.7 검증 중 발견) 청크 없는 conv 의 HWCF 필터 fan-out 도 공유 카운팅 lock | 주요 | §7.7 | §7.7 |

리뷰가 맞다고 확인한 것도 있다. 청크 패스의 피연산자·binding 대응, op 삭제 순서, FHWC 전치와 halo 맵 수식, pad 흡수 조건, `createPaddingDispatch` 재구성(같은 IR), split 가드의 IV 범위 계산이다.

#### 코드 변경(변경 전 / 변경 후)

**`AMDAIEDispatchRewriteUtils.cpp`** — `getFullTensorLoad`(#3). 변경 전:

```cpp
  // Only a plain full-tensor load (unit strides, zero offsets) is supported.
  for (OpFoldResult o : load.getMixedOffsets())
    if (!isConstantIntValue(o, 0)) return nullptr;
  for (OpFoldResult s : load.getMixedStrides())
    if (!isConstantIntValue(s, 1)) return nullptr;
  return load;
```

변경 후(쓰지 않게 된 `StaticValueUtils.h` include 도 뺐다):

```cpp
  // Only a plain full-tensor load is supported: zero offsets, unit strides and
  // the binding's whole shape, not rank-reduced. The callers grow or replace
  // the binding and reload all of it, which would change what a smaller or
  // rank-reducing slice reads.
  if (!load.isLoadOfWholeSource() ||
      load.getType() != load.getSourceType().asRankedTensorType())
    return nullptr;
  return load;
```

**`AMDAIEDispatchRewriteUtils.cpp`** — `eraseDeadGlobals`(#1). 변경 전:

```cpp
  for (IREE::Util::GlobalOp global : globals) {
    SmallVector<IREE::Util::GlobalLoadOp> loads;
```

변경 후:

```cpp
  // A global replaced for several dispatches is listed once for each of them;
  // it must be looked at (and erased) only once.
  llvm::SmallPtrSet<Operation *, 8> visited;
  for (IREE::Util::GlobalOp global : globals) {
    if (!visited.insert(global.getOperation()).second) continue;
    SmallVector<IREE::Util::GlobalLoadOp> loads;
```

**`AMDAIEChunkConvDispatches.cpp`** — epilogue 매처(#2). 변경 전:

```cpp
static bool isUnaryIdentityElementwise(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getNumParallelLoops() != op.getNumLoops())
    return false;
  ...
    if (!generic || !isUnaryIdentityElementwise(generic)) break;
```

변경 후:

```cpp
static bool isUnaryEpilogueOf(linalg::GenericOp op, Value v) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getDpsInputOperand(0)->get() != v ||
      op.payloadUsesValueFromOperand(op.getDpsInitOperand(0)) ||
      op.getNumParallelLoops() != op.getNumLoops())
    return false;
  ...
    if (!generic || !isUnaryEpilogueOf(generic, v)) break;
```

**`AMDAIEFoldBiasIntoContractionDispatches.cpp`** — bias 바이트(#5). 변경 전(`appendBiasChannel`, `appendBiasRow` 도 같은 꼴):

```cpp
  SmallVector<APFloat> values;
  for (APFloat v : bias.getValues<APFloat>()) {
    bool losesInfo;
    v.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
              &losesInfo);
    values.push_back(v);
  }
  ArrayRef<char> biasRaw =
      DenseElementsAttr::get(RankedTensorType::get({OC}, floatType), values)
          .getRawData();
```

변경 후(새 helper, 두 함수 모두 `SmallVector<char> biasRaw = getConvertedBiasBytes(bias, floatType);`):

```cpp
static SmallVector<char> getConvertedBiasBytes(DenseElementsAttr bias,
                                               FloatType floatType) {
  SmallVector<APFloat> values;
  ... (변환은 같음)
  auto attr = DenseElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(values.size())}, floatType),
      values);
  ArrayRef<char> raw = attr.getRawData();
  if (!attr.isSplat()) return SmallVector<char>(raw.begin(), raw.end());
  SmallVector<char> bytes;
  for (size_t i = 0, e = values.size(); i < e; ++i)
    bytes.append(raw.begin(), raw.end());
  return bytes;
}
```

**같은 파일** — 소비자 맵(#4)과 입력 확인(#6). 변경 전(`removeConvBias`; `removeBias` 도 같은 꼴):

```cpp
  auto newConsumer = rewriter.create<linalg::GenericOp>(
      loc, oldConsumer->getResultTypes(), ValueRange{convResult},
      oldConsumer.getDpsInits(), ArrayRef<AffineMap>{identity, identity},
      oldConsumer.getIteratorTypesArray());
  ...
  if (llvm::hasSingleElement(*newBody) &&
      yield.getOperand(0) == newBody->getArgument(0) &&
      newConsumer.getResult(0).getType() == convResult.getType())
    rewriter.replaceOp(newConsumer, convResult);
```

변경 후:

```cpp
  AffineMap initMap =
      oldConsumer.getMatchingIndexingMap(oldConsumer.getDpsInitOperand(0));
  auto newConsumer = rewriter.create<linalg::GenericOp>(
      loc, oldConsumer->getResultTypes(), ValueRange{convResult},
      oldConsumer.getDpsInits(), ArrayRef<AffineMap>{identity, initMap},
      oldConsumer.getIteratorTypesArray());
  ...
  if (initMap.isIdentity() && llvm::hasSingleElement(*newBody) &&
      yield.getOperand(0) == newBody->getArgument(0) &&
      newConsumer.getResult(0).getType() == convResult.getType())
    rewriter.replaceOp(newConsumer, convResult);
```

`getConvBiasInfo`·`getPostAddBiasInfo` 에는 변경 전 — 없음이던 한 줄을 더했다: `if (consumer.getDpsInputs()[convInput] != convResult) return std::nullopt;`(matmul 은 `mmInput`·`mmResult`).

**같은 파일** — 다른 op 도 읽는 binding(#7). 변경 전 — 없음(신규 추가). `foldConvBias` 와 matmul 경로에 다음 검사를 더했다(matmul 은 LHS·weight·bias):

```cpp
  auto onlyFeedsOneOp = [](BlockArgument arg, Operation *load) {
    return arg.hasOneUse() && load->hasOneUse();
  };
  if (!onlyFeedsOneOp(info.inArg, info.inLoad) ||
      !onlyFeedsOneOp(info.filterArg, info.filterLoad) ||
      (info.biasLoad && !onlyFeedsOneOp(info.biasArg, info.biasLoad)))
    return fail(
        "its input, filter or bias is also read by another op of the "
        "dispatch");
```

**`KernelDispatch.cpp`** — 지원하지 않는 init·epilogue(#10). 변경 전 — 없음(신규 추가). `setRootConfigForConvPipeline` 첫머리:

```cpp
  if (!linalgOp.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>()) {
    return linalgOp.emitOpError(
        "native conv pipeline: the output must start from a linalg.fill (a "
        "bias or partial sum as the init is not supported; the Flow bias "
        "fold handles a constant bias add)");
  }
  Value convResult = linalgOp->getResult(0);
  for (Operation *user : convResult.getUsers()) {
    if (isa<IREE::TensorExt::DispatchTensorStoreOp>(user) ||
        isUnaryElementwiseEpilogue(user, convResult))
      continue;
    return linalgOp.emitOpError("native conv pipeline: its result is read by '")
           << user->getName()
           << "', which is not a unary elementwise epilogue with identity "
              "maps (a residual or bias add, or a transposed output, is not "
              "supported)";
  }
```

**같은 파일** — 명명 conv(#8). 변경 전(`setConvRootConfig`):

```cpp
      succeeded(conv)) {
    return setRootConfigForConvPipeline(entryPointFn, linalgOp, *conv,
                                        getDeviceModel(targetDevice), numRows,
                                        numCols, stackSize);
  }
```

변경 후:

```cpp
      succeeded(conv)) {
    // The pipeline's copy insertion and producer fusion handle the generic
    // form only, which is what iree-compile hands the backend (named convs
    // are generalized before dispatch formation).
    return linalgOp.emitOpError(
        "native conv pipeline: needs the conv as a linalg.generic (generalize "
        "the named op first)");
  }
```

처음에는 InsertCopyOps 가 명명 conv 도 copy 대상으로 모으게 고쳤다. 크래시는 없어졌지만, 명명 conv 는 L2→L1 복사를 코어 forall 밖에서 버퍼 전체로 해서 `chooseConvTiling` 의 L1 모델과 맞지 않았다. 그래서 그 변경은 되돌리고 위 오류로 바꿨다.

**`Utils/AMDAIEUtils.cpp`** — `chooseConvTiling` 의 4 B 정렬(#9). 변경 전 — 없음(신규 추가). 탐색 루프에 `if (!cutsAligned(ocCores, ocTile, ohTile, owTile)) continue;` 와 HWCF 의 IC 청크 조건을 더했다. 조건은 "길이 `t` 의 타일이 원소 간격 `stride` 바이트인 차원을 자르면 `t * stride` 가 4 의 배수"다. 적용 대상은 NHWC 출력·epilogue(코어 타일·블록·청크), NHWC 입력 창(OW·OH 블록 시작, 청크), HWCF 필터의 OC·IC 절단이다(코드는 위 diff 그대로).

**`Target/AIETarget.h`** — 도움말(#11). 변경 전 `"chunks of output rows on array rows, output channels on columns"`, 변경 후 `"chunks of output rows on array columns, output-channel tiles on array rows (a conv without chunks spreads only output channels, over the rows of one column)"`.

**`scripts/debug/conv_dma_check.py`**(#12) — 리뷰어가 만든 다중 블록 검사기를 바탕으로 다시 썼다.
- 제어 코드(상수 bound 의 `scf.for`, `affine.apply`, SSA offset 의 L3 전송)를 해석합니다.
- circular 전송의 앞쪽 stride-0 차원을 fill 반복으로 봅니다.
- 코어마다 블록 b 의 출력 fill 과 입력·필터 fill b·K…b·K+K−1 을 짝지어 conv 의미와 비교합니다.
- 끝으로 모든 출력 원소가 정확히 한 번 쓰였는지 봅니다.
- 전송 길이가 맞지 않는 등 어긋나는 것은 모두 FAIL 입니다.
- `--index K` 로 로그의 K 번째 dispatch 를 고릅니다.
- L3 버퍼는 연결 방향으로 구분하므로 같은 타입의 입력·출력도 다룹니다. 패딩 버퍼 내부에 쓰는 출력은 원점을 찾아 검사합니다.

**`scripts/debug/aie_audit.py`**(#13, 15) — 다섯 가지를 바꿨다.
- 덤프가 빠진 dispatch 는 위반입니다.
- BD 차원마다 wrap·step 을 32-bit word 로 바꿔 aie-rt 의 AIE2P 표 값과 비교합니다(코어 256/8,192, memtile 1,024/131,072, shim 1,024/2^20). 바꾸는 방식은 가장 안쪽 차원의 크기와 바깥 차원의 stride 를 원소 바이트 / 4 배 하는 것입니다(mlir-aie `configureBdInBlock`).
- shim 의 stride-0 차원은 가장 바깥(반복, ≤ 256)일 때만 허용합니다.
- 반복 차원은 4 차원일 때 ≤ 63 입니다.
- `AcquireGreaterEqual N > 1` 인 lock 획득(공유 카운팅 lock 쌍)을 위반으로 셉니다.

변경 전 핵심 코드:

```python
BD_SIZE_MAX = 1023
...
        for size, stride in dims:
            if size > BD_SIZE_MAX:
                res["violations"].append(f"{kind} {cur_tile} BD dim size {size} > {BD_SIZE_MAX}")
...
        res = {"violations": [], "missing_dumps": sorted(set(PASSES.values()) - set(g))}
```

변경 후 핵심 코드:

```python
BD_WRAP_MAX = {"core": 256, "memtile": 1024, "shim": 1024}
BD_STEP_MAX = {"core": 8192, "memtile": 131072, "shim": 1 << 20}
...
        for i, (size, stride) in enumerate(dims):
            ...
            for what, words, lim in bd_dim_words(i == len(dims) - 1, size, stride, eb, kind):
                if words > lim:
                    res["violations"].append(f"{kind} {cur_tile} BD dim {i} {what} {words} words > {lim}")
...
        for kind in res["missing_dumps"]:
            res["violations"].append(f"missing {kind} dump: its checks did not run")
...
    counting = Counter(int(n) for n in re.findall(
        r"aie\.use_lock\(%[\w.]+, AcquireGreaterEqual, (\d+)\)", body) if int(n) > 1)
```

도구 검증:
- 고친 매핑 검사기는 이전 근거 로그를 모두 PASS 로 재현합니다(`sample_c8`, `c8_relu`, `c8_relu_ic256`, HWCF·접힘 사례, sweep 9 개). 블록이 여러 개라 전에는 못 보던 `conv5_1_folded` 도 PASS 입니다.
- 다음은 모두 FAIL 입니다: 일부러 오염시킨 `sample_c8_corrupt`, 리뷰어의 채널 누락 합성 사례, VGG 덤프의 L3 offset 변조, L1 stride 변조.
- audit 는 core BD wrap 300, shim stride-0 반복 300, 가운데 차원 stride 0 의 합성 위반을 모두 잡습니다. 덤프 4 개가 빠진 로그는 FAIL 입니다.

#### 분류 요약

| 파일 | 변경 | 분류 |
|---|---|---|
| `AMDAIEDispatchRewriteUtils.{h,cpp}` | `getFullTensorLoad` 강화, `eraseDeadGlobals` 방문 집합, include 정리 | 변경 |
| `AMDAIEChunkConvDispatches.cpp` | `isUnaryEpilogueOf`, `useFHWCFilter`(§7.7) | 변경·추가 |
| `AMDAIEFoldBiasIntoContractionDispatches.cpp` | `getConvertedBiasBytes`, init 맵 보존, 입력 확인, 다른 사용자 오류(matmul·conv) | 추가·변경 |
| `KernelDispatch.cpp` | init·epilogue 검사, 명명 conv 오류, TensorExt include | 추가·변경 |
| `Utils/AMDAIEUtils.{h,cpp}` | 4 B 정렬 조건, HWCF 행 1 개(§7.7), 문서 주석 | 추가·변경 |
| `Target/AIETarget.h` | 도움말 | 변경 |
| `AMDAIEInsertCopyOps.cpp` | 명명 conv 수집을 넣었다가 되돌림 | 유지(최종 변경 없음) |
| `scripts/debug/conv_dma_check.py` | 다중 블록 검사기로 교체 | 변경 |
| `scripts/debug/aie_audit.py` | 덤프 누락·BD 한계·shim 반복·lock 검사 | 변경 |
| lit: `chunk_conv_dispatches`, `fold_bias_into_contraction_dispatches`, `lowering_strategy_conv_native` | 사례 추가·기대값 갱신 | 추가·변경 |
| lit: `lowering_strategy_conv_native_errors.mlir` | 오류 3 건(비-fill init, residual, 명명 conv) | 추가 |

### 7.7 출력 join 과 필터 fan-out 의 lock 쌍(T3.3)

**문제**. 한 열의 코어 4 개는 출력 채널을 나눠 갖는다(G1). 그래서 각 코어가 열의 L2 출력 버퍼에 쓰는 `[oh, ow, ocTile]` 타일은 NHWC 에서 픽셀마다 번갈아 놓인다(offset 0/16/32/48).

- `AMDAIEObjFifoBufferization` 은 생산자 슬라이스가 쓰기 출력 전송을 "연속·오름차순"으로 나눌 때만 생산자별 lock 쌍을 준다(`partiesTileTransferInOrder`). 그 밖에는 공유 카운팅 쌍(init = 생산자 × depth, 소비자 `AcquireGreaterEqual 4`)으로 물러난다.
- 이 파일의 주석과 커밋 6499776a·dd16b81c 는 이 방식을 경쟁 상태라고 적는다. 앞서 달린 코어의 release 가 4 를 채우면 다른 코어의 슬라이스가 아직 낡은 버퍼가 L3 로 나간다.
- 11-08 의 VGG 는 conv 13 개가 모두 이 꼴이었다(lock init 8, `AcquireGreaterEqual 4` 가 dispatch 마다 14–16 개).
- 실제로 깨지려면 한 행이 한 블록 이상 뒤처져야 한다. 그래서 잠재적이지만 정확성이 타이밍에 기댄다. 블록이 하나인 dispatch(작은 시험 모델)는 노출되지 않는다.

**수정 1 — 출력 L2 배치**: 새 패스 `AMDAIEConvOutputSlicesInOrder` 를 ConvertToDma 바로 뒤에 둔다. conv 파이프라인 함수만 대상이다(translation_info 의 `"conv"` override 로 판별).

- L1 에서 쓰이고 L3 로 읽히는 L2 버퍼 하나를 찾는다. 다른 DMA 가 닿지 않아야 하고, 쓰는 쪽은 채널 일부를 쓰는 DMA 하나, 읽는 쪽은 버퍼 전체를 읽는 DMA 하나여야 한다.
- 버퍼 타입은 그대로 두고, 원소 배치만 `[n?, slice, oh, ow, ocTile]` 로 바꾼다.
  - 코어의 쓰기는 자기 슬라이스(연속)에 하도록 고친다. 채널 offset 이 slice × ocTile 이므로 그 차원의 stride 를 oh·ow 로 두면 된다.
  - 쓰기 출력은 슬라이스 순서로 연속해서 읽고, 같은 차원 순서로 NHWC L3 타일에 흩뿌린다(slice 차원의 L3 stride = ocTile).
- 그러면 슬라이스가 쓰기 출력을 순서대로 나누므로, 기존 경로가 생산자별 lock 쌍을 주고 소비자 전송을 생산자별 BD 로 쪼갠다(`createDMABlocks`).

변경 전 — 없음(신규 추가: 패스 파일, `Passes.td`·`Passes.h`·`PassDetail.h`·CMake 등록, lit `conv_output_slices_in_order.mlir` 3 사례). `Passes.cpp` 에는 다음을 더했다.

```cpp
  // The conv tile pipeline's cores write interleaved channel tiles of one L2
  // output buffer; store them as contiguous slices instead, so the join gets a
  // lock pair per core (no-op for other pipelines).
  passManager.addNestedPass<func::FuncOp>(
      createAMDAIEConvOutputSlicesInOrderPass());
```

lit 의 청크 사례(L2 `[2,2,4,8]`, 코어당 채널 4)에서 다시 쓴 패턴은 다음과 같다.
- 쓰기: `[%n, %oc, 0, 0, 0] [1, 1, 2, 4, 4] [64, 8, 16, 4, 1]`
- 쓰기 출력: L2 `[2, 2, 2, 4, 4] [64, 32, 16, 4, 1]` → L3 `[2, 2, 2, 4, 4] [64, 4, 32, 8, 1]`

**수정 2 — 필터 fan-out**. 검증 중에 같은 문제가 하나 더 나왔다. 청크가 없는 conv(OH 가 작아 1 열)는 HWCF 필터를 그대로 쓰면서 출력 채널을 행에 나눈다. HWCF 는 OC 가 가장 안쪽이라 각 행의 필터 슬라이스가 번갈아 놓이고, 이번에는 소비자 쪽(fan-out)에서 공유 카운팅 쌍이 생긴다(생산자 `AcquireGreaterEqual 4`).

- `chooseConvTiling`: HWCF 필터는 청크 수와 상관없이 행 1 개다. 변경 전 `if (!dims.filterOcFirst && nCores > 1) maxOcCores = 1;`, 변경 후 `if (!dims.filterOcFirst) maxOcCores = 1;`(이유는 주석에 적음).
- `AMDAIEChunkConvDispatches`: 청크를 못 만드는 conv 도 상수 HWCF 필터면 FHWC 사본으로 바꾼다(`useFHWCFilter`, 변경 전 — 없음). 이 경로는 dispatch 피연산자, 실행 파일의 binding, conv 의 필터 맵만 바꾼다. 그러면 1 열의 행들이 FHWC 의 연속 OC 슬라이스를 받아 소비자별 lock 쌍이 된다.

**검증**(컴파일·정적 IR 만):
- `pair16`(청크 8)의 memtile (0,1): 출력 join(MM2S 2)이 BD 8 개(생산자 4 × depth 2)로 나뉘고, 각 BD 가 자기 생산자의 lock 쌍(init 2) 만 획득·해제한다. pack-peel matmul 출력 join 과 같은 구조다.
- VGG `--conv native` 감사: 16 dispatch 위반 0 이다. 새 lock 검사 기준으로 counting 획득이 0 개다(수정 전 conv 13 개 모두 14–16 개).
- VGG conv 13 개 다중 블록 매핑 검사: 모두 PASS(출력 정확히 한 번 덮음).
- 청크가 없는 conv(`pair2`: OH 2, 1 열 4 코어, 패딩 버퍼 내부 `[1,1,0]` 에 쓰는 tied 출력)는 수정 2 전에 필터 fan-out 의 counting 획득이 dispatch 마다 2 개였고, 수정 뒤 0 개다. 필터가 FHWC 사본으로 바뀌었고 매핑 검사도 PASS 다(출력 원점 `[1,1,0]` 확인).
- `pair16`·`pair14`(청크 8·7)는 counting 0, 매핑 PASS 다.
- 기본 레시피(im2col)의 NPU PDI md5 는 `2781c8d4…` 그대로다. native PDI 는 4,382,240 → 4,547,312 B 이다(BD·lock 이 늘어남). md5 는 `697f0b8a…`(이후 회귀 기준)다.
- ctest 238/238.

### 7.8 세 번째 독립 리뷰와 수정

§7.7 을 마친 뒤 세 번째 리뷰를 워크플로로 돌렸다(`wf_2d27710a-38b`). 차원은 셋이다.

1. slice 패스와 lock 경로(§7.7)
2. §7.6 수정분(fold·codegen)
3. 검사 도구(`aie_audit.py`, `conv_dma_check.py`)

검증 에이전트가 재현으로 확인한 지적은 11 건이다. lock 경로(slice 패스, 생산자별 lock 쌍, HWCF 행 1 개 규칙, FHWC 사본)에서는 결함이 나오지 않았다. 30 개가 넘는 모양에서 counting 획득이 0 개였다. 나머지 지적은 대부분 검사 도구의 빈틈이다. 컴파일러 쪽은 둘이다.
- T1: 셔임 전송의 4 B 단위를 아무도 보지 않아, 정렬되지 않은 출력이 조용히 틀린 주소로 나갈 수 있었다.
- FC-1: 진단 문제다.

재현 파일은 `log_shjo/review2/` 에, 수정 확인 결과는 `log_shjo/impl/fixcheck/r3/` 에 있다.

| ID | 대상 | 지적 | 심각도 | 수정 | 확인 |
|---|---|---|---|---|---|
| T1 | 컴파일러(`AMDAIEControlCodeLowering`), 두 도구 | bf16 셔임 전송의 시작·안쪽 길이·stride 가 4 B 단위가 아니면 하강이 조용히 내림한다. OC 3 conv 의 패딩 내부 쓰기가 byte 30 대신 28 에서 시작한다 | 주요(조용한 오답) | 하강에서 op 오류(원소 수로 접어 word 가 아니면 word 단위로 다시 접어 보고, 그 결과가 BD 자리에 맞을 때만 받음). audit 에 sub-word 검사 | lit 10 건. 재현: rc 1 + 명확한 오류, 감사 FAIL |
| FC-1 | `KernelDispatch` | 연쇄 epilogue, init 을 읽는 epilogue 가 strategy 를 통과한 뒤 엉뚱한 메시지로 실패한다 | 경미(진단) | conv 결과의 독자는 op 하나(저장, 또는 저장만 되는 epilogue 하나). 청크 패스도 epilogue 1 개까지만 받음 | lit 8 건. 재현 11 개(거부 7, 정상 4) |
| FC-2 | `aie_audit.py` | 덤프 전에 실패한 dispatch 가 있어도 PASS 한다 | 경미 | 복구되지 않은 error 줄은 위반. 빈 덤프와 코어 0 도 위반 | z1·a_empty FAIL |
| T7 | `aie_audit.py` | 다른 쪽이 코어인 공유 카운팅 쌍은 acquire 가 1 이라 잡지 못한다 | 경미 | lock init 이 DMA BD 가 지키는 버퍼 수보다 크면 위반 | 재현 IR FAIL. VGG lock 280 개 대조 |
| T2 | `conv_dma_check.py` | 출력 원점을 데이터에서 추정하고 범위를 보지 않는다 | 주요 | 원점 항상 출력. 출력이 들어갈 자리, 같은 모양이면 0, `--origin` | m2 FAIL. m1a·m1b 는 `--origin 1,1,0` 으로 FAIL |
| T3 | `conv_dma_check.py` | 코어가 쓰지 않은 드레인 원소를 보지 않는다 | 경미 | 조각별 코어 쓰기 = 드레인 위치, 드레인 전체 = 출력 상자 | m10 FAIL |
| T4 | `conv_dma_check.py` | `K = IC // ICt` 가 내림이라 꼬리 채널(bias 포함)이 빠져도 PASS | 경미 | `IC % ICt == 0` | m4 FAIL |
| T5 | `conv_dma_check.py` | L3 전송이 어느 binding 을 읽고 쓰는지 보지 않는다 | 경미 | 역할별 binding 하나, 셋이 서로 다름, 역할의 모양, L3 범위 | m3·m3b·m9 FAIL |
| T6 | `conv_dma_check.py` | 반복 횟수(앞쪽 stride-0)를 objectFifo 의 공통 횟수와 대조하지 않는다 | 경미 | memtile 조각의 모든 쪽에서 반복 횟수가 같아야 함(L1 쪽은 하강이 따로 접어 제외) | m5 FAIL |
| LOCK-T1·T8 | `conv_dma_check.py` | 필터를 "3 인 차원 둘"로 찾아 3×3 이 아니면 거짓 FAIL | 경미 | 모양이 맞는 쌍으로 역할 결정 | 5×5 두 사례 PASS |


#### 원인과 수정 방향

**T1 — 셔임 BD 는 32-bit word 단위다.** 셔임 BD 의 주소 필드는 word 단위다(`BASE_ADDRESS_LOW` mask `0xFFFFFFFC`). 크기와 stride 도 word 로 센다.
- `AMDAIEControlCodeLowering` 은 `size * 16 / 32`, `stride * 16 / 32` 로 바꾸며 나머지를 버린다. byte offset 은 `npu.address_patch` 에 그대로 넘긴다.
- 리뷰 재현: OC 3 bf16 conv 가 다음 층의 패딩 버퍼 `4x4x3` 내부 `[1,1,0]` 에 쓴다. 시작이 원소 15 = byte 30 이라 BD 는 byte 28 부터 쓴다. 행마다 bf16 하나씩 앞으로 밀린다. 그래도 compile rc 0, audit PASS, 매핑 검사 PASS 였다.
- 청크 경로는 출력을 자기 버퍼에 쓰고 host insert 가 패딩 내부로 복사한다(§7.2). 그래서 이 문제는 청크가 없는 conv(OH 가 작은 층)의 tied 내부 쓰기에서만 생긴다. 또 원소가 4 B 보다 좁고 OC 가 홀수일 때만이다. VGG·pair 로그를 스캔했더니 이런 전송은 없었다. 앞서 낸 결과는 영향이 없다.
- 수정: 하강에서 4 B 단위가 아닌 셔임 접근을 op 오류로 멈춘다. 시작(reinterpret_cast offset 포함), 바깥 stride, 안쪽 길이가 모두 word 여야 하고, 안쪽 stride 는 1 이어야 한다.
  - 수정 검토(CF1, F1)를 거치며 한 가지를 더했다. 원소 수 한계로 접은 결과가 word 가 아니면, 안쪽 차원을 word 수 한계로 다시 접어 본다. 다시 접은 결과는 각 차원이 BD 자리의 한계 안에 있을 때만 받아들인다.
  - 정렬을 맞추는 변환(이런 conv 도 host insert 로 돌리기)은 §8 에 남긴다.

**FC-1 — epilogue 검사**. `isUnaryElementwiseEpilogue` 는 GEMV·pad 패스와 같이 쓰는 술어다. payload 가 init 을 읽는지는 보지 않는다.
- 그래서 init 을 읽는 epilogue 와 연쇄 epilogue(ReLU generic → truncf generic)가 strategy 를 통과했다.
- 그 뒤 `linalg.copy` verifier 오류, 진단 없는 bufferize 실패, "needs logic implemented" 로 이유 없이 실패했다.
- 공유 술어는 그대로 두고, conv 파이프라인에서 두 조건을 따로 본다. 청크 패스도 epilogue 를 하나까지만 받아, strategy 가 거부할 dispatch 를 쪼개지 않는다.

**검사 도구**. 리뷰어가 덤프를 손으로 바꾼 변이 로그(`log_shjo/review2/tools/mut/`)로 확인했다. 원래 검사기가 PASS 한 변이 9 개(m1a·m1b·m2·m3·m3b·m4·m5·m10, audit a1)와 빈 덤프(a_empty)가 그 예다.
- T6 은 처음에 "반복 횟수 = 채움 횟수"로 고쳤다가 VGG 12 개 conv 에서 거짓 FAIL 이 났다. 순환 DMA 의 반복 횟수는 BD 체인 한 바퀴의 lock 사이클 수다. 체인은 lock 이 허락하는 만큼 계속 돈다. 예: VGG #1 은 반복 32, 채움 288 = 32 × K 9.
- 그래서 하강이 실제로 쓰는 규칙으로 바꿨다. `getRepetitionCount` 가 사용자들의 최솟값을 lock 사이클로 쓰고, 남는 횟수는 `createDMABlocks` 가 한 사이클 안의 BD 로 풀어 같은 채움을 다시 보낸다. 따라서 memtile 조각의 모든 쪽에서 반복 횟수가 같아야 한다. L1 쪽은 하강이 따로 접으므로 비교하지 않는다(수정 검토 MF4).
- T2 의 내부 원점은 이 덤프에 없다. 패딩 버퍼 내부 쓰기는 `--origin H,W,C` 로 기대값을 줘야 m1a·m1b 같은 이동을 잡는다. 기본 실행은 원점을 출력하고, 같은 모양이면 0, 출력이 들어갈 자리가 있는지만 본다.

#### 코드 변경(변경 전 / 변경 후)

**`KernelDispatch.cpp`** — `setRootConfigForConvPipeline` 의 epilogue 검사(FC-1). 변경 전:

```cpp
  for (Operation *user : convResult.getUsers()) {
    if (isa<IREE::TensorExt::DispatchTensorStoreOp>(user) ||
        isUnaryElementwiseEpilogue(user, convResult))
      continue;
    return linalgOp.emitOpError("native conv pipeline: its result is read by '")
           << user->getName()
           << "', which is not a unary elementwise epilogue with identity "
              "maps (a residual or bias add, or a transposed output, is not "
              "supported)";
  }
```

변경 후(conv 결과의 독자는 op 하나. §7.8 의 수정 검토에서 넓힌 판):

```cpp
  // One output path: the result is stored, or read by one epilogue whose
  // result is stored. AMDAIEInsertCopyOps copies the accumulator out only when
  // no epilogue reads it, so a second reader (another epilogue, or a store of
  // the raw result next to the epilogue) has nowhere to go. Readers are ops,
  // not uses: an in-place epilogue takes the result as input and init.
  auto readersOf = [](Value v) {
    return SmallPtrSet<Operation *, 2>(v.getUsers().begin(),
                                       v.getUsers().end());
  };
  Value convResult = linalgOp->getResult(0);
  SmallPtrSet<Operation *, 2> convReaders = readersOf(convResult);
  if (convReaders.size() != 1) {
    return linalgOp.emitOpError("native conv pipeline: its result has ")
           << convReaders.size()
           << " readers (one store or one unary elementwise epilogue is "
              "supported)";
  }
  Operation *user = *convReaders.begin();
  if (!isa<IREE::TensorExt::DispatchTensorStoreOp>(user)) {
    if (!isUnaryElementwiseEpilogue(user, convResult)) {
      return linalgOp.emitOpError(
                 "native conv pipeline: its result is read by '")
             << user->getName()
             << "', which is not a unary elementwise epilogue with identity "
                "maps (a residual or bias add, or a transposed output, is not "
                "supported)";
    }
    // The epilogue computes a core tile of its own from the accumulator and
    // goes straight to the output, so it may not read its init and its result
    // may only be stored.
    auto epilogue = cast<linalg::GenericOp>(user);
    if (epilogue.payloadUsesValueFromOperand(epilogue.getDpsInitOperand(0))) {
      return linalgOp.emitOpError(
          "native conv pipeline: its epilogue reads its own init (only an "
          "epilogue of the conv result alone is supported)");
    }
    SmallPtrSet<Operation *, 2> epilogueReaders =
        readersOf(epilogue->getResult(0));
    if (epilogueReaders.size() != 1) {
      return linalgOp.emitOpError(
                 "native conv pipeline: its epilogue's result has ")
             << epilogueReaders.size() << " readers (one store is supported)";
    }
    Operation *epilogueUser = *epilogueReaders.begin();
    if (!isa<IREE::TensorExt::DispatchTensorStoreOp>(epilogueUser)) {
      return linalgOp.emitOpError(
                 "native conv pipeline: its epilogue's result is read by '")
             << epilogueUser->getName()
             << "' (a chain of epilogues is not supported; fuse them into one "
                "linalg.generic)";
    }
  }
```

같은 함수의 epilogue L1 예산(수정 검토의 CF2). 변경 전:

```cpp
  // A fused elementwise epilogue writes its own core tile next to the
  // accumulator (see `AMDAIEInsertCopyOps`).
  int64_t bytesEpilogue = 0;
  for (Operation *userOp : linalgOp->getUsers()) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(userOp);
    if (linalgUser && isElementwise(linalgUser) &&
        !isa<linalg::FillOp, linalg::CopyOp>(linalgUser))
      bytesEpilogue += bytesOf(linalgUser.getDpsInitOperand(0)->get());
  }
```

변경 후(in-place epilogue 가 사용 2 개로 두 번 잡히던 것을 op 하나로):

```cpp
  // A fused elementwise epilogue writes its own core tile next to the
  // accumulator (see `AMDAIEInsertCopyOps`), one tile even when it is in place.
  int64_t bytesEpilogue = 0;
  if (auto epilogue = dyn_cast<linalg::GenericOp>(user))
    bytesEpilogue = bytesOf(epilogue.getDpsInitOperand(0)->get());
```

**`AMDAIEChunkConvDispatches.cpp`** — `getConvChunkInfo` 의 epilogue(FC-1). 구조체 주석도 "a chain of unary elementwise generics" → "at most one unary elementwise generic ... the conv pipeline takes no chain" 으로 고쳤다. 변경 전:

```cpp
  Value v = info.conv->getResult(0);
  while (v.hasOneUse()) {
    auto generic = dyn_cast<linalg::GenericOp>(*v.getUsers().begin());
    if (!generic || !isUnaryEpilogueOf(generic, v)) break;
    info.epilogue.push_back(generic);
    v = generic->getResult(0);
  }
```

변경 후:

```cpp
  Value v = info.conv->getResult(0);
  if (v.hasOneUse()) {
    auto generic = dyn_cast<linalg::GenericOp>(*v.getUsers().begin());
    if (generic && isUnaryEpilogueOf(generic, v)) {
      info.epilogue.push_back(generic);
      v = generic->getResult(0);
    }
  }
```

**`AMDAIEControlCodeLowering.cpp`** — `insertWriteBdOps`, `foldLinearDims` 와 그 뒤(T1, 수정 검토의 CF1). 변경 전(검사 없음, 접기는 원소 수 상한만):

```cpp
    SmallVector<OpFoldResult> linearOffsets, linearSizes, linearStrides;
    (void)foldLinearDims(
        rewriter.getContext(), offsets, sizes, strides, linearOffsets,
        linearSizes, linearStrides, [&](size_t idxFromEnd, int64_t size) {
          return idxFromEnd < maxSizes.size() &&
                 size <= maxSizes[maxSizes.size() - idxFromEnd - 1];
        });
```

변경 후(sub-word 검사 추가. 원소 수로 접은 결과가 word 가 아니면 안쪽 차원을 word 수 상한으로 다시 접어 본다. 다시 접은 결과는 각 차원이 들어갈 BD 자리(반복 횟수, iteration, intra 차원)의 한계 안에 있어야 받아들인다. 지금 통과하는 패턴의 접기는 바뀌지 않는다):

```cpp
    SmallVector<OpFoldResult> linearOffsets, linearSizes, linearStrides;
    // `innermostInWords` limits a merged innermost dim by its size in words,
    // what its wrap counts once converted below, instead of in elements.
    auto foldDims = [&](bool innermostInWords) {
      linearOffsets.clear();
      linearSizes.clear();
      linearStrides.clear();
      (void)foldLinearDims(
          rewriter.getContext(), offsets, sizes, strides, linearOffsets,
          linearSizes, linearStrides, [&](size_t idxFromEnd, int64_t size) {
            if (innermostInWords && idxFromEnd == 0) {
              size = (size * elemWidthInBits + minStrideBitWidth - 1) /
                     minStrideBitWidth;
            }
            return idxFromEnd < maxSizes.size() &&
                   size <= maxSizes[maxSizes.size() - idxFromEnd - 1];
          });
    };
    foldDims(/*innermostInWords=*/false);

    // The shim BD counts its base address, strides and innermost size in
    // `minStrideBitWidth` words, and the conversion below floors whatever part
    // of a word is left. For elements narrower than a word that would silently
    // drop or shift data, so such an access must be whole words: the offset
    // and the outer strides, and a unit-stride innermost run (the whole
    // transfer when it folds to no dims at all, a single element). Returns
    // what breaks that, if anything.
    auto subWordProblem = [&]() -> std::optional<std::string> {
      std::string problem;
      llvm::raw_string_ostream os(problem);
      auto notWords = [&](StringRef what, int64_t numElems) {
        os << "accesses " << elemWidthInBits << "-bit elements with " << what
           << " of " << numElems << " elements, which is not a whole number of "
           << +minStrideBitWidth
           << "-bit words (the shim DMA counts its address, strides and "
              "innermost size in words)";
        return os.str();
      };
      auto isWords = [&](int64_t numElems) {
        return numElems * elemWidthInBits % minStrideBitWidth == 0;
      };
      int64_t elemOffset = bufferOffset + reinterpretElemOffset;
      if (!isWords(elemOffset)) return notWords("a base offset", elemOffset);
      if (linearSizes.empty() && !isWords(bufferLength))
        return notWords("an innermost run", bufferLength);
      for (size_t i = 0, e = linearSizes.size(); i < e; ++i) {
        int64_t size = getConstantIndexOrAssert(linearSizes[i]);
        int64_t stride = getConstantIndexOrAssert(linearStrides[i]);
        if (i + 1 == e) {
          if (size > 1 && stride != 1) {
            os << "accesses " << elemWidthInBits
               << "-bit elements with an innermost stride of " << stride
               << " elements (the shim DMA moves whole " << +minStrideBitWidth
               << "-bit words, so sub-word elements must be contiguous in the "
                  "innermost dim)";
            return os.str();
          }
          if (!isWords(size)) return notWords("an innermost run", size);
        } else if (size > 1 && stride != 0 && !isWords(stride)) {
          return notWords("a stride", stride);
        }
      }
      return std::nullopt;
    };
    // Whether each folded dim fits the BD dim it lands in (see the loop
    // below): the outermost is the repeat count (stride 0) or, when every
    // address dim is used, the iteration; the others are intra dims, the
    // innermost counted in words. An intra wrap field holds at most WrapMax (a
    // WrapMax + 1 that aie-rt accepts is encoded as 0); the highest intra dim
    // has no wrap field, but aie-rt still takes at most WrapMax + 1 there.
    FailureOr<uint16_t> wrapMax = deviceModel.getDmaBdProp<uint16_t>(
        tileType, 0, AMDAIE::AMDAIEDmaBdProp::WrapMax);
    FailureOr<uint8_t> iterWrapMax = deviceModel.getDmaBdProp<uint8_t>(
        tileType, 0, AMDAIE::AMDAIEDmaBdProp::IterWrapMax);
    if (failed(wrapMax) || failed(iterWrapMax))
      return op.emitOpError() << "could not get the DMA BD wrap limits";
    int64_t maxRepeatCount = deviceModel.getMaxRepeatCount(tileType);
    auto bdDimsProblem = [&]() -> std::optional<std::string> {
      size_t n = linearSizes.size();
      if (n > numAddrDim) return std::nullopt;  // Reported below.
      for (size_t i = 0; i < n; ++i) {
        int64_t size = getConstantIndexOrAssert(linearSizes[i]);
        int64_t stride = getConstantIndexOrAssert(linearStrides[i]);
        size_t fromInnermost = n - 1 - i;
        StringRef what = "an intra dim";
        int64_t limit = *wrapMax;
        if (i == 0 && (n == numAddrDim || stride == 0)) {
          what = stride == 0 ? "a repeat count" : "an iteration count";
          limit = stride == 0 ? maxRepeatCount : *iterWrapMax + 1;
        } else if (fromInnermost == 0) {
          what = "an innermost dim (in words)";
          size = size * elemWidthInBits / minStrideBitWidth;
        } else if (fromInnermost + 1 == numIntraAddrDim) {
          what = "a highest intra dim";
          limit = *wrapMax + 1;
        }
        if (size <= limit) continue;
        std::string problem;
        llvm::raw_string_ostream os(problem);
        os << "folds to a shim BD with " << what << " of " << size
           << ", more than its limit of " << limit;
        return os.str();
      }
      return std::nullopt;
    };
    if (elemWidthInBits < minStrideBitWidth && bufferLength > 0) {
      std::optional<std::string> problem = subWordProblem();
      // Contiguous rows of an odd number of elements stay unmerged when their
      // total passes the wrap counted in elements, yet the merged run may be
      // whole words within the wrap counted in words. Merging moves every
      // outer dim down one BD dim, so the refold must also fit the BD; if it
      // does not, the first fold's problem stands.
      if (problem) {
        std::string firstProblem = *problem;
        foldDims(/*innermostInWords=*/true);
        problem = subWordProblem();
        if (!problem) problem = bdDimsProblem();
        if (problem) problem = firstProblem;
      }
      if (problem) return op.emitOpError() << *problem;
    }
```

**lit**. 변경 전 — 없음(신규 추가). 기존 파일에 사례를 더했다.
- `controlcode_lowering.mlir`: bf16 셔임 접근 10 사례.
  - 오류 8 개: 시작 원소 15, 안쪽 stride 2, 안쪽 길이 3, 바깥 stride 5, 원소 1 개(접으면 차원 없음), reinterpret_cast offset 15, 1025 원소 행 2 개(word 로도 상한 초과), `[2, 1024, 26, 45]`(다시 접으면 1024 행 차원이 D1 로 내려가 맞지 않음).
  - 정상 2 개:
    - reinterpret_cast offset 15 + 패턴 offset 1 = 원소 16 이다. address_patch 32, buffer_length 8 word 로 내려간다.
    - 45 원소 행 26 개(1,170 원소 = 585 word)는 한 차원 585 word 로 내려간다.
    - 정상 사례마다 ADD1·ADD2 접두어를 끊었다.
  - 기존 bf16 사례(시작 32, 정렬됨)는 그대로 통과한다.
- `lowering_strategy_conv_native_errors.mlir`: 오류 5 사례를 더하고 머리 주석을 갱신했다.
  - `conv_epilogue_chain`(ReLU → truncf 두 generic)
  - `conv_epilogue_reads_init`(readwrite 출력에 `out = conv + out`)
  - `conv_two_epilogues`
  - `conv_raw_and_epilogue`(원본 저장 + epilogue)
  - `conv_epilogue_stored_twice`
- `lowering_strategy_conv_native.mlir`: `conv_inplace_relu`(결과가 입력이자 init 인 epilogue)와 `conv_relu_empty_init`(같은 ReLU, 빈 init). 둘 다 conv 파이프라인을 받고, 타일이 `[[7,7,50,…],[…,8],[0,0,50,…]]` 로 같아야 한다.
- `chunk_conv_dispatches.mlir`: "청크 안 함" 묶음에 `conv_chain`(OH 4 라 전에는 2 청크로 쪼개졌다)을 더했다. 원래 피연산자 그대로 남는지 본다.

**`scripts/debug/aie_audit.py`**. 독스트링에 새 검사와 "compile rc 는 여기서 보지 않는다"를 적었다.

FC-2 — `audit()` 의 진단. 변경 전:

```python
    for d in DIAGNOSTICS:
        n = text.count(d)
        if n:
            report["diagnostics"][d] = n
    if not groups:
        report["error"] = "no recognized IR dump headers: add the --print-flags flags to iree-compile"
        return report
```

변경 후(`count_errors` 는 새 함수다. error 줄을 센다. 다만 note 줄들 바로 뒤에 "순차 배치로 넘어간다"는 경고가 오는 오류는 bank-aware 배치가 복구한 것이라 따로 센다. 버퍼를 뱅크에 못 놓은 경우, 스택이 뱅크보다 큰 경우가 그렇다. 순차 배치마저 실패하면 그 오류는 그대로 센다):

```python
    for d in DIAGNOSTICS:
        n = text.count(d)
        if n:
            report["diagnostics"][d] = n
    n_err, n_bank = count_errors(text)
    if n_err:
        report["diagnostics"]["error diagnostics in the log"] = n_err
    report["bank_fallbacks"] = n_bank
    if not groups:
        report["error"] = "no recognized IR dump headers: add the --print-flags flags to iree-compile"
        report["violations"] = sum(report["diagnostics"].values())
        return report
```

dispatch 마다(변경 전 — 없음):

```python
        for kind, body in sorted(g.items()):
            if "func.func" not in body and "aie.device" not in body:
                res["violations"].append(f"{kind} dump holds no IR: its checks did not run")
        ...
        if res.get("cores") == 0:
            res["violations"].append("no core tiles")
```

T1 — `audit_shim`. 변경 전에는 접지 않은 차원(크기 1 만 뺌)으로 BD 한계를 봤고, sub-word 검사는 없었다. 변경 후:
- from_memref 의 memref offset 을 `base` 로 모은다.
- scf.for(상수 경계)·affine.apply 로 SSA offset 이 갖는 값을 구한다. 시작 바이트를 4 로 나눈 나머지만 추적한다.
- 차원은 하강과 같이 접는다(`fold_dims`, `shim_max_sizes`). 크기 1 차원을 빼고, 이어진 차원은 `DmaDimConfig::getMaxSizes` 한계 안에서 합친다.
  - 원소가 4 B 보다 좁고 그렇게 접은 결과가 word 가 아니면, 안쪽 차원을 word 수 한계로 다시 접는다. 하강과 같다.
- 반복·wrap·step 한계도 접힌 차원으로 본다. 자리마다 한계가 다르다(세 번째 수정 검토의 AF1-D2 반영).
  - D0·D1 의 wrap 은 1023 이하다. wrap 필드가 10 비트라, aie-rt 가 받는 1024 는 0 으로 쓰인다.
  - 3 개 intra 차원 중 가장 높은 D2 는 wrap 필드가 없지만 aie-rt 가 1024 이하만 받는다.
  - iteration 은 64 이하다(aie-rt 가 wrap−1 로 쓴다). 앞쪽 stride-0 반복 횟수는 256 이하다.

```python
            real = fold_dims(sizes, strides, eb or 4)
            problems = []
            if eb and eb < 4:
                # The start's bytes mod 4, over every value an offset takes; an
                # offset on a dim whose stride is whole words cannot change it.
                rest = {base.get(side.group(1), 0) * eb % 4}
                for tok, st in zip(offsets, strides):
                    if rest is None or st * eb % 4 == 0:
                        continue
                    vs = values(tok)
                    rest = None if vs is None else {(r + v * st * eb) % 4 for r in rest for v in vs}
                if rest is None:
                    unchecked += 1
                elif rest != {0}:
                    res["violations"].append(f"shim base offset not a multiple of 4 B: {line[:160]}")
                problems = sub_word_problems(real, eb)
                if problems:
                    real = fold_dims(sizes, strides, eb, True)
                    problems = sub_word_problems(real, eb)
            for problem in problems:
                res["violations"].append(f"{problem}: {line[:160]}")
```

값을 못 구한 offset 은 `shim_offsets_unchecked` 로 출력한다. 접힌 차원으로 바꾸자, 리뷰 변이 a3(`[64, 2, 2, 9] [36, 18, 9, 1]`)의 "반복 64 > 63" 이 거짓 FAIL 이었음이 드러났다. 하강은 이를 `[64, 36]` 두 차원으로 접어 반복 차원을 쓰지 않는다. 진짜 반복 초과(접히지 않는 4 차원, 반복 65)는 FAIL 한다(a3c). 반복 64 는 aie-rt 한계 안이라 PASS 한다(a3b).

BD — `audit_bds`(CF1 재현 조사 중 추가, 변경 전 — 없음). 코어·memtile BD 의 바깥 stride 와 안쪽 크기가 4 B 배수인지 본다. 런타임의 `configureDMABD`(`iree_aie_configure.cc`)가 직렬화 때 거부하는 규칙과 같다.

```python
        # The BD counts strides and the innermost size in 32-bit words
        # (configureDMABD in iree_aie_configure.cc rejects anything else).
        if dims and eb:
            for i, (size, stride) in enumerate(dims[:-1]):
                if stride * eb % 4:
                    res["violations"].append(f"{kind} {cur_tile} BD dim {i} stride {stride * eb} B not a multiple of 4")
            if dims[-1][0] * eb % 4:
                res["violations"].append(f"{kind} {cur_tile} BD innermost size {dims[-1][0] * eb} B not a multiple of 4")
```

T7 — `audit_aie`. 변경 전에는 `AcquireGreaterEqual N > 1` 만 셌다. 변경 후 다음을 더했다. BD 블록(`^bb` 라벨이나 `aie.end` 로 끊음)마다 쓰는 lock 과 `dma_bd` 버퍼를 모은다. lock 마다 지키는 버퍼 집합을 만들고, init 이 그보다 크면 위반이다.

```python
    over = sorted((lk, inits[lk], len(b)) for lk, b in guarded.items() if b and inits.get(lk, 0) > len(b))
    res["locks_init_over_buffers"] = len(over)
    for lk, init, nb in over[:10]:
        res["violations"].append(f"lock {lk} starts at {init} but its DMA BDs guard {nb} buffers: a shared "
                                 "counting pair (one party can run ahead into the next buffer)")
```

**`scripts/debug/conv_dma_check.py`**. 인자를 argparse 로 바꾸고 `--origin` 을 더했다. 독스트링에 새 검사, 반복 횟수의 뜻, 모델링하지 않는 경우(memtile 재사용)를 적었다.

LOCK-T1·T8 — 역할. 변경 전:

```python
    flt_ty = [t for t in read if shp[t].count(3) >= 2]
    in_ty = [t for t in read if t not in flt_ty]
    assert len(out_ty) == 1 and len(in_ty) == 1 and len(flt_ty) == 1, (out_ty, in_ty, flt_ty)
```

변경 후(리뷰 검증자의 수정과 같다):

```python
    def fits(i, f):
        n_, hi_, wi_, ic_ = ([1] + shp[i])[-4:]
        if len(shp[f]) != 4: return False
        kh_, kw_, icf_, oc_ = shp[f] if hwcf else (shp[f][1], shp[f][2], shp[f][3], shp[f][0])
        nb_, hb_, wb_, cb_ = ([1] + shp[out_ty[0]])[-4:]
        return (icf_ == ic_ and nb_ == n_ and 1 <= hi_ - kh_ + 1 <= hb_
                and 1 <= wi_ - kw_ + 1 <= wb_ and oc_ <= cb_)
    pairs = [(i, f) for i in read for f in read if i != f and fits(i, f)] if len(out_ty) == 1 else []
    assert len(out_ty) == 1 and len(read) == 2 and len(pairs) == 1, (out_ty, read, pairs)
    in_ty, flt_ty = [pairs[0][0]], [pairs[0][1]]
```

T4 — 변경 전 `K = IC // ICt`. 변경 후:

```python
        if IC % ICt:
            bad.append(f"{core}: IC {IC} is not a whole number of {ICt}-channel chunks"); continue
        K = IC // ICt
```

T2 — 원점. 변경 전에는 원점을 구해 쓰기만 하고, 출력이 버퍼와 같은 모양이면 출력하지도 않았다. 변경 후:

```python
    origin = (orig_h, orig_w, orig_c)
    if orig_h + OH > Hb or orig_w + OW > Wb or orig_c + OC > Cb:
        bad.append(f"output origin {list(origin)} leaves no room for {OH}x{OW}x{OC} in a {Hb}x{Wb}x{Cb} buffer")
    if (Hb, Wb, Cb) == (OH, OW, OC) and origin != (0, 0, 0):
        bad.append(f"output origin {list(origin)} in a buffer of the output's shape")
    if want_origin is not None and origin != want_origin:
        bad.append(f"output origin {list(origin)} != expected {list(want_origin)}")
```

T5 — binding 과 범위(변경 전 — 없음).
- 제어 코드 해석기가 L3 전송마다 lof 를 기록한다. 원소 범위가 그 lof 의 memref 밖이면 MISMATCH 다.
- 역할별로 lof → `reinterpret_cast`/`assume_alignment` → subspan 을 따라 binding 을 구한다.
- 역할마다 binding 은 하나여야 하고, 셋은 서로 달라야 한다. 읽는 lof 는 그 역할의 모양이어야 한다.
- 수정 검토(MF1·MF2)에서 둘을 더했다.
  - 전송의 기준(reinterpret_cast offset, 없으면 0)은 BD 주소가 되는 값이다. 이것이 subspan 의 offset 이 가리키는 피연산자 위치와 같아야 한다.
  - 입력·필터 binding 은 ReadOnly 이고 출력은 아니어야 한다.
  - 세 번째 수정 검토(MF1-ROLE-BASE)에서, 역할마다 기준이 하나여야 한다는 조건을 더했다. 같은 binding 의 두 subspan 을 각자 맞는 cast 로 쓰는 경우를 막는다.

T6 — 반복 횟수(변경 전 — 없음). memtile 조각의 모든 쪽에서 앞쪽 stride-0 곱이 같아야 한다. 입력·필터는 조각 읽기 쪽과 채움 쪽, 출력은 조각 쓰기 쪽과 드레인 쪽이다. 처음에는 L1 쪽도 비교했다. 그런데 L1 objectFifo 의 DMA 사용자는 그 전송 하나뿐이라, 하강은 그 쪽을 따로 다 접는다. 그래서 수정 검토(MF4)에서 L1 쪽 비교를 뺐다. L1 쪽만 바꾼 변이는 하강 결과 AIE IR 이 같았다.

```python
            reps = (repetitions(circ[n1]["src"]), repetitions(circ[n2]["tgt"]))
            if len(set(reps)) != 1:
                bad.append(f"{core}: {role} piece repetitions (read, fill) {reps} differ")
```

T3 — 드레인(변경 전 — 없음). 조각마다 코어들이 쓰는 위치가 드레인이 읽는 위치와 정확히 같아야 한다(개수까지). 모든 드레인이 L3 에 쓰는 원소를 모으면 출력 상자와 정확히 같아야 한다. 결과 줄에 `drained to L3 <n> exact <bool>` 과 binding 이 나온다.
- 수정 검토(MF3)에서 처음 판이 코어마다 첫 출력 연결만 본다는 점이 드러났다. 그래서 경로 밖 이동을 다 센다.
  - 코어마다 출력 연결 1 개, 입력 연결 2 개(입력·필터)여야 한다.
  - DMA 채널을 쓰는 연결은 모두 입력·필터·출력 경로 위에 있어야 한다. 셔임의 CTRL 연결은 제외한다.
  - L3 전송은 모두 어떤 역할의 것이어야 한다.
  - 조각의 쓰기는 그 조각을 대상으로 하는 모든 연결에서 모은다.

#### 검증(컴파일·정적 IR·lit/ctest 만, NPU 미사용)

빌드는 `JOBS=4 ./scripts/build/build.sh --target iree-opt iree-compile`(호스트 잠금 디렉터리를 마운트한 컨테이너)로 했다. 결과는 `log_shjo/impl/fixcheck/r3/` 에 있다.

- **lit·ctest**: 최종 빌드(v23)에서 conv·controlcode 관련 lit 11 개가 통과한다. `ctest -R amd-aie` 는 **238/238** 이다(`log_shjo/impl/ctest_v15.log`). 새 사례는 기존 파일에 들어가 테스트 수는 그대로다.
- **컴파일러 재현**:

| 입력 | 수정 전 | 수정 후 |
|---|---|---|
| `pair2_oc3`(OC 3 bf16 → 4×4×3 패딩 내부) | rc 0, BD 가 byte 28 부터 씀, audit·매핑 PASS | rc 1: `accesses 16-bit elements with a base offset of 15 elements, which is not a whole number of 32-bit words`, audit FAIL |
| `c_chain`(ReLU → truncf 두 generic) | strategy 통과 → `linalg.copy` verifier 오류 | strategy 오류: `its epilogue's result is read by 'linalg.generic' (a chain of epilogues is not supported ...)` |
| `c_chain32`(ReLU → ×2, f32) | 진단 없이 bufferize 실패 | 같은 strategy 오류 |
| `c_readinit`(`out = conv + out`) | `needs logic implemented` | strategy 오류: `its epilogue reads its own init ...` |
| `c_fused`(ReLU+truncf 한 generic), `c_none` | 컴파일, audit PASS | 그대로(rc 0, audit PASS) |
| `f1_siblings`(epilogue 둘), `f2_raw_and_epi`(원본 저장 + epilogue), `f3b_raw_twice`(원본 두 번 저장) | strategy 통과 → 늦은 실패 또는 크래시 | `its result has 2 readers (one store or one unary elementwise epilogue is supported)` |
| `f3_store_twice`(epilogue 결과 두 번 저장) | strategy 통과 → 크래시(exit 139) | `its epilogue's result has 2 readers (one store is supported)` |
| `f5_inplace`(결과가 입력이자 init), `f6_fill_init` | 컴파일, audit PASS | 그대로(rc 0, audit PASS) |
| `t1_allunit`(bf16 원소 1 개, 원소 15) | rc 0, buffer_length 0, 주소 byte 30 | 오류(base offset 15) |
| `R1`(reinterpret offset 15), `R2`(15 + 1) | R1 은 조용히 내림 | R1 오류. R2 는 address_patch 32 로 내려감 |

- **도구 변이 로그**(`log_shjo/review2/tools/mut/`):

| 로그 | 바꾼 것 | 수정 전 | 수정 후 |
|---|---|---|---|
| m0 | 없음(대조군, pair2 #0) | PASS | PASS(원점 `[1,1,0]` 출력) |
| m1a·m1b | 출력을 `[0,0,0]`·`[2,2,0]` 에 씀 | PASS | 기본 PASS, `--origin 1,1,0` 이면 FAIL |
| m2 | 열마다 출력 기준을 16 원소 밀어 버퍼 밖에 씀 | PASS | FAIL(L3 범위 밖, 원점) |
| m3 | 입력 DMA 가 출력 binding 을 읽음 | PASS | FAIL(역할이 binding 공유) |
| m3b | 필터 DMA 가 256 원소 입력 lof 를 2,304 원소 패턴으로 읽음 | PASS | FAIL(범위, 모양) |
| m4 | IC 24 중 16 채널만 이동 | PASS | FAIL(`IC 24 is not a whole number of 16-channel chunks`) |
| m5 | 조각→L1 반복 64, L3→조각 32 | PASS | FAIL(반복 횟수 다름) |
| m6–m9 | 필터 offset·채널 공유·입력 offset | FAIL | FAIL |
| m10 | 드레인이 80 원소를 L3 에 씀 | PASS | FAIL(조각 드레인 ≠ 코어 쓰기, 드레인 80 ≠ 64) |
| a1(audit) | bf16 셔임 읽기의 안쪽 stride 2 | PASS | FAIL |
| a3(audit) | `[64, 2, 2, 9] [36, 18, 9, 1]`, 반복 64 로 보던 패턴 | FAIL | PASS. 하강이 `[64, 36]` 두 차원으로 접으므로 거짓 FAIL 이었다 |
| a3c(audit, 새 변이) | 접히지 않는 4 차원, 반복 65 | — | FAIL(64 는 PASS) |
| a2(audit) | 코어 BD 256 word(경계값, 대조군) | PASS | PASS |
| a_empty(audit) | 헤더만 있는 빈 덤프 5 개 | PASS | FAIL(빈 덤프 5, 코어 0) |
| z1.log(audit) | dispatch_2 가 덤프 전에 실패, rc 1 | PASS | FAIL(error 3) |
| 공유 카운팅 쌍 IR(audit_aie) | 생산자 DMA 2 + 코어 소비자, init 4 | 위반 없음 | `lock %lock_0_2 starts at 4 but its DMA BDs guard 2 buffers` |
| 5×5 두 사례(k5 FHWC, conv5x5 HWCF) | 3×3 이 아닌 커널 | FAIL(거짓) | PASS |

- **실제 컴파일**(새 바이너리):
  - `pair2`·`pair16`·`pair14`: rc 0, audit PASS. dispatch 마다 counting 획득 0, init 초과 lock 0 이다. 매핑 6/6 PASS, 드레인 정확(`pair2` 는 `--origin 1,1,0`).
  - VGG `--conv native` 감사 컴파일: 115.4 s. 16 dispatch 위반 0, 복구된 bank 오류 252 개(따로 셈)다. lock 280 개가 모두 init 2 · 버퍼 2(또는 init 0)다. 셔임 offset 을 못 구한 전송은 0 개다.
  - VGG conv 13 개 매핑: 모두 PASS. binding 은 모두 입력 0 · 필터 1 · 출력 2 이고, 드레인이 출력 상자와 정확히 같다.
  - VGG `--conv native` 일반 컴파일: 92.4 s. PDI md5 `697f0b8a…` 로 기준과 같고, vmfb 286,833,681 B · PDI 4,547,312 B 도 그대로다. 수정이 VGG 의 코드 생성을 바꾸지 않았다. 독자 수 수정 뒤(v20, 90.1 s), word 재접기·epilogue 예산 수정 뒤(v21, 94.0 s), 재접기 자리 검사 뒤(v23, 89.8 s)에도 md5 가 같다.
  - VGG 기본(im2col): 110.8 s, PDI md5 `2781c8d4…` 그대로. v21(114.8 s)·v23(108.6 s)에서도 같다.
  - 도구 최종판 sweep(`log_shjo/impl/fixcheck/r3/r6probe/`): audit 은 로그 231 개 중 판정이 바뀐 것이 5 개이고 모두 의도한 것이다. 스택 20 KB 로그 3 개가 FAIL→PASS, word 로도 한계를 넘는 홀수 행 2 개가 PASS→FAIL 이다. 매핑 검사는 정상 조합 73 개, VGG 13 개, pair 6 개가 모두 PASS 다.
  - 셔임 검사가 다른 경로를 막지 않는지도 봤다. 공용 하강이라 matmul·GEMV·pooling 도 지난다. 아래 네 컴파일 모두 rc 0, error 진단 0 이다(`log_shjo/impl/fixcheck/r3/extra_compiles.log`).
    - MLP 2 층 bf16: 10.9 s
    - MLP 2 층 f32: 7.2 s
    - VGG `--conv native --pool accel`: 99.4 s. 복구된 bank 오류는 252 개로 같다.
    - VGG 기본 + `--pool accel`: 118.4 s
    - v21 에서 MLP bf16, VGG `--conv native --pool accel`(102.2 s)을 다시 컴파일했다. 둘 다 rc 0, error 0 이다(`log_shjo/impl/fixcheck/r3/v21/regress.log`).

#### 수정 검토(반증)

위 수정을 다시 반증 검토했다. 워크플로 세 번이다. 각 영역마다 리뷰어 하나가 지적을 찾고, 지적마다 검증자 하나가 재현으로 반박한다.

1. 첫 번째(`wf_0bd7b95d-a58`)는 세션 한도에 걸려 컴파일러 영역만 끝났다. 지적 3 건이 나왔고 모두 재현됐다.
   - FC-1 의 빈틈: epilogue 가 둘이거나, 원본 저장과 epilogue 가 함께 있으면 여전히 strategy 를 통과했다.
     - 수정: conv 결과의 독자를 op 하나로 제한한다. 저장 하나, 또는 epilogue 하나다. epilogue 결과도 저장 하나로 제한한다.
   - T1 의 빈틈: 원소 1 개 접근은 접으면 차원이 남지 않는다. 그래서 검사를 건너뛰고, buffer_length 0 · 주소 byte 30 으로 조용히 내려갔다.
     - 수정: 조건을 "전송 길이 > 0" 으로 바꿨다. 시작은 항상 보고, 차원이 없으면 전체 길이를 본다.
   - T1 lit 에 reinterpret_cast offset 사례가 없었다. 정상·오류 1 개씩 더했다.
2. 독자 수 수정의 첫 판은 사용(use) 수를 셌다. 그래서 결과를 입력이자 init 으로 쓰는 in-place epilogue(`f5_inplace`)를 잘못 거부했다. 서로 다른 op 수를 세도록 고치고, 정상 lit(`conv_inplace_relu`)으로 막았다.
3. 두 번째(`wf_54e82db9-fb2`)는 남은 두 도구 영역과, 방금 바꾼 컴파일러 부분을 다시 봤다. 에이전트 9 개가 모두 끝났다. 지적은 8 건이다. 검증자가 맡은 6 건은 모두 재현됐고(반박 불가), 상한에 걸린 2 건(MF3·MF4)은 리뷰어 재현 로그로 직접 확인했다.

| ID | 대상 | 지적 | 수정 | 확인 |
|---|---|---|---|---|
| CF1 | 하강 | sub-word 검사가 원소 수 1,023 상한으로 접힌 차원을 본다. 홀수 길이 행이 1,023 원소를 넘게 이어지면(OC 3 청크 conv 의 `[26, 45] [45, 1]`, 585 word) 합법인데 거부했다. T1 전에는 조용히 내림됐다 | 원소 수로 접은 결과가 word 가 아닐 때만 안쪽 차원을 word 수 상한으로 다시 접는다. 통과하던 패턴의 접기는 그대로다 | lit 2 건. 재현 모델의 셔임 전송은 585 word 로 내려간다(모델 전체는 아래처럼 코어 BD 에서 실패) |
| CF2 | strategy | in-place epilogue 가 L1 예산에 두 번 잡혀 타일이 작았다(`[7,7,20]` 대 `[7,7,50]`). 정확성 영향은 없다 | epilogue 예산을 독자 op 하나로 | lit(두 형태의 타일이 같음) |
| AF1 | audit | `fold_dims` 가 크기 상한 없이 합쳐, 하강이 남기는 홀수 행을 못 봤다 | 하강과 같은 상한으로 접고, word 로 다시 접는 것까지 따름. BD 한계 검사도 접힌 차원으로 | 로그 231 개 판정 비교 |
| AF2 | audit | bank-aware 배치의 다른 복구 오류(스택 > 뱅크)를 위반으로 셌다. rc 0 인 정상 컴파일이 FAIL 했다 | 오류 문구가 아니라 바로 뒤의 순차 배치 경고로 판단 | 스택 20 KB 로그 3 개 PASS |
| MF1 | 매핑 검사 | L3 기준 오프셋을 보지 않았다. 필터를 다른 conv 의 상수에서 읽거나 binding 밖을 읽어도 PASS 했다 | reinterpret_cast offset 이 subspan offset 과 같아야 함 | 재현 6 개 FAIL |
| MF2 | 매핑 검사 | 입출력 타입이 같으면 binding 이 뒤바뀌어도 PASS 했다 | 입력·필터 ReadOnly, 출력은 아님 | 재현 FAIL |
| MF3 | 매핑 검사 | 코어의 두 번째 출력 연결, 조각의 추가 writer 를 보지 않았다 | 연결·L3 전송을 모두 셈 | 재현 2 개 FAIL |
| MF4 | 매핑 검사 | L1 쪽 반복 횟수까지 비교해, 같은 AIE IR 로 내려가는 구조가 거짓 FAIL 했다 | 조각 쪽만 비교 | L1 쪽만 바꾼 변이 PASS, 조각 쪽 변이 FAIL |

CF1 재현 모델(`pair208x15_oc3`, 208×15 영상, 채널 8 → 3 → 16)을 v21 로 다시 컴파일했다. 셔임 단계는 통과하지만, 직렬화(CDO)에서 `` `stride` on dim 0, times element width (in bytes), should be a multiple of 4 bytes`` 로 멈춘다.
- 원인은 코어의 출력 BD 다. L1 출력 타일 26×15×3 bf16(1,170 원소)은 코어 wrap 한계(256 word)를 넘어 `[26, 45] [45, 1]` 행으로 나뉜다. 그런데 45 원소는 90 B 라 word 가 아니다. 런타임 BD 설정(`configureDMABD`, `iree_aie_configure.cc`)이 이를 거부한다.
- T1 전에도 이 모델은 같은 코어 BD 에서 실패했을 구조다. 그래서 회귀는 아니다. 셔임 쪽 잘못된 오류 메시지만 사라졌다.
- 이 규칙을 audit 의 BD 검사에 넣었다. 바깥 차원 stride 와 안쪽 크기가 4 B 배수여야 한다. 이제 덤프만으로 잡힌다(이 모델에서 코어 8 개의 BD). 로그 231 개 판정은 이 규칙으로 바뀐 것이 없다.
- 홀수 OC·홀수 OW 의 bf16 출력을 컴파일하려면, BD 를 word 단위로 다시 나누거나 타일 규칙을 더해야 한다. §8 에 남긴다.

4. 세 번째(`wf_99c8e8e8-9be`)는 위 수정분만 겨냥했다. 리뷰어 2 명, 검증자 3 명이다. 지적 3 건이 모두 재현됐다.

| ID | 대상 | 지적 | 수정 | 확인 |
|---|---|---|---|---|
| F1 | 하강 | word 재접기로 차원 수가 줄면 바깥 차원이 한 칸씩 내려간다. 내려간 자리의 한계를 보지 않아, D1 에 1024 가 들어가면 aie-rt 가 받고 10 비트 필드에 0 으로 쓴다(조용한 오답). 1025 이상이면 진단 없이 aie-rt 에서 실패한다 | 다시 접은 결과는 각 차원이 BD 자리의 한계 안에 있어야 받아들인다(반복 횟수 ≤ 256, iteration ≤ IterWrapMax+1, D0·D1 ≤ WrapMax, D2 ≤ WrapMax+1). 아니면 첫 접기의 오류를 낸다 | lit 1 건. 리뷰어 탐침 11 개 중 자리에 안 맞는 4 개는 오류, 나머지 6 개는 v21 과 같은 BD |
| AF1-D2 | audit | D2 의 wrap 검사를 건너뛰었다. aie-rt 는 D2 도 1024 를 넘으면 거부한다 | 자리별 한계: D0·D1 ≤ 1023, D2 ≤ 1024, iteration ≤ 64, 반복 횟수 ≤ 256 | D2 1100·2048 변이 FAIL, iteration 65 FAIL, 64 PASS |
| MF1-ROLE-BASE | 매핑 검사 | 한 역할이 같은 binding 의 서로 다른 subspan 두 개를 쓰면(각자 cast 는 맞음) PASS 했다 | 역할마다 기준이 하나여야 함 | 재현 2 개 FAIL, 대조군 PASS |

- 리뷰어는 무작위 탐침 1,600 개를 C++ 하강으로 내리고, BD 를 재생해 주소를 다시 계산하는 판정기와 비교했다. 다시 접힌 BD 344 개가 모두 요청한 word 순서를 만들었다. audit 의 접기 흉내도 1,600 개 모두에서 C++ 의 받아들임/거부와 같았다.
- in-place epilogue 예산을 두 번 세던 옛 계산이면, lit 사례가 76,160 B 로 L1 한계 64,512 B 를 넘는다. 그래서 새 lit 이 그 수정을 지킨다.
- 첫 접기만으로도 짝수 행이 D1 에 1024 로 내려가거나, 높은 D2(1024 초과)를 만드는 경로가 원래 있다. 기존 lit 이 그런 합성 패턴의 하강을 기대값으로 두고 있어(예: 한 차원 2,048 word), 이번에는 바꾸지 않았다. §8 에 남긴다.

리뷰어가 확인했다고 적은 것도 있다.
- 지금 거부하는 epilogue 형태는 전에도 모두 늦게 실패하거나 크래시했다. 정상 컴파일을 막은 경우는 없다.
- `linalg.index` 를 쓰는 epilogue, in-place epilogue, fill init 은 컴파일되고 감사·매핑이 PASS 한다.
- 로그 252 개에서 T7 이 잡는 lock 은 모두 실제 공유 쌍이다.
- 매핑 검사기는 sweep 의 정상 조합 73 개(커널 3·5·7, HWCF·FHWC, batch 접힘, 내부 출력, 상수 필터)를 모두 PASS 한다.

#### 분류 요약

| 파일 | 변경 | 분류 |
|---|---|---|
| `Transforms/KernelDispatch.cpp` | conv 파이프라인: 결과의 독자 op 하나, epilogue 의 init 읽기·연쇄·다중 저장 거부, epilogue L1 예산은 op 하나 | 변경 |
| `Transforms/AMDAIEChunkConvDispatches.cpp` | epilogue 1 개까지, 구조체 주석 | 변경 |
| `Transforms/AMDAIEControlCodeLowering.cpp` | sub-word 셔임 접근의 word 정렬 검사(op 오류, 원소 1 개 포함), word 재접기와 그 결과의 BD 자리 검사 | 추가 |
| `Transforms/test/controlcode_lowering.mlir` | bf16 오류 8 사례, 정상 2 사례(reinterpret offset, word 재접기) | 추가 |
| `Transforms/test/lowering_strategy_conv_native_errors.mlir` | epilogue 오류 5 사례, 머리 주석 | 추가·변경 |
| `Transforms/test/lowering_strategy_conv_native.mlir` | in-place·빈 init epilogue 정상 사례(같은 타일) | 추가 |
| `Transforms/test/chunk_conv_dispatches.mlir` | `conv_chain`(청크 안 함) | 추가 |
| `scripts/debug/aie_audit.py` | error 줄(복구 오류는 경고로 판단)·빈 덤프·코어 0, 하강과 같은 접기와 자리별 한계, sub-word 셔임 검사, 코어·memtile BD word 규칙, lock init 대 버퍼 | 추가·변경 |
| `scripts/debug/conv_dma_check.py` | 역할 판정, `--origin`·원점 검사, binding·범위·기준 오프셋·플래그·역할별 기준 하나, 조각 쪽 반복 횟수, 드레인 검사, 경로 밖 이동, IC 청크 | 추가·변경 |
| `isUnaryElementwiseEpilogue`(GEMV·pad 패스 공용) | 손대지 않음 | 유지 |

#### 의미와 한계

- T1 은 이제 컴파일 오류로 멈춘다. 홀수 OC 의 bf16 conv(청크 없음)가 패딩 버퍼 내부에 쓰는 모델은 컴파일되지 않는다. 행이 홀수 원소인 bf16 출력 타일은 코어 BD 에서도 막힌다(위 CF1 재현). 정렬을 맞추는 변환은 §8 에 남긴다. 이 검사는 하강 공용 경로라 matmul·GEMV·pooling 에도 적용된다. 확인한 범위는 위의 VGG·MLP 컴파일까지다.
- 매핑 검사는 패딩 버퍼의 내부 원점을 덤프에서 알 수 없다. 기대값은 `--origin` 으로 준다.
- memtile 이 L3 채움 하나를 여러 번 보내는 재사용 구조는 모델링하지 않아 FAIL 로 나온다. 지금 컴파일러는 만들지 않는다.
- audit 은 compile 의 종료 코드를 보지 못한다. 회귀 스크립트는 rc 를 따로 적는다.
- 셔임 하강의 첫 접기 결과에는 BD 자리 검사를 넣지 않았다(기존 lit 의 기대값과 얽힘). §8 에 남긴다.
- 수치 정확도는 여전히 검증하지 않았다(NPU 미실행).

### 7.9 OH 패딩(T2.4 일부)

**동기와 분석**. 청크 수 N 은 OH 의 약수 중 열 수 이하에서 가장 큰 것이었다(청크당 2 행 이상). 그래서 약수가 나쁜 OH 는 열을 거의 쓰지 못했다.
- 7, 13 같은 소수는 1 청크, 즉 1 열(코어 4 개)이었다. 52 는 4 청크 × 13 행이다.
- 모든 열은 자기 청크를 동시에 계산하므로, 층 시간은 청크 하나의 행 수 S 에 비례한다(열 안의 코어 4 개는 출력 채널을 나눈다).
- 그러니 열을 늘리는 의미는 S 를 줄이는 데 있다.

VGG 의 OH 28·14 층을 8 열로 늘려도 S 는 줄지 않는다. 그래서 §8 에 "8 열로 쓰는 OH 패딩"이라고 적었던 기대는 맞지 않았다.

| OH | 지금(약수) | 8 열로 패딩 | 비고 |
|---|---|---|---|
| 28 | 7 청크 × 4 행 | 8 청크 × 4 행(32 행) | 8 번째 청크는 패딩 행 4 개뿐. S 그대로 |
| 14 | 7 × 2 | 8 × 2(16 행) | 8 번째 청크는 패딩뿐 |
| 7 | 1 × 7 | 4 × 2(8 행) | S 7 → 2 |
| 13 | 1 × 13 | 7 × 2(14 행) | S 13 → 2 |
| 52 | 4 × 13 | 8 × 7(56 행) | S 13 → 7 |

28·14 층이 8 번째 열을 실제로 쓰려면, 행만이 아니라 OW 도 나눠야 한다(2 차원 청크).
- 예: 28×28 출력을 7 행 × 14 열 청크 8 개로 나누면 열당 98 화소다. 지금은 4×28 = 112 화소라 12.5 % 줄어든다.
- 이 방식은 입력 열 방향 halo 와 2 차원 gather 가 필요하다. 이번 작업에서는 하지 않았고 §8 에 남긴다.

**선택 규칙**(`chooseConvChunks`).
- S = max(2, ⌈OH / 열 수⌉): 열들이 허락하는 가장 적은 청크당 행 수다(2 행 이상은 기존 조건과 같다).
- N = ⌈OH / S⌉: 그 S 로 OH 를 덮는 가장 적은 청크 수다.
- N·S > OH 이면 남는 행을 패딩한다. 약수가 같은 S 를 주면 N 은 그 약수가 되고 패딩은 없다.
- 그래서 VGG 의 청크 선택은 바뀌지 않는다(224·112·56 → 8 청크, 28·14 → 7 청크).
- 모든 청크에 실제 행이 하나 이상 있어 패딩만 있는 청크는 없다(N = ⌈OH/S⌉ 이므로 패딩 행 < S).

**재작성**.
- 입력: halo gather 가 입력 아래에 0 행을 패딩 수만큼 덧댄 뒤(`tensor.pad`, host dispatch 안) 모은다. 채널 패딩(ones 채널)과 합쳐진 경우에도 같다. 패딩 행은 잘라낼 출력 행에만 쓰인다. 출력 행 g < OH 가 읽는 입력 행은 g + KH − 1 ≤ OH + KH − 2 로 원래 입력 안이다.
- 실행 파일: [N][S][OW][OC] 를 쓰는 7 루프 conv. 구조는 그대로이고 S 만 넘겨받는다.
- 출력: `flow.tensor.reshape` 로 [N·S][OW][OC], 패딩이 있으면 `flow.tensor.slice` 로 앞 OH 행(가장 바깥 차원의 연속 구간)을 잘라낸다. tied 출력이면 잘라낸 결과를 기존 insert dispatch 가 패딩 버퍼 내부로 복사한다.

#### 코드 변경(변경 전 / 변경 후)

**`AMDAIEChunkConvDispatches.cpp`** — 청크 선택. 변경 전:

```cpp
/// Number of row chunks for a conv with `oh` output rows on `numCols` array
/// columns: the largest divisor of `oh` up to `numCols` that leaves at least
/// two rows per chunk, 1 if there is none.
static int64_t chooseConvChunks(int64_t oh, int64_t numCols) {
  for (int64_t n = std::min(numCols, oh / 2); n > 1; --n)
    if (oh % n == 0) return n;
  return 1;
}
```

변경 후:

```cpp
/// Row chunks of a conv with `oh` output rows on `numCols` array columns (see
/// the file comment): the fewest rows per chunk the columns allow, at least
/// two, and the fewest chunks of that many rows that cover `oh`. Their rows
/// can pass `oh`; one chunk means no chunks.
struct ConvChunks {
  int64_t numChunks;
  int64_t rowsPerChunk;
};
static ConvChunks chooseConvChunks(int64_t oh, int64_t numCols) {
  numCols = std::max<int64_t>(numCols, 1);
  int64_t rows = std::max<int64_t>(2, (oh + numCols - 1) / numCols);
  return {(oh + rows - 1) / rows, rows};
}
```

halo gather(`createHaloSplitDispatch`)는 `padRows` 인자를 받는다. 변경 전 — 없음(신규 추가). 변경 후, 입력을 읽은 바로 뒤:

```cpp
  if (padRows > 0) {
    auto paddedType = RankedTensorType::get(
        {srcType.getDimSize(0) + padRows, width, srcType.getDimSize(2)},
        elemType);
    Value zero = arith::ConstantOp::create(fb, loc, fb.getZeroAttr(elemType));
    loaded = tensor::createPadHighOp(paddedType, loaded, zero,
                                     /*nofold=*/false, loc, fb);
  }
```

`chunkExecutable` 은 청크당 행 수를 받는다. 변경 전에는 안에서 `int64_t rowsPerChunk = oh / numChunks;` 로 구했고, 변경 후에는 인자 `int64_t rowsPerChunk` 로 받는다.

패스 본체 — 출력 되돌리기. 변경 전:

```cpp
    Value rows = IREE::Flow::TensorReshapeOp::create(
        rewriter, dispatch.getLoc(), valueType, newDispatch.getResult(0),
        /*source_dims=*/ValueRange{}, /*result_dims=*/ValueRange{});
    if (tiedDest)
```

변경 후:

```cpp
    auto rowsType = RankedTensorType::get({numChunks * rowsPerChunk, ow, oc},
                                          valueType.getElementType());
    Value rows = IREE::Flow::TensorReshapeOp::create(
        rewriter, dispatch.getLoc(), rowsType, newDispatch.getResult(0),
        /*source_dims=*/ValueRange{}, /*result_dims=*/ValueRange{});
    if (padRows > 0) {
      // The first OH rows: a contiguous outer-dim range of the chunks.
      auto index = [&](int64_t v) -> Value {
        return arith::ConstantIndexOp::create(rewriter, dispatch.getLoc(), v);
      };
      Value zero = index(0);
      rows = IREE::Flow::TensorSliceOp::create(
          rewriter, dispatch.getLoc(), valueType, rows,
          /*source_dims=*/ValueRange{},
          /*start_indices=*/ValueRange{zero, zero, zero},
          /*lengths=*/ValueRange{index(oh), index(ow), index(oc)},
          /*result_dims=*/ValueRange{});
    }
    if (tiedDest)
```

본체는 `ConvChunks` 를 받아 `padRows = numChunks * rowsPerChunk - oh` 를 구하고, 그것을 gather 에 넘긴다. 파일 머리 주석의 청크 규칙 설명도 새 규칙으로 바꿨다.

**lit**(`chunk_conv_dispatches.mlir`). 변경 전 — 없음(신규 추가). OH 5 사례 하나를 더했다. 3×3 conv(ones-채널 pad 와 합침, tied 출력)와 1×1 conv(그냥 출력) 두 dispatch 다.
- 둘 다 3 청크 × 2 행이 된다.
- halo dispatch 에 `tensor.pad … high[1, 0, 0]` 이 들어간다.
- `flow.tensor.reshape` 로 [6, 2, 2] 를 만든 뒤 `flow.tensor.slice` 로 [5, 2, 2] 를 잘라낸다. tied 쪽은 그 결과를 insert 가 [1, 1, 0] 에 넣는다.
- 기존 사례(OH 4·6·2)의 기대값은 그대로 통과한다.

#### 검증(컴파일·정적 IR·lit/ctest·CPU 수치, NPU 미사용)

- **lit·ctest**: conv 관련 lit 8 개 통과, `ctest -R amd-aie` 238/238(`log_shjo/impl/ctest_v16.log`).
- **NPU 컴파일**(`gen_pair.py`, 3×3 두 층, `log_shjo/impl/ohpad/npu_compile.log`):

| 모델 | 전(청크) | 후(청크 × 행, 코어) | 감사 | 매핑 검사 |
|---|---|---|---|---|
| OH 7 | 1 열, 7 행 | 4 × 2, 16 코어 | PASS(위반 0) | 2/2 PASS |
| OH 13 | 1 열, 13 행 | 7 × 2, 28 코어 | PASS | 2/2 PASS |
| OH 5 | 1 열, 5 행 | 3 × 2, 12 코어 | PASS | 2/2 PASS |

- **CPU 수치 비교**(`log_shjo/impl/ohpad/cpucheck/run_check.sh`).
  - 기준은 모델을 CPU 전용으로 컴파일한 것이다.
  - 비교 대상은 native 레시피의 Flow IR 에서 NPU dispatch 를 CPU 로 재지정해 이어 컴파일한 것이다. 같은 bf16 입력으로 local-host 에서 돌렸다.
  - Flow 재작성 전체(bias 접기, 청크, OH 패딩, FHWC, crop, insert)를 수치로 확인한 첫 결과다. NPU 코드 생성은 이 비교에 들어가지 않는다.

| 모델 | 재작성 | 결과 |
|---|---|---|
| OH 7, 13, 5 | 청크 + 패딩(halo 2, crop 2) | 기준과 비트 단위로 같음(max\|diff\| 0) |
| OH 16 | 청크, 패딩 없음 | 비트 단위로 같음 |
| OH 2 | 청크 없음 | 비트 단위로 같음 |
| OH 7, crop 시작을 한 행 민 변이(음성 대조) | — | 47 % 원소가 다름 → 비교가 헛돌지 않음 |
| VGG-16 전체(conv 13 개 청크, insert 8 개) | 청크, 패딩 없음 | logit 상관 0.9999985, max\|diff\| 0.0075(최대 logit 4.82 의 0.16 %), top-1·top-5 같음 |

  - 작은 모델이 비트 단위로 같은 이유는 `gen_pair.py` 의 가중치·bias 가 1/16 의 배수라 bf16·f32 에서 반올림이 생기지 않기 때문이다.
  - VGG 의 작은 차이는 bias 를 bf16 으로 반올림한 것과 누산 순서에서 오는 것으로 보인다(따로 분리해 확인하지는 않았다).
- **VGG 회귀**: native PDI md5 `697f0b8a…` 그대로(106.9 s). VGG 의 청크 선택은 바뀌지 않는다. 기본(im2col) PDI md5 `2781c8d4…` 도 그대로다(122.5 s).

#### 분류 요약

| 파일 | 변경 | 분류 |
|---|---|---|
| `Transforms/AMDAIEChunkConvDispatches.cpp` | 청크 선택(최소 S, 최소 N), gather 의 행 패딩, 실행 파일의 S 인자, 출력 crop, 머리 주석 | 변경·추가 |
| `Transforms/test/chunk_conv_dispatches.mlir` | OH 5 패딩 사례(tied·untied) | 추가 |
| 전략·파이프라인·하강 | 손대지 않음(청크 실행 파일의 모양은 그대로) | 유지 |
| `log_shjo/impl/ohpad/cpucheck/` | CPU 수치 비교 스크립트(`retarget.py`, `run_check.sh`, `run_check_vgg.sh`) | 추가(커밋하지 않는 작업 파일) |

## 8. 남은 일

1. **NPU 실행**: 11-09 에 했다(승인, 15 회 실행, [1109_native_conv_npu_run.md](1109_native_conv_npu_run.md)). VGG-16 native 는 CPU f32 기준 대비 corr 0.99999, maxabs 0.0257, top-1 일치다. 시간은 run 5.83 s(performance 모드)로 im2col 0.388 s 의 15 배이고, conv 13 개의 NPU 시간 5.54 s 가 거의 전부다(스칼라 코어, MAC 하나에 17–31 cycle). 단일 층 프로브 4 개(OH 패딩 포함)도 corr 0.99999 이상이다.
2. **T2.4 나머지**:
   - OH 패딩은 했다(§7.9). VGG 의 28·14 층은 8 열로 늘려도 청크당 행 수가 그대로라 7 열에 둔다. 8 번째 열을 쓰려면 OW 도 나누는 2 차원 청크가 필요하다(28×28 이면 열당 화소가 12.5 % 준다).
   - 청크 우선 활성값·필터 레이아웃(G16). IC 청크 8 층의 L3 연속 구간이 16 B 다.
   - insert 복사를 truncf host dispatch 와 합치기. tied 층마다 f32 출력 전체를 한 번 더 복사한다.
3. **T2.5 나머지**: fail-fast 배치. native 경로가 받지 않는 conv 도 affinity 단계에서 NPU 에 배치된다.
   - stride 2 의 3×3 conv + bias + ReLU 로 확인했다(`log_shjo/impl/chunk/stride2.log`). `getConvDims` 가 거부해서 bias 는 접히지 않고, 결국 코드 생성에서 `Unhandled pass pipeline in setRootConfig` 로 실패한다.
   - conv 로 인식된 뒤의 비-fill init, 단항이 아닌 epilogue, 명명 conv 는 이제 strategy 에서 이유를 밝힌 오류로 멈춘다(§7.6 #8·#10).
   - 남은 일은 affinity 단계에서 이런 conv 를 host 로 보내 모델 전체가 컴파일되게 하는 것이다. vulkan 을 함께 쓸 때의 장치별 선택도 여기에 속한다.
   - 11-09 에 하나 더 확인했다. 단일 층 모델처럼 conv 의 ReLU 가 NCHW 전치까지 맡으면(출력 맵 `(d2, d0, d1)`) 전략이 "transposed output is not supported" 로 멈춘다. 전치를 epilogue 에서 떼어 host dispatch 로 옮기는 것도 방법이다([1109_native_conv_npu_run.md](1109_native_conv_npu_run.md) §5.4).
4. **4 B 단위(§7.8 T1·CF1 의 후속)**: 홀수 OC 의 bf16 출력은 두 곳에서 막힌다.
   - 청크가 없는 conv 가 패딩 버퍼 내부에 쓰면 셔임 시작이 word 가 아니다. 이제 컴파일 오류로 멈춘다(전에는 조용히 2 B 앞에 썼다). 청크 경로처럼 자기 버퍼에 쓰고 host insert 로 옮기면 피할 수 있다.
   - 행(OW×OC)이 홀수 원소인 L1 출력 타일은 코어 wrap(256 word)을 넘을 때 홀수 행으로 나뉘어, 런타임 BD 설정이 거부한다. BD 를 word 단위로 다시 나누거나, `chooseConvTiling` 에 규칙을 더해야 한다.
   - 셔임 하강의 첫 접기는 접은 차원을 BD 자리의 한계와 대조하지 않는다. 짝수 행이 D1 에 1024 로 내려가면 aie-rt 가 0 으로 쓰고, D2 가 1024 를 넘으면 aie-rt 가 진단 없이 실패한다. word 재접기 결과에만 자리 검사를 넣었다(§7.8). 첫 접기에도 넣으려면 기존 lit 의 합성 패턴(한 차원 2,048 word 등) 기대값을 바꿔야 해서 업스트림과 같이 정한다.
   - 필요한 모델이 나오면 한다.
5. **L1 과 뱅크 배치**: 64 KB 가까이 쓰는 타일에서 bank-aware 배치가 실패해 순차 배치로 넘어간다(252 코어). 뱅크 단위 예산을 `chooseConvTiling` 에 넣을지 정해야 한다.
6. **커널 경로(T4.x, D5)**: 코어 코드는 스칼라 bf16 MAC 이다. conv ukernel(aie_api) 또는 IGEMM.
7. **커밋**: 아무것도 커밋하지 않았다. 순서(T1.0·T1.4 와 이번 변경 묶음)는 사용자 결정을 기다린다.
8. **matmul fold 경로·하강 공용 경로(팀 코드)**: 리뷰에서 나온 같은 결함들을 함께 고쳤다. bias 바이트, init 맵, 소비자 입력 확인, 다른 사용자가 있는 binding 의 오류다. §7.8 의 셔임 4 B 검사(`AMDAIEControlCodeLowering`)도 모든 파이프라인이 지나는 하강이다. 업스트림 PR 로 따로 낼지 정해야 한다.
