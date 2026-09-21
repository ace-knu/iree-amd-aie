# 2026-09-17 — d1 의 12-head broadcast 제거: CPU 물질화 없이 Q/K/V matmul 로 흡수

> `docs/2026-09-17_attention_softmax_npu_handoff.md` 의 "남은 CPU dispatch" 중 **d1 만**
> 다룬다. d0(input quantize+pad)와 d9(final dequant+bias)는 손대지 않았다.

## 0. 결과

| | 이전 | 이후 |
|---|---|---|
| 디스패치 | NPU 7 / **CPU 3** (10) | NPU 7 / **CPU 2** (9) |
| broadcast 의 L3 물질화 | `12x26624xi8` = **319,488 B** | **없음** (`26624 B` 를 12 번 읽음) |
| NPU 출력 | corr 0.9987321 | **비트 동일** (같은 SHA-256) |

broadcast 는 **NPU 디스패치로 옮긴 것이 아니라 없어졌다.** Q/K/V 세 matmul 이 각자
자기 디스패치 안에서 같은 `32x832` L3 버퍼를 head 마다 다시 읽는다 — DMA fan-out 이다.

⚠️ 과제에 적힌 목표치는 "최소 NPU 8 / CPU 2" 였지만, 실제 결과는 **NPU 7 / CPU 2** 로
그보다 하나 적다. broadcast 를 독립 NPU 디스패치로 만들면 8/2 가 되지만 319 KB 물질화가
그대로 남고 launch 가 하나 늘어난다(핸드오프 문서 §"d1" 의 경고). 흡수가 엄격히 더 낫다.

## 1. d1 이 CPU 디스패치가 되는 정확한 경로

flow 레벨 (`--compile-to=flow`):

```mlir
%5 = flow.dispatch @..._dispatch_1_elementwise_broadcast_12x26624_i8(%4)
       {stream.affinity = #hal.device.affinity<@cpu>}
       : (tensor<26624xi8>) -> tensor<12x26624xi8>
```

디스패치 안의 op 은 순수 복사 broadcast 다:

```mlir
linalg.generic {indexing_maps = [affine_map<(d0,d1,d2) -> (d1,d2)>,
                                 affine_map<(d0,d1,d2) -> (d0,d1,d2)>],
                iterator_types = ["parallel","parallel","parallel"]}
  ins(%x : tensor<32x832xi8>) outs(%e : tensor<12x32x832xi8>) { linalg.yield %in }
```

`gen_attn_headsplit.py` 가 만든 `MatMul(x[1,S,K], W[H,K,D])` 의 **배치 broadcast** 다.
Q·K·V 가 같은 `x` 를 쓰므로 소비자가 3 개다.

**세 갈래 전부 막혀 있었다** (`third_party/iree`, `DispatchCreation/FormDispatchRegions.cpp`):

| 경로 | 막는 곳 | 이유 |
|---|---|---|
| 소비자로 융합 (`fuseRootsWithConsumers`) | `isFusableWithConsumer:587` | `consumerFusionOp.getNumLoops() != getNumParallelLoops()` — matmul 은 리덕션이 있어 거절 |
| " (앞단) | `getFusableUses:407` | 비-aggressive 모드에선 **소비자가 정확히 1 개**여야 함. 여기는 3 개 |
| 생산자로 융합 (`fuseRootsWithProducers`) | `isFusableWithProducer:790` | 비-aggressive 모드에선 소비자의 **DPS init** 피연산자만 받음. matmul 의 LHS 는 input |

전부 막히면 broadcast 는 어느 디스패치에도 못 들어가고, `CloneProducersIntoDispatchRegions`
의 마지막 단계("디스패치 밖에 남은 `linalg.generic` 을 제 디스패치로 감싼다",
`CloneProducersIntoDispatchRegions.cpp:49-57")가 **혼자 남은 디스패치를 만든다.**
그 디스패치는 contraction 이 아니므로 `AMDAIEAssignDeviceAffinities` 가 CPU 로 보낸다.

`--iree-dispatch-creation-enable-aggressive-fusion` 도 답이 아니다: 실측으로 융합되지
않았고, [[feedback-npu-compile-flags-bert-int8]] 대로 실모델 llvm-cpu 를 깨뜨린다.

## 2. 먼저 시도한 두 가지 — 왜 안 되는지

### 2.1 broadcast 를 contraction 의 indexing map 으로 접기 (✗ 백엔드 assert)

LHS 에서 배치 차원을 아예 없애는 형태:

```mlir
linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d1,d3)>,   // 배치 없음
                                 affine_map<(d0,d1,d2,d3)->(d0,d3,d2)>,
                                 affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>], ...}
  ins(%x : tensor<32x832xi8>, %w : tensor<12x832x64xi8>)
```

npu4 컴파일이 **assert 로 죽는다**:

```
mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp:5264:
  commonPermutationOfPackAndUnPackOp: Assertion `outerPermutation.size() ==
  metadata.outerDimsPerm.size() && isPermutationVector(outerPermutation)' failed.
```

`KernelDispatch.cpp:356-381` 의 `setOuterPermA/B` 가 `isBatchMatmul` 하나로 **세 피연산자
전부**에 rank-3 outer perm(`{0,1,2}`/`{0,2,1}`)을 주는데, LHS 만 rank 2 라 어긋난다.
피연산자별 rank 인식으로 고칠 수는 있으나 그 뒤 pack/objectFifo/DMA 전 구간이 "배치
차원이 없는 피연산자"를 처음 보게 된다. 표면적이 너무 넓어 접었다.

### 2.2 broadcast 를 그대로 둔 채 디스패치 안으로만 넣기 (✗ 스케줄 붕괴)

손으로 병합한 `hal.executable.source`(§5 의 `bcast_src_fused.mlir`)로 백엔드 단독 시험:
컴파일은 `AMDAIEDistributeL1Allocations` 에서 실패하고, 그 직전 IR 이 이유를 보여준다.

| | L2 (memtile) LHS | L1 (core) LHS |
|---|---|---|
| 기준선 | `1x4x1x8x32` = **1 KB** (K-슬라이스, k 루프 안에서 재패킹) | `1x1x1x4x1x8x8` = 256 B |
| broadcast 포함 | `4x26x8x32` = **26 KB** (전체를 한 번에) | `4x26x4x1x8x8` + `1x4x26x4x1x8x8` = **53 KB** |

broadcast 가 compute op 으로 남으면 활성화 **전체**가 k 루프 **밖에서** L1 까지 올라간다
(코어 L1 은 64 KB). K 스트리밍이 통째로 사라진다.

## 3. 실제 해법 — 세 군데

핵심 관찰: **레벨-0 타일링이 끝나면 배치 타일 크기가 1 이라 broadcast 는 데이터를 하나도
옮기지 않는다.** `AMDAIETileAndFuse` 직후 IR:

```mlir
%s = tensor.extract_slice %3[%arg1, 0] [32, 832] [1, 1] : tensor<32x832xi8>
%b = linalg.generic <broadcast> ins(%s) outs(tensor<1x32x832xi8>)   // d0 extent = 1
%m = linalg.batch_matmul ins(%b, %ws) ...
```

즉 그 시점의 broadcast 는 **compute op 으로 쓰인 rank 확장 reshape** 일 뿐이다.

### (a) `Transforms/Cleanup.cpp` — unit-extent broadcast → `tensor.expand_shape`

`FoldUnitExtentBroadcastToExpandShape` 패턴 신규. 조건이 엄격하다:
단일 입력·단일 init, 리덕션 없음, 바디가 정확히 `linalg.yield <입력 원소>`,
init map 이 identity, 입력 map 이 **순서 보존** projected permutation(전치 금지),
그리고 **늘어나는 차원이 전부 extent 1**. 하나라도 어긋나면 복사로 남긴다.

의미 보존이 자명한 재작성이다(원소를 하나도 복제하지 않는 경우에만 발동).

### (b) `Transforms/AMDAIEPropagateDataLayout.cpp` — unit-dim reshape 위로 pack 을 올리지 않기

(a) 만으로는 부족했다. 이 패스의 control function 이 무조건 `true` 라
upstream `BubbleUpPackOpThroughReshapeOp` 가 `pack(expand_shape(x))` 를
**`expand_shape(pack(x))` 로 뒤집고, 그 과정에서 LHS 의 선행 unit 차원을 없앤다**:

```mlir
// before:  %pack = linalg.pack %expanded ... -> tensor<1x4x26x8x32xi8>
// after:   %pack = linalg.pack %3        ... -> tensor<4x26x8x32xi8>
//          %expanded = tensor.expand_shape %pack ... -> tensor<1x4x26x8x32xi8>
```

그러면 pack 이 matmul 의 **직접 생산자가 아니게 되어** 뒤따르는 K 루프 타일링이
`AMDAIEFuseProducerIntoLoop` 에서 pack 을 못 가져오고, §2.2 의 26 KB 스케줄이 나온다.

unit 차원만 넣고 빼는 reshape 는 레이아웃 변화가 아니라 view 이므로 올려봐야 얻는 게
없다. control function 에서 그 경우만 거절하도록 했다(`isUnitDimExpandShape`).

### (c) `Transforms/AMDAIEControlCodeLowering.cpp` — binding 까지 view op 을 통과해 찾기

(a)+(b) 뒤 스케줄은 기준선과 같아졌지만 DMA 단계에서 막혔다:

```
'amdaie.logicalobjectfifo.from_memref' op must operate on a
  `hal.interface.binding.subspan`
```

`memref.expand_shape` 가 subspan 과 subview 사이에 생겼기 때문이다. 이미
`memref.reinterpret_cast` / `memref.assume_alignment` 는 통과해서 보고 있었으므로,
**unit 차원만 다른 `expand_shape`/`collapse_shape`** 도 같이 통과하도록 했다.
그런 reshape 는 선형 레이아웃을 바꾸지 않고, 접근 패턴(offsets/sizes/strides)은 이미
평탄 버퍼 기준으로 계산되므로 안전하다.

### (d) `third_party/iree` — 순수 broadcast 를 소비자 디스패치로 복제

(a)(b)(c) 는 "디스패치 안에 들어온 broadcast" 를 처리한다. 들어가게 만드는 것은 별개다.
§1 의 세 경로가 전부 막혀 있으므로 남은 문 하나를 썼다: `isCloneableIntoDispatchOp`.

`CloneProducersIntoDispatchRegions` 는 cloneable 생산자를 **그것을 쓰는 모든 디스패치
리전에 복제**한다(소비자 수 제한 없음). IREE 는 이미 `LinalgExt::isBitExtendOp`
(dequant 류)를 같은 이유 — 재계산이 물질화보다 싸다 — 로 복제한다.

`RegionOpUtils.cpp` 에 `isBroadcastOfContractionOperand` 추가:
`isPureBroadcastOp`((a)와 같은 엄격한 순수 broadcast 판정, 단 extent 조건 없음) **이고**
`tensor.dim` 을 뺀 **모든** 사용처가 contraction 이며 init 이 아닌 피연산자로 읽을 때만
true. 소비자를 보고 판정하는 것은 `isAttentionMaskGenerator`/`isScatterIndicesGenerator`
와 같은 방식이다.

⚠️ 이건 `third_party/iree` 수정이다. **작업 브랜치 `bert` 에는 서브모듈 커밋과 포인터를
그대로 올린다.** 서브모듈 포인터를 올리면 안 되는 것은 **공유 브랜치 `bert-onnx`** 쪽
이야기이고([[feedback-branch-workflow-bert-onnx]]), 그쪽으로 옮길 때 쓸 패치를
`_local/share/0003-iree-clone-broadcast-into-contraction-consumers.patch` 로 함께 남겼다
(`git am` 으로 적용 확인함).

## 4. 검증

전부 `scripts/lock/with-npu-lock.sh` 로 감쌌다. NPU lock/컨테이너 점검은 시작·중간·끝에
수행했다.

### 4.1 정확도·결정성 (실 하드웨어)

`attn0_hsm_sm_nb.vmfb` (= `attn0_hsm_int8.mlir`, broadcast 흡수판):

| | 값 |
|---|---|
| ORT 대비 corr | **0.9987321** |
| maxdiff | **0.0474694** |
| mean abs diff | **0.0060457** |
| **기존 검증본(`attn0_hsm_sm.vmfb`) NPU 출력과** | **비트 동일** (SHA-256 `53463eda89b6f889…`) |
| 6 회 반복 | **단일 해시** |

세 수치가 `docs/2026-09-17_attention_softmax_npu_handoff.md` 의 기준선과 **소수점까지
같다.** broadcast 제거는 수치적으로 완전 중립이다.

tap 판(`attn0_hsmtap_sm_nb.vmfb`)도 같이 확인했다:

| | 값 |
|---|---|
| softmax tap vs ORT | corr **0.9992575**, maxdiff **3**, mean abs **0.0873210** |
| 최종 output vs ORT | corr 0.9987321, maxdiff 0.0474694 |
| tap 판 최종 output vs tap 없는 판 | **비트 동일** |
| 5 회 반복 (두 출력 각각) | **단일 해시** |

### 4.2 디스패치 배치

`iree-dump-module` 로 확인: NPU executable 에 7 개, CPU executable 에 2 개 = **9 개**.
flow 레벨도 같다:

```
d0 cpu  elementwise_32x768_f32xi8            (입력 quantize+pad — 경계)
d1 npu  batch_matmul 12x32x64x832   Q
d2 npu  batch_matmul 12x32x64x832   K
d3 npu  batch_matmul 12x32x64x832   V
d4 npu  batch_matmul 12x32x32x64    QK^T
d5 npu  softmax 12x32x32            (i8 ukernel)
d6 npu  batch_matmul 12x32x64x32    PV
d7 npu  batch_matmul 1x32x768x768   proj
d8 cpu  elementwise_32x768_f32xi8xf32 (최종 dequant+bias — 경계)
```

Q/K/V 의 LHS binding 이 `12x32x832xi8`(319,488 B) 에서 `32x832xi8`(26,624 B)로 바뀌었다.
transient 버퍼 319,488 B 와 그 CPU 쓰기가 사라졌다.

### 4.3 스케줄이 기준선과 같은지 (IR 대조)

마이크로 재현체(§5)로 `AMDAIEDistributeL1Allocations` 직전 IR 을 흡수판 vs
"이미 broadcast 된 LHS" 판으로 정규화 diff:

- `memref.alloc` **6 개 전부 shape 동일** (`1x4x1x8x32xi8` L2, `1x1x1x4x1x8x8xi8` L1 …).
- LHS pack 이 **k 루프 안에서 `[1,32,32]` K-슬라이스**로 도는 것까지 동일.
- 남는 차이는 의도한 것뿐: binding 이 `32x832` 이고 배치 offset 이 `%arg0` 대신 상수 0.
  = **12 개 head 가 같은 L3 영역을 읽는다.**

### 4.4 회귀

**바이트 불변 확인은 "저장된 산출물" 이 아니라 "변경 전 컴파일러로 방금 만든 것" 과
대조했다.** 저장본은 9/11·9/16 것이라 그 뒤 커밋들 때문에 이미 표류해 있었고, 그대로
비교하면 내 변경 탓으로 오판한다(실제로 처음엔 둘 다 달랐다).

변경을 stash → 재빌드 → 같은 두 모델 재컴파일 → 변경 복원·재빌드:

| 모델 | 변경 전 | 변경 후 | 판정 |
|---|---|---|---|
| `attn0_fuse` (head split 아님) | `b35a583d…` | `b35a583d…` | **동일** |
| `bb_kpadq` (full BERT 배포본) | `ba384851…` | `ba384851…` | **동일** |

(둘 다 저장본과는 다르지만 그 차이는 변경 전에도 있었다 = 기존 표류.)

`ctest -R amd-aie` (225 테스트): 실패 **5 개**.
같은 5 개가 **변경을 stash 한 빌드에서도 똑같이 실패**한다 — 새 실패 0.

```
controlcode_lowering / insert_cores / npu_dma_to_half_dma_cpy_nd /
split_logicalobjfifos_for_connection_reuse      ← 기존 4 개 (65069ce 이후)
matmul_elementwise_pack_peel_air_e2e (Timeout)  ← [[project-matmul-elementwise-fusion]] 의 기존 hang
```

⚠️ 마지막 것은 실 NPU 를 쓰는 샘플인데 **`ctest` 는 우리 lock 을 거치지 않는다.**
이번엔 NPU 가 free 였고 끝난 뒤 known-good 으로 건강을 재확인했지만(§4.5),
**공유 호스트에서 `ctest` 를 돌릴 때는 `Test/samples` 가 락 없이 장치를 잡는다는 걸
알고 있어야 한다.**

새 lit 테스트 2 개 추가, 둘 다 통과:
- `Transforms/test/cleanup_unit_broadcast.mlir` — 발동 1 + 음성 3
  (진짜 복제하는 broadcast / 전치가 섞인 것 / 바디가 계산하는 것)
- `Transforms/test/propagate_data_layout.mlir` 에 `pack_not_bubbled_through_unit_expand`

### 4.5 NPU 건강

세션 중 `attn0_hsm_sm.vmfb`(기존 검증본)를 두 번 돌려 매번 corr 0.9987321 /
maxdiff 0.0474694 로 기준선을 정확히 재현했다. 중간에 (b)(c) 를 넣기 전 판이
`ert state 8`(타임아웃)로 5 회 연속 실패했지만 그 뒤 장치는 정상이다.

## 5. 재현용 산출물

| 파일 | 내용 |
|---|---|
| `_local/int8_debug/out/attn0_hsm_sm_nb.vmfb` | 흡수판 (NPU 7 / CPU 2). MD5 `1b8490a5…`, 재컴파일 시 동일 |
| `_local/int8_debug/out/attn0_hsmtap_sm_nb.vmfb` | 같은 것 + softmax tap |
| `_local/scratch_bcast/bcast_lhs.mlir` | §2.1 의 broadcast-in-map 형태 (assert 재현체) |
| `_local/scratch_bcast/bcast_src_fused.mlir` | 손으로 병합한 `hal.executable.source`. 백엔드만 단독 시험 |
| `_local/scratch_bcast/bcast_src_plain.mlir` | 그 대조군(이미 broadcast 된 LHS) |
| `_local/scratch_bcast/build_attn.sh`, `run_attn.sh`, `run_tap.sh`, `regress.sh` | 컴파일·실행·회귀 |
| `_local/share/0003-...patch` | `third_party/iree` 변경 (팀원 공유용) |

컴파일 플래그는 [[feedback-npu-compile-flags-bert-int8]] 그대로이되
`--iree-dispatch-creation-no-fuse-into-contraction-conv-roots` 는 **빼고**(attention 은
융합을 켠다), `--iree-amdaie-enable-ukernels=softmax
--iree-amd-aie-enable-chess-for-ukernel=false` 를 더한다.

## 6. 남은 것

1. **성능은 아직 안 쟀다.** 디스패치가 10 → 9 로 줄고 CPU 가 319 KB 쓰기를 안 하지만,
   L3→NPU DMA 총량은 그대로다(12 head 가 같은 26 KB 를 각각 읽는다). 기대 이득은
   launch 하나 + CPU 작업 + transient 할당이다. [[feedback-npu-perf-measurement-method]]
   대로 `iree-benchmark-module` + 순서반전 교차로 재야 한다. 참고로 softmax 를 NPU 로
   올렸을 때 디스패치 하나 **늘어서** 7.2 % 느려졌으므로, 하나 **줄인** 이번 변경의
   부호는 반대일 것으로 보이나 **측정 전까지는 추정이다.**
2. **d0 / d9** — 모델 I/O 경계라 별개 판단이 필요하다(핸드오프 문서 그대로).
3. **§2.1 의 배치 없는 LHS** 는 여전히 더 깨끗한 표현이다. `setOuterPermA/B` 를
   피연산자별 rank 로 일반화하면 reshape 자체가 없어지고 (b)(c) 가 불필요해진다.
   지금은 그 뒤 pack/objectFifo/DMA 의 미지 구간이 넓어 미룬다.
4. **full BERT 는 범위 밖**이다. fusion 을 켜면 FC1 12 개가 여전히 실패한다
   (2026-09-17 핸드오프 문서 §범위와 주의).
