---
date: 2026-09-28
topic: [kfold, encoder-stack]
status: resolved
summary: "residual view(Slice) + IREE 패턴 2개로 12층 K-fold 복사 35→1, 176→142 (NPU 123 / CPU 19), NPU 출력 해시 불변. 남은 1개는 layer0 QKV(그래프 입력)"
---
# K-fold 활성 Pad — residual 을 view 로 읽게 해서 복사 제거 (2026-09-28)

`2026-09-25_kfold_activation_pad_trace.md` 의 시도 ①(Pad 를 생산자 레이아웃으로 올리고 다른
소비자는 Slice 로)을 이어서 끝냈다. 그때는 **FC1 의 Pad 는 사라졌지만 residual 용 Slice 가 새
복사 디스패치**가 되어 순이득 0 이었다. 오늘 그 Slice 를 복사가 아니라 **residual 디스패치 안의
부분 읽기**로 만들었다.

## 0. 결과

| 12층 모델 | 총 | NPU | CPU | K-fold 복사(`slow_memcpy`) |
|---|---:|---:|---:|---:|
| `enc12hsk2_k_i8` (어제) | 176 | 123 | 53 | 35 |
| **`enc12hv_k_i8`** (오늘) | **153** | **123** | **30** | **12** |

- 없어진 23 개 = **FC1 12 + QKV 11**(layer 1–11). 복사가 NPU 로 옮겨 간 게 아니라 사라졌다 —
  NPU 디스패치 수는 그대로다.
- NPU 출력: 입력 3 종 × (5/3/3) 회 전부 성공, SHA-256 이 기준(`976028ff…`/`ce50abfb…`/`80d864f1…`)과
  **동일**. ORT 도 3 입력 바이트 동일.
- 남은 12 개: **FC2 11**(아래 §4) + **layer 0 QKV 1**(생산자가 그래프 입력).

## 1. 왜 Slice 가 복사가 됐나

디스패치 형성 직전 IR:

```mlir
%inserted_slice_38 : tensor<32x832xi8>                        // LN① 가 넓힌 버퍼에 씀
%expanded = tensor.expand_shape %inserted_slice_38 -> tensor<1x1x32x832xi8>
%slice    = tensor.extract_slice %expanded [1,1,32,768]      -> tensor<1x32x768xi8>
%collapsed = tensor.collapse_shape %slice                    -> tensor<32x768xi8>
linalg.generic ins(..., %collapsed)                          // residual (LN② 디스패치)
```

`tensor.extract_slice` 는 원래 소비자 디스패치로 **복제 가능한** 연산이다
(`RegionOpUtils.cpp::isCloneableIntoDispatchOp`). 그런데 unit-dim 정리가 만든 `expand_shape` /
`collapse_shape` 가 앞뒤를 감싸고 있고, 이 둘은 복제 대상이 아니다(같은 함수의 TODO #8637).
복제는 소비자에서 역방향으로 걷다가 `collapse_shape` 에서 멈추고, slice 는 혼자 남아 자기
디스패치(`slow_memcpy`)가 된다.

## 2. 수정 — IREE 패턴 1 개

`collapse(slice(expand(x)))` 에서 expand 가 unit 차원만 끼워 넣었고 slice 가 그 차원들을 통째로
취하면 `slice(x)` 로 바꾼다. 원소 순서가 그대로이므로 정확히 같은 텐서다.

⚠️ **어디서 돌리느냐가 핵심이다.** 처음에 디스패치 형성 전(`SinkReshapes`)에 넣었더니 개수가
그대로였다: `x` 가 아직 넓히는 `tensor.insert_slice` 라서 `slice(insert_slice(v))` 가 정규화로
`v` 로 접히고, LN① 출력의 소비자가 다시 2 개가 되어 Pad 가 LN① 에 흡수되지 못했다. 값 기준으로는
같은 텐서라 디스패치 형성 전에는 컴파일러가 늘 이렇게 꿰뚫어 본다.

그래서 **디스패치 영역이 만들어진 뒤, 생산자를 영역으로 복제하기 직전**
(`CloneProducersIntoDispatchRegions` 첫 단계)에 돌린다. 이때 `x` 는 LN① 영역의 결과라 접히지
않고, 정리된 slice 는 곧바로 LN② 영역으로 복제된다. 패턴 드라이버는 이 rewrite 만 돌린다
(folding·상수 CSE 끔).

결과 — LN② 디스패치가 넓힌 버퍼를 직접 받아 앞 768 열만 읽는다:

```mlir
func.func @..._reduction_DxD_f32(%arg0: readonly:tensor<32x832xi8>, ...)
  %3 = dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 768]   // residual, view
```

백엔드는 이 부분 읽기를 그대로 처리했다(추가 수정 없음).

커밋: `third_party/iree` 로컬 작업 브랜치 `bert` 의 `539b39f`(로컬 전용 — 확정 전까지 push 안 함).
패치 사본 `_local/share/0007-iree-clone-slice-stranded-by-unit-reshapes.diff`.
lit: `clone_producers_into_dispatch_regions.mlir` 에 케이스 추가.

## 3. 모델 쪽 — `hoist_kfold_pad.py` 확장

`_local/int8_debug/tools/hoist_kfold_pad.py` 는 9/25 에 FC1·FC2 만 처리했다(Pad 앞에 Reshape 가
있는 경우). **QKV 는 Pad 가 이미 생산자 레이아웃에 있어서 "already sits on its producer's layout"
로 건너뛰었는데**, 그래도 residual 이 원래 텐서를 읽으니 생산자의 소비자가 2 개였다. 이 경우에도
residual 을 `Slice(padded)` 로 돌리게 했다.

12층 적용: FC1 12(hoist + residual Slice), FC2 11(hoist 만), **QKV 11(residual Slice 만)**,
layer 0 QKV 1 거부(그래프 입력), output projection 5 대상 아님(이미 흡수).

## 4. 남은 것 — FC2 11 개

FC2 의 Pad 는 residual 과 무관하다. 생산자가 FC1 디스패치에 융합된 GELU+quantize **꼬리**이고,
Pad 가 그 디스패치 루트(contraction)의 **결과를 소비**하는 형태라 루트 디스패치 안으로 들어가지
않는다(9/25 §6). view 방식으로는 풀리지 않고, 루트 디스패치가 넓힌 버퍼로 store 하게 만드는
별도 작업이 필요하다.

## 5. 회귀

| 검사 | 결과 |
|---|---|
| attention `attn0_kpadq_int8` vmfb MD5 | `b35a583d` **동일** |
| softmax-on-NPU `attn0_hsm_int8` vmfb MD5 | `1b8490a5` **동일** |
| `enc12hsk2_k_i8`(view 적용 전 모델) stream IR | **동일** |
| layer0 base 모델 flow IR | **동일** |
| layer0 view 모델(`layer0_i8h`) NPU 출력 | base 와 해시 **동일**(2 입력 × 3 회) |
| `ctest -R DispatchCreation` | **49/49 통과** |

## 6. 재현

```bash
# 모델
PYTHONPATH=_local/int8_debug/deps python3 _local/int8_debug/tools/hoist_kfold_pad.py \
  _local/int8_debug/out/enc12hsk2_k_i8.onnx _local/int8_debug/out/enc12hv_k_i8.onnx
# 디스패치 수 / vmfb / NPU
./scripts/docker/run-dev.sh /workspace/_local/padbe/stream_only.sh enc12hv_k_i8
_local/padbe/census2.sh _local/padbe/enc12hv_k_i8.stream.mlir
./scripts/docker/run-dev.sh /workspace/_local/padbe/build_final.sh enc12hv_k_i8
./scripts/docker/run-dev.sh /workspace/_local/padbe/run_hv.sh
```

---

# 7. (같은 날 추가) FC2 도 해결 — 생산자 쪽 unit-dim collapse

§4 에서 "FC2 는 view 로 안 풀리는 별도 작업"이라 했는데, **같은 종류의 문제**였다. 디스패치 형성
직전 IR 을 output projection(흡수됨)과 나란히 보면 차이가 하나다:

```mlir
// output projection — 생산자 결과에 바로 insert → PV 디스패치에 흡수
%ins = tensor.insert_slice %26 into %fill [32,12,64] -> tensor<32x13x64xi8>
// FC2 — 생산자(FC1 의 GELU+quantize 꼬리) 결과와 insert 사이에 unit-dim collapse
%c   = tensor.collapse_shape %42 : tensor<1x32x3072xi8> into tensor<32x3072xi8>
%ins = tensor.insert_slice %c into %fill [32,3072] -> tensor<32x3136xi8>
```

`FormDispatchRegions.cpp` 는 insert-into-fill 을 생산자와 묶을 때 **insert 의 source 가 생산자
결과 그 자체**여야 한다(`source.getDefiningOp() != producer` 면 거부). collapse 가 끼면 떨어진다.
§1 이 소비자 쪽 unit-dim reshape 였다면 이건 **생산자 쪽**이다.

수정: `insert(collapse(x) → fill)` → `collapse(insert(x → unit 차원 복원한 fill))`
(`SinkReshapes`, iree 로컬 `bert` `ada9339`). 두 번 헛디딘 기록:

1. 처음엔 fill 이 단일 사용일 때만 걸었더니 12 층에서 발동 안 함 — **12 층은 FC2 fill 하나를
   11 개 층이 CSE 로 공유**한다. 어차피 새 fill 을 만들므로 조건 제거.
2. 그다음엔 12 층이 153 → **158 로 악화**(CPU LN 8 개 층의 넓히기·residual slice 가 다시 복사).
   원인: 이 패턴을 `SinkReshapes` 의 기존 패턴과 **같은 greedy 드라이버**에 넣었는데, 기존 패턴은
   collapse 를 반대 방향(fill 위로)으로 움직인다. 섞이면 이미 흡수되던 넓히기가 다른 모양으로
   수렴한다. **기존 패턴이 끝난 뒤 별도 단계로, folding 없이** 돌리니 해결.

## 7.1 결과

| 12층 | 총 | NPU | CPU | K-fold 복사 |
|---|---:|---:|---:|---:|
| 9/27 `enc12hsk2` | 176 | 123 | 53 | 35 |
| 오늘 오전 (residual view) | 153 | 123 | 30 | 12 |
| **FC2 까지** | **142** | **123** | **19** | **1** |

남은 1 개는 layer 0 QKV(생산자가 그래프 입력). CPU 19 = K-fold 1 + LayerNorm 잔여 18.

| 검사 | 결과 |
|---|---|
| 12층 NPU, 3 입력 × (5/3/3) | 전부 성공, 해시 **기준과 동일** |
| layer0 base / view 모델 NPU | 오전 해시와 **동일** (base 도 FC2 복사 3→2 로 줄었는데 출력 동일) |
| attention vmfb MD5 | `b35a583d` / `1b8490a5` **동일** |
| `ctest -R DispatchCreation` | **49/49** |
| lit | `sink_reshapes.mlir` 에 단일·공유 fill 케이스 추가 |

⚠️ 9/27 모델(`enc12hsk2`, view 미적용)도 이 패턴으로 FC2 부분 IR 이 바뀐다. NPU 로 다시 돌려보진
않았다(같은 변경을 layer0 base 모델에서 확인).
