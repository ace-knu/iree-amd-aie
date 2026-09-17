# 2026-09-11 — K축 bias 접합, 2열로 고치니 성립한다 (앞선 "음성" 판정 뒤집음)

> `docs/2026-09-11_k_axis_bias_fold_negative.md` 의 음성 판정은 **1열 프로토타입의 한계**였다.
> 정확도 손실과 디스패치 증가가 **같은 원인(K축 열을 1개만 붙인 것)** 에서 나왔고, 둘 다 사라진다.

## 0. 결론

**bias 를 matmul 디스패치 안으로 넣는 것은 된다.** 2열 + K 정렬로 다시 만들었더니:

| | 원본 | K접합 1열 (어제) | **K접합 2열 + K=800** |
|---|---|---|---|
| NPU 디스패치 | 96 | 96 | **96** |
| CPU 디스패치 | 232 | 292 (+60) | **220 (−12)** |
| corr (NPU 실측, torch 대비) | 0.901986 | — | **0.920554** |
| maxdiff (NPU 실측) | 2.972 | — | **2.435** |
| corr (CPU/ORT, torch 대비) | 0.883656 | 0.910111 | **0.917955** |
| bias 표현 오차 | — | 16.5 LSB | **0.50 LSB** |

NPU 3회 실행 **비트 동일**(결정론적).

## 1. 무엇이 문제였나 — 열이 1개였다

`fold_bias_into_k.py` 는 K 축에 상수 열을 **하나**만 붙인다(768→769). 그러면
`b_i8[n] = round(bias[n] / (c·x_scale·w_scale))` 로 bias 를 **간격 `c` 인 int8 격자**에
욱여넣어야 하고, 레이어별 `c` 가 5~33 이라 오차가 커진다. 여기서 두 가지가 동시에 터진다.

1. **정확도**: bias 표현 오차 최대 **16.5 누산기 LSB**. 이 모델은 1 LSB flip 이 7% 출력
   오차로 번지는 knife-edge 라 e2e 에 그대로 보인다.
2. **디스패치**: K=769 가 배수 32 가 아니라 `AMDAIEPadContractionDispatches` 가 800 으로
   패딩한다(`AMDAIEUtils.cpp:92`, `getPackPeelReductionTile(1) = 32`). 패딩은 디스패치
   **밖** 호스트 `tensor.pad` 로 나가므로 CPU 디스패치가 +60 된다.

## 2. 고침 — 열 2개, 블록 32 정렬

```
x_aug = Concat(x_i8, [c, 1, 0×30])     # +32 열 → K=800 (배수 32, 패딩 패스 미발동)
w_aug = [w_i8 ; b1 ; b2 ; 0×30]        # +32 행
bias[n] ≈ (c·b1[n] + 1·b2[n]) · x_scale · w_scale
```

`b1 = clip(round(T/c))`, `b2 = clip(round(T − c·b1))`, `T = bias/(x_s·w_s)`.
상위/하위 분할이라 **해상도가 `c` 에서 1 로** 떨어진다.

| | 최대 bias 오차 (누산기 LSB) |
|---|---|
| 1열 | 16.50 |
| **2열 (c, 1)** | **0.50** |

**0.5 LSB 는 이 방식의 바닥이고, int32 bias 를 누산기에 더하는 교과서 방식의 바닥과 같다.**
x_aug 열 값이 int8 정수라 간격이 1 미만이 될 수 없기 때문이다. 즉 **정수 양자화 추론에서
도달 가능한 최적**이고, MatMulInteger 로 바꿔도 이보다 좋아지지 않는다.

⚠️ 앞선 분석에서 "3열이면 0.004 LSB" 라고 적었던 것은 **틀렸다** — 열 값 1/127 을 가정했는데
int8 로 표현 불가능하다. 2열이 끝이다.

증강 블록을 32 로 맞춘 덕에 남는 30열은 0 이라 결과에 기여하지 않고, K 패딩 패스가 아예
안 뜬다. 도구: `_local/int8_debug/tools/fold_bias_into_k2.py`.

## 3. 정확도 (e2e)

CPU/ORT, torch 정답 대비:

| 모델 | corr | maxdiff |
|---|---|---|
| 원본 QDQ (현 배포본) | 0.883656 | 3.468 |
| K접합 1열 | 0.910111 | 3.223 |
| **K접합 2열 + K=800** | **0.917955** | **2.520** |
| 통제군 (bias 를 f32 로 둠) | 0.921592 | 2.531 |

bias 양자화 손실이 **−0.0115 → −0.0036** 으로 줄었고, maxdiff 는 오히려 통제군보다 낮다.
남은 0.0036 은 통제군이 bias 를 f32 로 들고 있어서 생기는 차이라 **정수 누산 경로로는
원리상 도달 불가**다.

실 NPU: **corr 0.920554 / maxdiff 2.435** (원본 NPU 0.901986 / 2.972 대비 +0.0186).

## 4. 속도 — 미해결, 노이즈에 묻힘

교차 8라운드(`iree-benchmark-module`, 순서 반전, NPU 락):

- 평균 A(원본) 8551 ms vs B(2열) 8875 ms → **+3.79%, t=2.27 (df=7), 95% CI [−0.16%, +7.74%]**
- 중앙값 차이 +242 ms

**유의하지 않고, 측정이 오염됐다.** A 값이 7718~8930 으로 14% 흔들렸고(라운드 4·5 에서
호스트가 급격히 빨라졌다 느려짐), 측정 창 끝에 다른 사용자의 `cmake --build -j 6` 가 시작됐다.

**메커니즘상 예상 비용은 +1% 미만이다**: QKV 는 전체 matmul MAC 의 24.8% 이고 K 가 4.17%
늘었으므로 연산량 증가는 **+1.04%**, 게다가 이 워크로드는 벽시계의 대부분이 디스패치당
드라이버 오버헤드다(2026-08-31 측정). 반대로 CPU 디스패치는 12개 줄었다.
**+3.8% 를 설명할 메커니즘이 없다.** 호스트가 조용할 때 재측정할 것.

## 5. 곁가지 — 컴파일 플래그와 3-입력 문제

플래그 기록이 남아있지 않아 재구성했고, 원본을 재컴파일해 **192,388,801 바이트 /
디스패치 96+232 로 기존 `bert_base_fix.vmfb` 와 일치**함을 확인한 뒤 사용했다.

```
--iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local
--iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu
--iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=/workspace/llvm-aie
--iree-dispatch-creation-no-fuse-into-contraction-conv-roots
--iree-flow-enable-executable-deduplication=false
```

재구성 과정에서 두 가지가 드러났다.

- **`--iree-dispatch-creation-no-fuse-into-contraction-conv-roots` 를 빼면 원본조차
  컴파일이 깨진다**: `'amdaie.connection' op no producer DMA channel available`.
  즉 **현 파이프라인은 "elementwise 를 matmul 로 융합하지 말라"고 명시적으로 꺼둔 상태**이고,
  켜는 순간 DMA 채널이 고갈된다 — 3-입력 문제의 날것 그대로다. K축 접합이 이 플래그를
  건드리지 않고 bias 를 넣는 우회로라는 점이 여기서 확인된다.
- **`--iree-flow-enable-executable-deduplication=false` 를 빼면** FC2(K=3072)에서
  `'memref.subview' op has a non-zero base offset that could not be recovered from a
  backing subspan` 로 깨진다.

## 6. 다음

- 속도는 호스트 조용할 때 재측정 (현재 데이터로는 "차이 없음"도 "느려짐"도 단정 불가).
- QKV 36개에만 적용돼 있다. `proj`/`FC1`/`FC2` 까지 넓히면 남은 bias Add 86개가 더 빠진다.
- bias 가 그래프에서 사라졌으므로 이제 **MatMul → Reshape → Transpose 가 인접**해진다 —
  transpose 융합의 전제 조건이 갖춰졌다.
- 정확도 최대 레버는 여전히 per-channel(0.884→0.968, 백엔드 컴파일 실패):
  `docs/2026-09-10_per_channel_int8_blocked_in_npu_backend.md`. MatMulInteger 로 바꾸면
  스케일이 출력측 Mul 로 빠져 풀릴 가능성이 있다 — 미검증.

아티팩트: `_local/int8_debug/out/{ref_orig,bb_kbias2}.vmfb`, `bert_base_kbias2_int8.{onnx,mlir}`.

---

# 7. 나머지 36개로 확장 시도 — QKV 가 유일하게 이득인 곳이다

§6 의 "확대하면 더 빠진다" 예상은 **틀렸다.** 전부 만들어 재봤다.

## 7.1 결과

| 접합 범위 | matmul | Concat | NPU | CPU | corr (ORT) |
|---|---|---|---|---|---|
| 원본 | 0 | 0 | 96 | **232** | 0.883656 |
| **QKV (k2)** | 36 | 12 | 96 | **220** | **0.917955** |
| + proj (k5) | 48 | 24 | 96 | 244 | 0.904651 |
| + FC2 (k4) | 60 | 36 | 96 | 268 | 0.890387 |
| + FC1 (k3) | 72 | 48 | — | **컴파일 실패** | 0.904493 |

**확장할수록 디스패치도 정확도도 단조 악화한다.** QKV 에서 멈추는 것이 맞다.

## 7.2 왜 QKV 만 이득인가 — 활성 공유 비율

레이어당 활성(x_i8) 그룹을 세어보면:

| 그룹 | matmul 수 | Concat |
|---|---|---|
| embeddings/LayerNorm 출력 | **3** (Q, K, V) | 1 |
| attention/self/Reshape_3 출력 | 1 (proj) | 1 |
| attention/output/LayerNorm 출력 | 1 (FC1) | 1 |
| GELU Mul_1 출력 | 1 (FC2) | 1 |

**QKV 만 3개가 활성 하나를 공유해서 증강 Concat 비용이 3:1 로 분산된다.** 나머지는 1:1 이라
Concat 이 제거한 bias Add 보다 비싸다. 실측으로 **QKV 밖 12개를 추가할 때마다 CPU 디스패치
+24**, 정확히 Concat 12개가 추가되는 만큼이다.

⚠️ **단 이건 ONNX 레벨 프로토타입의 한계다.** Concat 은 "활성 뒤에 상수 32열을 붙인 사본"을
만드는 복사인데, **컴파일러 패스라면 활성 버퍼를 애초에 32열 넓게 잡고 그 자리를 상수로
채워두면 복사가 아예 없다**(상수 열은 루프 불변이고 DMA 에 구워넣을 수도 있다).
즉 **"확장하면 손해"는 프로토타입 산출물이지 K접합 원리의 한계가 아니다.**
확장하려면 패스로 가야 한다.

## 7.3 정확도는 bias 탓이 아니다 — 재양자화 제거의 위치 문제

통제군(재양자화 Q(m)/DQ(m) 만 제거하고 bias 는 f32 유지)을 두고 분리했다:

| | corr |
|---|---|
| 원본 | 0.883656 |
| 통제군A — QKV 36개 재양자화만 제거 | **0.921592** |
| k2 — QKV 36개 K접합 | 0.917955 |
| 통제군B — **전체 72개** 재양자화만 제거 | **0.886566** |
| k3 — 전체 72개 K접합 | 0.904493 |

**bias 의 int8 화는 범인이 아니다** — k3 는 자기 통제군(0.887)보다 오히려 **+0.018 좋다**.
진짜 원인은 **중간 재양자화 제거가 QKV 에서만 크게 이득(+0.038)이고 나머지 36개에서는
거의 0(+0.003)** 이라는 것이다. QKV 출력만 `QK^T` 라는 증폭기로 들어가기 때문으로 보인다
(`docs/2026-09-10_bert_e2e_int8_vectorized.md` §5 의 knife-edge 와 일치).
K접합은 구조상 재양자화 제거를 동반하므로 이 둘을 분리할 수 없다.

도구: `_local/int8_debug/tools/{fold_bias_into_k3.py,strip_all_qdq.py}`.
`fold_bias_into_k3.py <출력> [건너뛸 종류,..]` 로 범위를 고를 수 있다(종류는 정확 일치).

## 7.4 열 값도 int8 이다 — 적응적 다열 분해가 필요했다

2열 고정으로는 안 된다. **x_aug 의 열 값 `c` 도 int8 이라 127 을 넘을 수 없는데**,
`attention/output/dense` 는 `c=308` 이 필요하다(bias 동적 범위가 커서). 1열로는 표현 자체가
불가능하다. 그래서 그룹 잔차 최대치를 보고 열을 쌓는 방식으로 일반화했다:

```
V=[]; R=T
while max|R| > 0.5:
    v = min(127, max(1, ceil(max|R| / 127)))     # 열 값도 int8 상한 127
    b = clip(round(R/v), -128, 127); R -= v*b; V.append(v)
```

열 수 분포: **2열 45그룹, 3열 2, 4열 1** (32열 예산 중). 전 구간 오차 0.5 LSB 유지.

## 7.5 FC1 은 백엔드가 못 받는다 (별개 결함)

`intermediate/dense` (K 768→800, **N=3072**) 만 컴파일이 깨진다:

```
'amdaie.logicalobjectfifo.from_buffers' op could not create DMA operations
   AMDAIELowerToAIE.cpp:801  (createDMABlocks 실패)
```

- K=800, N=768 (QKV, proj) → **성공**
- K=3104, N=768 (FC2) → **성공** (3104=32×97 도 통과)
- K=800, **N=3072** (FC1) → **실패**

즉 K 정렬 문제가 아니라 **N=3072 와 증강된 K 의 조합**이다. FC1 은 N=3072 인 유일한
matmul 이다. 원본(K=768, N=3072)은 통과하므로 K 를 32 늘린 것이 방아쇠다.
백엔드 수정이 필요한 별개 항목.

## 7.6 결론

- **배포/사용은 `bb_kbias2.vmfb` (QKV 36개)** — CPU 232→220, NPU corr 0.902→0.921.
- ONNX 레벨에서 더 넓히는 것은 손해다. **넓히려면 컴파일러 패스로 가서 Concat 을 없애야 한다.**
- FC1 은 백엔드 DMA 결함에 막혀 있어 패스로 가더라도 별도 수정이 필요하다.

---

# 8. Concat 비용 제거 — 컴파일러 패스가 아니라 `Pad` 로 해결됐다

§7 에서 확장을 막은 것은 증강 Concat 비용이었다. 패스를 짜기 전에 **Concat 이 정확히 무엇으로
내려가는지** 최소 재현체로 봤고, 그 결과 패스 없이 풀렸다.

## 8.1 Concat 의 정체 — `slow_memcpy` 디스패치 2개

`_local/int8_debug/tools/gen_concat_repro.py` (K=768, N=768, 활성 Q→증강→DQ→MatMul):

| | 디스패치 |
|---|---|
| base (증강 없음) | 3 |
| **Concat 증강** | **5** |

늘어난 2개의 정체 (`--compile-to=flow`):

```
dispatch_1_slow_memcpy(readonly 32x768xi8  -> readwrite 1x32x800xi8)   # x 복사
dispatch_2_slow_memcpy(readonly 32x32xi8   -> readwrite 1x32x800xi8)   # 상수 열 복사(매 추론)
```

**2번은 컴파일 타임 상수를 매 추론마다 복사한다.** 전 모델의 "Concat 당 +2" 가 이것이다.

## 8.2 `Pad` 로 바꾸면 재현체에서는 0개

`FormDispatchRegions.cpp:369` 의 `isRootLikeOp` 은 `tensor::PadOp` 과 `tensor::ConcatOp` 을
둘 다 root 에서 제외하는데, 실제로는 **Pad 만 생산자에 흡수된다**:

```
dispatch_0_elementwise_32x768_f32xi8(readonly 32x768xf32 -> readwrite 32x800xi8)
```

QuantizeLinear 디스패치가 **800 폭 버퍼에 직접 쓴다.** 상수 열은 `fill_buffer`(HAL DMA fill,
연산 디스패치 아님)로 채워진다. base 와 디스패치 수가 같다.

다만 **열 값이 하나뿐**이므로 동적 범위를 열 개수로만 벌어야 한다:
`Σ_i 1·b_i[n] = round(T[n])`, `B = ceil(max|T|/127)`. 오차는 여전히 0.5 LSB.
중첩 Pad(값 2개)는 llvm-cpu 에서
`'linalg.generic' op write affecting operations on global resources ...` 로 깨진다.

도구: `_local/int8_debug/tools/fold_bias_into_k_pad.py <출력> [건너뛸종류,..]`.

## 8.3 ⚠️ K 는 32 가 아니라 **64 의 배수**여야 한다 (N=3072 에서)

§7.5 의 FC1 실패를 형상만 뽑아 스윕했다 (`gen_fc1_sweep.py`, K=768/N=3072):

| K | K/32 | 결과 |
|---|---|---|
| 768 | 24 | OK |
| **800** | **25** | **FAIL** |
| 832 | 26 | OK |
| **864** | **27** | **FAIL** |
| 896 | 28 | OK |
| 1024 | 32 | OK |

`'aie.memtile_dma' op cannot split this DMA access pattern into per-par...`
**K/32 가 홀수면 깨진다.** N=768 에서는 K=800(25, 홀수)도 통과하므로 N 의존이다.
`ALIGN=64` 로 올리니 전체 72개가 컴파일된다. `AMDAIEPadContractionDispatches` 의 K 패딩
배수(32)도 이 경우 부족하다는 뜻이라 **백엔드 쪽에 별도 보고할 항목**이다.

## 8.4 실측 — 그리고 "왜 QKV 만 이득인가" 의 진짜 규칙

| | NPU | CPU | 합계 | corr (NPU) |
|---|---|---|---|---|
| 원본 | 96 | 232 | 328 | 0.901986 |
| Concat · QKV 36 (k2) | 96 | 220 | 316 | 0.920554 |
| Concat · 60 (k4) | 96 | 268 | 364 | — |
| Pad · 전체 72 (kpad) | 96 | 232 | 328 | — |
| **Pad · QKV 36 (kpadq)** | 96 | **208** | **304** | **0.920554** |

NPU 3회 비트 동일(결정론적), 그리고 **Concat 판과 NPU 출력이 비트 동일**(maxdiff 0.00e+00) —
Pad 는 순수한 디스패치 최적화지 수치 변경이 아니다.

디스패치 종류별로 세면 규칙이 나온다 (`--compile-to=flow --mlir-elide-elementsattrs-if-larger=8`):

| 종류 | 원본 | Pad·전체72 |
|---|---|---|
| batch_matmul (NPU) | 96 | 96 |
| elementwise_24576 (재양자화) | 60 | 24 |
| slow_memcpy | 0 | 48 |

**규칙: 활성 하나를 공유하는 matmul G 개를 접으면 재양자화 −G, 증강 +1 → 순이득 −(G−1).**
`G=3` 인 **QKV 만 이득**이고 proj/FC1/FC2 는 `G=1` 이라 순이득 0 이다. §7 의 모든 수치가
이 한 줄로 설명된다.

## 8.5 남은 컴파일러 작업은 이제 정확히 특정된다

재현체에서는 Pad 가 생산자에 흡수되는데 **실 모델에서는 안 된다.** 생산자가 다르기 때문이다:

- 재현체: `elementwise` (단순 elementwise) → **흡수됨**
- 실 모델 QKV: `reduction_32x768_f32 -> 32x768xi8` (LayerNorm + Quantize 융합) → 안 됨
- 실 모델 proj: `elementwise_transpose_12x32x64_i32xi8` → 안 됨

**Pad 가 reduction/transpose 생산자에도 흡수되게 만들면** 증강이 완전히 공짜가 되고,
그러면 `G=1` 그룹도 −1 씩 이득이 되어 **전체 72개 접합이 −72 디스패치(CPU 232→160,
합계 328→256)** 가 된다. 지금 −24 인 것과 큰 차이다. 이것이 남은 유일한 컴파일러 작업이고,
`ConvertStridedInsertSliceToGeneric`(IREE Preprocessing)이 같은 종류의 선례다.

## 8.6 결론

- **쓸 것: `bb_kpadq.vmfb` (Pad · QKV 36개)** — NPU 96 / CPU 208 / 합계 304,
  NPU corr 0.902→0.921, 결정론적.
- Concat → Pad 교체만으로 증강 비용이 절반(디스패치 2→1)이 됐고, QKV 범위에서
  k2 대비 CPU 를 12개 더 줄였다. **패스를 짤 필요는 없었다.**
- 전체 72개로 넓히는 것은 **Pad 를 reduction 생산자에 융합**시키는 컴파일러 작업이
  선행되어야 의미가 있다.

---

# 9. Pad 를 생산자에 융합시키기 — 원인 특정 완료, 남은 건 IREE 코드 수정

§8.5 의 "패스를 짜자" 를 시작했는데, **먼저 기존 플래그와 최소 재현체로 원인을 좁혔더니
패스를 짜기 전에 알아야 할 것들이 나왔다.** 결론부터: 절반은 플래그로 해결됐고, 나머지
절반은 IREE `FormDispatchRegions` 의 한 줄짜리 제약이다.

## 9.1 N 이 큰 게 문제가 아니라 K 다 (질문 정정)

§8.3 의 제약은 **K** 에 대한 것이다. N=3072 는 이미 64 의 배수라 손댈 게 없고,
**K 를 64 배수로 맞추면 끝**이다(`ALIGN=64`, 이미 적용해 전체 72개 컴파일 성공).
N=3072 는 방아쇠일 뿐이고(N=768 에선 K=800 도 통과), 고치는 쪽은 K 다.
패드 열 수는 우리가 정하는 값이라 추가 비용도 없다.

## 9.2 기존 플래그가 이미 있다 — 절반은 해결

```
--iree-dispatch-creation-enable-fuse-padding-into-linalg-producer-ops
```
`FormDispatchRegionsPassOptions::fusePadWithProducers` (기본 false) 를 켠다.

| | 디스패치 | copy_buffer | fill_buffer |
|---|---|---|---|
| 원본 | 328 | 96 | 0 |
| Pad·QKV36 | 304 | 108 | 1 |
| **Pad·QKV36 + 플래그** | 304 | **96** | **0** |
| Pad·전체72 | 328 | 144 | 5 |
| **Pad·전체72 + 플래그** | 328 | **96** | **0** |

**증강이 만들던 여분 `copy_buffer`/`fill_buffer` 가 전부 사라져 원본과 같아진다.**
다만 **디스패치 수는 안 줄어든다** — Pad 가 생산자에 흡수되는 대신 `slow_memcpy` 에서
`tensor.pad` 한 개짜리 자기 디스패치로 바뀔 뿐이다. 그래도 순이득이므로 채택할 것.

## 9.3 왜 생산자에 흡수되지 않는가 — 최소 재현체로 특정

재현체를 단계적으로 실 모델에 맞춰가며 조건을 분리했다
(`gen_concat_repro.py` → `gen_pad_repro.py` → `gen_ln_pad_repro.py` →
`gen_multiuse_repro.py` → `gen_qkv_repro.py`, 각각 수 초 컴파일):

| 재현체 | Pad 융합 |
|---|---|
| 단순 elementwise 생산자, 소비자 1 | **됨** (디스패치 추가 0) |
| LayerNorm(reduction) 생산자, 소비자 1 | **됨** |
| + 생산자 결과가 residual 로도 쓰임(다중 사용), 소비자 1 | **됨** |
| **+ Pad 결과를 matmul 3개가 소비 (= 실 모델 Q/K/V)** | **안 됨 (+1 디스패치)** |

즉 **"reduction 생산자라서" 가 아니었다** — §8.5 의 추정은 틀렸다.
진짜 조건은 **Pad 를 사이에 둔 생산자-소비자 관계가 1:1 이 아닌 것**이다.

원인 지점: `FormDispatchRegions.cpp:396-411`

```cpp
// In non-aggressive mode, restrict fusion to producers whose results flow
// to a single consumer.
if (!aggressiveFusion) {
  llvm::SetVector<Operation *> consumers;
  for (Operation *user : op->getUsers()) { ... consumers.insert(user); }
  if (consumers.size() != 1) return {};      // <- 여기서 거부
}
```

BERT 에서 LayerNorm+Quantize 결과는 **Pad(QKV용) 와 residual 경로** 둘 다로 간다.

## 9.4 `aggressive-fusion` 은 막다른 길

`--iree-dispatch-creation-enable-aggressive-fusion` 은 위 제약을 푼다. 재현체에서
기준선이 4→3 으로 줄어 효과가 있다. 그러나 **Pad 비용 1 은 그대로**이고(B=0:3, B=64:4),
더 중요하게 **실 모델에서는 llvm-cpu 코드젠이 깨진다** — 원본과 kpadq 둘 다
`failed to run translation of source executable ... llvm-cpu`. 쓸 수 없다.

## 9.5 남은 것 — 결정 필요

Pad 를 생산자에 흡수시키려면 **`FormDispatchRegions` 의 위 제약을 Pad 소비자에 한해
완화**해야 한다(다중 소비자 생산자라도 Pad 는 흡수 허용). 이는 `third_party/iree`
서브모듈 수정 + 컴파일러 전체 재빌드가 필요하다.

**얻는 것**: Pad·QKV36 은 304→292(−12), **Pad·전체72 는 328→280(−48, −15%)**.
전체 72개 접합이 처음으로 순이득이 된다.

**재현/검증 수단은 갖춰져 있다**: `gen_qkv_repro.py` 가 수 초 만에 4 vs 3 으로
성공/실패를 판정한다.

## 9.6 현재 최선

**`bb_kpadq.vmfb` (Pad · QKV 36개) + 위 플래그** — NPU 96 / CPU 208 / 합계 304
(원본 328), NPU corr 0.902→0.921, 결정론적, Concat 판과 NPU 출력 비트 동일.
2D Pad(`Reshape→Pad→Reshape`)도 만들어 봤으나 수치는 동일하고 디스패치는 그대로라
복잡도만 늘어 채택하지 않는다.
