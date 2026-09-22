# 2026-09-18 — GELU 를 ukernel 없이 math 다항 근사로 NPU 에 올리고, 인코더 한 층을 융합 상태로 컴파일

> 목적: K-fold 를 `--align=64` 로 적용하면 FC1 의 오랜 실패가 풀리는지 확인하고, 인코더
> 레이어 하나를 **융합을 켠 채** 컴파일·실행한다.
>
> 결과: **FC1/FC2 가 NPU 로 가고 하드웨어에서 정확히 실행된다.** 도중에 드러난 두 개의
> 서로 다른 벽 중 하나는 K-fold 가, 다른 하나는 새 패스가 해결했다.

## 0. 한 줄 결론

| 구성 (융합 켬) | 결과 |
|---|---|
| K-fold 없음 | ❌ `'amdaie.connection' op no producer DMA channel available` (FC1 의 그 고질적 실패) |
| K-fold 적용 | ❌ → 위 에러 **사라짐**, `math.erf` 에 lowering 없음 |
| + `AMDAIEApproximateMathFunctions` | ❌ → erf 해결, 스칼라 `G_FMA` 합법화 실패 |
| + `math.fma` 확장 | ✅ **컴파일·실행 성공. NPU 9 / CPU 8 (17 디스패치), 5 회 단일 해시** |

## 1. 실험 대상 — 인코더 레이어 하나

`extract_attn_res_ln.py` 에 `--whole-layer` 추가. layer 0 전체(attention + residual/LN +
FC1/GELU/FC2 + residual/LN)를 잘라낸다. 73 노드, MatMul 8 개, LayerNormalization 2 개.
여기에 head-split + 범용 K-fold(`fold_bias_into_k_general.py`, `--align=64`)를 적용한다.

K-fold 가 접은 것 — **proj, FC1, FC2 세 개 전부**:

| matmul | K | K/32 | 열 수 | bias 오차 | 재양자화 헤드룸 |
|---|---|---:|---:|---|---|
| attention proj | 768 → **832** | 26 | 31 | ≤0.499 LSB | 12.9 LSB |
| **FC1** (N=3072) | 768 → **832** | 26 | 3 | ≤0.500 LSB | 2.8 LSB |
| **FC2** | 3072 → **3136** | 98 | 3 | ≤0.500 LSB | 1.6 LSB |

`K/32` 가 둘 다 짝수라 `docs/2026-09-11_k_axis_bias_fold_two_column_fix.md` §8.3 의
`aie.memtile_dma` 홀수 실패 조건에 걸리지 않는다.

## 2. 벽 1 — FC1 의 DMA 채널: K-fold 가 해결한다

융합을 켜고 풀 컴파일하면 K-fold 전에는 `no producer DMA channel available` 로 죽고,
K-fold 후에는 **그 에러가 나오지 않는다.**

`docs/2026-09-03_method_b_compiles_but_still_hangs.md` 가 특정한 메커니즘 그대로다 —
진짜 변수는 "bias 냐"가 아니라 **"세 번째 tensor 급 채널 수요가 생기느냐"** 였고,
bias 를 K 축으로 접으면 FC1 의 tensor 입력이 3 → 2 가 되어 채널 할당이 통과한다.

## 3. 벽 2 — GELU 의 `math.erf`: 새 패스

```
error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`
       registration for dialect for op: math.erf
```

코어에는 libm 이 없으므로 살아남은 `math` op 은 **lowering 자체가 없다.**

### 3.1 upstream `iree-codegen-math-transform` 를 그대로 쓰면 기존 경로가 깨진다

먼저 upstream 패스를 그대로 파이프라인에 넣어 봤다. **기준선 attention 모델이 깨졌다**:

```
LLVM ERROR: unable to legalize instruction: %356 = G_FCOPYSIGN ... (in function: core_7_5)
```

원인: `MathTransformPass` 는 다항 근사(`populateMathPolynomialApproximationPatterns`)
**뿐 아니라** rewrite 세트(`populateMathFunctionsRewritePatterns`)도 돌리는데, 거기에
`math.roundeven` 이 들어 있다. 그 확장은 `copysign` 을 거치고 peano aie2p 는 이를
합법화하지 못한다. 그리고 이 저장소에는 그 일을 하는 전용 패스
(`iree-amdaie-expand-roundeven`)가 이미 있다.

### 3.2 다항 근사만 돌리는 패스

`AMDAIEApproximateMathFunctions` (신규). 다항 근사 세트(erf, tanh, exp, log … 14 개)와
f32 승격만 적용하고 **rewrite 세트는 건드리지 않는다.** 타일링 **전**에 돌려서 근사식이
소비자와 함께 타일링·벡터화되게 한다.

### 3.3 벽 2b — 근사식이 내놓는 스칼라 `math.fma`

```
LLVM ERROR: unable to legalize instruction: %373 = G_FMA ... (in function: core_7_5)
```

aie2p 는 스칼라 `G_FMA` 를 선택하지 못한다. 스칼라 `fmul`/`fadd` 는 소프트 플로트
libcall 로 내려가고 벡터 형태는 코어의 MAC 유닛이 직접 처리하는데, **스칼라 fma 만
갈 곳이 없다.** 같은 패스에 `math.fma → arith.mulf + arith.addf` 패턴을 추가했다.
단일 반올림 보장을 잃지만 다항 근사는 거기에 기대지 않는다.

## 4. 결과

### 4.1 디스패치 배치 (융합 켬, K-fold 적용)

```
 d0  CPU  elementwise_24576_f32xi8      입력 quantize
 d1  CPU  slow_memcpy                   x 의 K 패딩 (residual 때문에 소비자 2 개)
 d2-d4 NPU batch_matmul_12x32x64x832    Q / K / V
 d5  NPU  batch_matmul_12x32x32x64      QK^T
 d6  NPU  softmax_12x32x32              i8 softmax ukernel
 d7  NPU  batch_matmul_12x32x64x32      PV
 d8  NPU  batch_matmul_1x32x768x832     output projection (bias 가 누산기 안)
 d9  CPU  reduction_32x768_f32          residual + LayerNorm (평균)
 d10 CPU  reduction_32x768_f32          LayerNorm (분산·정규화·quantize)
 d11 CPU  slow_memcpy                   FC1 활성의 K 패딩
 d12 NPU  batch_matmul_1x32x3072x832    **FC1 + GELU 다항 근사 융합**
 d13 CPU  slow_memcpy                   FC2 활성의 K 패딩
 d14 NPU  batch_matmul_1x32x768x3136    **FC2**
 d15 CPU  reduction_32x768_f32          residual + LayerNorm (평균)
 d16 CPU  reduction_32x768_f32          LayerNorm (분산·정규화·quantize)
```

**NPU 9 / CPU 8 (17 디스패치).** CPU 8 개 중 4 개가 LayerNorm 두 벌, 3 개가 K 패딩,
1 개가 입력 quantize 다.

### 4.2 하드웨어 (5 회, `with-npu-lock.sh`)

| vmfb | 구성 | 5 회 해시 | corr (ORT) | maxdiff |
|---|---|---|---|---|
| `L0_kfold_math3` | 융합 켬 + GELU NPU | **1 종** `a6f770d0…` | 0.9927183 | 0.548 |
| `L0_kfold_nf` | 융합 끔 (GELU·재양자화 CPU) | 1 종 | **1.0000000** | 1.2e-05 |
| `L0_base_nf` | 융합 끔, K-fold 없음 | 1 종 | 1.0000000 | 5e-06 |

⚠️ **두 줄의 정확도 차이는 세 가지 변화가 섞인 값이다** — bf16 softmax ukernel,
GELU 다항 근사, 융합된 정수 재양자화. 분리하려고 softmax ukernel 만 끄고 다시 빌드했으나
**program memory 초과**라는 다른 벽에 걸려 이 문서에서는 분해하지 못했다. 참고로 같은
NPU 기능 조합의 attention 단독 블록은 이미 corr 0.9987 이므로(9/18 문서), 0.9927 이
전부 GELU 근사 탓은 아니다.

## 5. 회귀

- **기준선 `attn0_hsm_int8` 재컴파일 vmfb MD5 = `1b8490a5b86d86bc89d48ff1f0cefe18`** —
  9/17 하드웨어 검증본과 **바이트 동일**. 새 패스가 기존 경로를 건드리지 않는다.
  (upstream `MathTransform` 를 그대로 썼을 때는 이 빌드가 아예 실패했다 — §3.1.)
- `ctest -R "amd.aie|amdaie"`: 204 lit 테스트 중 실패 4 + timeout 1.
  - 실패 4 개(`controlcode_lowering`, `insert_cores`, `npu_dma_to_half_dma_cpy_nd`,
    `split_logicalobjfifos_for_connection_reuse`)는 기존에 알려진 것과 같다.
  - timeout 1 개(`matmul_elementwise_pack_peel_air_e2e`)는 직접 돌려 보니 **88 초**
    걸려 lit 의 60 초 한도를 넘는 것이고, 실패하는 split 은 **전부 i32**
    (`failed to legalize unresolved materialization ... memref<2x2x32x32xi32>`,
    AIR 파이프라인)이다. 이 패스는 float transcendental 과 `math.fma` 만 다루므로
    원인이 될 수 없다. **단 bisect 로 확증하지는 않았다.**

## 6. 변경 파일

| 파일 | 내용 |
|---|---|
| `Transforms/AMDAIEApproximateMathFunctions.cpp` | 신규 패스 (다항 근사 + fma 확장) |
| `Transforms/Passes.{td,h,cpp}`, `PassDetail.h`, `CMakeLists.txt` | 등록 및 파이프라인 편입 |
| `_local/int8_debug/tools/extract_attn_res_ln.py` | `--whole-layer` 옵션 |

산출물: `_local/int8_debug/out/layer0_{m,hs,hsk}_int8.{onnx,mlir}`,
`_local/res/L0_{base_nf,kfold_nf,kfold_math3}.vmfb`.

## 7. 다음

- **정확도 분해** — bf16 softmax / GELU 근사 / 재양자화 중 무엇이 0.9927 을 만드는지.
  softmax ukernel 을 끈 빌드가 program memory 초과로 죽는 것부터 봐야 한다.
- **GELU 근사의 코어 비용** — 근사식이 벡터화됐는지, 코어 ELF 를 얼마나 먹는지 미측정.
  팀원의 bf16 경로가 더 싸다면 그쪽이 맞다.
- LayerNorm 4 개 디스패치가 이제 CPU 쪽 최대 덩어리다 — 별도 과제.
