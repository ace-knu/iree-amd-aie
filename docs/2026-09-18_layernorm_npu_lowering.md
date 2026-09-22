# 2026-09-18 — LayerNorm 을 NPU ukernel 까지 내리기 (컴파일러 lowering)

> `docs/2026-09-18_layernorm_ukernel_feasibility.md` 의 커널을 실제 그래프에 연결하는 작업.
> **M1(순수 LayerNorm)은 하드웨어까지 통과. M4(residual producer 융합)는 미완.**
> 컴파일러 변경 다수 + third_party 패치 1 개, **아직 커밋 안 함.**

## 0. 결과

| 검증 | 결과 |
|---|---|
| M1 `[1,32,768]` i8→DQ→LN→Q→i8 | ✅ **NPU 에서 실행** |
| 5 회 실행 | **단일 SHA** `4b95f867435282bb` |
| ORT(f32 레퍼런스) 대비 | **100.00 % 가 ±1 LSB 이내**, exact 90.15 %, corr 0.9998986 |
| ukernel 호출 | ✅ `iree_codegen.ukernel.generic "layernorm_i8_16x768"` → `func.call` |
| row-only 타일링 | ✅ 32 행 → 코어당 16 행 × 2 코어, feature 768 은 symbol 이라 불가분 |
| 코어당 L1 | 27,648 B (in 12,288 + γβ 3,072 + out 12,288) + 스택 4,096 = **31.7 KB / 64 KB** |
| 조건 미충족 시 | ✅ raise 하지 않고 **기존 generic/CPU 경로 유지** (실측: f32 residual 모델은 CPU 2 디스패치) |
| 기존 attention 회귀 | ✅ `attn0_hsm_int8` vmfb MD5 **`1b8490a5…`**, 9/17 검증본과 동일 |
| ctest | ✅ **새 실패 0** (기존 4 + 알려진 timeout 1) |

## 1. 설계 — 상수를 전부 raise 시점에 접는다

커널은 scale 인자를 받지 않는다. 두 가지 대수 정리 덕분이다.

- `(x·s − m·s)/sqrt(v·s² + ε) = (x − m)/sqrt(v + ε/s²)` → **입력 scale 은 상쇄**되고
  `epsilon_scaled = ε/s_in²` 안에만 남는다. op 속성으로 운반한다.
- **출력 scale 은 γ/β 에 접는다.** raise 가 `[2,768] bf16` 상수 하나를 만든다:
  행 0 = `γ/s_out`, 행 1 = `β/s_out`. custom_op 오퍼랜드는 이것 하나뿐이고 f32 γ/β 는
  남기지 않는다 — DMA 6 KB 가 아니라 **3 KB**.

확정 ABI(인자 순서는 `iree_codegen.ukernel.generic` 이 만드는 순서에 맞춤:
입력들(ptr, offset) → 출력들(ptr, offset) → 스칼라):

```c
void layernorm_i8_16x768(int8 *input,      int64_t input_offset,
                         bfloat16 *gamma_beta, int64_t gamma_beta_offset,
                         int8 *output,     int64_t output_offset,
                         float epsilon_scaled);
```

**residual 은 ABI 에 없다.**

## 2. 파이프라인 — 어디에 무엇을 붙였나

| 단계 | 한 일 |
|---|---|
| `AMDAIERaiseLayerNorm` (신규, preprocessing) | generic 13 개 체인을 `iree_linalg_ext.custom_op` 하나로. 반복자는 `[parallel]`(행) 하나, feature 와 packed 상수의 두 차원은 **symbol** = 절대 타일링 안 함 |
| `AMDAIEAssignDeviceAffinities` | 표시된 LayerNorm executable 을 NPU 로 (ukernel 플래그 게이트) |
| `AMDAIELowerExecutableTarget` | root 가 그 op 이면 GeneralCopyPipeline |
| `KernelDispatch` | `setRootConfigForLayerNormCopyPipeline`: 행만 타일링, 코어당 행 수는 L1 예산에서 |
| `AMDAIELowerToUKernels` | 표시된 custom_op → `layernorm_i8_MxN` (N=768, M∈{16,8}) |
| `AMDAIEInsertCores` | layernorm.o 를 링크하는 **코어에만** 스택 4096 |

### 2.1 ⚠️ `custom_op` 의 body 를 루트/경계로 보던 패스가 네 개

`AMDAIETileAndFuse`, `AMDAIEInsertCopyOps`, `AMDAIEBufferizeToAllocation`(+`DistributeL1Allocations`)
가 전부 post-order 로 body 안의 generic 을 먼저 집었다. custom_op 은 규정적 융합 단위이므로
**"parentOfType<CustomOp> 이면 skip, custom_op 자체를 타깃에 추가"** 로 통일했다.

### 2.2 목적지를 읽지 않게 해야 한다

region 의 출력 블록 인자를 마지막 generic 의 init 으로 쓰면 "destination 을 읽는다"로 판정돼
**같은 L2 버퍼가 코어 양쪽에 붙고** objectfifo 타일 할당이 실패한다
(`No source or target tiles found`). 새 `tensor.empty` 에 쓰고 그것을 yield 하면 사라진다.

### 2.3 스택 — 조용히 틀리지 않는다

`XCLBinGen` 은 `.stack_sizes` 를 **자동 반영하지 않고 검증만** 한다. 실측으로 확인:

```
An upper bound for the stack size of the core (col=0, row=2), inferred from the
object file, is 1152 bytes. The assigned memory for the stack is 1024 bytes,
which is insufficient (1152 > 1024).
```

즉 부족하면 **컴파일이 실패**한다(조용히 틀리지 않는다). 그래서 `--iree-amdaie-stack-size`
(전역, 모든 코어의 L1 을 먹음)를 올리는 대신 **`AMDAIEInsertCores` 에서 layernorm.o 를
링크하는 코어에만** 4096 을 준다.

## 3. ⚠️ 상류 IREE 버그 하나 (패치로 공유)

`CustomOp::getIterationDomainForDimensions` 가 심볼 차원의 범위를
`concatMap.getResult(symbol + numSymbols)` 에서 읽는데, `getDimExprsForSymbols` 는 심볼 `k` 를
차원 `k + numDims` 로 번호 매긴다. **심볼 수와 루프 수가 같을 때만 우연히 맞는다.**
LayerNorm 은 루프 1 개 · 심볼 2 개라 `ArrayRef` 범위를 넘어 `iree-compile` 이 assert 로 죽는다.

`_local/share/0004-iree-custom-op-symbol-range-index.patch` (`third_party/iree` 에 적용).

## 4. 미완 — M4 residual producer 융합

두 형태를 다 만들어 재봤다.

| 모델 | 결과 |
|---|---|
| **m4a**: i8 + i8 → (정수 add) → DQ → LN → Q | LN 은 **NPU**, 그러나 add 는 **CPU 별도 디스패치로 남음** |
| **m2**: DQ, DQ → f32 add → LN → Q (실제 BERT 형태) | raise **안 함**, CPU 2 디스패치 (의도된 보수적 동작) |

**m4a 가 융합되지 않는 구조적 이유**: IREE 는 elementwise producer 를 **1-D 로 선형화**한다
(`linalg.generic (d0)->(d0)` on 24576). custom_op 은 행 구조 `[32,768]` 이라 반복 공간이 맞지
않아 `FormDispatchRegions` 의 producer 융합이 성립하지 않는다.

**m2 가 더 근본적인 문제**: 실제 BERT 의 residual 은 두 dequantize 사이의 **f32 덧셈**이다.
커널이 i8 을 받는 한, producer 가 i8 을 내놓아야 하는데 그러려면 **덧셈 결과를 i8 로 재양자화**
하는 그래프 변환이 필요하다 — 수치가 바뀌는 변환이라 지시 없이 하지 않았다.
(residual 을 ABI 에 넣는 것은 금지사항이므로 대안이 아니다.)

선택지는 둘이다. ① producer 가 정수 도메인에서 i8 을 내도록 재양자화를 넣는다(오차 1 회 추가),
② 커널에 i32 입력 변형을 추가해 producer 가 i32 를 넘기게 한다(추가 양자화 없음, L1 4 배).

## 5. 검증 재현

```
# 컴파일 + 실행 (M1)
_local/ln_lower/build.sh ln_m1 m1_uk5 "--iree-amdaie-enable-ukernels=layernorm"
scripts/lock/with-npu-lock.sh build/tools/iree-run-module --device=amdxdna \
  --module=_local/ln_lower/m1_uk5.vmfb --function=lnm1 \
  --input=@_local/ln_lower/m1_x.npy --output=@out.npy
```

모델 생성은 `_local/ln_lower/gen_ln_min.py` (m1/m2), `gen_m4.py` (m4a).

## 6. 남은 일

- M4 의 producer 형태 결정(§4 의 ①/②) 후 실 BERT `attn0res` 연결.
- M=8 tail 타일 경로 실측(현재 M=16 만 하드웨어 확인).
- 성능 측정은 아직 하지 않았다.
