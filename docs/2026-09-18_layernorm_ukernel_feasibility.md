# 2026-09-18 — AIE2P LayerNorm 마이크로커널 성립성 검증 (standalone)

> 범위는 **standalone ukernel 구현·검증까지**다. generic LayerNorm chain 을 custom op 으로
> raise 하거나 dispatch/tile 파이프라인을 고치는 일은 하지 않았고, residual add 도 ABI 에
> 넣지 않았다. **컴파일러 패스 변경 없음.**

## 0. 결론

**성립한다.** 소프트 플로트 호출 0 개, `.text` 1,840 B, 스택 2,816 B, 반복 실행 단일 해시,
그리고 **모든 출력 원소가 float64 레퍼런스의 1 LSB 이내**(exact 89.7 %, corr 0.99990).

| | M=16 | M=8 |
|---|---|---|
| 반복 실행 | **5 회 단일 해시** `3d02a14c…` | **3 회 단일 해시** |
| exact | 89.67 % | 89.06 % |
| **\|Δ\| ≤ 1 LSB** | **100.00 %** | **100.00 %** |
| max\|Δ\| | **1** | **1** |
| corr | 0.9998956 | 0.9998915 |

⚠️ **이 커널은 코어 스택 4 KB 이상을 요구한다.** mlir-aie 와 IREE 둘 다 기본값이
0x400 = 1 KB 다. §5 가 이 문제로 하루를 쓴 기록이다.

## 1. ABI — scale 두 개가 인자에서 사라진다

```c
void layernorm_i8_16x768(int8 *input,  int64_t input_offset,
                         int8 *output, int64_t output_offset,
                         bfloat16 *gamma_beta, int64_t gamma_beta_offset,
                         float epsilon_scaled);
```

`input_scale` 도 `output_scale` 도 인자에 없다. 두 가지 대수 정리 덕분이고, 이것이 코어에서
스칼라 float 연산을 0 으로 만드는 핵심이다.

- **입력 scale 은 상쇄된다.** m, v 를 *정수* 표본의 평균·분산이라 하면
  `(x·s − m·s)/sqrt(v·s² + ε) = (x − m)/sqrt(v + ε/s²)`.
  즉 `s` 는 `epsilon_scaled = ε/s²` 안에만 남는다.
- **출력 scale 은 γ/β 에 접힌다.** 둘 다 컴파일 타임 상수이므로 호출자가
  `gamma_beta = [γ/s_out (768), β/s_out (768)]` 를 bf16 으로 넘긴다.

offset 은 전부 `int64_t` — softmax 가 `unsigned` 로 썼다가 출력이 0 이 됐던 사고를
반복하지 않는다.

### 1.1 bare-pointer 진입점

`layernorm_i8_bare_MxN(int8*, int8*, bfloat16*, float)`. softmax 가 두 ABI 를 노출한 것과
같은 이유이고, 여기엔 강제하는 제약도 있다: **aiecc 는 `aie.core` 안의 외부 호출 인자가
4 개를 넘고 그중 정수가 하나라도 있으면 assert 로 죽는다**
(`eraseOp ... expected that op has no uses`). 실측 경계는 `(memref, memref, memref, f32)`
통과, `i64` 하나만 더해도 실패. IREE 의 (포인터, offset) ABI 는 손으로 쓴 mlir-aie 설계에서
표현 자체가 불가능하다.

## 2. 알고리즘 — 통계는 정수 MAC, 출력만 bf16

행 하나(N=768)에 대해:

1. **통계** — int8 입력을 **그대로** 64 레인씩 읽어 코어의 int8 MAC 배열로 Σx 와 Σx² 를
   `v64acc32` 에 모은다(`mac_elem_64(x, ones)`, `mac_elem_64(x, x)`). bf16 스크래치를
   전혀 건드리지 않는다. 768 개 int8 의 Σx ≤ 97,536, Σx² ≤ 12,386,304 이고 64 레인 각각은
   12 청크분만 담으므로 **전 구간이 int32 에서 정확**하다. 64 레인은 메모리에 한 번
   내려 스칼라 정수 덧셈으로 접는다.
2. **행 스칼라** — `fix2float` 로 정확히 float 화한 뒤 평균·분산·rsqrt 를 **f32 레인**에서
   계산한다(`broadcast_to_v16float`, `mul_elem_16_accuracy_safe`, `sub`, `add`), 마지막에
   `invsqrt` NLF 단일 명령. 스칼라는 splat/추출만 하고 산술은 하지 않는다.
   `var = E[x²] − mean²` 의 상쇄 위험은 두 합이 정확한 정수라 없다.
3. **확장** — 출력 패스용으로만 int8 행을 bf16 으로 넓힌다(스칼라 `fix2float`, aie2p 에
   벡터 int→float 이 없다). int8 정수는 bf16 에서 정확하므로 무손실.
4. **출력** — `sub(ups(x), mean_acc)` 로 **f32 에서** 중심화, bf16 으로 내려
   `mac_elem_32(centred, rstd·γ', β')`, `+0.5` 후 `bfloat16_to_int` → `ssrs`(int8 포화).

bf16 반올림은 정확히 셋(중심화 결과, `rstd·γ'` 곱, 출력 직전 `+0.5`)뿐이고 평균은 f32
정밀도를 유지한다. γ/β 를 bf16 으로 받는 ABI 선택의 비용은 작다 — 레퍼런스를 bf16 γ/β 로
바꾸면 exact 가 89.67 % → 89.99 % 로 움직일 뿐이다.

### 2.1 accfloat 에서 f32 스칼라를 꺼내는 법

`to_v16float` 과 `to_v8float` 은 **둘 다 instruction selection 에서 실패한다**
(`cannot select ... llvm.aie2p.v32accfloat.to.v16float`). peano 가 intrinsic 을 선언만 하고
내리지 못한다. 무손실 경로는 **저장**뿐이고, 실제로 변환 없는 `vst bmll0, [p]` 한 개로
컴파일된다(디스어셈블리 확인). softmax 처럼 bf16 을 거쳐 꺼내면 행 합(~10⁵)에서 bf16 의
1 mantissa 가 ~256 이라 평균이 그만큼 망가진다.

## 3. 검증 결과

| 항목 | 결과 |
|---|---|
| **A. 컴파일** | ✅ in-tree 레시피 그대로 |
| **B. soft-float helper** | ✅ **0 개** (`llvm-nm -u` 비어 있음) |
| **C. `.text`** | **1,840 B** (+ bare wrapper 48 B) — 16 KB program memory 의 11 % |
| **D. 스택** | **2,816 B** → 코어 스택 **4 KB 이상 필요** |
| **E. 정확도** | 전 원소 1 LSB 이내, exact 89.7 %, corr 0.99990 |
| **F. M=16 / M=8** | 둘 다 위와 같음 |
| **G. 반복성** | M=16 5 회 / M=8 3 회 **단일 해시** |

L1 예산: `[16,768]` i8 입출력 더블 버퍼 49,152 B + 스택 2,816 B + γβ 3,072 B ≈ **54.7 KB**
(코어 64 KB). M=8 이면 ≈ 30.1 KB.

## 4. 재현 방법

`_local/layernorm_uk/`:

| 파일 | 역할 |
|---|---|
| `gen.py <M> [seed]` | IRON MLIR + 입력 + numpy 레퍼런스 2 종 생성 |
| `build.sh <M>` | ukernel `.o` → aiecc xclbin → XRT 호스트 |
| `host.cpp` | `x.bin` 입력, `out_<r>.bin` 출력 |
| `probe*.cc/mlir` | ABI 배선·행별 통계·버퍼 주소 진단 프로브 |

- aiecc 에 `--no-xchesscc --no-xbridge --aie-generate-npu-insts` 가 필요하다(없으면 chess 를
  찾다가 실패). 툴체인은 호스트 `~/NPU/mlir-aie/ironenv/bin/aiecc`.
- 커널은 `aie.device(npu2)` 의 코어 하나에서 돌고 **`aie.core` 에 `stack_size = 8192`** 를
  붙인다(§5).
- γ/β 는 `aie.buffer` 의 `initial_value` 로 굽는다 — 컴파일러 통합에서도 원하는 배치다
  (컴파일 타임 상수를 여러 코어에 DMA broadcast 하는 것은 순손실이고, broadcast 는
  `docs/2026-09-03_method_b_compiles_but_still_hangs.md` 의 hang 패턴을 부른다).

## 5. ⚠️ 원인이었던 것 — 코어 스택 오버플로

처음에는 **타일당 한두 행만** Σx² 가 과대평가되고 나머지 행은 비트 정확했다. 결정론적이고,
데이터와 무관하고, 주변 코드를 바꾸면 희생 행이 옮겨 다녔다.

진짜 원인은 **`aie.core` 의 `stack_size` 기본값이 `0x400` = 1 KB** 인데 이 커널의 프레임이
2.8 KB 라는 것이었다. 넘쳐도 fault 가 나지 않고, 프레임이 끝을 넘어가 **먼저 실행되는
행들의 통계만 조용히 망가진다.** `stack_size = 8192` 로 올리자 그대로 전 행이 정상이 됐다.

돌아보면 가장 강한 단서는 **스크래치를 늘릴 때마다 손상 행이 늘어난 것**이었다
(스크래치 이중화 1 → 3 행, 정수 레인 버퍼 추가 1 → 2 행). 그때 스택을 의심했어야 했다.
아래는 원인을 찾기 전에 시도해서 **효과가 없던** 것들이다 — 같은 함정에 다시 빠지지 않도록 남긴다.

| 시도 | 결과 |
|---|---|
| `__asm__ volatile("" ::: "memory")` | aie2p 가 인라인 asm 을 못 내림 (`unable to translate instruction: call`) |
| `__atomic_signal_fence` | `cannot select: G_FENCE` |
| 스크래치를 `volatile` 로 쓰기 | 오브젝트 **바이트 동일** |
| 벡터 읽기에 `may_alias` | 변화 없음 |
| 누산기를 0 대신 첫 청크로 시드 | 변화 없음 |
| 읽기 슬롯 분리 / 저장-읽기 거리 확대 | 변화 없음 |
| `#pragma clang loop pipeline(disable)` | 변화 없음 |
| 행 스크래치 더블 버퍼링 | **악화** (스택을 더 먹으니까) |

**컴파일러 통합 시 반드시 할 것**: IREE 의 `--iree-amdaie-stack-size` 기본값도 1024 이므로
이 커널을 쓰는 dispatch 는 **4096 이상**으로 올려야 한다. 참고로 XCLBinGen 은 ELF 의
`.stack_sizes` 메타데이터에서 실제 사용량을 읽으므로(`XCLBinGen.cpp:126 getMaxStackSize`),
자동으로 맞춰 줄 여지가 있는지 확인할 값어치가 있다.

## 6. 산출물

| 경로 | 내용 |
|---|---|
| `compiler/.../Target/uKernels/npu4/peano/layernorm.cc` | 커널 (신규, 미커밋) |
| `_local/layernorm_uk/{gen.py,build.sh,host.cpp}` | standalone 재현 하네스 |
| `_local/layernorm_uk/probe{,2,3}.cc` | ABI·통계·주소 진단 프로브 |
