# peano-v22 브랜치 단계별 정리

`peano-v22` 브랜치에서 진행한 단계별 변경사항과 그 결과(정확도, 컴파일/실행 시간, dispatch별 시간)를 정리한다.
모든 수치는 같은 호스트(npu4, amdxdna `timeout_in_sec=60`)에서 같은 입력으로 측정했다.

| 단계 | 컴파일러 | Peano | ukernel | 벡터화 |
|---|---|---|---|---|
| **S0 기존** | dev `f7a5c86f` | v19 `31d2aa6e` | off | off |
| **S1 Peano v22** | 1단계 완료(`8c8a9b31`) | v22 `83ba9ea9` | off | off |
| **S3 ukernel on** | 2-A 완료 + conv4 수정(`990bcab2`) | v22 | `all` | off |

- S2(2-A 완료 + ukernel off)는 S1과 결과가 같아(아래 "2-A 변경의 ukernel off 영향") 따로 분석하지 않는다.
  S1의 dispatch별 측정은 2-A 완료 컴파일러에서 ukernel을 끈 빌드로 했다.
- 2-B(벡터화 on)는 현재 배치에서 켰을 때의 동작만 확인했다(아래 "2-B 단계"). 코드 변경은 없다.
- 브랜치 마무리로 ukernel 기본 컴파일러를 peano로 바꿨다(`9177dcf6`). 이 커밋에서 chess 플래그 없이 최종 검증했다:
  ctest 221/221, CI 41/41, ukernel off MLP 0.999995 / VGG 0.99999,
  ukernel `all` MLP 0.999913(컴파일 8.4s / 실행 0.32s), VGG 0.99997(컴파일 98.6s / 실행 2.39s, top-1 일치).
- 추가 작업으로 bf16 변환 옵션 없이 f32 그대로 e2e를 실행하도록 고쳤다(아래 "추가 작업: f32 e2e").

## 요약

| | S0 기존 | S1 Peano v22 | S3 ukernel on |
|---|---|---|---|
| VGG-16 정확도 (logit corr vs f32) | 1.00000 | 0.99999 | 0.99997 (top-5 일치) |
| VGG-16 컴파일 | 122.6s | 70s | 99.2s |
| VGG-16 실행 (wall) | 74.5s | 8.8s | **2.29s** |
| └ NPU 연산 (23 dispatch 합) | 72.30s | 6.65s | **0.124s** |
| └ dispatch 고정 비용 (hwctx 생성/해제) | 1.77s | 1.75s | 1.80s |
| MLP 정확도 (corr vs llvm-cpu) | 0.999995 | 0.999995 | 0.999913 |
| MLP 컴파일 / 실행 | 7.6s / 0.33s (\*\*) | 4.4s / 0.22s | 8.3s / 0.22s |
| ctest / npu4 CI | 221 / 43 | 221 / 43 | 221 / 41 (\*) |

(\*) upstream `matmul_64_64_64_bf16` ukernel_peano 테스트 2개를 제외했다(2-A 홀수 블록 참고).

(\*\*) S0 MLP 실행 시간은 이전 기록값이다. S1/S3은 같은 방법(첫 실행 제외)으로 측정했고, 둘 다 dispatch 2개의 고정 비용(약 150ms)이 대부분이다.

- VGG 실행 시간은 S0 대비 **32배**(74.5s → 2.29s) 줄었다. NPU 연산만 보면 **583배**(72.3s → 0.124s)다.
- S3에서는 실행 시간의 79%(1.80s / 2.29s)가 연산이 아니라 dispatch 고정 비용이다. 다음 개선의 가장 큰 대상이다.
  이 비용의 대부분은 NPU power mode(default)의 영향이다. performance mode에서는 VGG 실행이 0.44s가 된다(아래 "NPU power mode의 영향").
- 같은 VGG를 CPU only(f32, `znver5`)로 실행하면 0.34s다(함수 호출 0.096s). NPU 최적화의 목표 수치다(아래 "CPU only 기준선").
- 벡터화를 켜도 S3 결과는 같다. 앞으로 NPU 벡터 연산은 ukernel을 우선 추가하고, 적절하지 않을 때만 aievec을 확장한다.

---

## NPU에서 matmul이 실행되는 방식

VGG에서 NPU가 실행하는 것은 matmul뿐이다(아래 "VGG 연산의 타입과 실행 위치"). 단계마다 그 matmul의 곱셈과 누적이 다음과 같이 실행된다.

| 단계 | 입력 | 곱셈 | 누적 | 출력 |
|---|---|---|---|---|
| S0 | bf16 → f32 확장(정수 shift) | soft-float `__mulsf3` 호출 | soft-float `__addsf3` 호출 | f32 |
| S1 | bf16 값 하나를 vector register에 broadcast(`vbcst.16`) | `vmul.f` bf16 x bf16 → f32, lane 1개만 사용 | `vadd.f` f32, lane 1개만 사용 | f32 |
| S3 | bf16 64개 → bfp16 블록으로 변환 | `mac_8x8_8x8T`: 명령 하나에 MAC 512개(bfp16 x bfp16) | 같은 명령 안에서 f32 accumulator에 누적 | f32 |

- AIE2P 코어에는 scalar 부동소수점 연산기가 없다. 그래서 scalar 부동소수점 연산은 정수 명령으로 흉내 내거나(S0) vector unit으로 처리한다(S1).
- **S1은 벡터화가 아니다.** vector unit을 쓰지만 명령 하나에 곱셈 하나이고, lane 하나만 의미 있는 값이다.
  곱셈과 덧셈 한 쌍마다 broadcast, 곱셈, 덧셈, lane 추출(`vextract.32`) 명령이 함께 실행된다.
  - IR은 "bf16 → f32 확장 후 f32 곱셈"이지만, `vmul.f`가 bf16 두 개를 곱해 바로 f32 결과를 낸다. bf16끼리의 곱은 f32로 정확히 표현되므로 결과는 같다.
  - 이 lowering은 matmul만이 아니라 코어 코드의 모든 scalar 부동소수점 연산에 적용된다(backend의 instruction selection).
- **S3에서 처음으로 SIMD가 된다.** 컴파일러 벡터화 pass(aievec 경로)는 꺼져 있고, ukernel이 손으로 쓴 벡터 코드로 matmul을 대신한다.
  bf16 → bfp16 변환에서 블록 안의 작은 값이 mantissa 비트를 잃으므로, 세 단계 중 S3만 곱셈 정밀도가 bf16보다 낮다(2-A 정밀도 참고).

| conv3_2 (3136x256x2304) | S0 | S1 | S3 |
|---|---|---|---|
| 명령 하나당 MAC 수 | 1 미만(함수 하나에 수십 명령) | 약 1 | 512 |
| 실측 처리량 | 0.5 GFLOP/s | 7.3 GFLOP/s | 653 GFLOP/s |

### VGG 연산의 타입과 실행 위치

`--iree-amdaie-demote-contraction-inputs-to-bf16`은 contraction(matmul)의 입력 A와 B만 bf16으로 낮춘다. 출력과 누적은 f32다.
`AMDAIEAssignDeviceAffinities`는 contraction과 convolution만 NPU에 두고 나머지를 CPU에 둔다([`device_placement.md`](../device_placement.md)).

| 연산 | 타입 | 실행 위치 |
|---|---|---|
| matmul (im2col conv 13개, FC 3개) | 입력 bf16, 누적과 출력 f32 | NPU |
| im2col | f32 입력을 펼치면서 bf16으로 변환 | CPU |
| bias add + ReLU | f32 | CPU |
| maxpool | f32 | CPU |
| layout transpose | f32 | CPU |
| FC 사이의 bias + ReLU + bf16 변환 | f32 → bf16 | CPU |
| 마지막 bias | f32 | CPU |

- matmul 이외의 f32 연산은 모두 CPU에서 실행되고, 합쳐서 약 0.1s다(dispatch별 분석 참고).

---

## 1단계: S0 → S1 (Peano v19 → v22)

### 변경사항

| 커밋 | 내용 |
|---|---|
| `7d485626` | Peano opt에 `-unroll-full-max-count=0`(full unroll 끔) |
| `b12e2511` | stack check의 한계(`.stack_sizes`가 없는 helper 함수)를 주석으로 기록 |
| `8c8a9b31` | Peano를 v22(`22.0.0.2026101201+83ba9ea9`)로 고정, fork release에 mirror |

- **full unroll을 끈 이유**: v22에서 full unroll이 켜지면 두 가지 문제가 생겼다.
  - VGG 코어 `.text`가 8304B → 17680B로 늘어 16KB program memory를 넘는다.
  - f32 8x4x8 kernel의 stack frame이 128B → 896B로 늘어, 1024B stack을 조용히 넘친다(필요 1088B).

### 결과

- ctest 221/221, npu4 CI 43/43, f32 random 정확, MLP/VGG 정확도 유지(v19 출력과의 corr 0.9999994).
- **NPU 연산이 dispatch마다 6~16배 빨라졌다.** v19는 scalar 부동소수점 곱셈과 덧셈을 soft-float 함수 호출(`__mulsf3`, `__addsf3`)로
  lowering했고, v22는 lane 하나만 쓰는 vector unit 명령(`vbcst.16`, `vmul.f`, `vadd.f`)으로 lowering한다(위 "NPU에서 matmul이 실행되는 방식").
  곱셈 한 번의 비용이 줄었을 뿐 연산은 여전히 scalar다. 코어를 disassemble해서 확인했다.
- 컴파일 시간도 123s → 70s로 줄었다. 원인은 따로 분리해 측정하지 않았다.

---

## 2-A 단계: S1 → S3 (ukernel on, 벡터화 off)

ukernel을 켜면 matmul이 lane 하나짜리 scalar 연산 대신, 명령 하나에 MAC 512개를 처리하는 bfp16 벡터 MAC(`mac_8x8_8x8T`)을 쓰는 ukernel로 바뀐다. 켜는 과정에서 네 가지 문제를 해결했고, dispatch별 분석에서 찾은 DMA 문제 하나를 더 고쳤다.

### 변경사항

| 문제 | 커밋 | 원인과 해결 |
|---|---|---|
| 홀수 블록 | `462325de` | bf16 ukernel이 M/N을 2x2로 unroll해서 M/8, N/8이 홀수면 tile 밖을 읽고 쓴다. ukernel on일 때 M/N을 rows/cols x 2 x 8의 배수로 패딩하고, 짝수 블록 tile을 고르고, 홀수 블록 tile은 ukernel에서 제외하고, `static_assert`를 추가했다. 홀수 블록에 기대어 통과하던 upstream CI 2개는 제외했다. |
| memtile join race | `6499776a`, `579ab66f` (bert 브랜치 cherry-pick), `dd16b81c` | 4개 코어 행이 하나의 L2 버퍼에 쓸 때 lock pair 하나를 공유해서, 빠른 행이 shim이 읽기 전에 ping 버퍼를 덮어썼다. ukernel로 코어가 빨라지면서 드러났다. producer마다 lock pair를 두고 BD를 producer별로 나눴다. producer가 연속으로 tile하지 않는 4-level 배치는 공유 lock pair를 유지한다. |
| 크기 조합 부족 | `92736f26` | ukernel `.cc`에 고정 크기 목록만 있어서 목록 밖의 크기는 링크 실패했다. LowerToUKernels가 `ukernel_instance` attribute에 template 인자를 기록하고, XCLBinGen이 이를 `-DUK_INSTANCES`로 넘겨 dispatch가 쓰는 크기만 정의한다. 고정 목록은 삭제했다. 같은 커밋에서 bf16 ukernel에 transpose_b 변형(B를 `[n][k]` 블록으로 읽음)을 추가해 FC도 ukernel을 쓴다. |
| 정밀도 | `cff1e56e` | bf16 → bfp16 변환이 코어 기본 rounding인 floor로 동작해 곱마다 한쪽으로 치우친 오차가 생기고, 층을 지나며 쌓였다(VGG corr 0.979). 함수 동안 round-to-nearest-even으로 바꾸고 끝에서 되돌린다. |
| K 방향 DMA loop (conv4_2/4_3) | `990bcab2` | 아래 "conv4_2/4_3 DMA 문제" 참고. |

- **정밀도 분석**: bfp16(8개 원소가 exponent 하나를 공유, int8 mantissa)을 numpy로 emulation해서 NPU 출력과 비트 단위로 일치시켰다.

  | VGG emulation | corr | top-5 |
  |---|---|---|
  | bfp16 + floor (수정 전 NPU) | 0.97917 | 2~5위 바뀜 |
  | bfp16 + 일반 반올림 | 0.99974 | 2·3위 바뀜 |
  | bfp16 + RNE (수정 후 NPU) | 0.99997 | f32와 같음 |

  RNE에서 블록 최댓값(양수)이 127을 넘게 반올림되면, 하드웨어는 exponent를 1 올리고 블록을 다시 양자화한다.

### 결과

- ctest 221/221, npu4 CI 41/41, f32 random 정확.
- VGG: corr 0.99997, top-5가 f32 reference와 같다. 23개 NPU dispatch(conv 13개, FC 10개) 모두 ukernel을 쓴다.
- MLP: corr 0.999913.
- 실행 시간은 크게 줄었지만, 컴파일 시간은 70s → 99.2s로 늘었다. ukernel `.cc` 하나를 Peano로 컴파일하는 데
  약 1.2s(대부분 intrinsic header parsing)가 걸리고, dispatch마다 matmul과 zero_fill을 컴파일하므로 VGG에서 46회다.

### conv4_2/4_3 DMA 문제 (`990bcab2`)

dispatch별 분석에서 conv4_2/4_3(784x512x4608)만 100.8ms로, K가 절반인 conv4_1(6.7ms)보다 15배 느렸다.

- **재현**: 784x512xK를 단독 실행하면 K=4096까지 8.4ms, K=4224부터 100ms다. 결과는 모든 K에서 정확하다.
  코어당 K tile이 16이라 K 방향 반복 횟수(K/16)가 256을 넘는 지점이다.
- **원인**: `AMDAIEDmaLoopSubsumption`은 control-code loop를 DMA 접근 패턴의 차원으로 흡수해 DMA 명령 하나로 만든다.
  이 pass가 반복 횟수가 shim repeat count(256)를 넘는 loop는 모두 흡수하지 않았다.
  흡수되지 않은 loop는 `AMDAIEControlCodeLoopUnroll`에서 펼쳐져, 반복마다 16x16짜리 DMA 12개와 wait가 생긴다.
  NPU 명령열이 45KB(4,959줄)에서 42MB(4,722,683줄)로 늘고, 펌웨어 처리에 약 92ms가 더 걸렸다.
- **잘못된 점**: repeat count는 stride가 0인 차원(같은 데이터 반복)만 제한한다.
  K loop처럼 주소를 옮기는 loop는 stride가 0이 아닌 일반 차원이 되고, 그 한계는 `isValidAccessPattern`이 따로 검사한다.
  `isValidAccessPattern`이 stride 0 차원을 검사하지 않아서 생긴 구멍을, loop 반복 횟수 전체를 막는 방식으로 막고 있었다.
- **수정**: 흡수 후 접근 패턴에서 stride 0 차원의 크기만 repeat count와 비교한다.
  기존 lit test(같은 주소를 257·1024번 반복)는 그대로 거부되고, 주소를 옮기는 300회 loop가 흡수되는지 확인하는 test를 추가했다.
- **결과**: 784x512x4608 단독 실행 102ms → 9.1ms, VGG의 conv4_2/4_3은 100.8ms → 9.5/9.3ms다.
  VGG 컴파일 시간도 113.5s → 99.2s로 줄었다(펼친 control code를 처리하던 시간).

### 2-A 변경의 ukernel off 영향

ukernel off일 때 영향을 주는 변경은 memtile lock pair 변경뿐이다. 2-A 완료 후 ukernel off 결과는 S1과 같다:
ctest 221/221, CI 41/41, MLP corr 0.999995, VGG corr 0.99999 / 컴파일 71s / 실행 8.9s.

## 2-B 단계: 벡터화 on (현재 배치 그대로)

`--iree-amdaie-enable-vectorization-passes=true`(컴파일러 기본값)로 `990bcab2`를 그대로 실행했다. 코드 변경은 없다.

| 구성 | MLP | VGG |
|---|---|---|
| ukernel `all` + 벡터화 on | 컴파일 8.4s / 실행 0.31s / corr 0.999913 | 컴파일 100.7s / 실행 2.41s / corr 0.99997, top-1 일치 |
| ukernel off + 벡터화 on | 컴파일 실패 | 컴파일 실패 |

- **ukernel `all`**: 벡터화 off와 결과가 같다. VGG NPU dispatch 23개의 core LLVM IR이 벡터화 on/off에서 동일하다.
  matmul은 ukernel이 처리하고, elementwise 연산(bias, ReLU, pool 등)은 CPU에 배치되어 NPU에서 벡터화할 대상이 없다.
- **ukernel off**: aievec이 bf16 x bf16 → f32 `vector.contract`를 lowering하지 못해 실패한다
  (`'vector.contract' op has matmul operand types`). npu4의 알려진 제약이다(`run.py`의 TODO).

### 앞으로의 벡터화 방향

aievec 경로를 새로 만드는 것보다 ukernel이 간단하고 효과적이어서, 다음 순서를 원칙으로 한다.

1. **ukernel 우선**: npu4에서는 ukernel `all`을 권장한다. 컴파일러 기본값은 바꾸지 않는다(npu1과 CI 전체 영향 미검증).
2. **지원하지 않는 입력**: 먼저 ukernel(template + matcher)을 추가한다.
   예: dev_GEMV 통합 후 NPU로 오는 pool, matmul epilogue의 bias + ReLU fusion, 추가 타입 조합.
3. **ukernel이 적절하지 않을 때만 aievec lowering을 추가한다.** 판단 기준 예:
   - 임의의 elementwise 연산 체인을 fusion해야 해서 커널 조합이 계속 늘어나는 경우
   - 타입과 shape 조합이 많아 instance 수와 컴파일 시간을 감당하기 어려운 경우
   - 연산이 단순해서 codegen만으로 충분히 빠른 경우
4. **그 사이에는 scalar fallback으로 동작해야 한다.** 현재는 ukernel matcher가 받지 않는 bf16 matmul
   (odd-block tile, transpose_a, npu1의 transpose_b)이 벡터화 on(기본값)에서 컴파일에 실패한다.
   해당 dispatch의 벡터화를 끄는 fallback은 backlog다.

---

## 추가 작업: f32 e2e (bf16 변환 없이)

`--iree-amdaie-demote-contraction-inputs-to-bf16`을 주면 지금까지처럼 matmul 입력을 bf16으로 낮추고, 빼면 f32 그대로 NPU에서 실행한다.
옵션을 줬을 때의 동작은 바뀌지 않는다(아래 "결과"의 bf16 행, dispatch binary 비교). 벡터화는 두 경우 모두 off다.

### 변경사항

| 문제 | 커밋 | 원인과 해결 |
|---|---|---|
| VGG f32 컴파일 실패 | `b492865a` | 패딩 pass가 vector instruction이 없는 타입(f32 등)을 건너뛰어, conv5(M=196)와 FC(M=1)가 tile로 나누어떨어지지 않았다. KernelDispatch가 실제로 쓰는 scalar pack 크기(4x4x8)로 패딩한다. 두 곳이 같은 값을 쓰도록 `getScalarMatmulPackSize()`로 묶었다. |
| M 방향 반복이 256을 넘는 matmul의 hang | `18cafea6`, `2119b20f` | 아래 "BD chain hang" 참고. 타입과 무관하다(bf16 81920x64x576에서도 재현). f32는 tile이 작아 conv1_2(50176x64x576)에서 반복이 392가 되어 드러났다. |
| f32 + ukernel `all` 컴파일 실패 | `65ece963` | ukernel matcher가 타입을 검사하지 않아, 템플릿이 없는 타입(f32, i8 x i8 → i16 등)도 ukernel로 보내 ukernel 컴파일이 실패했다. npu4 peano 템플릿 표 `hasUKernelTemplate`을 두고, matcher(matmul, zero_fill, trunci), KernelDispatch, 패딩이 이 표로 ukernel 사용 여부를 판단한다. 템플릿이 없으면 ukernel off와 같은 경로로 간다. npu4 밖의 장치는 동작이 그대로다. |

- **BD chain hang**: M 반복이 256(shim repeat count)을 넘으면 RHS를 반복마다 DMA로 보낸다.
  - `AMDAIEInsertDmaBdChain`은 같은 connection의 DMA를 BD chain으로 묶고, chain의 마지막 DMA에서만 queue에 push한다.
    chain이 다른 connection의 wait를 사이에 두면, wait 대상(LHS)이 끝나려면 코어가 RHS를 받아야 하는데 그 RHS는 wait 뒤에야 push되어 deadlock이 된다.
    chain을 그것이 넘는 wait마다 끊는다(`2119b20f`).
  - 같은 shape에서 `AMDAIEFoldDmaWaits`가 wait를 접을 때 token을 가진 DMA의 BD 하나만 재사용 검사를 했다.
    chain의 다른 BD를 그 chain이 끝나기 전에 다시 programming하는 경우가 IR에 308개 있었다(chain 분할 없이도 존재, hang에 가려져 있었음).
    chain의 모든 BD와, 접힌 wait와 남는 wait 사이에서 다시 programming되는 BD까지 검사한다(`18cafea6`).
  - chain 분할로 chain이 짧아지지만, bf16 VGG의 NPU 연산 시간은 그대로다(ukernel off 6.65s, `all` 0.124s → 0.125s).

### 결과

`65ece963`에서 측정했다. VGG 실행은 3회 중앙값(bf16 ukernel `all`은 strace 2회)이다.

| 구성 | VGG 정확도 | VGG 컴파일 | VGG 실행 (wall) | └ NPU 연산 | MLP 정확도 | MLP 컴파일 / 실행 |
|---|---|---|---|---|---|---|
| f32, ukernel off | corr 1.00000, top-1 일치 | 64.1s | 35.1s | 32.73s | 1.000000 | 3.7s / 0.21s |
| f32, ukernel `all` | corr 1.00000, top-1 일치 | 84.5s | 35.1s | 32.74s | 1.000000 | 5.9s / 0.33s |
| bf16, ukernel off | corr 0.99999, top-1 일치 | 64.9s | 8.8s | 6.65s | 0.999995 | 4.2s / 0.19s |
| bf16, ukernel `all` | corr 0.99997, top-1 일치 | 99.6s | 2.28s | 0.125s | 0.999913 | 8.5s / 0.31s |

- 정확도는 VGG가 ONNX f32 reference 대비 logit corr, MLP가 llvm-cpu 대비 corr이다. f32는 MLP max abs 차이 1.4e-6이다.
- f32 ukernel `all`의 컴파일이 20s 더 긴 것은 zero_fill ukernel 때문이다. f32 matmul은 템플릿이 없어 ukernel을 쓰지 않지만
  누적값 초기화(fill)는 f32 템플릿이 있어 ukernel로 가고, `zero_fill.cc`를 NPU dispatch마다(VGG 23번) Peano로 컴파일한다.
  실행 시간은 같으므로 f32에서는 `--ukernel none`을 권장한다(`scripts/onnx/README.md`).
- bf16 ukernel `all`의 VGG dispatch 23개는 이 작업 전(`2119b20f`)과 core IR, NPU 명령열, pdi가 byte 단위로 같다.
- i8 x i8 → i16 matmul(256x256x256)은 ukernel off와 `all` 모두 numpy와 정확히 일치한다(이전에는 `all`에서 컴파일 실패).
- ctest 221/221.

### dispatch별 NPU 연산 시간

| dispatch | 층 | M x N x K | f32 off (ms) | f32 `all` (ms) | bf16 off (ms) | bf16 `all` (ms) | f32 / bf16 off | f32 / bf16 `all` |
|---|---|---|---|---|---|---|---|---|
| d2 | conv1_1 | 50176x64x27 | 229.0 | 229.0 | 56.0 | 6.65 | 4.1x | 34x |
| d6 | conv1_2 | 50176x64x576 | 3630.3 | 3630.6 | 543.0 | 7.69 | 6.7x | 472x |
| d10 | conv2_1 | 12544x128x576 | 1921.9 | 1922.2 | 277.7 | 4.22 | 6.9x | 456x |
| d14 | conv2_2 | 12544x128x1152 | 3490.5 | 3490.8 | 518.6 | 6.01 | 6.7x | 581x |
| d18 | conv3_1 | 3136x256x1152 | 1834.5 | 1834.7 | 263.4 | 4.17 | 7.0x | 440x |
| d22 | conv3_2 | 3136x256x2304 | 3463.9 | 3465.0 | 508.5 | 6.11 | 6.8x | 567x |
| d26 | conv3_3 | 3136x256x2304 | 3484.6 | 3484.1 | 508.9 | 6.11 | 6.8x | 571x |
| d30 | conv4_1 | 784x512x2304 | 1661.3 | 1661.9 | 261.2 | 6.13 | 6.4x | 271x |
| d34 | conv4_2 | 784x512x4608 | 3373.4 | 3374.0 | 512.7 | 9.49 | 6.6x | 356x |
| d38 | conv4_3 | 784x512x4608 | 3282.1 | 3282.5 | 512.4 | 10.94 | 6.4x | 300x |
| d42 | conv5_1 | 196x512x4608 | 908.2 | 908.4 | 272.0 | 4.49 | 3.3x | 202x |
| d46 | conv5_2 | 196x512x4608 | 933.9 | 934.4 | 272.4 | 4.24 | 3.4x | 220x |
| d50 | conv5_3 | 196x512x4608 | 923.1 | 922.4 | 272.6 | 3.97 | 3.4x | 233x |
| d54 x8 | fc1 (N-split 8개, 각각) | 1x512x25088 | 374~375 | 374~375 | 210~211 | 4.14~4.70 | 1.8x | 80~90x |
| d56 | fc2 | 1x4096x4096 | 477.7 | 478.2 | 146.1 | 5.17 | 3.3x | 92x |
| d58 | fc3 | 1x1000x4096 | 119.8 | 120.1 | 39.4 | 3.88 | 3.0x | 31x |

- **f32는 ukernel off에서도 bf16 off보다 큰 conv에서 약 6.7배 느리다.** AIE2P에는 f32 x f32 곱셈 명령이 없어서,
  f32 곱셈은 soft-float `__mulsf3` 호출이 된다. bf16 x bf16 → f32는 `vmul.f` 명령 하나다(MLP 코어 disassemble로 확인).
  덧셈은 둘 다 `vadd.f`다.
- **f32 ukernel off는 S0(Peano v19)보다 약 2배 빠르다**(VGG NPU 연산 72.3s → 32.7s, conv3_2 7145ms → 3464ms).
  같은 tile loop를 두 Peano로 컴파일해 비교했다.

  | 구성 | 곱셈 | 덧셈 | MAC당 soft-float 호출 |
  |---|---|---|---|
  | S0: v19, bf16 입력(f32로 확장) | `__mulsf3` | `__addsf3` | 2 |
  | v22, f32 입력 | `__mulsf3` | `vadd.f` | 1 |
  | S1: v22, bf16 입력 | `vmul.f` | `vadd.f` | 0 |

  - 덧셈이 `vadd.f`가 된 것은 Peano 버전 차이다. v22가 scalar f32 덧셈을 lane 1개짜리 vector 명령으로 lowering한다.
  - 곱셈이 soft-float로 남는 것은 하드웨어 제약이다. AIE2P의 부동소수점 곱셈 명령은 bf16 x bf16 → f32뿐이다.
- **f32 ukernel `all`은 off와 같다.** f32 matmul 템플릿이 없어서 matmul은 ukernel off 경로를 탄다.
  컴파일이 20s 늘어난 것은 f32 zero_fill이 ukernel을 쓰기 때문이다(dispatch마다 ukernel `.cc` 컴파일).
- conv5와 FC는 비율이 작다. f32는 scalar pack(4)으로, bf16은 instruction 크기(8)로 패딩해서 계산량이 다르다
  (예: FC의 M=1을 f32는 16, bf16 off는 32로 패딩).
- **f32는 기준값(reference) 경로로 유지하고, f32 matmul ukernel은 만들지 않는다.**
  성능은 하드웨어가 지원하는 벡터/행렬 연산(bf16 → bfp16 행렬 곱셈 ukernel)으로 내고, f32와 비교한다.
  - AIE2P에는 f32 곱셈 하드웨어가 없다. `mac_elem_16(v16float, ...)`도 f32를 bf16 세 조각으로 나눠
    bf16 곱셈 9번(기본 `accuracy_safe`), 6번(`fast`), 3번(`low`)으로 흉내 낸 것이다.
    행렬 곱셈은 bfp16(8개 원소가 exponent 공유)뿐이라 이 방식으로 f32 정밀도를 낼 수 없다.
  - mlir-aie(`aie_kernels/aie2p/mm.cc`)와 aie_api에도 AIE2P용 f32 입력 matmul 커널이 없다. 직접 작성하면 정확도 검증과 유지보수 부담이 생긴다.

---

## dispatch별 시간 분석 (VGG-16)

### 측정 방법

- amdxdna HAL driver는 IREE device profiling을 구현하지 않았다(`device.cc`의 `profiling_begin`이 unimplemented).
- 대신 host에서 `strace -f -ttt -T -X raw -e trace=ioctl`로 `iree-run-module`을 실행하고, NPU queue thread의 ioctl을 dispatch 단위로 묶었다.
  - `CREATE_HWCTX`(nr 0x40) → BO 준비 → `EXEC_CMD`(0x46) → `SYNCOBJ_TIMELINE_WAIT` → `DESTROY_HWCTX`(0x41)
  - NPU 연산 시간 = `EXEC_CMD` 시작부터 wait 종료까지
  - NPU dispatch는 program 순서대로 실행되므로 k번째 명령을 k번째 NPU dispatch에 대응시켰다.
  - NPU 명령 사이의 빈 시간은 CPU 작업(im2col, maxpool, bias+ReLU, transpose 등)이다.
- 단계마다 3회 측정해 중앙값을 썼다. strace overhead는 wall 기준 약 1%다.
- 이후 같은 분해(dispatch별 NPU/CPU 시간, hwctx 생성/해제)는 `scripts/onnx/run.py --verbose`로 얻을 수 있다.
  console tracing을 켠 runtime(`build-trace/`)의 로그를 해석한다(`scripts/onnx/README.md`).

### 시간 구성

| 구성 (중앙값) | S0 | S1 | S3 |
|---|---|---|---|
| NPU 연산 | 72.30s | 6.65s | 0.124s |
| hwctx 생성 | 0.88s | 0.87s | 0.87s |
| hwctx 해제 | 0.89s | 0.88s | 0.93s |
| CPU 작업 (dispatch 사이) | 0.10s | 0.10s | 0.12s |
| BO 준비, 기타 | 0.14s | 0.12s | 0.07s |
| 그 외 (module load, 입출력) | 0.2s | 0.2s | 0.2s |
| **wall** | **74.5s** | **8.8s** | **2.29s** |

- hwctx 생성과 해제는 dispatch마다 약 80ms(68~81ms)씩, 단계와 상관없이 일정하다. 23 dispatch에서 약 1.8s다.
  runtime(HAL driver)이 dispatch마다 hardware context를 만들고 지운다. 컴파일러 변경으로는 줄지 않는다.
- 단독 matmul에서 보이던 약 109ms 고정 비용의 대부분도 이것이다.

### dispatch별 NPU 연산 시간

| dispatch | 층 | M x N x K | S0 (ms) | S1 (ms) | S3 (ms) | S0→S1 | S1→S3 | S3 GFLOP/s |
|---|---|---|---|---|---|---|---|---|
| d2 | conv1_1 | 50176x64x27 | 503.6 | 55.9 | 6.78 | 9.0x | 8.2x | 26 |
| d6 | conv1_2 | 50176x64x576 | 7882.3 | 543.0 | 7.67 | 14.5x | 70.8x | 483 |
| d10 | conv2_1 | 12544x128x576 | 4438.6 | 277.0 | 4.63 | 16.0x | 59.8x | 399 |
| d14 | conv2_2 | 12544x128x1152 | 7267.3 | 518.3 | 6.00 | 14.0x | 86.4x | 617 |
| d18 | conv3_1 | 3136x256x1152 | 4034.5 | 262.1 | 5.08 | 15.4x | 51.6x | 364 |
| d22 | conv3_2 | 3136x256x2304 | 7145.3 | 508.9 | 5.67 | 14.0x | 89.8x | 653 |
| d26 | conv3_3 | 3136x256x2304 | 7246.1 | 508.7 | 5.57 | 14.2x | 91.4x | 665 |
| d30 | conv4_1 | 784x512x2304 | 3292.2 | 260.8 | 6.36 | 12.6x | 41.0x | 291 |
| d34 | conv4_2 | 784x512x4608 | 6843.3 | 512.8 | 9.45 (수정 전 100.80) | 13.3x | 54.3x | 391 |
| d38 | conv4_3 | 784x512x4608 | 6409.9 | 512.4 | 9.34 (수정 전 100.77) | 12.5x | 54.8x | 396 |
| d42 | conv5_1 | 196x512x4608 | 1699.4 | 272.2 | 4.70 | 6.2x | 57.9x | 197 |
| d46 | conv5_2 | 196x512x4608 | 1834.7 | 271.6 | 3.81 | 6.8x | 71.4x | 243 |
| d50 | conv5_3 | 196x512x4608 | 1781.9 | 272.2 | 4.10 | 6.5x | 66.4x | 225 |
| d54 x8 | fc1 (N-split 8개, 각각) | 1x512x25088 | 1228.6 | 210.4 | 4.17~5.50 | 5.8x | 38~51x | 5~6 |
| d56 | fc2 | 1x4096x4096 | 1684.9 | 145.8 | 4.77 | 11.6x | 30.6x | 7.0 |
| d58 | fc3 | 1x1000x4096 | 411.3 | 39.8 | 3.21 | 10.3x | 12.4x | 2.6 |

- **S0 → S1**: dispatch마다 6~16배다. scalar 곱셈과 덧셈이 soft-float 호출에서 lane 하나짜리 vector unit 명령으로 바뀐 효과다.
- **S1 → S3**: 큰 conv는 40~90배다. scalar 연산(명령당 MAC 약 1개)이 SIMD 연산(명령당 MAC 512개)으로 바뀐 효과다.
  - conv1_1(K=27, 32로 패딩)과 fc3(N=1000)은 연산량이 작아 상대적으로 덜 빨라졌다.
  - fc는 M=1을 64로 패딩해서 계산하므로 GFLOP/s가 낮다(실제 유효 연산은 1/64).
- conv4_2/4_3은 처음 측정에서 100.8ms였고, 이 분석으로 찾은 DMA 문제를 고쳐 9.4ms가 되었다(2-A의 "conv4_2/4_3 DMA 문제").
- S3의 작은 차이(±1ms)는 측정 편차 수준이다. 수정 전후로 conv4_2/4_3 이외의 dispatch는 바뀌지 않았다.

### 남은 병목 (S3, wall 2.29s 기준)

| 항목 | 시간 | 비중 | 성격 |
|---|---|---|---|
| hwctx 생성/해제 | 1.80s | 79% | runtime. dispatch마다 context를 재사용하면 대부분 없앨 수 있다 |
| NPU 연산 (23 dispatch) | 0.12s | 5% | ukernel 연산 |
| CPU 작업, BO 준비, module load 등 | 0.37s | 16% | im2col, maxpool, bias+ReLU 등은 CPU에서 실행된다 |

### NPU power mode의 영향

같은 S3 VGG(bf16, ukernel all)를 NPU power mode만 바꿔 5회씩 실행했다. power mode는 host에서
`sudo xrt-smi configure --pmode <mode>`로 바꾼다(performance = amdxdna driver의 `high`).

| | default | performance |
|---|---|---|
| VGG 실행 (wall, 중앙값) | 2.256s | **0.439s** (5.1배) |
| hwctx 생성 / 해제 (23 dispatch 합) | 0.827s / 0.933s | 0.002s / 0.003s |
| NPU 연산 (23 dispatch 합) | 0.125s | 0.101s |
| 정확도 (corr) | 0.99997 | 0.99997 |

- dispatch 고정 비용(hwctx 생성/해제, dispatch마다 약 36~40ms씩)은 대부분 default mode의 power management 때문이다.
  performance mode에서는 거의 사라지고, NPU 연산도 약 20% 빨라진다.
- power mode 설정에는 root 권한(CAP_SYS_ADMIN)이 필요하다. 컨테이너 안에서 iree의 `--amdxdna_power_mode`로 바꾸면 errno 13으로 실패한다.
- 위 표들의 시간은 모두 default mode에서 측정했다. 측정값을 비교할 때는 power mode를 함께 기록한다.

### CPU only 기준선 (NPU 최적화 목표)

같은 VGG를 CPU에서만 f32로 실행한 시간이다. accuracy 기준값 모듈(`reference_device`)을 그대로 쓴다.
이 머신의 CPU는 Ryzen AI 9 HX 370이다(Zen 5, 24 thread, AVX-512).
두 모듈 모두 CPU 코드는 `--iree-llvmcpu-target-cpu=host`(= `znver5`)로 컴파일했다.
이전에는 NPU+CPU 모듈의 CPU 부분만 generic CPU(SSE2)였다. 그때 cpu dispatch 합은 0.067s, 실행 시간은 2.32s였다.

| | CPU only, f32 | NPU+CPU, bf16 ukernel all (default mode) |
|---|---|---|
| 컴파일 | 5.0~5.4s | 100.9s |
| 실행 (wall, module load 포함, 중앙값) | **0.34s** | 2.26s |
| 함수 호출 (`--verbose` total) | **0.096s** | 2.097s |
| dispatch 연산 | cpu 59개 0.085s | npu 23개 0.196s + cpu 69개 0.051s |
| host | 0.011s | 1.849s (hwctx 생성/해제 1.766s) |
| 정확도 (corr) | 기준값 | 0.99997, top-1 일치 |

- 현재 NPU+CPU는 CPU only보다 모든 항목에서 느리다. performance mode(0.439s)에서도 CPU only 0.34s보다 느리다.
- NPU 연산(0.196s, `--verbose`로 잰 dispatch별 submit + wait 합)만 따져도 CPU only의 전체 dispatch 시간(0.085s)보다 길다.
- **NPU 최적화의 목표는 이 CPU only 수치다.** 실행 0.34s, 함수 호출 0.096s를 기준으로 비교한다.
  hwctx 재사용(dispatch 고정 비용)이 첫 번째 대상이고, 그다음이 NPU 연산과 CPU-NPU 사이 dispatch다.

---

## 알려진 한계와 남은 작업

- **ukernel 기본 컴파일러는 peano이고, chess는 이 브랜치의 범위 밖이다.**
  `useChessForUKernel` 기본값을 `true` → `false`로 바꿔 `--iree-amd-aie-enable-chess-for-ukernel` 없이 peano로 ukernel을 컴파일한다.
  chess ukernel 경로(`--iree-amd-aie-enable-chess-for-ukernel=true`)는 그대로 있지만 `UK_INSTANCES`와 transpose_b를 받지 않고,
  이 브랜치에서 검증하지 않았다(Vitis가 없는 환경). CI(`run.py`)는 플래그를 명시하므로 영향이 없다.
- **ukernel이 지원하는 타입 조합**: npu4 peano matmul은 i8 x i8 → i32와 bf16 x bf16 → f32뿐이고, transpose_b는 bf16만 지원한다.
  템플릿이 없는 타입은 ukernel off 경로로 간다(`hasUKernelTemplate`, `65ece963`).
  목표는 하드웨어가 지원하는 모든 타입 조합이다. 템플릿을 추가하면 `hasUKernelTemplate` 표에도 추가한다(`AMDAIEUtils.cpp`, `matmul.cc`의 TODO).
  f32는 기준값 경로로 두고 f32 matmul ukernel은 만들지 않는다(위 "추가 작업: f32 e2e").
- **npu4 밖의 장치**(npu1)는 `hasUKernelTemplate` 표를 적용하지 않고 이전 동작을 유지한다.
  npu1의 transpose_b matmul은 KernelDispatch가 ukernel용으로 tile하지만 matcher는 ukernel로 보내지 않는다(검증 불가, TODO).
- **BD chain 분할 후 token 없는 push 수**: 분할된 앞 조각은 token 없이 push되어, channel별 대기 task 수가 queue depth로 제한되지 않는다.
  측정한 최댓값은 3이다(81920x64x576, f32 50176x64x576). 원래 코드에도 있던 한계라서(ukernel 4096x512x512에서 이미 4) backlog다.
- **4-level 배치의 memtile join**은 공유 lock pair를 유지해서 race가 잠재해 있다(producer chunk별 BD로 나누는 방안은 backlog).
- ukernel off에서 FC 64x4096x25088을 단독으로 실행하면 DMA cliff(ert state 6)가 난다. VGG는 N-split으로 피한다.
- **벡터화 on에서는 ukernel matcher가 받지 않는 bf16 matmul이 컴파일에 실패한다**(aievec bf16 `vector.contract` 미지원).
  해당: vector instruction 수가 홀수인 tile(odd-block), transpose_a, npu1의 transpose_b. 현재 VGG와 MLP는 해당 없음(전부 ukernel).
  위 "앞으로의 벡터화 방향"에 따라 ukernel 추가를 우선하고, scalar fallback은 backlog다.
- **dispatch 고정 비용**은 default power mode에서 VGG 실행의 약 80%다. performance mode로 대부분 없어지지만 root 권한이 필요하다.
  default mode에서 줄이려면 runtime에서 dispatch마다 hwctx를 만들고 지우지 않고 재사용해야 한다(backlog, 위 "NPU power mode의 영향").
- VGG의 elementwise 연산은 모두 CPU에서 실행된다. NPU로 옮기려면(dev_GEMV 통합 후 pool 등) 해당 ukernel이 필요하다.
