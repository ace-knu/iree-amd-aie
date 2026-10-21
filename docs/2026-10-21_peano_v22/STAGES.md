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

## dispatch별 시간 분석 (VGG-16)

### 측정 방법

- amdxdna HAL driver는 IREE device profiling을 구현하지 않았다(`device.cc`의 `profiling_begin`이 unimplemented).
- 대신 host에서 `strace -f -ttt -T -X raw -e trace=ioctl`로 `iree-run-module`을 실행하고, NPU queue thread의 ioctl을 dispatch 단위로 묶었다.
  - `CREATE_HWCTX`(nr 0x40) → BO 준비 → `EXEC_CMD`(0x46) → `SYNCOBJ_TIMELINE_WAIT` → `DESTROY_HWCTX`(0x41)
  - NPU 연산 시간 = `EXEC_CMD` 시작부터 wait 종료까지
  - NPU dispatch는 program 순서대로 실행되므로 k번째 명령을 k번째 NPU dispatch에 대응시켰다.
  - NPU 명령 사이의 빈 시간은 CPU 작업(im2col, maxpool, bias+ReLU, transpose 등)이다.
- 단계마다 3회 측정해 중앙값을 썼다. strace overhead는 wall 기준 약 1%다.

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

---

## 알려진 한계와 남은 작업

- **ukernel 기본 컴파일러는 peano이고, chess는 이 브랜치의 범위 밖이다.**
  `useChessForUKernel` 기본값을 `true` → `false`로 바꿔 `--iree-amd-aie-enable-chess-for-ukernel` 없이 peano로 ukernel을 컴파일한다.
  chess ukernel 경로(`--iree-amd-aie-enable-chess-for-ukernel=true`)는 그대로 있지만 `UK_INSTANCES`와 transpose_b를 받지 않고,
  이 브랜치에서 검증하지 않았다(Vitis가 없는 환경). CI(`run.py`)는 플래그를 명시하므로 영향이 없다.
- **ukernel이 지원하는 타입 조합**: npu4 peano는 i8 x i8 → i32와 bf16 x bf16 → f32뿐이고, transpose_b는 bf16만 지원한다.
  목표는 하드웨어가 지원하는 모든 타입 조합이다(`AMDAIELowerToUKernels.cpp`, `matmul.cc`의 TODO).
  일반 matmul matcher는 타입을 검사하지 않아서, 지원하지 않는 타입에 ukernel을 켜면 ukernel 컴파일이 실패한다(기존 동작).
- **4-level 배치의 memtile join**은 공유 lock pair를 유지해서 race가 잠재해 있다(producer chunk별 BD로 나누는 방안은 backlog).
- ukernel off에서 FC 64x4096x25088을 단독으로 실행하면 DMA cliff(ert state 6)가 난다. VGG는 N-split으로 피한다.
- **벡터화 on에서는 ukernel matcher가 받지 않는 bf16 matmul이 컴파일에 실패한다**(aievec bf16 `vector.contract` 미지원).
  해당: vector instruction 수가 홀수인 tile(odd-block), transpose_a, npu1의 transpose_b. 현재 VGG와 MLP는 해당 없음(전부 ukernel).
  위 "앞으로의 벡터화 방향"에 따라 ukernel 추가를 우선하고, scalar fallback은 backlog다.
- VGG의 elementwise 연산은 모두 CPU에서 실행된다. NPU로 옮기려면(dev_GEMV 통합 후 pool 등) 해당 ukernel이 필요하다.
