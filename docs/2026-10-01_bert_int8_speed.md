# BERT int8 속도 — 기존 모델 vs 정확도 개선 모델 (2026-10-01)

정확도 작업(인계 §4, work log 은 바깥 main 체크아웃의 `docs/2026-09-29_int8_accuracy_work_log.md`) 전후 두 모델의
NPU 실행 속도를 비교하고, 차이가 어디서 나는지 측정으로 추적했다. 이 문서에는 **측정한 것만**
적는다.

---

## 0. 요약

| | 기존 | 개선 |
|---|---:|---:|
| 정확도 (fp32 대비 corr, NPU 64 문장) | 0.6559 | 0.9448 |
| 1 회 추론 | **400 ms** | **386 ms** (−3.5 %) |
| 상수 버퍼 invalidate 생략 (실험) | **328 ms** | **332 ms** |

- 두 모델의 차이 14 ms 는 NPU 가 아니라 **호스트 쪽**에서 났다. NPU 실행 시간은 개선 모델이 4 ms 길다.
- 호스트 시간은 디스패치가 끝날 때마다 드라이버가 바인딩 전체를 `clflush` 로 invalidate 하는 데
  쓰인다. 90 MB 상수(가중치) 버퍼를 바인딩한 NPU 디스패치가 기존 95 개, 개선 72 개다.
- 실험으로 상수 버퍼(`SHARING_IMMUTABLE`)의 invalidate 만 건너뛰자 두 모델 모두 14~18 % 빨라졌고,
  출력은 두 모델 × 64 문장 모두 바이트 동일했다.

---

## 1. 대상과 방법

| | 기존 | 개선 |
|---|---|---|
| 모델 | `_local/wj_models/v3/bertx_int8.onnx` (인계 시점) | SmoothQuant + ①②③ + LN ukernel f32 + FC2 per-channel (work log §7.5) |
| 컴파일러 | `1549c7f6` (인계 시점, 2026-09-29 컴파일) | `30bd07b0` (`bert-int8-accuracy`) |
| vmfb | 컨테이너 `/tmp/bert_v3.vmfb`, 204,537,343 B | 컨테이너 `/tmp/full_hy.vmfb`, 212,345,391 B |

- 측정 도구: `iree-benchmark-module --benchmark_repetitions=5` (반복당 2 회 실행), 입력은 평가 문장
  0 번(`/tmp/sents64/in_00.npy`) 하나.
- 런타임은 `_local/wj/build` (`CMAKE_BUILD_TYPE=Release`). 출력의 `Library was built as DEBUG`
  경고는 Google Benchmark 라이브러리 자체에 대한 것이다.
- 장치: `--device=amdxdna --device=local-task`. 모든 NPU 실행은 `with-npu-lock.sh` 안에서, 측정마다
  락을 따로 몇 초씩만 잡았다.

---

## 2. 결과 1 — 기본 속도

| | 평균 | 중앙값 | stddev | cv | 호스트 CPU 시간 |
|---|---:|---:|---:|---:|---:|
| 기존 | 400 ms | 400 ms | 2.02 ms | 0.50 % | 85.2 ms |
| 개선 | 386 ms | 386 ms | 1.33 ms | 0.34 % | 67.6 ms |

차이 14 ms 는 stddev 의 7~10 배다. 실험 빌드로 다시 잰 기본 모드(§7)도 399 / 387 ms 로 같았다.

---

## 3. 두 모듈의 구조는 같다

`iree-dump-module` 로 VM 코드를 비교했다.

| | 기존 | 개선 |
|---|---:|---:|
| `command_buffer.dispatch` | 136 | 136 |
| — NPU / CPU | 131 / 5 | 131 / 5 |
| — NPU 종류 (`batch_matmul` / `reduction` / `softmax`) | 96 / 23 / 12 | 96 / 23 / 12 |
| `execution_barrier` | 117 | 117 |
| `fill_buffer` | 47 | 47 |
| `copy_buffer` | 3 | 3 |
| command buffer | 5 | 5 |

dispatch · fill · barrier 의 **순서까지 동일**하다. NPU 종류는 실행 파일 안의 export 이름
(`main_graphasync_dispatch_<n>_<종류>_<ordinal>`) 에서 셌다.

---

## 4. 결과 2 — NPU 시간과 호스트 시간 나누기

amdxdna 드라이버는 기본 설정에서 NPU 디스패치마다 제출 후 완료를 기다린다(`submit_and_wait`).
그래서 `strace -f -T -e trace=ioctl` 로 대기 ioctl(`DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`) 시간을 재면
디스패치별 NPU 시간이 나온다. 대기 ioctl 은 393 번 = NPU 131 개 × 3 회(워밍업 1 + 측정 2) 였다.
(strace 아래 실행은 420 / 409 ms 로, 차이가 그대로 보인다.)

| (측정 2 회 평균) | 기존 | 개선 | 차이 |
|---|---:|---:|---:|
| NPU 대기 합 (131 개) | 278.6 ms | 282.5 ms | **+4.0 ms** |
| 디스패치 사이 호스트 시간 합 | 139.9 ms | 125.1 ms | **−14.8 ms** |

### 4.1 NPU 쪽

한 층의 NPU 디스패치 11 개 (0 층, ms):

| 층 내 순서 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 종류 | bmm | bmm | bmm | bmm | softmax | bmm | bmm | **reduction** | bmm | bmm | **reduction** |
| 기존 | 2.09 | 2.08 | 2.08 | 0.43 | 0.30 | 1.99 | 1.99 | 0.40 | 9.49 | 2.07 | 0.40 |
| 개선 | 2.08 | 2.08 | 2.07 | 0.42 | 0.30 | 1.99 | 2.00 | 0.37 | 9.78 | 2.15 | 0.38 |

늘어난 시간은 거의 전부 층 내 8 번 `batch_matmul` 이다(층마다 9.30~9.51 → 9.62~9.82 ms,
+0.29~0.32 ms × 12 층).

### 4.2 호스트 쪽

호스트 시간이 줄어든 곳은 층마다 **두 `reduction` 디스패치 직후**뿐이다.

| 다음 디스패치까지의 틈 | 기존 | 개선 |
|---|---:|---:|
| 층 내 7 번 `reduction` 뒤 | 1.34~1.39 ms | 0.50~0.59 ms |
| 층 내 10 번 `reduction` 뒤 | 1.26~1.28 ms | 0.48~0.51 ms |
| 전체 틈 중앙값 | 1.266 ms | 1.243 ms |

이 틈 안의 ioctl 은 두 모델이 종류·순서·개수까지 같다. 줄어든 것은 대기 ioctl 이 끝난 직후
**시스템 콜이 하나도 없는 구간**이다(예: 0 층 7 번 뒤 1.166 ms → 0.397 ms).

---

## 5. 그 구간에서 하는 일 — 바인딩 invalidate

드라이버는 디스패치가 끝나면 **모든 바인딩**을 host 쪽으로 sync 한다:

- `runtime/src/iree-amd-aie/driver/amdxdna/direct_command_buffer.cc` 의 `normal_run`,
  `// Sync the bindings back to the host.` 루프 → 바인딩마다 `iree_hal_amdxdna_buffer_invalidate_range`
- 이것이 `shim/linux/kmq/bo.cpp` 의 `bo::sync` 로 가서, `AMDXDNA_BO_SHMEM` / `AMDXDNA_BO_CMD` 는
  사용자 공간 `clflush_data`(64 B 마다 `_mm_clflush`) 를 돈다. 시스템 콜이 없다.

### 5.1 바인딩 길이

`iree-run-module --trace_execution` 으로 디스패치마다 바인딩 (버퍼, 오프셋, 길이) 를 뽑았다.

| 같은 디스패치 (0 층 7 번 `reduction`) | 기존 | 개선 |
|---|---|---|
| 바인딩 | 중간 버퍼 + **상수 버퍼 90,313,728 B** | 중간 버퍼만 |
| 바인딩 길이 합 | 90,747,904 B | 561,152 B |

| 90 MB 상수 버퍼를 바인딩하는 NPU 디스패치 | 기존 | 개선 |
|---|---:|---:|
| `batch_matmul` | 72 | 72 |
| `reduction` | 23 | 0 |
| **합** | **95** | **72** |

### 5.2 상수 버퍼의 표시

| 바인딩 | 개수 (기존 / 개선) | usage |
|---|---:|---|
| 상수 버퍼 | 95 / 72 | `0x9080c03` — `SHARING_IMMUTABLE` (bit 19) 포함 |
| 그 밖의 바인딩 전부 | 263 / 263 | `0x9000c03` — `SHARING_IMMUTABLE` 없음 |

두 버퍼 모두 allowed access 는 `0x7`(쓰기 포함)이어서, 접근 권한으로는 구별되지 않는다.

### 5.3 상수 버퍼의 내용

기존 ONNX 모델의 int8 initializer 를 합산했다.

| initializer | 개수 | 크기 |
|---|---:|---:|
| `kfold_w_*` — (832, 768), (832, 3072), (3136, 768) | 36 | 67.24 MB |
| `hsplit*_w`, `ksplit*_w` — (12, 832, 64) | 36 | 23.01 MB |
| 합 | | **90.25 MB** |

실행 중 바인딩 길이 90.31 MB 와 0.06 MB 차이다. 행 수가 768 / 3072 가 아니라 832 / 3136 인 것은
bias 를 K 축에 접은(`fold_bias_into_k*`) 결과다.

### 5.4 clflush 비용

같은 호스트에서 90,313,728 B 버퍼를 `_mm_clflush` 로 비우는 시간을 따로 쟀다(코어 1 개).

| | 시간 |
|---|---:|
| 90,313,728 B (3 회) | 0.736 ~ 0.752 ms |
| 0.5 MB | 0.012 ms |

---

## 6. 정리 — 3.5 % 차이의 출처

| | 값 |
|---|---:|
| 개선 모델에서 사라진 상수 바인딩 (`reduction` 23 개) | 23 × ~0.75 ms ≈ 17 ms |
| 개선 모델에서 늘어난 NPU 시간 (층 내 8 번 `batch_matmul`) | +4 ms |
| 측정된 차이 | −14 ms |

---

## 7. 실험 — 상수 버퍼 invalidate 생략

먼저 `direct_command_buffer.cc` 의 위 루프에 환경변수 `AMDXDNA_EXP_RO` 를 넣어 시험했다(이 실험 코드는 커밋하지
않았고, §7.3 의 형태로 바꿔 커밋했다). 변수가 없으면 원래 동작과 같다.

- `=1`: 바인딩마다 usage 와 길이를 stderr 로 출력 (§5.2 의 표가 이 출력)
- `=2`: usage 에 `SHARING_IMMUTABLE` 이 있는 바인딩은 invalidate 를 건너뜀

### 7.1 출력

`AMDXDNA_EXP_RO=0` 과 `=2` 로 평가 문장 64 개(`quant/eval_ids.npy`)를 돌려 출력을 바이트 비교하고,
fp32 모델(`_local/int8_debug/bert_base_L12.onnx`) 대비 corr 을 계산했다(실제 토큰 위치만).

| 모델 | 바이트 비교 | corr 기본 | corr 생략 |
|---|---:|---:|---:|
| 기존 | **64/64 동일** | 0.6559 (최저 0.5906) | 0.6559 (최저 0.5906) |
| 개선 | **64/64 동일** | 0.9448 (최저 0.9315) | 0.9448 (최저 0.9315) |

- 개선 모델의 기본 모드 출력은 실험 수정 전에 0.9448 을 잰 NPU 출력과도 64/64 바이트 동일하다.
- 기존 모델의 앞 16 문장 corr 은 0.6633 으로, 인계 시점 NPU 16 문장 값과 같다.

### 7.2 속도

| | 기본 (`=0`) | 생략 (`=2`) | 차이 | 호스트 CPU 시간 |
|---|---:|---:|---:|---|
| 기존 | 399 ms (cv 0.16 %) | **328 ms** (cv 0.19 %) | −71 ms (−17.8 %) | 85.0 → 13.8 ms |
| 개선 | 387 ms (cv 0.46 %) | **332 ms** (cv 0.21 %) | −55 ms (−14.2 %) | 67.7 → 13.5 ms |

생략한 상수 바인딩 수 × 90 MB clflush 시간 (§5.4) 과 맞는다: 95 × 0.74 ≈ 70 ms, 72 × 0.74 ≈ 53 ms.
생략 모드에서는 두 모델의 차이가 개선 모델 +4 ms 로, §4.1 의 NPU 시간 차이와 같다.

### 7.3 기본 동작으로 바꿔 커밋

IREE HAL 은 `IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE` 을 "초기화 뒤 바뀌지 않는 버퍼이고, 구현은 바뀔 수 있는 버퍼에
필요한 동기화 · 전송을 생략해도 된다" 고 정의한다(`iree/hal/buffer.h`). 그래서 환경변수 없이, 이 표시가 있는 바인딩은
항상 invalidate 를 건너뛰도록 바꿨다. `iree_hal_amdxdna_invalidate_binding` 하나로 두 호출 지점에 같은 규칙을 쓴다.

| 호출 지점 | 확인 |
|---|---|
| 기본 경로 (`normal_run`) | 개선 모델 64 문장 출력이 실험 코드를 넣기 전 출력(`/tmp/sents64_hy`)과 **64/64 바이트 동일** |
| cmd_chain 경로 (`--amdxdna_cmd_chain=1`, `flush_chains`) | 실행 못 함. 수정 전 코드에서도 `hardware context creation failed: errno 22` 로 같은 자리에서 실패한다 |

커밋 상태로 다시 잰 속도는 기존 334 ms (cv 0.55 %), 개선 341 ms (cv 0.95 %) 였다. 이때 다른 사용자의
`iree-compile` 이 CPU 394 % 를 쓰고 있었다(load average 2.94). §7.2 의 328 / 332 ms 는 이런 부하가 없을 때 쟀다.

---

## 8. 확인 범위와 현재 상태

- 속도는 입력 1 개(문장 0 번, 32 토큰), 바이트 동일은 평가 문장 64 개로 확인했다. 다른 모델은 보지
  않았다.
- 런타임 수정(§7.3)은 `_local/wj` 의 로컬 브랜치 `bert-int8-accuracy` 에 이 문서와 함께 커밋했다. 푸시는 하지 않았다.
  `iree-run-module` / `iree-benchmark-module` 이 그 상태로 빌드돼 있다.
- `iree-benchmark-module` 은 이번에 `_local/wj/build` 에 새로 빌드했다(원래 없었다).
