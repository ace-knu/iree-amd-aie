# S12. Vulkan: semaphore의 호스트 쪽 값은 앞으로만 올린다 (런타임 수정)

- 구분: 런타임 드라이버 수정(IREE 서브모듈 Vulkan HAL). 수정 대상은 업스트림 코드(#23695 `native_semaphore.cc`, `completion_watcher.cc`)다.
  - 브랜치 `fix/vulkan-semaphore-monotonic`(통합 브랜치 `integrate/vulkan-order-link-true` 기준, 2026-11-06). 커밋: fork `d79945b`, 상위 포인터 `fdeae865`.
- 파일(`third_party/iree/runtime/src/iree/hal/drivers/vulkan/`): `native_semaphore.cc/.h`, `completion_watcher.cc`.
- 이유 문서(로컬 세션 기록, 팀 git 미포함): `docs/2026-11-05/2_Vulkan_alloca_대기/`(챕터 01 원인 확정, 챕터 02 설계, 챕터 03 구현·검증).
- 해결하는 문제: dev CPU+GPU에서 드물게(GoogLeNet 자연 반복 46,500회 중 18회, 0.02–0.08%) 실행이 끝나지 않는다.
  - 메인 스레드가 Vulkan `queue_alloca` 호스트 대기, BRIDGE의 남의 semaphore 대기, VM `hal.fence.await` 중 한 곳에서 영원히 기다린다.
  - 이것은 CLAUDE.md 6절의 "P2 멈춤"이다.

## 0. 규칙
- **R1 관찰**: Vulkan에서 읽은 값(semaphore 조회, 호스트 대기 뒤 반영, 완료 감시)은 호스트 쪽 값(`iree_async_semaphore_t::timeline_value`)을 **더 클 때만** 올린다(CAS). 올린 스레드가 그 값으로 대기자를 깨운다. 같거나 작은 값은 아무것도 하지 않는다.
- **R2 호스트 신호**: 호스트 신호는 호스트 쪽 값을 먼저 올리고(엄격 CAS, 값마다 한 번), 그다음 `vkSignalSemaphore`, 마지막에 깨운다.
- 이 규칙이 필요한 이유
  - 대기자 등록(`iree_async_semaphore_acquire_timepoint`)과 깨우기(`iree_async_semaphore_dispatch_timepoints`)는 "값을 올린 쪽이 한 번 깨우고, 등록 쪽은 넣은 뒤 값을 다시 본다"로 엇갈림을 막는다.
  - 이 방식은 값이 줄지 않을 때만 맞다.
  - 수정 전에는 오래된 작은 값으로 덮어써 값이 줄었다(1 → 0). 그 직후 등록한 대기자는 다시 깨워 줄 쪽이 없었다.
- 업스트림 main의 새 Vulkan 드라이버와 같은 규칙이다(`semaphore.c` `iree_hal_vulkan_semaphore_observe_native_value`·`iree_hal_vulkan_semaphore_signal`, 커밋 `9e2ac9142e`). HIP(`event_semaphore.c` 603-613, 729-740, 774)도 같다.

## 1. 흐름상 위치
런타임 Vulkan HAL에서 호스트 쪽 값을 쓰는 네 곳이다.

| # | 위치 | 누가 부르나 | 이전 동작(#23695) | 이후 |
|---|---|---|---|---|
| ① | `completion_watcher.cc` scan 3단계 | 감시 스레드(GPU 완료를 관찰) | 무조건 덮어쓰기 + 깨우기 | R1 |
| ③ | `native_semaphore.cc` query | 아무 스레드(예: CPU 큐가 작업을 넣으며 대기 semaphore 조회, VM fence 조회) | 같음 | R1, 반환값 = 반영 뒤 호스트 쪽 값 |
| ④ | `native_semaphore.cc` wait(`vkWaitSemaphores` 뒤) | 호스트 대기 | 같음 | R1 |
| ② | `native_semaphore.cc` signal | 호스트 신호(예: Vulkan `queue_alloca`가 할당 뒤 신호) | `vkSignalSemaphore` → 엄격 CAS | R2(순서 바꿈) |

세 관찰 자리(①③④)는 새 함수 `iree_hal_vulkan_native_semaphore_observe_value(semaphore, value)` 하나를 쓴다.

## 2. 입력 → 동작 → 출력

### 2-1. `iree_hal_vulkan_native_semaphore_observe_value(semaphore, value)`
- 입력: Vulkan semaphore(그 안의 async semaphore)와 Vulkan에서 읽은 값. 실패 값(`>= IREE_HAL_SEMAPHORE_FAILURE_VALUE`)은 호출자가 미리 거른다.
- 동작: `timeline_value`를 읽어 `value`가 더 크면 CAS로 바꾼다(경합이면 다시). 바꿨으면 `dispatch_timepoints(value)`.
- 출력: 반영 뒤 호스트 쪽 값(`max(이전 값, value)`).

### 2-2. 조회(query)
실패 확인 → `vkGetSemaphoreCounterValue` → 실패 값 처리(그대로) → S11의 `release_before_visible`(그대로) → 2-1 → 그 반환값을 돌려준다.

### 2-3. 대기(wait)
`vkWaitSemaphores` → S11의 `release_completed`(그대로) → 실패 확인 → 값 조회 → 2-1.

### 2-4. 완료 감시 scan
1단계(값 관찰)·2단계(끝난 제출 해제, S11)는 그대로다. 3단계에서 완료된 항목마다 2-1을 부르고 항목을 놓는다.

### 2-5. 호스트 신호(signal)
`iree_async_semaphore_advance_timeline`(엄격 CAS) → 실패면 그 상태를 돌려준다(Vulkan은 건드리지 않음) → `vkSignalSemaphore` → 실패면 semaphore를 실패 처리하고 돌려준다 → `dispatch_timepoints`.

## 3. 지원 범위와 지원하지 않는 경우
| 경우 | 동작 |
|---|---|
| 모든 Vulkan semaphore(GPU 단독, CPU+GPU, CPU+NPU+GPU, lead CPU·GPU, 링크 표 true/false) | 위 규칙 |
| 같은 값을 여러 스레드가 관찰 | 처음 올린 스레드만 깨움, 나머지는 아무것도 안 함 |
| 이미 지나간 작은 값을 관찰(늦게 돌아온 조회 등) | 무시 |
| 같은 값을 두 번 호스트 신호, 또는 GPU가 신호한 값을 호스트가 다시 신호(프로그램 오류) | `INVALID_ARGUMENT`를 돌려줌, Vulkan 값은 그대로(수정 전: Vulkan을 올리고 semaphore를 실패 처리한 뒤 OK) |
| `vkSignalSemaphore` 실패(장치 손실) | semaphore 실패 처리, 오류 반환. 호스트 쪽 값은 이미 올라가 있음(업스트림과 같음) |
| 실패한 semaphore | 수정 전과 같음(실패 값은 호스트 쪽 값에 쓰지 않음) |
| 다른 HAL(local-task, HIP, CUDA, amdxdna, Metal) | 바꾸지 않음 |

## 4. 기존 대비 변경점
| | 업스트림 #23695(수정 전) | 이후 |
|---|---|---|
| 관찰 ①③④ | `iree_atomic_store` 무조건 + 매번 깨움 | CAS 앞으로만 + 올렸을 때만 깨움 |
| 조회 반환값 | 방금 읽은 Vulkan 값 | 반영 뒤 호스트 쪽 값(둘 중 큰 값) |
| 호스트 신호 순서 | Vulkan → 호스트 | 호스트 → Vulkan |
| 비용 | 쓰기 1 + 잠금 1 | 읽기 1 + CAS 1(+ 올렸을 때만 잠금 1) |
| 공용 코드(`iree/async/`) | — | 바꾸지 않음 |

- **업스트림 새 코드로 바뀔 때**: 업스트림 main은 Vulkan HAL을 다시 쓰면서 `native_semaphore.cc`·`completion_watcher.cc`를 없앴다(`97815d1`, 2026-05-15).
  - 새 드라이버에는 R1·R2가 이미 있다(`9e2ac9142e`). 새 드라이버도 처음 판 `9c6ab501c9`에는 같은 덮어쓰기와 신호 순서가 있었고, 다음 커밋에서 고쳤다.
  - 그래서 들여오면 이 수정은 파일과 함께 사라지되 **같은 동작이 유지된다. 다시 넣을 것이 없다**(S11과 다름).
  - 들여온 뒤에는 6절의 조회 지연 진단만 다시 돌려 확인한다.

## 5. 일반성 점검
| 항목 | 의존 | 이유·상태 |
|---|---|---|
| 규칙과 적용 자리 | 장치·모델·원소 타입·장치 순서 의존 없음 | 호스트 쪽 값을 쓰는 모든 자리 |
| 멈춤이 드러나는 조건 | 다른 장치 큐가 Vulkan semaphore를 조회하는 다중 장치 + 조회가 GPU 완료와 겹침(조회 호출이 0.16–3.2 ms 늦게 돌아오거나 조회 뒤 스레드가 밀려남) | 수정은 조건과 무관 |
| 잠금 | 새 잠금 없음 | 깨우기는 수정 전과 같은 함수(semaphore mutex 안에서 떼고 밖에서 콜백) |

## 6. 검증 근거 (2026-11-06, `docs/2026-11-05/2_Vulkan_alloca_대기/03_뿌리수정_구현검증/`, 전원 모드 default)
- 수정 전(챕터 01): GoogLeNet CPU+GPU 등 자연 반복 46,500회에서 멈춤 18(Vulkan 대기 13 / BRIDGE 1 / VM fence 4)이 나왔고, 모두 역행과 1:1이었다. 조회 읽기→반영 사이 2 ms 지연에서는 19/19 멈췄다.
- 수정 뒤 결과는 아래 표와 같다. V1–V3은 수정 + 검증용 계측 빌드이고, V4–V5는 계측을 뺀 수정 빌드다.

| # | 항목 | 결과 |
|---|---|---|
| V1 | 조회 지연 2 ms, GoogLeNet 링크 true lead CPU 300 + lead GPU 300 | **600/600 통과**(멈춤 0, 역행 0). "Vulkan에서 읽은 값이 이미 지난 값"(수정 전이라면 역행)이 600회차 모두에서 일어났고 무시됨(4,024 / 815회) |
| V2 | 호스트 신호의 두 단계 사이 2 ms 지연, lead CPU 300 + lead GPU 300 | 600/600 통과, 신호 실패 0. 그 틈에 관찰이 겹친 일은 없었음(해 없음까지만 확인) |
| V3 | 자연 반복 18,000회(링크 true lead GPU 6,000·lead CPU 6,000, 링크 false lead CPU 3,000, 최소 재현 링크 false 3,000) | **멈춤 0**, 역행 0(수정 전 빈도로 기대 7.5건, 0건 확률 약 0.06%). 수정 전이라면 역행했을 자연 경합 4번 모두 무시됨 |
| V4 | 회귀 v10 × 2 | 수동 28/28·자동 28/28, 2회 |
| V4 | 12조합(MLP·VGG16 × CPU / CPU+GPU / CPU+NPU bf16·f32 / CPU+NPU+GPU / CPU+GPU+NPU) | 출력 sha256 12/12가 dev 기록과 같음 |
| V4 | ctest `iree/target/amd-aie/`, `iree-amd-aie/`, SPIR-V, `iree/hal/drivers/vulkan` | 309/309 |
| V4 | dispatch별 장치 배정(12조합), ctest 전체 실패·미실행 목록 | 12/12 dev와 같음, 목록 dev와 같음(1,303개, diff 0) |
| V4 | 자연 반복 lead GPU 3,000(계측 없는 빌드) | 멈춤 0 |
| V5 | 성능(한 프로세스 20회 × 5회차, 수정 전과 번갈아) | 수정/수정 전 0.97–1.03(GoogLeNet lead CPU·GPU, VGG16, p1b 두 방향), 흔들림 범위 안 |

- 검증용 계측의 역행 검출기는 처음 판에 비교 순서 오탐이 있었다. 호스트 쪽 값을 먼저 읽고 기록된 최대값을 나중에 읽었기 때문이다(자연 2,834회차에 한 번). 2판에서 최대값을 먼저 읽도록 고친 뒤 V3을 처음부터 다시 돌렸다. 제품 코드와 무관하다.
- 업스트림 새 드라이버를 들여온 뒤에는 V1(조회 지연 진단)만 다시 돌려 확인한다.

## 7. 알려진 한계
| # | 한계 | 확인 | 영향 | 방향 |
|---|---|---|---|---|
| L-time | 이 수정은 멈춤만 없앤다. 평소의 Vulkan `queue_alloca` 호스트 대기 시간(GoogLeNet lead CPU 48–61 ms)은 그대로다 | 챕터 01·02 계측 | 성능 | Feature 2(대기 제거), 비차단 BRIDGE(별도) |
| L-slowq | `vkGetSemaphoreCounterValue`가 GPU 완료 순간 0.16–3.2 ms 늦게 돌아오는 일이 있다(경합의 틈). 원인(커널·드라이버) 미조사 | 챕터 01 5-6 | 수정 뒤에는 무해(늦은 값은 무시됨) | 별도 후보 |
| L-metal | 업스트림 Metal(`shared_event.m:118`)에도 같은 덮어쓰기가 있다 | 코드 읽기 | 이 장비와 무관 | 기록만 |
