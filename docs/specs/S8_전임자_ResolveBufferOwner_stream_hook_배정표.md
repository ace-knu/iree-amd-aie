# S8. 전임자: `AMDAIEResolveBufferOwner` + stream hook + 역할·링크 표 + `accelIdx` (신규 Pass·IREE 수정, **우리는 검증만**)

- 구분: **전임자(최민규) 코드, 9/28 인수인계 기준선**. 우리가 바꾼 것은 `linkFlags` 주석 한 곳(O17, 10/2 세션1 챕터00)뿐이다. 나머지는 검증만 했다.
- 파일
  - 플러그인: `Transforms/AMDAIEResolveBufferOwner.cpp`(N02, 9/28 기준선과 바이트 동일), 등록 `PluginRegistration.cpp`(Pp01), `Passes.td`·`Passes.h`·`PassDetail.h`·`CMakeLists.txt`(Pp03~07), 역할·링크 표 `Transforms/Utils/AMDAIEDevicePlacementUtils.cpp`(Pp08 + O17), `AMDAIEAssignDeviceAffinities.cpp` `accelIdx = 0`(Pp02)
  - IREE 서브모듈: stream hook API `compiler/src/iree/compiler/PluginAPI/Client.h`(Pi02), `PluginManager.h`(Pi03), 호출 `Pipelines/Pipelines.cpp:416-418`(Pi01)
- 이유·검증 문서: `docs/2026-09-28/01_인수인계_코드현황_파악/README.md` 3-2, `03_소유자결정_topology_검증보강/README.md`, `04_transparent_access_방향정의_확인/README.md`, `05_경계버퍼_전케이스_정리/README.md`, `docs/2026-10-19/2_발표피드백_조사/01_…`(topology 피드백 조사)
- 1-3 기록: "10/19 형 확인" — 요구사항은 "topology 표를 보고 소유자를 고른다"이고 지금 구조가 충족한다. NPU import는 필요할 때 한다.

## 1. 흐름상 위치

```mermaid
flowchart LR
  T["역할·링크 표<br/>deviceRole / linkFlags<br/>(DevicePlacementUtils)"] --> A["preprocessing hook: AssignDeviceAffinities 1회차<br/>stream.topology 주입"]
  A --> SA["Stream: ScheduleAllocation<br/>경계 버퍼 = optimal<[사용 장치들]>"]
  SA --> H["IREE Stream 단계 끝<br/>pipelineExtensions->extendStreamTransformPassPipeline (신설 hook)"]
  H --> RB["ResolveBufferOwner<br/>optimal → 자격자 1명이면 확정"] --> AI["AnnotateInputDevices (S5)"]
  RB --> RT["HAL: ResolveTopologyQueries<br/>(남은 optimal은 런타임 alloc_select)"]
```

- **stream hook**: IREE에 없던 플러그인 확장점이다. `Client.h`에 `extendStreamTransformPassPipeline` 가상 함수를 추가했고, `PluginManager.h`는 모든 세션에 이를 전달하며, `Pipelines.cpp`는 `buildStreamTransformPassPipeline` 직후·`afterPhase(Stream)` 전에 호출한다.
- **ResolveBufferOwner**: 그 hook의 첫 패스다. `ScheduleAllocation`이 경계 버퍼에 `#hal.device.optimal<[...]>`(실제 사용 장치 집합, `ScheduleAllocation.cpp` `joinOR(usageAffinities)`)을 붙인 뒤에 돈다.

## 2. 입력 → 동작 → 출력

### 2-1. `AMDAIEResolveBufferOwner` (`AMDAIEResolveBufferOwner.cpp`, 53줄)
- **입력**: `stream.topology`, `AffinityOpInterface` op(alloca·dealloca 등) 중 affinity가 `#hal.device.optimal<[X1..Xn]>`인 것
- **동작**: 후보 X마다 "모든 후보 Y가 `(Y → X) transparent_access`"인지 본다(같은 장치는 true, `HALAttrs.cpp:1129-1131`). 자격자가 **정확히 1명**이면 그 장치로 affinity를 바꾼다.
- **출력**: 확정된 단일 affinity. 0명이나 2명 이상이면 `optimal` 그대로 → 런타임 `hal.allocator.select`가 lead(`--device_lead_allocator`, 기본 0)로 고른다.
- topology가 없으면 무동작.

### 2-2. 역할·링크 표 (`AMDAIEDevicePlacementUtils.cpp`)
- `deviceRole(backend)`: `llvm-cpu` → Host, `amd-aie` → Accelerator, `vulkan-spirv` → Accelerator(10/20 시점에는 `// TEMP: experiment` 표식. dev 통합 때 표식을 떼고 정식 역할로 둠, `e7060e40`), 그 외 Unknown.
- `linkFlags(src, dst)` = `(src → dst)`의 `{unified_memory, transparent_access}`. "src가 dst 메모리를 복사 없이 읽는다".

| src → dst | transparent | 근거(주석) |
|---|---|---|
| CPU → NPU | true | amdxdna BO는 host-local·device-visible, CPU가 host-map |
| NPU → CPU | **false** | amdxdna HAL allocator에 `import_buffer` 없음(드라이버 소프트웨어 한계, 커널 드라이버는 dma-buf·userptr 지원) |
| CPU → GPU | true | Vulkan dispatch 메모리가 DEVICE_LOCAL\|HOST_VISIBLE\|HOST_COHERENT(이 플랫폼). 외장 GPU면 false여야 함 |
| GPU → CPU | true | host 포인터 import + BRIDGE(S3). (O17: 옛 TODO를 "검증됨 cg_topo lead=0"으로 교체) → 통합 후: **false**. BRIDGE가 기록 시점에 import해서 아직 메모리가 없는 transient CPU 버퍼를 못 가져오므로, CPU·GPU 공유 버퍼는 GPU가 소유(`e3d86485`) |
| GPU → NPU | true | amdxdna dma-buf export → Vulkan import(S9) |
| NPU → GPU | **false** | amdxdna import 없음, Vulkan export 없음 |
| 그 외 | false·false | 보수적 기본값(복사) |

### 2-3. `accelIdx = 0` (`AMDAIEAssignDeviceAffinities.cpp`)
- 라운드로빈 시작 가속기를 HEAD의 1에서 0으로 바꿨다(9/7 mtime). 첫 contraction → 선언 순서 첫 가속기.

## 3. 지원 범위와 지원하지 않는 경우

| 경우 | 동작 |
|---|---|
| 자격자 1명 | 확정. 지금 표에서는 NPU→CPU·NPU→GPU 칸이 false라, NPU가 낀 경계는 NPU로 정해진다 |
| 자격자 2명(CPU↔GPU 서로 true) | `optimal` 유지 → 런타임 lead가 고른다(lead = 선언 순서 의존. 1-3 "10/19 형 확인" 질문 3 **미확인**) |
| 자격자 0명 | `optimal` 유지 → 런타임이 고르고, 읽을 수 없는 장치가 생기면 실행 시 실패할 수 있다(지금 표에서는 생기지 않음) |
| 3장치 이상 | 같은 규칙(∀Y (Y→X)). 업스트림 `ResolveTopologyQueries`는 ∃X ∀Y (X→Y)로 양화가 달라 3장치 이상에서 다를 수 있다. 지금은 이 패스가 먼저 확정해서 가려져 있다(CLAUDE.md 3-2) |
| 표가 바뀜(예: NPU import 구현) | 표 한 칸만 바꾸면 소유자가 자동으로 바뀌는 구조(9/28 발표 남은 과제 3) |

## 4. 추가 목적 / 기존 대비 변경점

- **추가 목적**: 런타임의 `"amdxdna"` 문자열 규칙(후보에 NPU가 있으면 NPU)을 없애고, 컴파일러가 topology 표로 소유자를 고르게 한다(9/21 피드백 ① "하드코딩 제거, topology 기반 할당"). 런타임 문자열 규칙은 9/28 기준선에서 이미 지워져 있었다(9/28 01 3-2 참고).
- **기존 대비**: HEAD에는 hook·패스가 없다. 그래서 `optimal`이 모두 런타임 lead/문자열 규칙으로 갔다.

## 5. 일반성 점검

- 패스 본문은 장치 이름을 보지 않는다(topology만).
- 표는 **백엔드 이름 문자열**(`llvm-cpu`, `amd-aie`, `vulkan-spirv`)을 키로 쓴다. "표가 유일한 진실 원천" 설계라 이름이 이 한 곳에만 모여 있다. 다만 다음 두 전제가 깔려 있다.
  - **플랫폼 전제**: CPU→GPU true는 UMA(890M) 전제다. 외장 GPU에서는 성립하지 않는다(주석 NOTE).
  - **vulkan-spirv 역할**: 10/20 시점에는 Accelerator로 둔 것이 TEMP 표식이었다. dev 통합 때 정식 Accelerator로 바꿨다(`e7060e40`).
- `accelIdx` 시작값은 정책 상수다. 라운드로빈 자체가 선언 순서 × dispatch 순서에 의존하고, NPU 3입력 문제 회피가 이 우연에 기대고 있다(CLAUDE.md 3절, 우회 W11).

## 6. 검증 근거 (우리가 한 것)

- 9/28 챕터 03: topology 소유자 결정 검증 보강, 회귀 v2(order 케이스로 선언 순서를 바꿔도 표대로 정해지는지).
- 9/28 챕터 04: transparent_access 방향 정의 확인(src가 dst 메모리 접근).
- 9/28 챕터 05: 경계 버퍼 전 케이스 표(→ 이 세션 경계버퍼_전수표_동기화.md(세션 문서 `docs/2026-10-20/1_변경장부_명세_커밋준비/02_구현_명세/경계버퍼_전수표_동기화.md`, 팀 git 미포함)).
- 회귀: chain2 3-dev 32768, mlp_reduce 3-dev 1, cg_topo lead=0/1 2, mm_probe lead=0/1 256(기준선 6개) → 현재 26×2 PASS.
- 10/19 세션 2 챕터 01: 결정 로직이 topology 표만 읽는다는 것 확인.
