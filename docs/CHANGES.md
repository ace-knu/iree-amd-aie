# 변경 장부 (CHANGES.md)

git HEAD 대비 **모든 변경**을 한곳에 모은 장부다. 소스(플러그인·IREE 서브모듈)를 고치는 세션은 끝날 때 이 파일을 갱신한다(CLAUDE.md 1-4의 4, 3-1-2).

- 첫 판: 2026-10-20 세션 1 챕터 01 (`docs/2026-10-20/1_변경장부_명세_커밋준비/01_변경_장부/`)
- 주제별 명세: `docs/2026-10-20/1_변경장부_명세_커밋준비/02_구현_명세/` (아래 "명세" 열의 S1~S10)

## 0. 기준

| 저장소 | 기준 커밋 | 작업 트리 상태 |
|---|---|---|
| 플러그인 `ace-knu/iree-amd-aie` | HEAD `44e6bcd4` (브랜치 `junho/npu-gpu-dmabuf`, `origin/dev` `f7a5c86f` + Junho Kwak 8/31 커밋 3개) | 추적 파일 20개 수정, 새 파일 5개(+ 스크립트·문서) |
| IREE 서브모듈 `third_party/iree` (`ace-knu/iree`) | `c73928f0b9` (detached, `origin/amd-aie-flow-hook`·`origin/bert-onnx`에 포함) | 추적 파일 28개 수정, 빈 새 파일 6개(출처 불명). **서브모듈 포인터는 HEAD와 같다**(상위 저장소에서 서브모듈은 "변경 없음"으로 보인다. 변경은 서브모듈 작업 트리 안에만 있다) |

출처 구분 방법(확인함):
- **전임자** = 9/28 인수인계 시점 작업 트리 − HEAD. 9/28 기준선 patch(`docs/2026-09-28/01_인수인계_코드현황_파악/baseline_plugin.patch`, `baseline_iree.patch`, `untracked_AMDAIEResolveBufferOwner.cpp`)를 HEAD 파일에 적용해 기준선 파일을 재구성했다(두 patch 모두 깨끗이 적용됨).
- **우리** = 현재 작업 트리 − 기준선. 덩어리마다 세션 코드 patch(`변경내역*.patch`)의 추가·삭제 줄과 대조해 출처 세션을 정했다(대조 스크립트·결과: 챕터 01 폴더 `attribute.py`, `attribution.tsv`). **출처를 못 찾은 덩어리는 0개다.**
- 근거 diff: 챕터 01 폴더 `current_plugin.patch`·`current_iree.patch`(HEAD 대비 전체), `ours_since_0928_plugin.patch`·`ours_since_0928_iree.patch`(기준선 대비 우리 몫).

## 1. 규모 요약

| 구분 | 플러그인 | IREE 서브모듈 | 합계 |
|---|---|---|---|
| HEAD 대비 수정 파일 | 20 | 28 | 48 |
| 그중 전임자 몫이 있는 파일 | 15 (모델 2 포함) | 23 | 38 |
| 그중 우리 몫이 있는 파일 | 14 | 8 | 22 |
| (둘 다 있는 파일) | 9 | 3 | 12 |
| 전임자 덩어리(9/28 기준선 hunk) | 51 (기능 19, 디버그 25, 기능+디버그 6, 바이너리 1) | 57 (기능 30, 디버그 1, 기능+디버그 22, 빈 줄만 4) | 108 |
| 우리 덩어리(기준선 대비 hunk) | 24 | 32 | 56 |
| 우리 변경 줄(+/− 합, 빈 줄 제외) | 509 | 830 | 1,339 |

분류 기준: "디버그" = 추가 줄이 모두 로그(`[T]`/`[DBG]`/`[BR]` 출력, 타이밍 변수)나 잔재(`g_last_bindings`, `execinfo`). `TEMP:`·`#if 0`·`EXPERIMENT`는 동작을 바꾸므로 "기능"으로 셌다(4절 D6~D8, D11).
| 새 소스·테스트 파일 | 5 (우리 4, 전임자 1) | 0 | 5 |

디버그·임시 표식(현재 작업 트리, 수정·신규 소스 대상 grep): `[T]` 30, `[DBG]` 21, `[BR]` 19, `BRIDGE` 주석 17, `TEMP:` 2(코드 표식), `EXPERIMENT` 5, 환경 변수 스위치 3. 9/28 대비 `[BR]` 3줄이 늘었고(우리: BRIDGE 오류 로그 이동, 매핑 없는 바인딩 로그, 입력 자동 배치 로그), 나머지는 전임자 것 그대로다. 4절 참고.

## 2. 우리 변경 — 덩어리별 (9/28 기준선 이후)

줄 번호는 **현재 작업 트리** 기준(`+시작,길이`). 명세 S번호는 챕터 02 문서.

### 2-1. 플러그인 (`compiler/plugins/target/AMD-AIE/iree-amd-aie/`, `runtime/src/iree-amd-aie/`)

| ID | 파일 | 덩어리 | 내용 | 출처(세션·챕터) | 이유 문서 | 명세 |
|---|---|---|---|---|---|---|
| O01 | `PluginRegistration.cpp` | +74,24 | (a) stream hook에 `AMDAIEAnnotateInputDevices` 등록(`ResolveBufferOwner` 바로 뒤) (b) flow hook에서 `AssignDeviceAffinities`에 옵션 2개 전달 | (a) 9/29 세션2 챕터02 (b) 10/1 세션2 챕터02 + 10/2 세션2 챕터02 | `docs/2026-09-29/2_VGG16상수묶음확인_input_device자동화/02_input_device_자동화_구현/README.md`, `docs/2026-10-01/2_C안_1단계_S1_respect_marks/02_구현과_기본값_불변_검증/README.md`, `docs/2026-10-02/2_f32_3장치_VGG16_임의배정/02_배정_지정_플래그/README.md` | S5, S6, S7 |
| O02 | `Target/AIETarget.h` | +67,15 | 옵션 필드 `respectAffinityMarks`, `placementMap` | 10/1 세션2 챕터02, 10/2 세션2 챕터02 | 위 10/1·10/2 문서 | S6, S7 |
| O03 | `Target/AIETarget.h` | +187,27 | CLI 옵션 `--iree-amdaie-respect-affinity-marks`, `--iree-amdaie-placement-map` | 10/1, 10/2 (같음) | 같음 | S6, S7 |
| O04 | `Target/uKernels/npu4/peano/matmul.cc` | +185,7 | `matmul_vectorized_bf16_f32` 첫 줄 `set_rnd(rnd_conv_even);` | 10/19 세션1 챕터04 | `docs/2026-10-19/1_NPU_bf16_벡터화_범위조사/04_U1_반올림_설정_확인/README.md` | S10 |
| O05 | `Transforms/AMDAIEAssignDeviceAffinities.cpp` | +33,10 | include 4개 (StringExtras, StringSwitch, MemoryBuffer, raw_ostream) | 10/2 세션2 챕터02 | 10/2 문서 | S7 |
| O06 | 같은 파일 | +115,240 | `readExecutableMark`(respect-marks) / placement-map 규칙 구조체·파서·`ruleMatches`(placement-map) / 옵션 생성자 | 10/1(표식 읽기·생성자) + 10/2(map) | 10/1·10/2 문서 | S6, S7 |
| O07 | 같은 파일 | +395,102 | dispatch walk 앞단: placement-map 분기 → respect-marks 분기 → (기존) 라운드로빈 | 10/2 + 10/1 | 같음 | S6, S7 |
| O08 | 같은 파일 | +504,67 | `markFailed` 실패 처리(10/1), 쓰이지 않은 번호 규칙 오류 + stderr 배정 요약(10/2) | 10/1 + 10/2 | 같음 | S6, S7 |
| O09 | 같은 파일 | +583,12 | topology 주입 대상에 "첫 번째가 아닌 host"도 포함(placement-map일 때만) | 10/2 | 10/2 문서 | S7 |
| O10 | 같은 파일 | +607,9 | `createAMDAIEAssignDeviceAffinitiesPass(options)` | 10/1 | 10/1 문서 | S6 |
| O11 | `Transforms/CMakeLists.txt` | +50,7 | `AMDAIEAnnotateInputDevices.cpp` 추가 | 9/29 세션2 챕터02 | 9/29 문서 | S5 |
| O12 | `Transforms/PassDetail.h` | +25,7 | `GEN_PASS_DEF_AMDAIEANNOTATEINPUTDEVICES` | 9/29 | 9/29 문서 | S5 |
| O13 | `Transforms/Passes.h` | +82,14 | 배정 패스 생성자 옵션 인자(10/1), AnnotateInputDevices 생성자(9/29) | 10/1 + 9/29 | 각 문서 | S5, S6 |
| O14 | `Transforms/Passes.td` | +85,22 | 배정 패스 옵션 `respect-marks`, `placement-map` | 10/1 + 10/2 | 각 문서 | S6, S7 |
| O15 | `Transforms/Passes.td` | +110,13 | `AMDAIEAnnotateInputDevices` 정의 | 9/29 | 9/29 문서 | S5 |
| O16 | `Transforms/test/CMakeLists.txt` | +10,15 | lit 3개 등록(annotate 9/29, placement_map 10/2, respect_marks 10/1) | 9/29, 10/2, 10/1 | 각 문서 | S5, S6, S7 |
| O17 | `Transforms/Utils/AMDAIEDevicePlacementUtils.cpp` | +66,12 | `linkFlags` GPU→CPU 줄의 옛 TODO 주석을 "BRIDGE로 지원됨, cg_topo lead=0 검증"으로 교체(**주석만**) | 10/2 세션1 챕터00 | `docs/2026-10-02/1_C안_2단계_연산단위배정/00_사전정리/README.md` | S8 |
| O18 | `runtime/…/amdxdna/allocator.cc` | +39,19 | allocator 구조체에 unowned `parent_device` | 9/28 챕터10 | `docs/2026-09-28/10_amdxdna_동기할당_placement_기록/README.md` | S2 |
| O19 | 같은 파일 | +144,11 | 동기 할당 placement에 `.device = parent_device` | 9/28 챕터10 | 같음 | S2 |
| O20 | 같은 파일 | +204,7 | `iree_hal_amdxdna_allocator_create` 시그니처에 `parent_device` | 9/28 챕터10 | 같음 | S2 |
| O21 | 같은 파일 | +214,8 | 생성자 호출 | 9/28 챕터10 | 같음 | S2 |
| O22 | `runtime/…/amdxdna/allocator.h` | +12,10 | 선언·주석 | 9/28 챕터10 | 같음 | S2 |
| O23 | `runtime/…/amdxdna/device.cc` | +119,8 | `allocator_create`에 device 넘김 | 9/28 챕터10 | 같음 | S2 |
| O24 | `runtime/…/amdxdna/test/allocator_test.cc` | +16,8 | 테스트를 새 시그니처에 맞춤(`/*parent_device=*/nullptr`) | 9/29 세션1 챕터02 | `docs/2026-09-29/1_구조전환_offset회귀_input_device준비/02_allocator_test_시그니처수정/README.md` | S2 |

### 2-2. IREE 서브모듈 (`third_party/iree/`)

| ID | 파일 | 덩어리 | 내용 | 출처 | 이유 문서 | 명세 |
|---|---|---|---|---|---|---|
| I01 | `compiler/…/Stream/Transforms/ElideAsyncCopies.cpp` | +723,16 | `hasPinnedPlacement(Value)` 공개 진입점 | 9/28 챕터06 | `docs/2026-09-28/06_ElideAsyncCopies_양방향판정_수정/README.md` | S1 |
| I02 | 같은 파일 | +838,62 | `hasPinnedPlacement` 본체(import/상수/global/lifetime constant·variable, 같은 lifetime transfer·clone은 따라감) | 9/28 챕터06 | 같음 | S1 |
| I03 | 같은 파일 | +1100,38 | `isSafeToElideTransferOp` topology 판정을 양방향으로 | 9/28 챕터06 | 같음 | S1 |
| I04 | `compiler/…/Stream/Transforms/test/elide_async_copies_topology.mlir` | +239,153 | lit 5모듈(비대칭 transient 2, 고정 import 보존 2(직접·cast/clone), 고정 import elide 1, 상수 보존 1) | 9/28 챕터06 | 같음 | S1 |
| I05 | `runtime/…/vulkan/direct_command_buffer.cc` | +7,11 | include `<cinttypes>`, `<cstdlib>`, `<cstring>` | 9/28 챕터11, 10/2 세션3 챕터01 | 각 문서 | S3, S4 |
| I06 | 같은 파일 | +19,11 | include `native_buffer.h`, `sparse_buffer.h` | 9/28 챕터09 | `docs/2026-09-28/09_Vulkan_남의버퍼_명시적에러(D)/README.md` | S3 |
| I07 | 같은 파일 | +74,217 | `is_vulkan_buffer`·`foreign_buffer_error`·`require_vulkan_buffer`(D) + `bridge_error`·**`resolve_buffer_ref`**(A, 전임자 dispatch BRIDGE 본문을 옮겨 공통화, 실패 시 에러 반환, `byte_offset` 반영) | 9/28 챕터09 + 11 | 09 문서, `docs/2026-09-28/11_Vulkan_copy_fill_update_BRIDGE(A)/README.md` | S3 |
| I08 | 같은 파일 | +580,13 | `execution_barrier`의 buffer barrier에 남의 버퍼면 에러 | 9/28 챕터09 | 09 문서 | S3 |
| I09 | 같은 파일 | +680,13 | `wait_events`의 buffer barrier에 같은 검사 | 9/28 챕터09 | 09 문서 | S3 |
| I10 | 같은 파일 | +781,20 | `fill_buffer` target BRIDGE | 9/28 챕터11 | 11 문서 | S3 |
| I11 | 같은 파일 | +930,13 | `update_buffer` target BRIDGE | 9/28 챕터11 | 11 문서 | S3 |
| I12 | 같은 파일 | +944,7 | 위 update의 ref 교체 | 9/28 챕터11 | 11 문서 | S3 |
| I13 | 같은 파일 | +992,17 | `copy_buffer` source/target BRIDGE | 9/28 챕터11 | 11 문서 | S3 |
| I14 | 같은 파일 | +1012,7 | copy 기록에 resolved ref 사용 | 9/28 챕터11 | 11 문서 | S3 |
| I15 | 같은 파일 | +1036,19 | `iree_hal_vulkan_br_dump_enabled()`(환경 변수 `IREE_VULKAN_BR_DUMP=1`) | 10/2 세션3 챕터01 | `docs/2026-10-02/3_Vulkan_map_겹침_수정_VGG16_3장치/01_디버그로그_map_분리_가설확인/README.md` | S4 |
| I16 | 같은 파일 | +1064,18 | 덤프 꺼짐이면 바인딩 값을 map하지 않고 ref 정보만 `[BR]`로 출력 | 10/2 세션3 챕터01 | 같음 | S4 |
| I17 | 같은 파일 | +1099,30 | dispatch 바인딩 BRIDGE를 `resolve_buffer_ref` 호출로 교체(전임자 99줄 삭제), 모든 바인딩·indirect 인자 Vulkan 버퍼 검사 | 9/28 챕터11 + 09 | 09·11 문서 | S3 |
| I18 | `runtime/…/vulkan/native_buffer.cc` | +11,7 | include `mutex.h` | 10/2 세션3 챕터02 | `docs/2026-10-02/3_Vulkan_map_겹침_수정_VGG16_3장치/02_map_참조수_근본수정/README.md` | S4 |
| I19 | 같은 파일 | +20,13 | `mapping_mutex`, `mapped_ptr` 필드 | 10/2 세션3 챕터02 | 같음 | S4 |
| I20 | 같은 파일 | +39,10 | `iree_hal_vulkan_native_buffer_isa` | 9/28 챕터09 | 09 문서 | S3 |
| I21 | 같은 파일 | +74,8 | 필드 초기화 | 10/2 세션3 챕터02 | 같음 | S4 |
| I22 | 같은 파일 | +93,14 | destroy에서 영구 매핑 해제 | 10/2 세션3 챕터02 | 같음 | S4 |
| I23 | 같은 파일 | +147,43 | `map_range`: 처음 한 번 `vkMapMemory(0, VK_WHOLE_SIZE)`, 이후 `mapped_ptr + offset`, non-coherent 읽기는 invalidate | 10/2 세션3 챕터02 | 같음 | S4 |
| I24 | 같은 파일 | +207,25 | `unmap_range`: unmap 안 함, non-coherent 쓰기는 flush | 10/2 세션3 챕터02 | 같음 | S4 |
| I25 | `runtime/…/vulkan/native_buffer.h` | +45,9 | `_isa` 선언 | 9/28 챕터09 | 09 문서 | S3 |
| I26 | `runtime/…/vulkan/sparse_buffer.cc` | +31,10 | `iree_hal_vulkan_sparse_buffer_isa` | 9/28 챕터09 | 09 문서 | S3 |
| I27 | `runtime/…/vulkan/sparse_buffer.h` | +35,9 | 선언 | 9/28 챕터09 | 09 문서 | S3 |
| I28 | `runtime/…/tooling/run_module.c` | +52,8 | `--input_device` 도움말에 reflection 대체 설명 | 9/29 세션2 챕터02 | `docs/2026-09-29/2_VGG16상수묶음확인_input_device자동화/02_input_device_자동화_구현/README.md` | S5 |
| I29 | 같은 파일 | +218,32 | `iree_tooling_find_device_by_id`(장치 ID 패턴 `<id>*`로 `--device` 목록 검색) | 9/29 | 같음 | S5 |
| I30 | 같은 파일 | +275,28 | 플래그가 없으면 reflection `amdaie.input_devices` 사용 | 9/29 | 같음 | S5 |
| I31 | 같은 파일 | +307,66 | 항목별 처리: `""` 둠, `?` 에러, `a|b` 동률은 lead가 있으면 lead, 아니면 첫 후보, 목록에 없는 ID는 에러 | 9/29 | 같음 | S5 |
| I32 | 같은 파일 | +411,9 | `[BR] input[i] -> device N (ok, flag/reflection)` 로그 | 9/29 | 같음 | S5 |

### 2-3. 새 파일 (untracked)

| ID | 경로 | 출처 | 내용 | 이유 문서 | 명세 |
|---|---|---|---|---|---|
| N01 | `compiler/…/Transforms/AMDAIEAnnotateInputDevices.cpp` | 우리, 9/29 세션2 챕터02 | 입력 장치 reflection 기록 패스 | 9/29 세션1 챕터04(설계), 세션2 챕터02(구현) | S5 |
| N02 | `compiler/…/Transforms/AMDAIEResolveBufferOwner.cpp` | **전임자**(9/28 기준선 사본과 바이트 동일, 확인함 `cmp`) | 소유자 결정 패스 | `docs/2026-09-28/01_인수인계_코드현황_파악/README.md` 3-2, `03_소유자결정_topology_검증보강/` | S8 |
| N03 | `compiler/…/Transforms/test/annotate_input_devices.mlir` | 우리, 9/29 | lit | 9/29 세션2 챕터02 | S5 |
| N04 | `compiler/…/Transforms/test/assign_device_affinities_respect_marks.mlir` | 우리, 10/1 | lit | 10/1 세션2 챕터02 | S6 |
| N05 | `compiler/…/Transforms/test/assign_device_affinities_placement_map.mlir` | 우리, 10/2 | lit | 10/2 세션2 챕터02 | S7 |
| N06 | `scripts/lock/{check-containers,host-load,lock-common,status,with-npu-lock}.sh` | **팀**(`origin/bert-onnx` 커밋 `49514d9c` 계열에서 `git show`로 꺼냄, 9/29) | 공용 NPU·빌드 잠금 | `docs/2026-09-29/3_VGG16_2장치재현_3장치/03_…`, CLAUDE.md 0 "공용 잠금" | 커밋 대상 아님(팀 브랜치에 이미 있음) |
| N07 | `models/mlp_parallel/debug_out/` (12페이즈 덤프 + vmfb, 9/1) | 전임자 임시 산출물 | 디버그 덤프 | 9/28 01 3-1 | 커밋 제외 |
| N08 | `third_party/iree/runtime/src/iree/hal/drivers/vulkan/{access,, memory_type,, memory_types.dispatch_idx, type, type,, usage,}` (0바이트 6개, 8/27 21:04) | **출처 불명**(전임자 시기). 이름이 C 코드 조각이라 셸 리다이렉트 실수로 생긴 파일로 **추정** | 없음 | 9/28 01 `untracked_files.txt` | 커밋 제외 |
| N09 | `docs/` 아래 날짜 폴더(9/28~10/20), `docs/regression/`, `docs/CHANGES.md` | 우리 | 작업 문서·회귀 스크립트 | — | 커밋 여부는 챕터 04 결정 |

### 2-4. 통합 브랜치 변경 (`integrate/3devices-f32+aie2p-bf16`, 규민 merge `fe7f132e` 이후)

줄 번호는 통합 브랜치 worktree 기준이다(`+시작,길이`). 위 2-1~2-3과 달리 기준은 merge `fe7f132e`다.

| ID | 파일 | 덩어리 | 내용 | 출처(세션·챕터) | 이유 문서 | 명세 |
|---|---|---|---|---|---|---|
| R01 | `runtime/…/aie_runtime/iree_aie_runtime.h` | +609,6 | `getRoundingModeCtrlReg()` 선언 | 10/21 세션1 챕터04(10/22 작업) | `docs/2026-10-21/1_규민브랜치_통합/03b_반올림_설계초안/`, `04_반올림_구현/` | S11 |
| R02 | `runtime/…/aie_runtime/iree_aie_runtime.cc` | +623,12 | 정의: AIE2 6, AIE2P 1, 그 밖 `nullopt` | 같음 | 같음 | S11 |
| R03 | `compiler/…/aie/Passes.td` | +56,7 | `AMDAIECoreToStandard` 옵션 `rounding-mode`(기본 `auto`) | 같음 | 같음 | S11 |
| R04 | `compiler/…/aie/AMDAIECoreToStandard.cpp` | +10 / +118,26 / +145,2 / +170,15 / +276,36 / +331,15 | include, 모드 값 표·`CoreRoundingMode`, `coreToStd` 진입부 `call @llvm.<arch>.set.ctrl.reg`, 모드 결정(auto·none·이름·오류·chess 경고), 선언 생성 | 같음 | 같음 | S11 |
| R05 | `iree-amd-aie/Target/XCLBinGen.h` | +32,2 | `aie2xclbin` 끝 인자 `coreRoundingMode = "auto"` | 같음 | 같음 | S11 |
| R06 | `iree-amd-aie/Target/XCLBinGen.cpp` | +1300,2 / +1311 / +1568,2 / +1588 | `generateUnifiedObject`·`aie2xclbin`에 전달, `options.roundingMode` | 같음 | 같음 | S11 |
| R07 | `iree-amd-aie/Target/AIETarget.h` | +43,4 / +138,10 | 필드 `coreRoundingMode{"auto"}`, CLI `--iree-amdaie-core-rounding-mode` | 같음 | 같음 | S11 |
| R08 | `iree-amd-aie/Target/AIETarget.cpp` | +656,2 | `aie2xclbin` 호출에 옵션 전달 | 같음 | 같음 | S11 |
| R09 | `compiler/…/aie/test/core_rounding_mode.mlir` | 새 파일 | lit(RUN 6줄) | 같음 | 같음 | S11 |
| R10 | `docs/specs/S11_코어_반올림_설정.md`, `docs/specs/README.md`, 이 파일 | 새 파일·행 추가 | 명세·장부 | 같음 | — | — |
| R11 | `tools/bfp16_emulator/bfp_strict_v3.py` | 새 파일 | AIE2P bfp16 MAC 하드웨어 흉내 V3.1(양자화 + 공통 지수 정렬 누산 + 비정규 처리). 컴파일러·런타임 코드가 아닌 시험 도구. 검증한 파일과 바이트 동일(sha256 `9b2d3337…3941`) | 10/22 세션1 챕터03~05 | `docs/2026-10-22/1_반올림_검증과_미해결조사/03_…`, `04_…`, `05_…` | S12 |
| R12 | `tools/bfp16_emulator/README.md` | 새 파일 | 사용법·제약·함수 사용·규칙 요약 | 같음 | 같음 | S12 |
| R13 | `docs/specs/S12_AIE2P_bfp16_MAC_하드웨어모델.md`, `docs/specs/README.md`, 이 파일 | 새 파일·행 추가 | 명세·장부 | 같음 | — | — |
| R14 | `tools/bfp16_emulator/bfp_strict_v3.py` | 마지막 줄 | `main()`을 `if __name__ == "__main__":` 안으로(import 가능하게). CLI 출력 바이트 동일 확인 | 10/22 세션2 챕터02 | `docs/2026-10-22/2_회귀_벡터화_편입/02_구현/` | S12 |
| R15 | `tools/bfp16_emulator/README.md`, `docs/specs/S12_…md` | 함수 사용·이력 | exec 방식 → import 방식, 이력 한 줄 | 같음 | 같음 | S12 |
| R16 | `docs/regression/run_regression.sh` | 머리 주석, 단계 분기, V0, 고장 주입 | 회귀 v9: `REG_LEVEL=base/quick(기본)/full`, V0(26케이스 난수 입력·출력 전체·CPU 기준), `REG_FAULT=out_corrupt`. `compile_flags`는 `lib.sh`로 옮김(내용 같음). base 26×2 결과 행 v8과 같음 | 같음 | 같음 | `docs/regression/README.md` |
| R17 | `docs/regression/lib.sh`, `ext/{run_ext.sh,gen_inputs.py,judge.py,stamp_run.py}` | 새 파일 | 공용 함수, 새 시험 V1~V6·F1~F3 실행기, 시드 입력 생성, 판정기(흉내 V3.1 import), 시각 도장 | 같음 | 같음 | 같음 |
| R18 | `docs/regression/models/vec/*.mlir`(4), `models/chain3_ngc.mlir` | 새 파일 | V1~V3 matmul 모델, V6 NPU→GPU→CPU 직결 모델 | 같음 | 같음 | 같음 |
| R19 | `docs/regression/vgg16/*`(5) | 새 파일 | F1~F3 레시피 사본·placement map·기준 sha(`expected_sha256.txt`, 바꿀 때 근거 + 사용자 승인) | 같음 | 같음 | 같음 |
| R20 | `docs/regression/README.md` | 앞부분 다시 씀, 버전 이력 v9 | 1-4 형식(위치·입출력·단계·판정·시험·검출력·갱신 규칙) | 같음 | 같음 | — |
| R21 | `compiler/…/Target/AIETarget.cpp` | +29 / +373,3 / +391,12 / +711,12 / +724 / +735,4 | `llvm/ADT/StringMap.h`, `Flatbuffer1dStringArrayConverter::addEntryDedup`(바이트 같으면 같은 번호), 컨트롤 패킷 + amdxdna HAL이면 모든 entry의 base PDI를 넣고 같은 PDI는 번호 공유(전: 첫 entry만). XRT HAL·컨트롤 패킷 끔은 그대로 | 10/22 세션6(B1) | `docs/2026-10-22/6_B1_컨트롤패킷_PDI묶음/`(원인: `docs/2026-10-22/5_hwctx_수명_설계조사/`) | S13 |
| R22 | `runtime/…/amdxdna/executable_internal.h` | +30,6 | `kernel_params.context_cu_name`(같은 PDI를 쓰는 첫 entry 이름) | 같음 | 같음 | S13 |
| R23 | `runtime/…/amdxdna/executable.cc` | +483,4 / +508,3 | 적재 때 PDI 번호별 대표 CU 이름 계산 | 같음 | 같음 | S13 |
| R24 | `runtime/…/amdxdna/direct_command_buffer.cc` | +1012,8 / +1022 / +1024,2 | 컨트롤 패킷 dispatch가 (PDI, 대표 이름)으로 묶음 컨텍스트를 찾고 CU를 연다(전: entry 이름) | 같음 | 같음 | S13 |
| R25 | `runtime/…/amdxdna/device.cc` | +69,7 / +82 / +85 / +1107,2 / +1113,39 / +1158,4 | 캐시 항목에 마지막 사용 순번, 생성 실패(NOT_FOUND·RESOURCE_EXHAUSTED·UNAVAILABLE) 때 쓰이지 않는 캐시 컨텍스트 LRU 축출 후 재시도, `[T] ctx EVICT` 로그, 실패 정보 덧붙임 | 같음 | 같음 | S13 |
| R26 | `docs/specs/S13_컨트롤패킷_배선묶음_컨텍스트.md`, `docs/specs/README.md`, 이 파일 | 새 파일·행 추가 | 명세·장부 | 같음 | — | — |
| R27 | `docs/regression/ext/run_ext.sh`, `ext/gen_inputs.py` | V7 절(`ctrlpkt_case`), `vgg_case` 다섯째 인자(ctx MISS 기대 수), F4 한 줄, 머리 주석 / `sig` 모드 | 회귀 v10: 같은 모델을 컨트롤 패킷 끔·켬으로 컴파일·실행해 비트 동일 + `[T] ctx MISS` 수 판정. 입력은 함수 시그니처대로 시드 11 | 10/22 세션6(B1) 05 | `docs/2026-10-22/6_B1_컨트롤패킷_PDI묶음/05_회귀추가/` | `docs/regression/README.md`, S13 |
| R28 | `docs/regression/models/ctrlpkt/{pa,chain4}.mlir` | 새 파일 | V7 모델(배선 3종·1종) | 같음 | 같음 | 같음 |
| R29 | `docs/regression/vgg16/flags_b_vec_ctrlpkt.txt`, `vgg16/expected_sha256.txt`, `docs/regression/README.md`, 이 파일 | 새 파일·한 줄 추가·v10 | F4 레시피(F2 + 컨트롤 패킷 두 플래그), 기준 sha `ctrlpkt_vgg16 6ccb1a0a…`(사용자 승인), README v10 | 같음 | 같음 | 같음 |
| R30 | (merge `c521e6d2`) `docs/patch_aie2p_bf16_vectorized.md`, `docs/repro_aie2p_bf16_vectorized.md`, `scripts/repro/aie2p_bf16/{matmul,vgg16}.sh` | 새 파일 4개(**출처 규민** `f55542b4`·`3e7af2a1`) | 규민 브랜치 `aie2p-bf16-vectorized`를 다시 merge. 함께 온 `17986af8`은 우리 `822f0fed`(R01~R09)의 cherry-pick이라 코드 변경 없음(결과 트리 = 앞 tip + 새 파일 4개, 충돌 없음) | 10/22 세션7 | `docs/2026-10-22/7_규민문서_통합_재현/01_fetch_merge_빌드/` | — |
| R31 | `docs/regression/ext/judge.py` | 모드 `race` 추가 | 출력을 64행 조각으로 나눠 기대값(V3.1 흉내 또는 정수 정답)과 비트 비교, 2웨이브 뒤 기대값과 같은 조각을 레이스 흔적으로 셈(규민 `matmul.sh` 판정과 같은 방식) | 10/22 세션8 | `docs/2026-10-22/8_회귀V1_M4096_레이스/` | `docs/regression/README.md` |
| R32 | `docs/regression/ext/run_ext.sh`, `docs/regression/models/vec/mm_f32_4096x64x32.mlir` | V1 모양 목록·`RACE_SHAPES`, 새 파일 | 회귀 v11: V1에 4096x64x32 rand·relu(V3.1 100% + race) | 같음 | 같음 | 같음 |
| R33 | `docs/regression/README.md`, 이 파일 | v11 | 판정·시험 표, 검출력 메모, 버전 이력 | 같음 | 같음 | 같음 |
| R34 | `scripts/npu/getpm.py` | 새 파일 | NPU 전원 모드 읽기 전용 도구(`GET_INFO`/`GET_POWER_MODE`, 세션 4 T6 도구를 옮기고 인자·시각·실패 처리 추가) | 10/22 세션 12 | `docs/2026-10-22/12_NPU_power_mode_기록_turbo재측정/02_모드기록_구현/` | `docs/regression/README.md` 3절 |
| R35 | `docs/regression/run_regression.sh`, `docs/regression/README.md`, 이 파일 | 머리 주석·`npu_pm`/`pm_line`·summary 한 줄 / v12 | 회귀 v12: summary에 `npu power_mode: start=… end=…`(다르면 `WARN pm changed`). 판정·rc 무관 | 같음 | 같음 | 같음 |

## 3. 전임자 변경 — 덩어리별 (HEAD → 9/28 기준선)

줄 번호는 **9/28 기준선 patch의 hunk 머리**(그때 작업 트리 줄)다. "현존" = 그 덩어리의 추가 줄이 지금 파일에 남아 있는 정도(확인함, 줄 문자열 대조). "분류": 기능 / 디버그(로그만) / 기능+디버그(한 덩어리에 섞임). 내용 설명과 판정(동작·로그·보류)은 `docs/2026-09-28/01_인수인계_코드현황_파악/README.md` 3-2(파일 목록), 3-4(디버그 인벤토리 A·B·C)를 따른다.

이유 문서 연결(덩어리 묶음별):
- stream hook(Pi01~03), `ResolveBufferOwner` 등록(Pp01, Pp03~07), `linkFlags`(Pp08): 9/28 01 3-2, `docs/2026-09-28/03_소유자결정_topology_검증보강/README.md`, `04_transparent_access_방향정의_확인/README.md` → 명세 S8
- `accelIdx = 0`(Pp02): 9/28 01 3-5, CLAUDE.md 3절 "배정" → 명세 S8(배정 부분)
- amdxdna dma-buf export(Pp12~17 allocator, Pp46 native.h, Pp47~48 native_linux_kmq), hwctx 캐시(Pp43), Vulkan 남의 버퍼 dispatch BRIDGE(Pi13~14), Vulkan dma-buf import·memory type·확장(Pi10, Pi21~33, Pi37~40), heap export DISCARD 제거(Pi04), Vulkan 세마포어 BRIDGE(L1)(Pi15~20 direct_command_queue, Pi41~42 vulkan_device), allocator.select·fence 로그(Pi43~45), `--input_device`·device group(Pi46~49 context_util, Pi55~57 run_module): 9/28 01 3-2·3-4 A, `docs/2026-09-28/08_Vulkan_copy_buffer_BRIDGE_조사/README.md`, `docs/2026-10-19/3_세마포어_Host중개_조사/` → 명세 S9
- mlp_2layer 모델(Pp09~11): 9/28 01 3-5 "판단 보류" → 명세 없음(모델)
- 로그만(디버그): 9/28 01 3-4 B → 4절

**이유 문서가 9/28 01의 "목적(추정)" 한 줄뿐인 덩어리**(전임자가 남긴 설명 없음, 우리가 코드로 추정): Vulkan `base_buffer.c` memory type HOST_VISIBLE 우선(Pi10, TEMP), `native_allocator.cc` 무조건 `MAPPING`(Pi25)·msync 검사 `#if 0`(Pi30), `module.c` allocator.select 로그(Pi43~45), `buffer.c`·`command_buffer_validation.c` 로그(Pi05~09). 이 중 동작에 영향을 주는 것(Pi10, Pi25, Pi30)은 명세 S9에 "근거 문서 없음"으로 표시했다.

### 3-1. 플러그인 (51덩어리)

| ID | 파일 | 9/28 hunk | 함수 | +/− | 분류 | 현존 | 첫 추가 줄 |
|---|---|---|---|---|---|---|---|
| Pp01 | `iree-amd-aie/PluginRegistration.cpp` | `-70,6 +70,12` | struct AMDAIESession | +4/-0 | 기능 | 전부 | void extendStreamTransformPassPipeline(OpPassManager &passMa |
| Pp02 | `Transforms/AMDAIEAssignDeviceAffinities.cpp` | `-159,7 +159,7` | void AMDAIEAssignDeviceAffinitiesPass::r | +1/-1 | 기능 | 전부 | unsigned accelIdx = 0; |
| Pp03 | `Transforms/CMakeLists.txt` | `-49,6 +49,7` | iree_cc_library( | +1/-0 | 기능 | 전부 | "AMDAIEResolveBufferOwner.cpp" |
| Pp04 | `Transforms/CMakeLists.txt` | `-133,6 +134,7` | iree_cc_library( | +1/-0 | 기능 | 전부 | iree::compiler::Dialect::Stream::IR |
| Pp05 | `Transforms/PassDetail.h` | `-24,6 +24,7` | namespace mlir::iree_compiler::AMDAIE { | +1/-0 | 기능 | 전부 | #define GEN_PASS_DEF_AMDAIERESOLVEBUFFEROWNER |
| Pp06 | `Transforms/Passes.h` | `-83,6 +83,7` | std::unique_ptr<Pass> createAMDAIEAssign | +1/-0 | 기능 | 전부 | std::unique_ptr<Pass> createAMDAIEResolveBufferOwnerPass(); |
| Pp07 | `Transforms/Passes.td` | `-87,6 +87,13` | def AMDAIEAssignDeviceAffinities : | +5/-0 | 기능 | 전부 | def AMDAIEResolveBufferOwner : |
| Pp08 | `Utils/AMDAIEDevicePlacementUtils.cpp` | `-26,43 +26,72` | DeviceRole deviceRole(StringRef backend) | +53/-28 | 기능 | 44/53줄 | // 'transparent_access' on a link (src -> dst) means "src ca |
| Pp09 | `mlp_2layer/export_mlp_2layer.py` | `-1,33 +1,26` |  | +3/-18 | 기능 | 전부 | Weights w1, w2 are runtime inputs (x, w1, w2), not baked ini |
| Pp10 | `mlp_2layer/export_mlp_2layer.py` | `-35,11 +28,11` | graph = h.make_graph( | +2/-3 | 기능 | 전부 | [vi("x"), vi("w1"), vi("w2")],  # weights are runtime inputs |
| Pp11 | `mlp_2layer/mlp_2layer.onnx` | `BINARY` |  | +0/-0 | 바이너리 | - |  |
| Pp12 | `amdxdna/allocator.cc` | `-5,6 +5,8` |  | +2/-0 | 기능 | 전부 | #include <unistd.h> |
| Pp13 | `amdxdna/allocator.cc` | `-13,6 +15,26` |  | +18/-0 | 기능 | 전부 | static iree_status_t iree_hal_amdxdna_allocator_export_buffe |
| Pp14 | `amdxdna/allocator.cc` | `-40,7 +62,9` | iree_hal_amdxdna_allocator_query_buffer_ | +3/-1 | 기능 | 전부 | // Every amdxdna allocation is a DRM BO exportable via PRIME |
| Pp15 | `amdxdna/allocator.cc` | `-80,16 +104,22` | iree_hal_amdxdna_allocator_query_buffer_ | +5/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp16 | `amdxdna/allocator.cc` | `-125,6 +155,21` | static iree_status_t iree_hal_amdxdna_al | +13/-0 | 기능+디버그 | 전부 | if (getenv("IREE_AMDXDNA_TEST_EXPORT")) { |
| Pp17 | `amdxdna/allocator.cc` | `-238,7 +283,7` | const iree_hal_allocator_vtable_t iree_h | +1/-1 | 기능 | 전부 | .export_buffer = iree_hal_amdxdna_allocator_export_buffer, |
| Pp18 | `amdxdna/buffer.cc` | `-96,6 +96,7` | static iree_status_t iree_hal_amdxdna_bu | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp19 | `amdxdna/buffer.cc` | `-149,6 +150,7` | static iree_status_t iree_hal_amdxdna_bu | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp20 | `amdxdna/device.cc` | `-473,6 +473,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp21 | `amdxdna/device.cc` | `-617,6 +618,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp22 | `amdxdna/device.cc` | `-703,6 +705,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp23 | `amdxdna/device.cc` | `-874,6 +877,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp24 | `amdxdna/device.cc` | `-921,6 +925,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp25 | `amdxdna/device.cc` | `-940,6 +945,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp26 | `amdxdna/device.cc` | `-972,6 +978,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[DBG] amdxdna queue_dispatch called\n"); |
| Pp27 | `amdxdna/device.cc` | `-1092,15 +1099,24` | iree_status_t iree_hal_amdxdna_device_ge | +9/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[T] ctx HIT  '%.*s' t=%.3fms\n", |
| Pp28 | `amdxdna/direct_command_buffer.cc` | `-1,3 +1,4` |  | +1/-0 | 기능 | 전부 | #include <cstdlib> |
| Pp29 | `amdxdna/direct_command_buffer.cc` | `-179,6 +180,7` | static void iree_hal_amdxdna_direct_comm | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp30 | `amdxdna/direct_command_buffer.cc` | `-194,6 +196,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp31 | `amdxdna/direct_command_buffer.cc` | `-215,6 +218,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp32 | `amdxdna/direct_command_buffer.cc` | `-227,6 +231,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp33 | `amdxdna/direct_command_buffer.cc` | `-242,6 +247,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp34 | `amdxdna/direct_command_buffer.cc` | `-254,6 +260,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp35 | `amdxdna/direct_command_buffer.cc` | `-286,6 +293,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp36 | `amdxdna/direct_command_buffer.cc` | `-324,6 +332,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp37 | `amdxdna/direct_command_buffer.cc` | `-343,6 +352,14` | static iree_status_t iree_hal_amdxdna_di | +6/-0 | 기능+디버그 | 전부 | const float* f = (const float*)source_mapping.contents.data; |
| Pp38 | `amdxdna/direct_command_buffer.cc` | `-533,6 +550,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp39 | `amdxdna/direct_command_buffer.cc` | `-713,6 +731,7` | static iree_status_t iree_hal_amdxdna_su | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp40 | `amdxdna/direct_command_buffer.cc` | `-774,6 +793,8` | static iree_status_t iree_hal_amdxdna_di | +2/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp41 | `amdxdna/direct_command_buffer.cc` | `-861,6 +882,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp42 | `amdxdna/direct_command_buffer.cc` | `-938,6 +960,9` | static iree_status_t iree_hal_amdxdna_di | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[T] dispatch enter t=%.3fms\n", iree_time_n |
| Pp43 | `amdxdna/direct_command_buffer.cc` | `-975,12 +1000,24` | static iree_status_t iree_hal_amdxdna_di | +17/-6 | 기능 | 전부 | // Reuse a cached hw context keyed on (PDI, kernel) instead  |
| Pp44 | `amdxdna/direct_command_buffer.cc` | `-1051,6 +1088,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] dispatch exit t=%.3fms\n", iree_time_no |
| Pp45 | `amdxdna/direct_command_buffer.cc` | `-1058,6 +1096,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp46 | `amdxdna/native.h` | `-125,6 +125,12` | uint64_t iree_hal_amdxdna_native_buffer_ | +4/-0 | 기능 | 전부 | // Exports the buffer's underlying BO as a dma-buf file desc |
| Pp47 | `amdxdna/native_linux_kmq.cc` | `-4,6 +4,7` |  | +1/-0 | 기능 | 전부 | #include <fcntl.h> |
| Pp48 | `amdxdna/native_linux_kmq.cc` | `-416,6 +417,28` | iree_device_size_t iree_hal_amdxdna_nati | +17/-0 | 기능 | 전부 | iree_status_t iree_hal_amdxdna_native_buffer_export_fd( |
| Pp49 | `amdxdna/native_linux_kmq.cc` | `-452,10 +475,12` | iree_status_t iree_hal_amdxdna_native_co | +2/-0 | 디버그 | 전부 | int64_t tc0 = iree_time_now(); |
| Pp50 | `amdxdna/native_linux_kmq.cc` | `-465,7 +490,9` | iree_status_t iree_hal_amdxdna_native_co | +2/-0 | 디버그 | 전부 | int64_t td0 = iree_time_now(); |
| Pp51 | `amdxdna/native_linux_kmq.cc` | `-577,11 +604,19` | iree_status_t iree_hal_amdxdna_native_qu | +8/-0 | 기능+디버그 | 전부 | int64_t t0 = iree_time_now(); |

### 3-2. IREE 서브모듈 (57덩어리)

| ID | 파일 | 9/28 hunk | 함수 | +/− | 분류 | 현존 | 첫 추가 줄 |
|---|---|---|---|---|---|---|---|
| Pi01 | `Pipelines/Pipelines.cpp` | `-413,6 +413,9` | void buildIREEVMTransformPassPipeline( | +2/-0 | 기능 | 전부 | if (hooks.pipelineExtensions) { |
| Pi02 | `PluginAPI/Client.h` | `-88,6 +88,12` | public: | +5/-0 | 기능 | 전부 | // Adds passes to the \|buildStreamTransformPassPipeline\| p |
| Pi03 | `PluginAPI/PluginManager.h` | `-140,6 +140,12` | public: | +3/-0 | 기능 | 전부 | void extendStreamTransformPassPipeline(OpPassManager &passMa |
| Pi04 | `hal/allocator_heap.c` | `-275,11 +275,14` | static iree_status_t iree_hal_heap_alloc | +7/-4 | 기능 | 전부 | // BRIDGE: never request DISCARD when exporting: the content |
| Pi05 | `hal/buffer.c` | `-5,7 +5,7` |  | +1/-1 | 기능 | 전부 | #include <stdio.h> |
| Pi06 | `hal/buffer.c` | `-275,6 +275,10` | IREE_API_EXPORT iree_status_t iree_hal_b | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] mem_type actual=0x%x expected=0x%x\n" |
| Pi07 | `hal/buffer.c` | `-339,6 +343,9` | iree_hal_buffer_validate_usage(iree_hal_ | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] usage allowed=0x%x required=0x%x\n", |
| Pi08 | `hal/buffer.c` | `-878,6 +885,12` | IREE_API_EXPORT iree_status_t iree_hal_b | +6/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] map_persistent buf=%p dev=%p type=0x% |
| Pi09 | `hal/command_buffer_validation.c` | `-118,6 +118,16` | static iree_status_t iree_hal_command_bu | +9/-0 | 기능+디버그 | 전부 | if (!iree_all_bits_set(iree_hal_buffer_allowed_usage(binding |
| Pi10 | `vulkan/base_buffer.c` | `-137,10 +137,24` | static void iree_hal_vulkan_populate_dis | +15/-1 | 기능 | 전부 | // TEMP: experiment. Prefer host-visible device-local memory |
| Pi11 | `vulkan/base_buffer.c` | `-296,6 +310,13` | iree_status_t iree_hal_vulkan_populate_m | +6/-0 | 기능+디버그 | 전부 | fprintf(stderr,                                    // ← 여기 |
| Pi12 | `vulkan/completion_watcher.cc` | `-340,6 +340,7` | static int iree_hal_vulkan_completion_wa | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[BR] vk complete sem=%p value=%llu t=%.3fms |
| Pi13 | `vulkan/direct_command_buffer.cc` | `-1,3 +1,4` |  | +1/-0 | 기능 | 전부 | #include <stdio.h> |
| Pi14 | `vulkan/direct_command_buffer.cc` | `-782,15 +783,176` | static iree_status_t iree_hal_vulkan_dir | +129/-0 | 기능+디버그 | 108/129줄 | iree_hal_buffer_ref_t g_last_bindings[8]; |
| Pi15 | `vulkan/direct_command_queue.cc` | `-5,6 +5,10` |  | +3/-0 | 기능+디버그 | 전부 | extern iree_hal_buffer_ref_t g_last_bindings[8]; |
| Pi16 | `vulkan/direct_command_queue.cc` | `-28,7 +32,8` | DirectCommandQueue::~DirectCommandQueue( | +2/-1 | 기능 | 전부 | VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena*  |
| Pi17 | `vulkan/direct_command_queue.cc` | `-41,22 +46,47` | iree_status_t DirectCommandQueue::Transl | +30/-7 | 기능+디버그 | 전부 | iree_host_size_t wait_count = 0; |
| Pi18 | `vulkan/direct_command_queue.cc` | `-68,24 +98,24` | iree_status_t DirectCommandQueue::Transl | +4/-4 | 기능 | 전부 | static_cast<uint32_t>(wait_count); |
| Pi19 | `vulkan/direct_command_queue.cc` | `-103,17 +133,66` | iree_status_t DirectCommandQueue::Submit | +42/-2 | 기능+디버그 | 전부 | bool any_foreign_signal = false; |
| Pi20 | `vulkan/direct_command_queue.h` | `-35,7 +35,8` | class DirectCommandQueue final : public  | +2/-1 | 기능 | 전부 | VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena*  |
| Pi21 | `vulkan/dynamic_symbol_tables.h` | `-243,7 +243,7` | namespace vulkan { | +1/-1 | 기능 | 전부 | DEV_PFN(OPTIONAL, vkGetMemoryFdPropertiesKHR)                |
| Pi22 | `vulkan/extensibility_util.cc` | `-213,6 +213,16` | iree_hal_vulkan_populate_enabled_device_ | +6/-0 | 기능 | 전부 | } else if (strcmp(extension_name, |
| Pi23 | `vulkan/extensibility_util.h` | `-84,6 +84,13` | typedef struct iree_hal_vulkan_device_ex | +4/-0 | 기능 | 전부 | // VK_KHR_external_memory_fd is enabled. |
| Pi24 | `vulkan/native_allocator.cc` | `-1,3 +1,5` |  | +2/-0 | 기능+디버그 | 전부 | #include <execinfo.h> |
| Pi25 | `vulkan/native_allocator.cc` | `-198,7 +200,11` | iree_hal_vulkan_native_allocator_query_b | +5/-1 | 기능 | 전부 | // BRIDGE: on UMA parts (890M) every allocation lands in hos |
| Pi26 | `vulkan/native_allocator.cc` | `-288,6 +294,30` | static iree_status_t iree_hal_vulkan_nat | +19/-0 | 기능+디버그 | 전부 | // BRIDGE: report the memory type the buffer actually landed |
| Pi27 | `vulkan/native_allocator.cc` | `-295,7 +325,7` | static iree_status_t iree_hal_vulkan_nat | +1/-1 | 기능 | 전부 | placement, actual_type, params->access, actual_usage, alloca |
| Pi28 | `vulkan/native_allocator.cc` | `-333,7 +363,7` | static iree_status_t iree_hal_vulkan_nat | +1/-1 | 기능 | 전부 | placement, actual_type, params->access, actual_usage, alloca |
| Pi29 | `vulkan/native_allocator.cc` | `-586,7 +616,14` | static VkDeviceSize iree_hal_vulkan_nati | +5/-1 | 기능+디버그 | 전부 | // [BR] Read the pointer value as a pointer, then convert. T |
| Pi30 | `vulkan/native_allocator.cc` | `-626,7 +663,7` | static iree_status_t iree_hal_vulkan_nat | +1/-1 | 기능 | 전부 | #if 0 && defined(IREE_PLATFORM_LINUX) |
| Pi31 | `vulkan/native_allocator.cc` | `-723,6 +760,10` | static iree_status_t iree_hal_vulkan_nat | +0/-0 | 빈 줄만 | - |  |
| Pi32 | `vulkan/native_allocator.cc` | `-790,6 +831,83` | static void iree_hal_vulkan_native_alloc | +60/-0 | 기능 | 전부 | static iree_status_t iree_hal_vulkan_native_allocator_import |
| Pi33 | `vulkan/native_allocator.cc` | `-865,6 +983,14` | static iree_status_t iree_hal_vulkan_nat | +4/-0 | 기능 | 전부 | case IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD: |
| Pi34 | `vulkan/native_buffer.cc` | `-112,6 +112,13` | static iree_status_t iree_hal_vulkan_nat | +7/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] vkmap buf=%p dev=%p type=0x%x usage=0 |
| Pi35 | `vulkan/native_semaphore.cc` | `-7,7 +7,7` |  | +1/-1 | 기능 | 전부 | #include <cstdio> |
| Pi36 | `vulkan/native_semaphore.cc` | `-30,6 +30,10` | extern const iree_hal_semaphore_vtable_t | +4/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] semaphore_cast: value=%p vtable=%p ex |
| Pi37 | `vulkan/vulkan_device.cc` | `-6,6 +6,9` |  | +1/-0 | 기능 | 전부 | #include <stdio.h> |
| Pi38 | `vulkan/vulkan_device.cc` | `-227,6 +230,18` | IREE_API_EXPORT iree_status_t iree_hal_v | +9/-0 | 기능 | 전부 | // VK_KHR_external_memory_fd: |
| Pi39 | `vulkan/vulkan_device.cc` | `-1046,6 +1061,9` | static iree_status_t iree_hal_vulkan_que | +0/-0 | 빈 줄만 | - |  |
| Pi40 | `vulkan/vulkan_device.cc` | `-1146,6 +1164,12` | iree_status_t iree_hal_vulkan_device_cre | +5/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] vk ext: memory_fd=%d dma_buf=%d host= |
| Pi41 | `vulkan/vulkan_device.cc` | `-1912,7 +1936,28` | static iree_status_t iree_hal_vulkan_dev | +19/-1 | 기능 | 전부 | // BRIDGE: only native semaphores can be watched; foreign on |
| Pi42 | `vulkan/vulkan_device.cc` | `-1921,7 +1966,7` | static iree_status_t iree_hal_vulkan_dev | +1/-1 | 기능 | 전부 | device->completion_watcher, &native_signal_list, resource_se |
| Pi43 | `hal/module.c` | `-351,6 +351,11` | IREE_VM_ABI_EXPORT(iree_hal_module_alloc | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] alloc_select: no policy, pairs=%d ->  |
| Pi44 | `hal/module.c` | `-368,12 +373,21` | IREE_VM_ABI_EXPORT(iree_hal_module_alloc | +8/-1 | 기능+디버그 | 전부 | for (iree_host_size_t i = 0; i < pair_count; ++i) { |
| Pi45 | `hal/module.c` | `-1661,6 +1675,12` | IREE_VM_ABI_EXPORT(iree_hal_module_fence | +4/-0 | 기능+디버그 | 전부 | iree_string_view_t dev_id = iree_hal_device_id(device); |
| Pi46 | `tooling/context_util.c` | `-6,8 +6,11` |  | +2/-0 | 기능 | 전부 | iree_hal_device_group_t* iree_tooling_experimental_device_gr |
| Pi47 | `tooling/context_util.c` | `-185,6 +188,8` | static iree_status_t iree_hal_module_dev | +0/-0 | 빈 줄만 | - |  |
| Pi48 | `tooling/context_util.c` | `-463,6 +468,9` | static iree_status_t iree_tooling_load_h | +3/-0 | 기능 | 전부 | // EXPERIMENT: expose the group so run_module can place inpu |
| Pi49 | `tooling/context_util.h` | `-106,6 +106,9` | iree_status_t iree_tooling_create_instan | +2/-0 | 기능 | 전부 | // EXPERIMENT: device group created from --device flags (NUL |
| Pi50 | `tooling/function_util.c` | `-5,7 +5,7` |  | +1/-1 | 기능 | 전부 | #include <stdio.h> |
| Pi51 | `tooling/function_util.c` | `-16,6 +16,8` | iree_status_t iree_tooling_append_async_ | +2/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] abi.model='%.*s'\n", (int)model.size, |
| Pi52 | `tooling/function_util.c` | `-42,6 +44,9` | iree_status_t iree_tooling_append_async_ | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] async_fence sem=%p dev=%p\n", |
| Pi53 | `tooling/function_util.c` | `-94,6 +99,7` | static iree_status_t iree_tooling_setup_ | +0/-0 | 빈 줄만 | - |  |
| Pi54 | `tooling/function_util.c` | `-128,6 +134,8` | static iree_status_t iree_tooling_submit | +2/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] transfer_fence sem=%p dev=%p\n", |
| Pi55 | `tooling/run_module.c` | `-6,6 +6,9` |  | +2/-0 | 기능 | 전부 | #include <stdlib.h> |
| Pi56 | `tooling/run_module.c` | `-45,6 +48,12` | IREE_FLAG_LIST( | +5/-0 | 기능 | 전부 | IREE_FLAG_LIST( |
| Pi57 | `tooling/run_module.c` | `-237,6 +246,77` | static iree_status_t iree_tooling_run_fu | +59/-0 | 기능+디버그 | 53/59줄 | // EXPERIMENT: relocate inputs to per-input devices. |

현존이 "전부"가 아닌 덩어리(우리가 덮어쓴 것, 확인함):
- Vulkan `direct_command_buffer.cc` dispatch BRIDGE(Pi14, 108/129줄): 9/28 챕터11이 본문을 `resolve_buffer_ref`로 옮기며 실패 시 `continue`(조용한 실패)를 에러 반환으로 바꿈(I07, I17).
- `AMDAIEDevicePlacementUtils.cpp` `linkFlags`(Pp08, 44/53줄): 10/2 세션1 챕터00이 GPU→CPU 주석의 옛 TODO를 교체(O17).
- `run_module.c` `--input_device`(Pi57, 53/59줄): 9/29 세션2 챕터02가 reflection 경로를 넣으며 반복문·로그를 바꿈(I30~I32).

## 4. 디버그·임시 코드 (현재 작업 트리)

| 묶음 | 위치 | 출처 | 기본 동작 | 의존하는 곳 | 비고 |
|---|---|---|---|---|---|
| D1 `[T]` 타이밍 로그 30줄 | amdxdna `allocator.cc`, `buffer.cc`, `device.cc`, `direct_command_buffer.cc`, `native_linux_kmq.cc` | 전임자 | 항상 stderr | 없음(확인함: 회귀 스크립트 grep 0) | 9/28 01 3-4 B |
| D2 `[DBG]` 로그 21줄 | amdxdna 위 파일들, IREE `buffer.c`, `command_buffer_validation.c`, vulkan `base_buffer.c`·`native_buffer.cc`·`native_semaphore.cc`·`vulkan_device.cc`, `modules/hal/module.c`, `tooling/function_util.c` | 전임자 | 항상 stderr | **회귀 스크립트가 `module.c`의 `[DBG] alloc_select` 로그로 lead 선택을 판정**(`docs/regression/run_regression.sh:154`) | 지우면 회귀 판정이 깨진다 |
| D3 `[BR]` 로그 19줄 | vulkan `direct_command_buffer.cc`(BRIDGE import 결과 5, dispatch 바인딩 4), `direct_command_queue.cc` 3, `completion_watcher.cc` 1, `native_allocator.cc` 1(+주석 1), amdxdna `direct_command_buffer.cc` 1, `run_module.c` 2 | 전임자 15 + 우리 4(I07의 3줄은 전임자 로그를 옮긴 것, I16 1줄, I31 1줄, I32는 전임자 줄 수정) | 항상 stderr. dispatch 바인딩 값 덤프(map)는 `IREE_VULKAN_BR_DUMP=1`일 때만(I15) | 회귀 스크립트가 `[BR] input[i]`(placed 열, `:174`)와 `[BR] amdxdna copy`(xdna_copy 지표, `:283`)를 읽음 | 지우면 회귀 지표가 깨진다 |
| D4 `g_last_bindings[8]` 전역 | vulkan `direct_command_buffer.cc`, `direct_command_queue.cc` extern | 전임자 | 쓰기만, 읽는 곳 없음(9/28 확인), 스레드 안전 아님 | 없음 | 잔재 |
| D5 `#include <execinfo.h>` | vulkan `native_allocator.cc:1` | 전임자 | 쓰는 곳 없음 | 없음 | 잔재 |
| D6 `TEMP: experiment` | 플러그인 `AMDAIEDevicePlacementUtils.cpp:24`(vulkan-spirv → Accelerator, **HEAD에 이미 커밋됨**), vulkan `base_buffer.c:140`(HOST_VISIBLE 우선) | 전임자 | **동작** | GPU 배정 전제, CPU→GPU HOST_VISIBLE 표의 전제 | 9/28 01 3-4 A |
| D7 `#if 0 &&` msync 검사 끔 | vulkan `native_allocator.cc:666` | 전임자 | **동작**(host ptr import 전 검사 생략) | GPU→CPU host ptr import | 9/28 01 3-5 부채 |
| D8 무조건 `MAPPING` | vulkan `native_allocator.cc:203-207` | 전임자 | **동작** | CPU가 GPU 버퍼를 host-map | UMA 전제, 외장 GPU에선 성립 안 함 |
| D9 환경 변수 스위치 | `IREE_AMDXDNA_TEST_EXPORT`(allocator.cc:168, 테스트 훅), `IREE_AMDXDNA_CACHE_HWCTX`(direct_command_buffer.cc:1008, hwctx 캐시 opt-in), `IREE_VULKAN_BR_DUMP`(vulkan direct_command_buffer.cc:1046, 우리) | 전임자 2 + 우리 1 | 기본 꺼짐 | 없음 | — |
| D10 `[amdaie placement-map]` stderr 요약 | `AMDAIEAssignDeviceAffinities.cpp:558-566` | 우리(10/2) | **`--iree-amdaie-placement-map`을 줄 때만** | 실험 기록 | 실험 옵션의 출력이라 디버그로 분류하지 않음(챕터 04에서 판단) |
| D11 `EXPERIMENT` 표식 5곳 | `run_module.c`(`--input_device`), `context_util.c/.h`(device group 전역) | 전임자 | **동작** | 회귀 수동 모드, 입력 자동 배치(S5) | 9/28 01 3-4 A |

## 5. 코드가 아닌 우회 (배정 맵, 플래그, IR 편집, 설정 주입, 배정으로 피하기)

전체 목록(32개, 근거 줄 포함)은 `docs/2026-10-20/1_변경장부_명세_커밋준비/01_변경_장부/우회_목록.md`. 아래는 요약이다. "CLAUDE.md" 열은 **2026-10-20 수정 후 줄 번호**(1-3 행, 3-2 문단)다. 상태: **씀** = 지금 실험·데모에 씀, **기본** = 회귀·F3 등 기본 경로가 의존, 안 씀 = 기록만.

| ID | 우회 | 종류 | 피하는 것 | 원인 규명 | 없앨 계획 | 상태 | CLAUDE.md(1-3·3-2) |
|---|---|---|---|---|---|---|---|
| W01 | 맵 NPU1/GPU15/CPU43 (`docs/2026-10-02/2_…/03_…/*.map`) | 맵 | NPU f32 코드젠 실패 | 표면만 | 3-2 없앨 계획 1~3 | 안 씀(FC GPU 2배) | L305, L311-313 |
| W02 | 맵 t4 1/12/46 (`docs/2026-10-02/3_…/01b_…/map_t4_fc_cpu.txt`) | 맵 | NPU f32 실패 + GPU M=1 2배 | GPU 쪽 규명 | 3-2 L311-313, L325 | 데모 | L321, L325 |
| W03 | 맵 NPU10/GPU3 | 맵 | — | — | — | 안 씀(실행 실패) | L321 |
| W04 | **맵 NPU9/GPU4/CPU46** (`docs/2026-10-18/2_…/02c_…/…npu9_gpu4.map`) | 맵 | d6 반복 392, conv5 ②, FC ③, GPU 2배 | 반복 392 이유는 규명, **멈춤 기전 미확인**, ②③ 설계 수준 미규명 | L311·L312, L325 | **씀**(10/20 세마포어 시험, 커밋 후 시험 목록) | L152, L154, L321 |
| W05·W06 | 맵 NPU13·NPU16 (`docs/2026-10-18/4_…/03_…/*.map`) | 맵 | 편집 IR(W22·W23)용 | W22와 같음 | L311-313 | 데모 | L321 |
| W07 | NPU가 f32로 못 하는 dispatch를 배정으로 피함 | 배정 | ①②③ + DMA 반복 | 표면만 → **챕터 03** | L311-313 | 씀 | L143, L275, L154 |
| W08 | FC(M=1)를 GPU에 안 보냄 | 배정 | SPIR-V SubgroupReduce subgroup 크기 미기록 | **규명** | L325(서브모듈 수정, 일정 미정) | 씀 | L325 |
| W09 | d6(conv1_2)를 GPU로 | 배정 | 제어 코드 반복 392 > 256 | 392 이유 규명, 멈춤 기전 미확인 | L312 | 씀 | L147, L152 |
| W10 | 반복 ≤ 256 규칙 | 배정(규칙) | 256 초과 펼침 멈춤 | **미확인**(fc6 782 반례) | L312, L152 "추가 조사" | 씀 | L152, L321 |
| W11 | NPU 3입력 dispatch를 라운드로빈 순서로 피함(전임자) | 배정 | `no producer DMA channel available` | **미규명** | **없음**("고치려 하지 말 것") | **기본**(회귀 기준선) | L244, L296 |
| W12 | lead 기본값(0) 유지 → CPU↔GPU 동률은 GPU 소유 | 런타임 선택 | BRIDGE 기록 시점 transient export | 코드 경로 규명 | **없음**(수정 방향만) | **기본** | L324, L151 질문 3 |
| W13 | `-unroll-threshold=0` | 플래그 | ① soft-float 펼침 코드 16 KB 초과 | 규명 | L313(벡터화 후 뺌) | 씀 | L146, L314-319 |
| W14 | dedup off | 플래그 | 정적 offset만 지원하는 `address_patch` | 규명 | **없음**(팀 백엔드 제약) | 씀(VGG) | L298 |
| W15 | no-fuse-into-contraction | 플래그 | 융합 contraction tile 불가(팀 기록) | 팀 기록뿐 | **없음** | 씀(VGG) | L298 |
| W16 | vectorization off | 플래그 | aievec bf16/f32 미지원 | 규명(10/19 세션1) | **명시 없음**(벡터화 작업 연계 추정) | 씀(VGG) | L298 |
| W17 | im2col·channels-last·detach | 플래그(레시피) | NPU conv 경로 | 팀 기록 | 없음(레시피) | 씀(VGG) | L298 |
| W18 | `--device_lead_allocator=2` | 플래그 | Vulkan map SIGSEGV | 근본 수정됨 | — | 안 씀 | L324 |
| W19 | `--input_device` 수동 | 플래그 | 자동 배치 전 수단 | — | 없음(비교용) | **기본**(회귀 수동 모드) | L70 |
| W20 | `--iree-hal-external-resources-mappable=true` | 플래그(F3) | **docs에 이유 없음** | **미규명** | 없음 | **기본** | L57-65 |
| W21 | `--iree-amdaie-stack-size=2048` | 플래그(F3) | stack 부족(일반론) | 3장치 필요성 미확인 | 없음 | **기본** | L57-65 |
| W22 | conv5 M 196→224 패딩 IR 편집 | IR 편집 | ② 팀 pad가 f32 건너뜀 | 표면만 → **챕터 03** | L311(1) | 데모(vmfb만) | L145, L154 |
| W23 | FC M 1→32 `linalg.generic` 패딩, fc6 N 512×8 분할, fc8 N 1000→1024 | IR 편집 | ③ M=1 pack, N 1000 `ConvertToDma` 실패, 팀 N-split f32 건너뜀 | 표면만 | L311(1) | 데모 | L145, L154 |
| W24 | FC 패딩 v1(`tensor.pad`) | IR 편집 | — | 0 splat CSE 실패(규명) | — | 폐기 | — |
| W25·W26 | 단독 손 패딩 모델 8개, conv1_2 M 2분할 | IR 편집 | 검증용 | — | — | 안 씀 | L321 |
| W27 | conv1_2 1단계 타일 M 256 executable 치환(`--iree-hal-substitute-executable-source`) | 설정 주입 | 반복 392 | 392 이유 규명, 멈춤 미확인 | L312(타일 ≤ 256) | 데모 | L147, L152 |
| W28 | `REG_OUT=/home/mkchoi/iree-work/regression/…` | 환경 변수 | `REG_ROOT` 기본 `/tmp` | 규명 | 제안만(미승인) | **기본** | L55 |
| W29 | `RADV_PERFTEST=cswave32` | 환경 변수 | GPU 2배 | 효과 없음 | — | 안 씀 | L325 |
| W30 | `IREE_VULKAN_BR_DUMP=1` | 환경 변수(우리 코드) | 디버그 map 충돌 | 근본 수정됨 | 챕터 04 "빼야 할 것" | 기본 꺼짐 | L250, L323 |
| W31·W32 | `IREE_AMDXDNA_CACHE_HWCTX`, `IREE_AMDXDNA_TEST_EXPORT` | 환경 변수(전임자) | hwctx 캐시 opt-in, export 테스트 | 전임자 주석뿐 | 없음 | 안 씀 | — |

코드이지만 우회 수단인 것: K1 ukernel `set_rnd`(O04, 팀 반영 전, 규민 상의), K2 `--iree-amdaie-placement-map`(W01~W06의 통로, S7), K3 `--iree-amdaie-respect-affinity-marks`(디버그, S6), K4 `accelIdx = 0`(전임자, W11의 바탕).

- **원인 미규명 우회**: W10(DMA 멈춤 기전) / W07·W22·W23(f32 패딩 설계 수준·배수 → 챕터 03에서 조사) / W23의 N 1000 실패 / W11 NPU 3입력 / W20 mappable / W21 stack-size / W15 no-fuse
- **없앨 계획이 없는 우회**: W11, W12, W14, W15, W16, W17, W19, W20, W21, W31·W32, K4

## 6. 커밋 계획 대응 (2026-10-20 세션 1 챕터 04, 승인 대기)

| 커밋 | 장부 ID |
|---|---|
| K00a 인수인계 기준선(컴파일러) | Pp01~08, Pi01~03, N02 |
| K00b 인수인계 기준선(런타임) | Pp12~51 중 기능 덩어리, Pi04~57(U3 = Pi25 제외안), 로그 전용 덩어리 처리는 계획 3-5 |
| K01 S1 | I01~I04 | 
| K02 S3-D / K04 S3-A | I06, I08, I09, I17(일부), I20, I25~I27 / I05, I07, I10~I14, I17 |
| K03 S2 / K05 S2 test | O18~O23 / O24 |
| K06 S5 | O01a, O11~O13, O15, O16, N01, N03, I28~I32 |
| K07 S6 / K09 S7 | O01b~O03, O06~O08, O10, O13, O14, N04 / O05~O09, N05 |
| K08 S8 주석 | O17 |
| K10 / K11 S4 | I15, I16 / I18, I19, I21~I24 |
| K12 S10(보류 추천) | O04 |
| 제외 | N06(lock 스크립트), N07(덤프), N08(빈 파일 6개), Pp09~11(모델, 추천), Pi25(U3, 추천) |

## 7. 갱신 이력

| 날짜 | 세션 | 바뀐 것 |
|---|---|---|
| 2026-10-20 | 세션 1 챕터 01 | 첫 판(9/28 기준선 대비 우리 56덩어리, 전임자 108덩어리, 새 파일 9묶음, 디버그 11묶음, 우회 목록) |
| 2026-10-22 | 10/21 세션 1 챕터 04 | **통합 브랜치** 2-4절 R01~R10: 코어 반올림 모드 설정(S11). AIE2P 기본 conv_even, AIE2 기본 넣지 않음. 회귀 26×2·VGG16 2·3장치 출력 그대로, vmfb는 바뀜(NPU 코어 명령 1개). 작업 폴더(`3devices-f32`)에는 없다 |
| 2026-10-22 | 10/22 세션 1 챕터 05 | **통합 브랜치** 2-4절 R11~R13: AIE2P bfp16 MAC 흉내 V3.1(`tools/bfp16_emulator/`)과 명세 S12. 코드 변경 없음(시험 도구). VGG16 실제 dispatch 4개(conv1_1·conv4_2·conv5_2·fc6) 포함 41경우 비트 일치 |
| 2026-10-22 | 10/22 세션 2 챕터 02 | **통합 브랜치** 2-4절 R14~R20: 회귀 v9(3단계, V0, V1~V6, F1~F3), 흉내 import 보호. 컴파일러·런타임 변경 없음. quick 219 s·full 529 s 모두 PASS |
| 2026-10-22 | 10/22 세션 6 | **통합 브랜치** 2-4절 R21~R26: 컨트롤 패킷 경로의 배선(base PDI) 묶음별 PDI·컨텍스트(S13)와 컨텍스트 생성 실패 때 LRU 축출. 컨트롤 패킷 끔은 그대로(VGG16 vmfb `10abd446…`·출력 `6ccb1a0a…` 동일, 회귀 quick 전부 PASS). 켬: 배선 3종 모델이 고치기 전 틀림(corr 0.229) → 고친 뒤 기준과 비트 동일 |
| 2026-10-22 | 10/22 세션 6 챕터 05 | **통합 브랜치** 2-4절 R27~R29: 회귀 v10(V7 컨트롤 패킷 `ctrlpkt_pa_3routes`·`ctrlpkt_chain4`, F4 `ctrlpkt_vgg16`). 컴파일러·런타임 변경 없음. quick 245 s 전부 PASS(EXT 20), F4 단독 PASS |
| 2026-10-22 | 10/22 세션 7·8 | **통합 브랜치** 2-4절 R30: 규민 브랜치 재 merge `c521e6d2`(문서·스크립트 4개, 반올림 cherry-pick은 코드 변경 없음). R31~R33: 회귀 v11(V1 4096x64x32 + `race` 판정). 컴파일러·런타임 변경 없음. quick 249 s 전부 PASS(EXT 22) |
| 2026-10-22 | 10/22 세션 12 | **통합 브랜치** 2-4절 R34~R35: NPU 전원 모드 기록(`scripts/npu/getpm.py`, 회귀 v12 summary 줄). 컴파일러·런타임 변경 없음. 잠금 래퍼 `scripts/lock/with-npu-lock.sh`의 같은 기록은 브랜치 밖 파일(빌드 사본만)이라 장부 행 없음 |
| 2026-10-20 | 세션 1 CP1 후·챕터 04 | 소스 변경 없음. 6절 커밋 계획 대응 추가. 명세 알려진 한계(S7 L1, S3·S9 L-fd) 추가. U1~U4 명세(`04_커밋_계획/01_…`) |
