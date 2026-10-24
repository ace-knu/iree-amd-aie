# 3-device 회귀 테스트 (tests/regression)

CPU(local-task) + NPU(amdxdna) + GPU(vulkan, gfx1150) 3장치 작업의 회귀 테스트다. 스크립트와 모델은 여기서만 관리한다(v9부터 `tests/regression/`, v6~v8은 `docs/regression/`). 3장치 관련 변경(배치, 버퍼 소유, BRIDGE, HAL 드라이버)이 있을 때 ctest와 MLP/VGG e2e에 더해 돌린다.

## 구성
| 파일 | 내용 |
|---|---|
| `run_regression.sh` | 회귀 스크립트. 케이스마다 `.mlir`을 새로 컴파일하고(vmfb 재사용 안 함) `iree-run-module`로 실행한 뒤, 첫 출력 텐서의 첫 원소를 기대값과 비교한다 |
| `models/*.mlir` | 회귀 입력 모델 13개 (아래 표) |
| `models/src/export_mlp_2layer_const.py` | `mlp_2layer_const*.mlir`을 만든 ONNX export 스크립트(출처 보존용. 스크립트가 읽지는 않음) |
| `input_device_expected.md` | 케이스별 `--input_device` 기대값 표 (`--input_device` 자동화의 비교 기준. `docs/2026-09-29/1_…/04_…`에서 작성) |

## 실행 방법
호스트에서 dev 컨테이너로 실행한다(NPU, GPU 전달은 `run-dev.sh`가 한다). 산출물은 `build/regression/regression_YYYYMMDD_HHMM/`에 쓴다:
```bash
./scripts/docker/run-dev.sh bash tests/regression/run_regression.sh
```
- 케이스 이름 필터: 마지막에 정규식 인자를 준다. 예: `run_regression.sh '^fanout3'`
- 환경 변수: `REG_OUT`(산출물 폴더 전체 경로), `REG_TIMEOUT`(케이스별 제한 시간, 기본 600초), `REG_EXTRA_CASES`(`run_case` 줄만 담은 실험용 bash 파일을 source), `IREE_BUILD`(기본 `/workspace/build`), `REG_ROOT`(산출물 상위 폴더, 기본 `$IREE_BUILD/regression`), `MODEL_DIR`(기본 이 폴더의 `models/`)
- 결과: `$REG_OUT/summary.txt`. 26케이스 기준 약 2~12분 걸린다(컴파일 캐시 없음, NPU 코드젠 시간에 따라 다름).
- **입력 배치 모드** (v8~): **기본은 두 모드를 차례로 모두 돌린다**(`REG_INPUT_DEVICE=both`). 결과는 `$REG_OUT/manual/`, `$REG_OUT/auto/`에 남고, 합친 요약은 `$REG_OUT/summary.txt`의 마지막 줄 `BOTH: manual PASS=… / auto PASS=…`이다. 한쪽만 돌리려면 `REG_INPUT_DEVICE=manual` 또는 `auto`를 준다. `manual`은 케이스에 적힌 `--input_device`를 붙이고, `auto`는 `--input_device`를 빼고 실행한다. 그러면 `iree-run-module`이 컴파일러의 reflection `amdaie.input_devices`로 배치한다(`docs/2026-09-29/2_VGG16상수묶음확인_input_device자동화/02_input_device_자동화_구현`). 두 모드 모두 26/26이 기준이다.
  - 한쪽만: `./scripts/docker/run-dev.sh env REG_INPUT_DEVICE=manual bash tests/regression/run_regression.sh`(또는 `auto`).
  - `--input_device`가 없으면 `iree-run-module`은 **reflection 자동 배치가 기본**이다. 그래서 수동 모드에서 입력을 lead에 두려는 케이스는 lead 인덱스를 명시한다(2장치 4케이스).

**판정 (PASS 조건)**: 컴파일 rc 0 **그리고** 실행 rc 0 **그리고** 첫 원소가 기대값과 상대 오차 1e-3 이내. v8까지 있던 `alloc_select` 후보 조건과 지표 열(`cmd.copy`, `xdna_copy`, `placed`)은 런타임 디버그 로그에 기대던 것이라 로그를 지우면서 v9에서 뺐다.

## 케이스 목록 (v9, 26케이스)
약어: 순서 a=amdxdna, v=vulkan, l=local(실행 시 local-task). 입력은 전부 `128x128xf32`. `--input_device`와 lead(`--device_lead_allocator`)는 스크립트에 적힌 값이다.

| # | 케이스 | 모델 | 컴파일 순서 | `--input_device` | lead | 기대값 | 무엇을 검증하는가 | 편입 |
|---|---|---|---|---|---|---|---|---|
| 1 | chain2_3dev | chain2_npufirst.mlir | a,v,l | 0,0,1 | - | 32768 | 손으로 쓴 affinity chain2(NPU matmul → GPU matmul). NPU 경계 소유 | v1 (09-28/02) |
| 2 | mlp_reduce_3dev | mlp_reduce.mlir | a,v,l | 0,0,1 | - | 1 | 자동 배정 matmul → softmax(CPU) → matmul | v1 |
| 3 | cg_topo_lead0 | mlp_reduce.mlir | l,v | 0,0,0 (v8~, lead 인덱스. v7까지 `-`) | 0 | 2 | CPU+GPU 2장치, 소유자 동률 → lead가 고름 (CPU). 수동 모드에서 CPU 입력을 GPU가 host 포인터로 import | v1 |
| 4 | cg_topo_lead1 | mlp_reduce.mlir | l,v | 1,1,1 (v8~) | 1 | 2 | 같음 (GPU) | v1 |
| 5 | mm_probe_lead0 | mm_probe.mlir | l,v | 0,0 (v8~) | 0 | 256 | 단일 matmul, CPU+GPU | v1 |
| 6 | mm_probe_lead1 | mm_probe.mlir | l,v | 1,1 (v8~) | 1 | 256 | 같음 | v1 |
| 7 | order_chain2_lav | chain2_order_lav.mlir | l,a,v | 1,1,2 | - | 32768 | 선언 순서가 바뀌어도 NPU 경계 소유가 topology로 확정 | v2 (09-28/03) |
| 8 | order_chain2auto_avl | chain2_auto.mlir | a,v,l | 0,0,1 | - | 32768 | 자동 배정 chain2, 순서 변형 | v2 |
| 9 | order_chain2auto_vla | chain2_auto.mlir | v,l,a | 0,0,2 | - | 32768 | GPU → NPU 경계(GPU가 NPU BO에 dma-buf로 씀) | v2 |
| 10 | order_chain2auto_val | chain2_auto.mlir | v,a,l | 0,0,1 | - | 32768 | 같음, 다른 순서 | v2 |
| 11 | order_mlp_reduce_vla | mlp_reduce.mlir | v,l,a | 0,0,2 | - | 1 | CPU → NPU 경계 | v2 |
| 12 | order_mlp_reduce_lav | mlp_reduce.mlir | l,a,v | 1,1,2 | - | 1 | 순서 변형 | v2 |
| 13 | order_mlp_reduce_val | mlp_reduce.mlir | v,a,l | 0,0,1 | - | 1 | 순서 변형 | v2 |
| 14 | fanout3cpu_avl | fanout3_cpu.mlir | a,v,l | 2,0,1 | - | 1 | 3후보 경계 `optimal<[npu,gpu,cpu]>`, CPU 생산자 → NPU 소유 | v3 (09-28/05) |
| 15 | fanout3_val_lead1 | fanout3.mlir | v,a,l | 0,0,1 | 1 | 16384 | 3후보 경계, GPU 생산자, lead=amdxdna | v3 |
| 16 | fanout3_avl | fanout3.mlir | a,v,l | 0,0,1 | - | 16384 | NPU 생산자 3후보 경계(09-28/06 이후 무복사) | v3 |
| 17 | mlp2const_avl | mlp_2layer_const.mlir | a,v,l | 0 | - | 16384 | 상수 가중치(장치별 전용, 비-splat) | v3 |
| 18 | mlp2constshared_avl | mlp_2layer_const_shared.mlir | a,v,l | 0 | - | 16384 | 공유 상수(NPU 상수를 GPU가 dma-buf로 읽음) | v5 (09-28/10) |
| 19 | order_chain2_vla | chain2_order_vla.mlir | v,l,a | 2,2,0 | - | 32768 | NPU로 재배치한 입력(동기 할당)을 GPU가 읽음, GPU import → NPU는 NPU가 복사 | v5 |
| 20 | order_chain2_val | chain2_order_val.mlir | v,a,l | 1,1,0 | - | 32768 | 같음, 다른 순서 | v5 |
| 21 | fanout3_val_lead0 | fanout3.mlir | v,a,l | 0,0,1 | 0 | 16384 | lead=vulkan 툴링 출력 전송: CPU 출력을 Vulkan copy_buffer BRIDGE로 옮김 | v6 (09-28/11) |
| 22 | fanout3_vla_lead0 | fanout3.mlir | v,l,a | 0,0,2 | 0 | 16384 | 같음, 다른 순서 | v6 |
| 23 | fanout3_avl_lead1 | fanout3.mlir | a,v,l | 0,0,1 | 1 | 16384 | 같음, lead=vulkan이 index 1 | v6 |
| 24 | c2e_avl_lead1 | cpu2export.mlir | a,v,l | 0,2,2 | 1 | 4 | CPU 출력 2개가 한 외부 할당에 packing되어 subview로 export됨. lead=vulkan 전송에서 **byte_offset=65536 BRIDGE import** (subspan offset 재계산 검증) | v7 (09-29/1/03) |
| 25 | c2e_avl_lead2 | cpu2export.mlir | a,v,l | 0,2,2 | 2 | 4 | 같은 모델, lead=local-task 대조군(BRIDGE 안 탐) | v7 |
| 26 | n2o_avl | npu2out.mlir | a,v,l | 0,0,0 | - | 4194304 | 한 입력을 여러 장치가 import(arg0 → {NPU, GPU}, arg2 → {GPU, NPU}). 입력은 모든 importer가 읽을 수 있는 NPU에 있어야 함. 자동 모드는 reflection `amdxdna,amdxdna,amdxdna` | v8 (09-29/2/02) |

기대값 계산: 입력 1(또는 명시된 2), 128×128 matmul이면 원소 = 128 × 값. 예: chain2 = (1·1·128)·2·128 = 32768, softmax(균등) = 1/128. 모델별 근거는 `docs/2026-09-28/02`, `05` README에 있다.
스크립트는 **첫 출력 텐서의 첫 원소만** 비교한다. fanout3 계열의 두 번째 출력(s = 0.0078125), cpu2export의 두 번째·세 번째 출력(0.0078125, 128)은 챕터별로 손으로 확인했다. cpu2export 기대값: z=2 → s2 = z·z = 4 (offset을 잃으면 s1 = 1/128을 읽음).

## 버전 이력
| 버전 | 날짜 / 문서 | 케이스 수 | 변경 |
|---|---|---|---|
| v1 | 2026-09-28, `docs/2026-09-28/02_회귀테스트_기준선_재현` | 6 | 9/28 발표 기준선 6개를 쉘 히스토리에서 복원. `.mlir`을 새로 컴파일하는 구조 |
| v2 | 2026-09-28, `docs/2026-09-28/03_소유자결정_topology_검증보강` | 13 | 디바이스 선언 순서 뒤집기 7개 추가. alloc_select 후보에 amdxdna가 있으면 FAIL 판정 추가 |
| v3 | 2026-09-28, `docs/2026-09-28/05_경계버퍼_전케이스_정리` | 17 | 3후보 경계 3개, 상수 가중치 1개 추가 |
| v4 | 2026-09-28, `docs/2026-09-28/06_ElideAsyncCopies_양방향판정_수정` | 17 | 지표 열 `cmd.copy`, `xdna_copy` 추가(판정 불변) |
| v5 | 2026-09-29, `docs/2026-09-28/10_amdxdna_동기할당_placement_기록` | 20 | 공유 상수 1개, 재배치 입력 2개 편입 |
| v6 | 2026-09-29, `docs/2026-09-28/11_Vulkan_copy_fill_update_BRIDGE(A)` | 23 | lead=vulkan 출력 전송 3개 편입 |
| v6 (이전) | 2026-09-29, `docs/2026-09-29/1_구조전환_offset회귀_input_device준비/01_문서구조전환_회귀스크립트이전` | 23 | `docs/regression/`으로 이전. 헤더 주석(위치, 실행 예)과 오래된 주석 1줄만 수정, 케이스·판정 불변 |
| v7 | 2026-09-29, `docs/2026-09-29/1_구조전환_offset회귀_input_device준비/03_BRIDGE_subspan_offset_회귀케이스` | 25 | BRIDGE subspan offset 케이스 2개(cpu2export lead=vulkan / lead=local-task) 편입 |
| v8 | 2026-09-29, `docs/2026-09-29/2_VGG16상수묶음확인_input_device자동화/02_input_device_자동화_구현` | 26 | `REG_INPUT_DEVICE=auto` 모드, `placed` 지표 열, npu2out 편입, 2장치 4케이스의 수동 `--input_device`를 lead 인덱스로 명시(자동 배치가 기본이 되어서) |
| v8 (기본 모드) | 2026-09-29, 같은 챕터 | 26 | 기본 실행을 수동+자동 둘 다로 변경(`REG_INPUT_DEVICE=both`), 한쪽만은 `manual`/`auto` |
| v9 | 2026-10-25, dev 통합(3devices-integration) | 26 | `tests/regression/`으로 이전. 런타임 디버그 로그 제거에 맞춰 판정을 rc + 기대값으로 줄이고 `alloc_select`·지표 열과 stream 단계 별도 컴파일을 뺌. 기본 `IREE_BUILD=/workspace/build`, 산출물은 `build/regression/`. `--iree-amdaie-stack-size=2048`을 뺌(기본 1024로 26/26), `--iree-hal-external-resources-mappable=true`를 뺌(CPU↔GPU 링크 표 수정 + Vulkan mapping usage 수정 뒤 어느 lead든 26/26) |
