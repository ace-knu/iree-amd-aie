# --input_device 기대값 표 (회귀 v8, 26케이스)

`--input_device` 자동화의 비교 기준이다. 자동화한 뒤에는 각 케이스에서 툴링이 고른 장치가 이 표의 "현재 `--input_device`" 열과 같아야 하고, `--input_device` 없이 PASS해야 한다.
작성: `docs/2026-09-29/1_구조전환_offset회귀_input_device준비/04_input_device_자동화_설계` (생성 스크립트 `input_device_table.py`, 입력 = v7 회귀의 `<key>.stream.mlir`).

- "IR import 장치": public 함수 인자 `%argK`를 쓰는 `stream.tensor.import on(#hal.device.affinity<@__device_N>) %argK`의 N. 컴파일 순서(`--iree-hal-target-device`) 기준 인덱스다. 한 인자가 여러 장치에서 import되면 `0+1`처럼 적는다.
- 2장치(l,v) 케이스는 현재 `--input_device`를 쓰지 않는다. 입력은 lead 장치에 할당되고 GPU가 host 포인터로 import한다.

| case | 컴파일 순서 | 현재 `--input_device` | IR import 장치 (인자별) | 인자별 장치 집합 크기 | 일치 |
|---|---|---|---|---|---|
| chain2_3dev | amdxdna,vulkan,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| mlp_reduce_3dev | amdxdna,vulkan,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| cg_topo_lead0 | local,vulkan | - | 1,1,1 | 1 | (수동 지정 없음) |
| cg_topo_lead1 | local,vulkan | - | 1,1,1 | 1 | (수동 지정 없음) |
| mm_probe_lead0 | local,vulkan | - | 1,1 | 1 | (수동 지정 없음) |
| mm_probe_lead1 | local,vulkan | - | 1,1 | 1 | (수동 지정 없음) |
| order_chain2_lav | local,amdxdna,vulkan | 1,1,2 | 1,1,2 | 1 | ✓ |
| order_chain2auto_avl | amdxdna,vulkan,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| order_chain2auto_vla | vulkan,local,amdxdna | 0,0,2 | 0,0,2 | 1 | ✓ |
| order_chain2auto_val | vulkan,amdxdna,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| order_mlp_reduce_vla | vulkan,local,amdxdna | 0,0,2 | 0,0,2 | 1 | ✓ |
| order_mlp_reduce_lav | local,amdxdna,vulkan | 1,1,2 | 1,1,2 | 1 | ✓ |
| order_mlp_reduce_val | vulkan,amdxdna,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| fanout3cpu_avl | amdxdna,vulkan,local | 2,0,1 | 2,0,1 | 1 | ✓ |
| fanout3_val_lead1 | vulkan,amdxdna,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| fanout3_avl | amdxdna,vulkan,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| mlp2const_avl | amdxdna,vulkan,local | 0 | 0 | 1 | ✓ |
| mlp2constshared_avl | amdxdna,vulkan,local | 0 | 0 | 1 | ✓ |
| order_chain2_vla | vulkan,local,amdxdna | 2,2,0 | 2,2,0 | 1 | ✓ |
| order_chain2_val | vulkan,amdxdna,local | 1,1,0 | 1,1,0 | 1 | ✓ |
| fanout3_val_lead0 | vulkan,amdxdna,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| fanout3_vla_lead0 | vulkan,local,amdxdna | 0,0,2 | 0,0,2 | 1 | ✓ |
| fanout3_avl_lead1 | amdxdna,vulkan,local | 0,0,1 | 0,0,1 | 1 | ✓ |
| c2e_avl_lead1 | amdxdna,vulkan,local | 0,2,2 | 0,2,2 | 1 | ✓ |
| c2e_avl_lead2 | amdxdna,vulkan,local | 0,2,2 | 0,2,2 | 1 | ✓ |

## 회귀 밖 참고 케이스 (한 입력을 여러 장치가 import)
| case | 컴파일 순서 | IR import 장치 (인자별) | 규칙으로 정한 배치 | 결과 |
|---|---|---|---|---|
| npu2out (`docs/2026-09-29/1_…/03_…/models/npu2out.mlir`) | amdxdna,vulkan,local | 0+1, 0, 1+0 | **0,0,0** (나머지 importer 전원이 읽을 수 있는 장치 = NPU) | 0,0,0 → PASS 4194304 / 0,0,1 → NPU assertion (`buffer.cc:249`) |

## v8 갱신 (2026-09-29, 세션 2 챕터 02): 자동 배치 결과
- 컴파일러 reflection(`amdaie.input_device_ordinals`)은 위 "IR import 장치" 열과 **26/26 같다**(npu2out은 규칙대로 `0,0,0`).
- 수동 모드에서 2장치 4케이스는 이제 `--input_device`에 lead 인덱스를 적는다(cg_topo_lead0 `0,0,0`, cg_topo_lead1 `1,1,1`, mm_probe_lead0 `0,0`, mm_probe_lead1 `1,1`). 플래그가 없으면 자동 배치가 기본이 되기 때문이다.
- 자동 모드 기대 배치(`placed` 열, `L` = lead에 둠): 3장치 22케이스는 수동 모드와 같다. 2장치 lead=local-task 두 케이스(cg_topo_lead0, mm_probe_lead0)만 입력이 GPU로 재배치된다(`1,1,1`, `1,1`). 이것은 설계 결정 (A)이다(reflection = IR import 장치).
