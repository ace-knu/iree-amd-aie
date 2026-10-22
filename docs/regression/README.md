# 3-device 회귀 테스트 (docs/regression)

CPU(local-task) + NPU(amdxdna) + GPU(vulkan, gfx1150) 3장치 작업과 NPU 벡터화(bf16·int8, 반올림 설정 S11, 흉내 V3.1 S12)의 회귀 테스트다. 이 폴더는 날짜와 상관없는 고정 위치이고, 스크립트와 모델은 여기서만 관리한다(v6부터). v1~v6의 이전 사본은 `docs/2026-09-28/02_회귀테스트_기준선_재현/`에 있고, 더 이상 갱신하지 않는다. **현재 v10**(2026-10-22, 3단계 + V0 + V1~V7 + F1~F4).

## 1. 흐름상 위치·무엇을 하는가 (1-4 형식)
- **위치**: 컴파일러·런타임 밖의 시험이다. 현재 빌드(`$IREE_BUILD`, 기본 `/workspace/build-gpu`)의 `iree-compile`·`iree-run-module`로 모델을 **매번 새로 컴파일**하고(vmfb 재사용 안 함), 실제 NPU·GPU·CPU에서 실행해 출력을 판정한다. 컨테이너(`iree-aie-gpu:dev`, `/workspace` = 저장소 루트) 안에서 NPU 잠금으로 감싸 돈다.
- **입력**: `models/`의 모델, 실행 때 시드로 만든 입력(`ext/gen_inputs.py`, 저장소에 npy 없음), full 단계는 저장소 밖 VGG16 파일(아래 4절).
- **동작**: 단계(`REG_LEVEL`)에 따라 base(26×2) → V0(26케이스 난수 입력·출력 전체) → ext(V1~V7, full은 F1~F4)를 차례로 돈다.
- **출력**: `$REG_OUT/summary.txt`(마지막 줄 = 한 줄 요약), 단계별 폴더(`manual/`, `auto/`, `v0/`, `ext/`). 하나라도 FAIL·SKIP이면 rc 1.

## 2. 구성
| 파일 | 내용 |
|---|---|
| `run_regression.sh` | 회귀 진입점. `REG_LEVEL` 해석, base 26케이스(수동·자동 두 모드), V0(`REG_V0=1`로 자기 자신을 다시 부름), ext 호출, 요약 합치기 |
| `lib.sh` | 공용 함수 `compile_flags`(v8 `run_regression.sh` 안에 있던 것 그대로), `alloc_cands`(alloc_select 후보 장치) |
| `ext/run_ext.sh` | 새 시험 V1~V7, F1~F4 실행기(단독 실행도 됨: `bash ext/run_ext.sh <quick 또는 full> <out_dir> [필터]`) |
| `ext/gen_inputs.py` | 시드 입력 생성(V0 `default_rng(7000)`, V1·V2 10/21 생성기 시드 3, V3 `default_rng(7)`, V4~V6 `default_rng(1000+s)`, V7 `sig` 모드 = 함수 시그니처대로 `default_rng(11)` U(−1,1)·0.25) |
| `ext/judge.py` | 판정기(bfp·bfp_neg·int·ref·allclose·same·order·sha). 흉내 V3.1은 `tools/bfp16_emulator/bfp_strict_v3.py`를 **import**(복사 안 함) |
| `ext/stamp_run.py` | 명령 실행 중 stderr 줄마다 시각을 붙임(V5 순서 검사) |
| `models/*.mlir` | base·V0·V4~V6 모델 14개(v8 13개 + `chain3_ngc.mlir`) |
| `models/vec/*.mlir` | V1·V2 f32 matmul 3모양(demote로 bf16), V3 int8 matmul |
| `models/ctrlpkt/*.mlir` | V7 컨트롤 패킷 모델: `pa.mlir`(matmul 3개, 배선 base PDI 3종), `chain4.mlir`(matmul 4개, 배선 1종) |
| `vgg16/` | F1~F4 레시피 사본(`flags_b.txt`, `flags_b_vec.txt`, `flags_b_vec_ctrlpkt.txt`, `flags_nodemote_unroll0.txt`, `npu9_gpu4.map`)과 **기준 sha** `expected_sha256.txt` |
| `models/src/export_mlp_2layer_const.py` | `mlp_2layer_const*.mlir`을 만든 ONNX export 스크립트(출처 보존용) |
| `input_device_expected.md` | 케이스별 `--input_device` 기대값 표 |

## 3. 단계와 실행 방법
| 단계 | 내용 | 시간(2026-10-22 실측) | 산출물 |
|---|---|---|---|
| `base` | 26케이스 × 수동·자동 (v8과 같음, 이전 결과와 비교용) | 104 s | 16 MB |
| **`quick`(기본)** | base + **V0**(26케이스, 난수 입력·출력 전체) + **V1~V7**(ext 20행) | **245 s**(v10, v9는 219 s) | 100 MB |
| `full` | quick + **F1~F4**(VGG16 4개) | **605 s**(v10, v9는 529 s) | 102 MB(VGG vmfb 약 1.4 GB는 판정 뒤 지움) |

```bash
# 컨테이너 안(/workspace = 저장소, /work = ~/iree-work, 잠금 디렉토리·--ulimit memlock=-1·장치 마운트)
mkdir -p /home/mkchoi/iree-work/regression/<이름>      # 호스트에서 먼저
REG_LEVEL=quick REG_OUT=/work/regression/<이름> ./scripts/lock/with-npu-lock.sh bash docs/regression/run_regression.sh [케이스 이름 정규식]
```
- 이름 필터는 base·V0·ext 모두에 적용된다. 접두: base는 v8 이름 그대로이고, ext는 `vec_` `rnd_` `i8_` `mlp_` `edge_` `sem_` `ctrlpkt_` `vgg_`(F4는 `ctrlpkt_vgg16`).
- `REG_INPUT_DEVICE=manual|auto`를 직접 주면 그 모드의 **base만** 돈다(v8과 같음, `REG_LEVEL` 무시).
- 환경 변수:

| 변수 | 내용 |
|---|---|
| `REG_LEVEL` | `base` / `quick`(기본) / `full` |
| `REG_OUT` | 산출물 폴더(기본 `$REG_ROOT/regression_YYYYMMDD_HHMM`, `REG_ROOT` 기본 `/tmp` → CLAUDE.md대로 `/home/mkchoi/iree-work/regression/<이름>`을 준다) |
| `REG_TIMEOUT` | 케이스별 제한 시간(기본 600 s) |
| `REG_EXTRA_CASES` | base에 더할 `run_case` 줄 파일 |
| `IREE_BUILD` | 빌드 폴더(기본 `/workspace/build-gpu`) |
| `MODEL_DIR` | 모델 폴더(기본 `models/`) |
| `REG_PYTHON` | 판정용 python(기본 `/opt/venv/bin/python3`) |
| `REG_EMU_DIR` | 흉내 폴더(기본 `tools/bfp16_emulator`) |
| `REG_VGG_MLIR` | VGG16 모델(기본 `/work/vgg16/vgg.mlir`) |
| `REG_VGG_INPUT` | VGG16 입력(기본 `/workspace/models/vgg16/input.npy`) |
| `REG_KEEP_VGG_VMFB=1` | VGG vmfb를 지우지 않음 |
| `REG_FAULT` | **검출력 확인용**(평소 비움, 6절) |

- 입력 배치 모드(v8~): base는 수동(`--input_device` 붙임)과 자동(빼고 reflection 배치) 두 모드를 모두 돈다. V0·ext는 **한 번만** 돌고, 기존 케이스와 같은 `--input_device`를 붙인다.

## 4. 판정 (PASS 조건)
| 단계 | PASS 조건 |
|---|---|
| base | 실행 rc 0 **그리고** 첫 출력 텐서의 첫 원소가 기대값과 상대 1e-3 이내 **그리고** `alloc_select` 후보에 amdxdna가 없음(v8과 같음) |
| V0 | 실행 rc 0, CPU 기준 실행 rc 0, **출력 전부**가 원소별 rel = max\|out − ref\|/max\|ref\| ≤ 1e-5(NaN 없음), `alloc_select` 후보에 amdxdna가 없음 |
| V1 | 흉내 V3.1(`half_even` = 기본 crRnd conv_even)과 **원소 100% 비트 일치** |
| V2 | floor V3.1과 100% 비트 일치 **그리고** conv_even V3.1과 비트 일치 50% 미만(옵션이 실제로 먹는지) |
| V3 | 정수 정답 `a@b`(int64)와 원소 전부 같음(10/21 기준 sha256은 참고로 찍음) |
| V4·V6 | 출력 전부가 float64 기준 식과 rel ≤ 1e-5, NaN 없음, `alloc_select` 후보에 amdxdna가 없음 |
| V5 | 시드 0·1·2 × 2회 전부: 정확(V4와 같은 기준), 같은 시드 두 회 비트 동일, 로그 순서(NPU→GPU: GPU `vk submit`이 NPU `[T] submit '…' wait=` 뒤 / GPU→NPU: NPU `[T] submit begin`이 GPU `vk complete` 뒤). 로그가 없으면 FAIL |
| V7 | 같은 모델을 컨트롤 패킷 끔·켬으로 각각 컴파일·실행해 **출력 비트 동일**(`same`) **그리고** 켬 실행 로그의 `[T] ctx MISS` 줄 수 = 배선 종류 수(hw context 생성 수) |
| F1~F3 | 출력 npy sha256 = `vgg16/expected_sha256.txt`. vmfb sha256은 지표(입력 경로·빌드에 따라 바뀜). VGG 입력 파일이 없으면 **SKIP**(PASS로 세지 않고 rc 1) |
| F4 | F1~F3 조건 **그리고** `[T] ctx MISS` 줄 수 = 6 |

**지표 열(base, 판정에 안 씀)**: `cmd.copy`, `xdna_copy`, `placed`(v8과 같음).

## 5. 시험 목록
### 5-1. base·V0 — v8 26케이스 (아래 "케이스 목록 (v8, 26케이스)")
- **V0**는 같은 26케이스를 다음처럼 바꿔 한 번 더 돈다:
  - 상수 입력 대신 난수 입력 `v0_in<i>.npy`(128×128 f32, `default_rng(7000)`의 N(0,1)·0.3)
  - 출력은 **전부** npy로 받는다
  - 같은 모델을 CPU 하나(`local`)로 컴파일해 같은 입력으로 돌린 결과를 기준으로 원소별 비교한다
- 손으로 쓴 affinity 모델 4개(`chain2_npufirst`, `chain2_order_lav/vla/val`, 케이스 chain2_3dev·order_chain2_lav·order_chain2_vla·order_chain2_val)는 `@__device_1/2`를 가리켜 CPU 하나로 컴파일할 수 없다. 그래서 같은 함수((a@b)@c)의 자동 배정판 `chain2_auto.mlir`의 CPU 결과를 기준으로 쓴다.
- 결과 개수는 모델 `return` 줄로 센다(fanout3·fanout3_cpu·npu2out 2개, cpu2export 3개).
- 2026-10-22 실측: 26/26, rel 최대 3.85e-07.
- **왜 넣었나**: base는 첫 원소 하나만, 상수 입력으로 비교한다. 예를 들어 mlp_reduce는 softmax가 균등이라 기대값이 1이어서 행렬 곱이 틀려도 통과할 수 있다(CLAUDE.md 1-3 "10/22 발견").

### 5-2. ext — 새 시험 (v9, V7·F4는 v10)
| # | 케이스 | 모델·장치 | 입력 | 판정 | 무엇을 지키나 |
|---|---|---|---|---|---|
| V1 | `vec_bf16_{256x64x32,3072x256x1152,768x512x2304}_{rand,relu}` (6) | `models/vec/mm_f32_*`, NPU+CPU, demote, 벡터화 켬, 반올림 기본 | 10/21 생성기 시드 3(rand: N·0.5/N·0.2, relu: 희소·넓은 범위) | V3.1 비트 100% | 규민 bf16 벡터화 하강(bfp16), 반올림 기본 conv_even(S11), 흉내 V3.1(S12) |
| V2 | `rnd_floor_3072x256x1152_relu` | 위 + `--iree-amdaie-core-rounding-mode=floor` | relu | floor V3.1 100%, conv_even과 < 50% | 반올림 옵션이 실제로 코어에 들어감 |
| V3 | `i8_3136x64x576` | `models/vec/mm_i8_*`, NPU+CPU | `default_rng(7)` int8 | 정수 정답 | int8 벡터화, crRnd가 int8 결과를 바꾸지 않음 |
| V4 | `mlp_reduce_3dev_rand`, `mlp_cg_topo_lead{0,1}_rand` | `mlp_reduce.mlir`, a,v,l / l,v | r128 시드 0 | float64 `softmax(a@b)@c` rel ≤ 1e-5 | mlp_reduce 강화(base는 상수 입력 첫 원소만) |
| V5 | `sem_n2g_chain2auto_avl`, `sem_g2n_chain2auto_vla`, `sem_c2n_mlpreduce_vla` | `chain2_auto.mlir` a,v,l / v,l,a, `mlp_reduce.mlir` v,l,a | r128 시드 0·1·2 × 2회 | 정확 + 비트 동일 + 순서 | 세마포어(10/19 세션 3 T1a·T2c·T2d의 작은 판) |
| V6 | `edge_fanout3cpu_avl`, `edge_fanout3_val_lead1`, `edge_fanout3_avl`, `edge_chain3_ngc_avl` | fanout3 계열(B7·B8·B9), 새 `chain3_ngc.mlir`(NPU→GPU→CPU 직결) | r128 시드 0 | 출력 전부 float64 rel ≤ 1e-5 | 10/20 경계 버퍼 표의 미시험 경우(소비자 둘, 직결 체인) |
| V7 | `ctrlpkt_pa_3routes`, `ctrlpkt_chain4` | `models/ctrlpkt/{pa,chain4}.mlir`, NPU+CPU, 레시피 `vgg16/flags_b_vec.txt` ± `--iree-amdaie-enable-control-packet=true --iree-amdaie-packet-flow-strategy=auto` | `sig` 시드 11 | 끔과 비트 동일 + ctx MISS 3 / 1 | 컨트롤 패킷 배선 묶음(S13): entry마다 자기 base PDI 컨텍스트에서 실행, 같은 배선은 컨텍스트 공유. **pa는 고치기 전 빌드에서 corr 0.229로 틀렸다**(검출력 확인됨) |
| F1 | `vgg_2dev_novec` | 외부 `vgg.mlir`, `vgg16/flags_b.txt`(팀 bf16 2장치) | `input.npy` | sha `d5f1ebb9…` | 팀 bf16 VGG16 2장치(벡터화 끔) |
| F2 | `vgg_2dev_vec` | `flags_b_vec.txt`(위 + 벡터화 한 줄 켬) | 같음 | sha `6ccb1a0a…` | 벡터화 bf16 VGG16, conv_even(corr 0.999973) |
| F3 | `vgg_3dev_npu9` | `flags_nodemote_unroll0.txt` + `npu9_gpu4.map` | 같음 | sha `e4dcc1a0…` | f32 3장치 NPU9/GPU4/CPU46 |
| F4 | `ctrlpkt_vgg16` | `flags_b_vec_ctrlpkt.txt`(F2 + 컨트롤 패킷 두 플래그) | 같음 | sha `6ccb1a0a…`(F2와 같음) + ctx MISS 6 | VGG16 bf16 벡터화를 컨트롤 패킷으로(배선 6종, S13) |

- `chain3_ngc`의 장치 배정(matmul1 → NPU, matmul2 → GPU, softmax → CPU)은 2026-10-22에 stream IR(`stream.cmd.execute on(@__device_N)`)로 확인했다.
- **F3는 우회 레시피 기준이다**: `-unroll-threshold=0`(Peano opt 루프 펼침 억제)과 `placement-map`(dispatch별 배정 지정, 디버그 도구)을 쓴다(CLAUDE.md 3-2, 우회 W10 계열). 우회가 없어지면(f32 pad·타일 루프 ≤ 256 등 컴파일러 수정) 레시피와 기준 sha를 **바꿔야 한다**.
- **기준값 갱신 규칙(D6)**: 의도한 수치 변경(예: 반올림 기본값, 코드젠 변경)으로 VGG 출력 sha가 바뀌면 `vgg16/expected_sha256.txt` **한 곳만** 고친다. 바꿀 때는 근거(세션 문서 경로·원인·새 정확도)를 아래 버전 이력에 적고 **사용자 승인을 받은 뒤** 고친다. V1~V6는 기준을 그때 계산하므로(흉내·정수 정답·float64) 고칠 기준값이 없다.
- 미시험으로 남긴 것: GPU→CPU 전용 순서(CPU 쪽 실행 로그가 없어 로그로 가를 수 없음. 정확도만 V4 cg_topo), 같은 장치 안·상수·해제(10/20 표 B12·E3·E4).

## 6. 검출력 확인 (`REG_FAULT`, 평소 비움)
| 값 | 하는 일 | 2026-10-22 결과 |
|---|---|---|
| `v1_floor` | V1 컴파일에 floor 옵션을 넣고 판정은 그대로(conv_even) | V1 **6/6 FAIL**(비트 일치 0~0.12%) |
| `v1_v1model` | V1 판정을 옛 누산 모델(블록 합 f32 RNE)로 | relu **3/3 FAIL**(86.5~99.99%), rand 3/3 PASS. 판정이 V3.1 규칙에 민감함, rand만으로는 못 가림 |
| `v4_ref` | V4 기준을 다른 시드 입력으로 계산 | V4 **3/3 FAIL**(rel 1.34) |
| `out_corrupt` | `REG_FAULT_CASE`(기본 mlp_reduce_3dev)의 첫 출력 (1,1) 원소를 실행 뒤 바꿈(base는 stdout 텍스트, V0는 npy) | **base 수동·자동 PASS(못 잡음), V0 FAIL**(rel 1.00e-02) |

## 케이스 목록 (v8, 26케이스 — base와 V0)
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
| v9 | 2026-10-22, `docs/2026-10-22/2_회귀_벡터화_편입/02_구현` | 26×2 + V0 26 + ext 18(+F 3) | 3단계 `REG_LEVEL` = base / quick(기본) / full. **V0**(26케이스 난수 입력·출력 전체·CPU 기준), **V1~V6**(bf16 벡터화 V3.1 비트 일치, 반올림 floor 옵션, int8, mlp_reduce 강화, 세마포어 대표, 미시험 경계 버퍼 + NPU→GPU→CPU 직결 `chain3_ngc.mlir`), **F1~F3**(VGG16 출력 sha). `compile_flags`를 `lib.sh`로 옮김(내용 같음). base 26×2 결과 행은 v8과 같음(10/22 확인). 흉내는 `tools/bfp16_emulator` import(같은 날 `__main__` 보호 추가) |
| v10 | 2026-10-22, `docs/2026-10-22/6_B1_컨트롤패킷_PDI묶음/05_회귀추가` | 26×2 + V0 26 + ext 20(+F 4) | **V7**(컨트롤 패킷 배선 묶음, 명세 S13): `ctrlpkt_pa_3routes`(배선 3종, 고치기 전 빌드에서 corr 0.229로 틀리던 모델), `ctrlpkt_chain4`(배선 1종) — 끔과 비트 동일 + `[T] ctx MISS` 수. **F4** `ctrlpkt_vgg16`(F2 레시피 + 컨트롤 패킷, 출력 sha `6ccb1a0a…` = F2와 같음, ctx MISS 6. 기준 sha 추가는 사용자 승인). 새 파일 `models/ctrlpkt/{pa,chain4}.mlir`, `vgg16/flags_b_vec_ctrlpkt.txt`, `gen_inputs.py` `sig` 모드. quick 245 s 전부 PASS(manual·auto·V0 26, EXT 20). full 605 s 전부 PASS(EXT 24, F1~F4 sha 기준과 같음) |
