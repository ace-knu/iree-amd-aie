# 검증: 네이티브 conv 첫 NPU 실측 문서의 수치 재추출

작성 2026-11-09 (검증 세션)

대상: [1109_native_conv_npu_run.md](1109_native_conv_npu_run.md). 구현 세션의 표와 결론을 원자료에서 다시 뽑아 대조했다. 읽기 전용이고 NPU 는 쓰지 않았다. 재추출 결과는 `log_shjo/verify/1109_npu_run/` 에 있다.

## 1. 결론

문서의 수치는 전부 원자료와 일치한다. 불일치 0, 반올림 경계 차이 4 건(±0.01 ms, 아래 §3.2).

## 2. 원자료

| 항목 | 파일 |
|---|---|
| VGG native 실행 | `build/onnx/vgg16-12-conv_native-ukernel_all-vectorize_off-numeric_bf16-pool_cpu/{result.json,trace.log,config.json}` (12:57) |
| VGG im2col 실행 | `build/onnx/vgg16-12-conv_im2col-ukernel_all-vectorize_off-numeric_bf16-pool_cpu/{result.json,trace.log,config.json}` (12:58) |
| 프로브 4 개 | `build/onnx/ncp_*-conv_native-…/result.json` (12:55–12:56) |
| 프로브 감사 | `log_shjo/impl/npu/audit/probe_audit.log` |
| VGG 감사(코어 수·IC') | `log_shjo/impl/fixcheck/r3/vgg_audit.txt` (11-08 22:10, 최신) |
| 실행 명령 기록 | 구현 세션 transcript(`~/.claude/projects/…/8f162ec6-….jsonl`)의 Bash 호출, 03:55–03:58 UTC = 12:55–12:58 KST |

## 3. 대조

### 3.1 result.json — 일치

| | 문서 | result.json |
|---|---|---|
| native run | 5.827 (5.827, 5.819, 5.837) | 5.827 [5.827, 5.819, 5.837] |
| native corr / maxabs / top-1 | 0.999991 / 0.0257 / 일치 | 0.99999109 / 0.025738 / true |
| im2col run | 0.388 (0.413, 0.376, 0.388) | 0.388 [0.413, 0.376, 0.388] |
| im2col corr / maxabs / top-1 | 0.999955 / 0.0479 / 일치 | 0.99995532 / 0.047857 / true |
| ncp_c7 run / corr / maxabs | 0.06 / 0.99999 / 0.0024 | 0.055 / 0.9999949 / 0.002404 |
| ncp_c52 | 0.20 / 1.00000 / 0.0030 | 0.204 / 0.9999952 / 0.002962 |
| ncp_c42 | 0.69 / 1.00000 / 0.0030 | 0.687 / 0.9999951 / 0.003016 |
| ncp_c12 | 0.78 / 1.00000 / 0.0034 | 0.784 / 0.9999953 / 0.003416 |

config.json 의 commit 은 두 VGG 모두 `786d1d19`, 옵션 `conv native|im2col · ukernel all · vectorize off · numeric bf16 · pool cpu`, 장치 `local-host, amdxdna-npu4`. 문서 §2 와 같다.

### 3.2 trace.log → dispatch 표 — 일치

`scripts/onnx/dispatch_times.py` 의 `parse/format_report` 를 trace.log 에 직접 적용했다(`log_shjo/verify/1109_npu_run/vgg_{native,im2col}_dispatch.txt`). 구현 세션의 `log_shjo/impl/npu/run/*_dispatch.txt` 와 63 줄 / 84 줄 전부 같다.

| | native | im2col |
|---|---|---|
| NPU | 16 개 5.559 s | 16 개 0.084 s |
| CPU | 47 개 0.022 s | 68 개 0.053 s |
| host (hwctx) | 0.045 s (0.011 s) | 0.078 s (0.008 s) |
| total | 5.625 s | 0.215 s |
| conv 13 개 | 5,544.3 ms | 69.13 ms |
| FC 3 개 | 14.4 ms | 14.7 ms |

층별 ms 는 문서 §4.1 과 같다. 반올림 경계에 걸린 값 4 건은 원값을 적어 둔다: conv3_2 im2col 5.0845 → 문서 5.08, conv4_1 im2col 4.925 → 4.92, conv4_2 native 643.450 → 643.5, conv4_2 im2col 8.365 → 8.37. 모두 ±0.005 ms 이내다.

### 3.3 MAC · cycle/MAC — 일치

MAC = OH·OW·OC·9·IC 로 VGG 형상에서 독립 계산했고, IC' 는 감사 로그의 `IC=` 값을 썼다(conv1_1 은 3→8, 나머지는 IC+8: 72, 136, 264, 520). cycle/MAC = ms × 1.8 GHz × cores / MAC(IC'). 13 층 모두 문서 값과 소수 첫째 자리까지 같다(`log_shjo/verify/1109_npu_run/layer_table_recalc.txt`; 그 파일의 conv1_1 행은 IC'=11 로 잘못 가정한 값이라 30.7 로 읽는다).

코어 수는 `vgg_audit.txt` 의 dispatch 순서대로 `cores: 32` ×7(conv1_1–conv3_3), `cores: 28` ×6(conv4_1–conv5_3), `cores: 32` ×3(FC) 이다. 문서 표와 같다.

### 3.4 NPU 실행 횟수 15 회 — 일치

잠금(`/run/lock/iree-amd-aie/`)에는 감사 로그가 없고 커널 로그는 읽을 수 없어서(journalctl -k 권한 없음), 구현 세션 transcript 의 Bash 호출에서 `run-onnx.sh run` 을 셌다. `run.py` 는 호출마다 warmup + runs 번 실행하고 `--verbose` 면 trace 1 회를 더한다.

| 시각(KST) | 호출 | 실행 |
|---|---|---|
| 12:55:51 | `ncp_c7_7x64x64 --runs 1 --warmup 0` | 1 |
| 12:56:07 | `ncp_c52, ncp_c42, ncp_c12` 각 `--runs 1 --warmup 0` | 3 |
| 12:56:24 | VGG native `--runs 1 --warmup 0` (정확도) | 1 |
| 12:57:07 | VGG native `--verbose` (warmup 1 + runs 3 + trace 1) | 5 |
| 12:58:10 | VGG im2col `--verbose` | 5 |
| | 합 | **15** |

실행 전 확인도 기록대로다: 매 호출 앞에 `scripts/lock/status.sh` 와 IRON `cs/npu_guard.py status`, 12:56:55 와 12:58:10 에 `xrt-smi examine -r platform` 의 power mode, VGG 시간 측정 앞뒤에 `uptime` 과 CPU 30 % 이상 프로세스 목록.

## 4. 검증하지 않은 것

- 수치 정확도의 NPU 재실행(승인 없음). 위는 기록된 출력의 재해석이다.
- 전원 모드 값 자체(transcript 의 명령은 확인했으나 그 출력은 다시 읽지 않았다).
- 실행 당시 다른 사용자의 NPU 사용(사후에는 확인 불가; 문서의 npu_guard 0/16 기록만 있다).
