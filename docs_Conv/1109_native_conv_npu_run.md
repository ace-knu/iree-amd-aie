# 네이티브 conv(스칼라 코어)의 첫 NPU 실측

작성 2026-11-09

벡터화 전에, 스칼라 코어로 완성한 native conv 파이프라인을 NPU 에서 처음 돌렸다. 지시: "백터화를 도입하기 전에 스칼라 연산으로 Conv 파이프라인이 완성되면 NPU에서 실측을 진행해봐".

## 1. 요약

- native conv 는 NPU 에서 끝까지 돈다. VGG-16 출력은 CPU f32 기준 대비 corr 0.99999, maxabs 0.0257 이고 top-1 이 같다. 단일 층 프로브 4 개(OH 패딩 포함)도 corr 0.99999–1.00000 이다.
- 시간은 스칼라 코어가 다 쓴다. VGG 실행 5.83 s 중 NPU 가 5.56 s 이고, conv 13 개가 5.54 s 다. 코어당 MAC 하나에 17–31 cycle 이 든다(1.8 GHz 가정).
- 같은 조건의 im2col(기본 레시피, bf16 matmul ukernel)은 0.388 s 이고, 그 conv 는 NPU 에서 0.069 s 다. native 스칼라 conv 는 층별로 19–124 배, 합으로 80 배 느리다.
- 정확도는 native 가 더 좋다(corr 0.999991 대 0.999955, maxabs 0.026 대 0.048).
- 단일 층 모델처럼 conv 의 ReLU 가 NCHW 전치까지 맡는 dispatch 는 컴파일 오류로 멈춘다(§5.4). VGG 에는 없는 경우다.
- 멈춤(hang)이나 실행 오류는 없었다.

## 2. 조건

| 항목 | 값 |
|---|---|
| 코드 | `conv-npu` 브랜치(기준 `786d1d19`) + 미커밋 native conv 변경 전부(OH 패딩까지). `config.json` 의 commit 은 `786d1d19` 로 적힌다 |
| 바이너리 | `build/lib/libIREECompiler.so` 2026-11-09 10:53(그 뒤 소스 변경 없음 확인). 런타임 `iree-run-module` 11-03, `build-trace` 11-03(본 빌드보다 새것) |
| 실행 | `./scripts/docker/run-onnx.sh run <dir>`(npu.lock 안에서), 장치 `local-host,amdxdna-npu4`, 옵션 `ukernel all · vectorize off · numeric bf16 · pool cpu`, `conv native` 또는 `im2col` |
| 입력 | `run.py` 의 고정 seed 난수(정규분포 × 0.2) |
| 기준값 | 같은 모델을 llvm-cpu, f32 로 컴파일한 결과(`build/onnx/ref/`) |
| 전원 모드 | performance(`xrt-smi examine -r platform`, 측정 전후 확인) |
| 장치 확인 | 매 실행 전 `scripts/lock/status.sh`(NPU free)와 IRON `cs/npu_guard.py status`(전 사용자 context 0/16) |
| 호스트 부하 | 시간 측정 동안 load average 0.7–1.1. 앞서 다른 세션의 IRON selftest(가짜 백엔드, CPU 약 20 코어)가 돌았으나 측정 전에 끝났다 |
| NPU 실행 횟수 | 15 회: 프로브 4 개 × 1, VGG native 정확도 1, VGG native 시간 5(warmup 1 + runs 3 + trace 1), VGG im2col 시간 5 |

실행한 vmfb 의 PDI 는 정적 검사를 거친 것과 같다. VGG native `697f0b8a…`, im2col `2781c8d4…`(§7.8–§7.9 의 회귀 md5 와 같음). 프로브는 dump 플래그를 붙여 같은 모델을 다시 컴파일한 것과 md5 가 같고, 그 덤프로 감사·매핑 검사를 했다.

## 3. 단일 층 프로브

`log_shjo/impl/npu/gen_conv_onnx.py` 로 만든 NCHW ONNX: Conv 3×3(pad 1, bias) + ReLU + MaxPool 2×2/2. pool 은 CPU 에서 돈다.

| 모델 | VGG 대응 | 배치(정적) | 감사 | 매핑 검사 | run(1 회) | corr | maxabs |
|---|---|---|---|---|---|---|---|
| `ncp_c7_7x64x64` | OH 패딩 경로(4 청크 × 2 행, 8 행 중 1 행 패딩) | 16 코어, 4 열 | PASS | PASS | 0.06 s | 0.99999 | 0.0024 |
| `ncp_c52_14x512x512` | conv5_x | 28 코어, 7 열, 블록 8 × IC 13 | PASS | PASS | 0.20 s | 1.00000 | 0.0030 |
| `ncp_c42_28x512x512` | conv4_2·4_3 | 28 코어, 7 열, 블록 16 × IC 13 | PASS | PASS | 0.69 s | 1.00000 | 0.0030 |
| `ncp_c12_224x64x64` | conv1_2 | 32 코어, 8 열, 블록 32 × IC 9 | PASS | PASS | 0.78 s | 1.00000 | 0.0034 |

run 은 정확도 확인용 1 회이고 시간 측정이 아니다.

## 4. VGG-16

| | native(스칼라 코어) | im2col(기본) |
|---|---|---|
| 출력 디렉터리 | `build/onnx/vgg16-12-conv_native-ukernel_all-vectorize_off-numeric_bf16-pool_cpu/` | `build/onnx/vgg16-12-conv_im2col-ukernel_all-vectorize_off-numeric_bf16-pool_cpu/` |
| run(3 회 중앙값) | **5.827 s**(5.827, 5.819, 5.837) | **0.388 s**(0.413, 0.376, 0.388) |
| corr / maxabs / top-1 | 0.999991 / 0.0257 / 일치 | 0.999955 / 0.0479 / 일치 |
| NPU dispatch | 16 개 5.559 s(conv 13 개 5.544 s, FC 3 개 0.014 s) | 16 개 0.084 s(conv matmul 13 개 0.069 s, FC 3 개 0.015 s) |
| CPU dispatch | 47 개 0.022 s | 68 개 0.053 s |
| host(hwctx 생성·해제) | 0.045 s(0.011 s) | 0.078 s(0.008 s) |
| trace 합계 | 5.625 s | 0.215 s |

요약 줄:

```
vgg16-12: compile 106.9 s | run 5.83 s (median of 3) | corr 0.99999, maxabs 0.0257, top-1 match   (native)
vgg16-12: compile 122.5 s | run 0.388 s (median of 3) | corr 0.99996, maxabs 0.0479, top-1 match  (im2col)
```

### 4.1 층별 NPU 시간

trace 의 dispatch 시간(submit + wait). MAC 은 OH·OW·OC·9·IC 이고, 패딩한 IC'(bias 접기)로 센 값으로 cycle/MAC 을 냈다.

| 층 | native ms | im2col ms | 배 | MAC(G, 원래 IC) | MAC(G, IC') | 코어 | cycle/MAC(1.8 GHz 가정) |
|---|---|---|---|---|---|---|---|
| conv1_1 | 123.2 | 6.48 | 19 | 0.087 | 0.231 | 32 | 30.7 |
| conv1_2 | 738.1 | 9.81 | 75 | 1.850 | 2.081 | 32 | 20.4 |
| conv2_1 | 370.3 | 4.40 | 84 | 0.925 | 1.040 | 32 | 20.5 |
| conv2_2 | 627.0 | 5.07 | 124 | 1.850 | 1.965 | 32 | 18.4 |
| conv3_1 | 331.8 | 3.85 | 86 | 0.925 | 0.983 | 32 | 19.5 |
| conv3_2 | 604.6 | 5.08 | 119 | 1.850 | 1.907 | 32 | 18.3 |
| conv3_3 | 604.7 | 5.04 | 120 | 1.850 | 1.907 | 32 | 18.3 |
| conv4_1 | 370.2 | 4.92 | 75 | 0.925 | 0.954 | 28 | 19.6 |
| conv4_2 | 643.5 | 8.37 | 77 | 1.850 | 1.879 | 28 | 17.3 |
| conv4_3 | 643.4 | 8.50 | 76 | 1.850 | 1.879 | 28 | 17.3 |
| conv5_1 | 162.4 | 2.42 | 67 | 0.462 | 0.470 | 28 | 17.4 |
| conv5_2 | 162.7 | 2.44 | 67 | 0.462 | 0.470 | 28 | 17.5 |
| conv5_3 | 162.3 | 2.75 | 59 | 0.462 | 0.470 | 28 | 17.4 |
| 합 | 5,544.3 | 69.13 | 80 | | | | |

## 5. 해석

### 5.1 스칼라 코어가 시간을 정한다

- 코어당 MAC 하나에 17–31 cycle 이 든다. IC 조각이 작을수록(conv1_1 은 8 채널 1 조각) 루프 부담이 커서 cycle 이 늘어난다.
- 그래서 데이터 이동 쪽 관찰(필터 재전송, 블록 사이 대기, 7 열 층의 필터 집중, f32 출력)은 지금 시간에 드러나지 않는다. 7 열 층(conv4_x, conv5_x)의 cycle/MAC 은 17.3–19.6 으로 8 열 층과 같다.
- im2col 의 conv NPU 시간(0.069 s)에 맞추려면 conv 코어가 약 80 배 빨라져야 한다. 벡터 커널(T4.x)이 다음 일이다.

### 5.2 정확도

- native 가 im2col 보다 기준에 가깝다(corr 0.999991 대 0.999955, maxabs 0.026 대 0.048).
- 추정: native 스칼라 코어는 bf16 × bf16 곱을 f32 로 바로 누적한다. im2col 의 matmul ukernel 은 bfp16 에뮬레이션(블록 공유 지수)으로 곱한다. 벡터 커널도 bfp16 을 쓰면 이 차이는 줄어들 수 있다.
- native 의 차이는 입력·가중치의 bf16 반올림과 bias 의 bf16 반올림(bias 접기)에서 온다.

### 5.3 host 쪽

- native 는 CPU dispatch 가 47 개(0.022 s)로 im2col 의 68 개(0.053 s)보다 적다. im2col 의 CPU gather·pad·crop 대신 halo gather 와 insert 만 남기 때문이다.
- performance 전원 모드라 hwctx 생성·해제가 합 0.008–0.011 s 로 작다(default 모드 문서값은 dispatch 당 약 80 ms).

### 5.4 단일 층 NCHW 출력은 컴파일 오류

- Conv + ReLU 만 있는 단일 층 ONNX(출력 NCHW)는 4 개 모두 컴파일에서 멈췄다(NPU 미사용).
- conv 결과를 읽는 ReLU 가 NHWC → NCHW 전치까지 해서(출력 맵 `(d2, d0, d1)`), 전략이 "transposed output is not supported" 로 거부한다(§7.6 의 의도한 오류). 청크 패스도 이 dispatch 를 청크로 나누지 않는다.
- VGG 는 conv 뒤에 CPU pooling 이 와서 이 경우가 없다. 프로브는 MaxPool 을 붙여 피했다.
- 고치는 길: Flow 단계에서 전치를 epilogue 에서 떼어 host dispatch 로 옮기거나, T2.5(받지 못하는 dispatch 를 affinity 단계에서 host 로 보내기)로 처리한다. [1108_native_conv_impl.md](1108_native_conv_impl.md) §8 의 3 번에 더했다.

## 6. 재현

```bash
# 프로브(컴파일은 NPU 미사용)
docker run --rm --user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$PWD:/workspace" -w /workspace -e HOME=/workspace iree-amd-aie:dev-shjo \
  bash -c 'source /opt/venv/bin/activate && python3 log_shjo/impl/npu/gen_conv_onnx.py log_shjo/impl/npu/models ncp_c12_224x64x64 224 64 64'
./scripts/docker/run-onnx.sh compile log_shjo/impl/npu/models/ncp_c12_224x64x64.onnx --conv native --numeric bf16
./scripts/docker/run-onnx.sh run build/onnx/ncp_c12_224x64x64-conv_native-ukernel_all-vectorize_off-numeric_bf16-pool_cpu --runs 1 --warmup 0
# VGG(승인 필요, 5 회 실행)
./scripts/docker/run-onnx.sh run build/onnx/vgg16-12-conv_native-ukernel_all-vectorize_off-numeric_bf16-pool_cpu --verbose
```

산출물: `log_shjo/impl/npu/`(모델 생성기, 프로브 감사 로그 `audit/probe_audit.log`, dispatch 표 `run/*.txt`, 층별 표 `run/layer_table.txt`). 프로브 출력 디렉터리는 `build/onnx/ncp_*-conv_native-…/`, 감사용 덤프는 `build/onnx/ncp_*-audit/`.
