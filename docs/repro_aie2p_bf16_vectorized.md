# 재현 방법: AIE2P (npu4) bf16 matmul 벡터화

브랜치 `aie2p-bf16-vectorized`의 변경(커밋 1 `9b4f831a` 벡터화, 커밋 2 `d2846c0d` memtile
join lock 수정)을 NPU에서 직접 확인하는 절차입니다. 각 커밋이 무엇을 바꿨는지는
[패치 노트](patch_aie2p_bf16_vectorized.md)를 보세요. 재현 스크립트는
[scripts/repro/aie2p_bf16/](../scripts/repro/aie2p_bf16/)에 있습니다.


순서는 **호스트에서 준비 → 컨테이너 진입 → 컨테이너 안에서 붙여넣기**입니다. 컨테이너 안
명령은 모두 저장소 루트(`/workspace`, 진입하면 바로 이 위치)에서 실행합니다. 결과물(mlir,
vmfb, 입출력 npy)은 `build/repro_aie2p_bf16/` 아래에 생기며 git에는 들어가지 않습니다.

전제: [docs/2026-07-06_env_setup/USER_GUIDE.md](2026-07-06_env_setup/USER_GUIDE.md)대로
개발 이미지(`./scripts/docker/build-dev.sh`), `build/` 구성, `llvm-aie/`(Peano)가 준비되어
있어야 합니다.

## 0. 호스트에서: 브랜치·모델 준비 후 컨테이너 진입

```bash
# 저장소 루트에서. 작업 트리에 커밋 안 한 수정이 없어야 합니다
git fetch origin
git checkout aie2p-bf16-vectorized
git pull
git submodule update --init

# VGG16 모델 (528 MB, git에 없음. vgg16.sh에만 필요)
[ -f models/vgg16/vgg16-12.onnx ] || curl -fSL -o models/vgg16/vgg16-12.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-12.onnx

# 개발 컨테이너 진입 (NPU, memlock 설정 포함)
./scripts/docker/run-dev.sh
```

## 1. 컨테이너 안에서: 빌드

```bash
./scripts/build/build.sh --target iree-compile iree-run-module iree-benchmark-module
```

## 2. 컨테이너 안에서: 경우별 실행

```bash
# R1. bf16 하강 자체: 256x32x64 matmul (L2 웨이브 1개)
bash scripts/repro/aie2p_bf16/matmul.sh bf16 256

# R2. join 레이스, int8 (bf16 무관): 4096x32x64, 5회 실행
bash scripts/repro/aie2p_bf16/matmul.sh i8 4096

# R3. join 레이스, bf16: 4096x32x64, 5회 실행
bash scripts/repro/aie2p_bf16/matmul.sh bf16 4096

# R4. VGG16 첫 conv+relu 레이어, 벡터화 ON, 3회 실행
bash scripts/repro/aie2p_bf16/vgg16.sh layer1 on

# R5. VGG16 전체, 벡터화 ON, 3회 실행
bash scripts/repro/aie2p_bf16/vgg16.sh full on

# R6. VGG16 전체, 벡터화 OFF (비교 기준). 실행 1회 약 69초
bash scripts/repro/aie2p_bf16/vgg16.sh full off 1

# R7. 성능: 1회 실행 후 iree-benchmark-module 10회 반복 (OFF는 12분 정도 걸림)
BENCH=1 bash scripts/repro/aie2p_bf16/vgg16.sh full on 1
BENCH=1 bash scripts/repro/aie2p_bf16/vgg16.sh full off 1

# (참고) matmul도 같은 방식으로 성능 측정 가능
BENCH=1 bash scripts/repro/aie2p_bf16/matmul.sh bf16 4096 1
```

| # | 커밋 1까지 (`9b4f831a`) | 커밋 2까지 (`d2846c0d`, 브랜치 최신) |
|---|---|---|
| R1 | `RESULT: PASS`, corr 약 0.9999 | 같음 |
| R2 | `RESULT: FAIL`, 실행마다 손상 블록 14~40개, 대부분 `equal to wave+2 result` | `RESULT: PASS`, 모든 실행에서 손상 0 |
| R3 | `RESULT: FAIL` | `RESULT: PASS`, 모든 실행에서 손상 0 |
| R4 | corr 약 0.857 | corr 0.99995, `runs bit-identical: True` |
| R5 | corr 약 0.770, top-1 오답 | corr 0.97917, `top-1 gold=892 got=892 match=True`, `runs bit-identical: True` |
| R6 | corr 1.00000, top-1 정답 | 같음 |
| R7 | — | ON `real_time_mean` 약 3,300–3,700 ms, OFF 약 68,800 ms |

출력 읽는 법:

- `matmul.sh`는 실행마다 `corr`, `max|err|`, 손상된 64행 블록 수를 찍고, 그중 "같은 core row의
  2웨이브 뒤 정답과 같은" 블록 수를 괄호 안에 찍습니다. 손상 블록이 하나라도 있으면 마지막 줄이
  `RESULT: FAIL`입니다. R2·R3는 타이밍 레이스라 손상 위치와 개수가 실행마다 다릅니다.
- `vgg16.sh`는 실행마다 torch 대비 corr(전체 모델이면 top-1도)을 찍고, 마지막에 반복 실행 결과가
  비트 단위로 같은지 찍습니다.
- R7은 실행마다 흔들림(cv 10~20%)이 커서, 두 빌드를 비교할 때는 번갈아 여러 번 잽니다.

### 시간 출력

두 스크립트 모두 시간을 찍습니다.

| 출력 | 무엇을 잰 것인가 |
|---|---|
| `compile: N ms` | `iree-compile` 한 번에 걸린 시간 |
| `run i: N ms` | `iree-run-module` 프로세스 하나가 시작해서 끝날 때까지 걸린 시간. 모듈 로드, NPU 준비, 입출력 파일 읽기·쓰기가 모두 들어 있어서 순수 계산 시간보다 깁니다. 첫 실행이 보통 더 느립니다 |
| `real_time_mean` 등 (`BENCH=1`일 때만) | `iree-benchmark-module`이 같은 프로세스 안에서 함수를 10번 호출한 시간의 평균·중앙값·표준편차·변동계수. **빌드끼리 성능을 비교할 때는 이 값을 씁니다** |

예 (커밋 2 이후 빌드, npu4): `matmul.sh i8 4096`은 compile 약 2 s, run 약 0.2–0.4 s,
`real_time_mean` 약 150 ms. `vgg16.sh full on`은 compile 약 44 s, run 약 3.1–3.7 s.

## (선택) 커밋 1 상태로 레이스 확인하기

lock 수정 전 상태에서 R2가 FAIL하는 것을 보려면 커밋 1로 내려가 다시 빌드합니다. 커밋 1에는
재현 스크립트가 없어서, 먼저 git이 무시하는 `build/` 아래로 복사해 두고 거기서 실행합니다.
끝나면 브랜치 최신으로 돌아와 다시 빌드합니다. 모두 컨테이너 안에서 붙여넣으면 됩니다.

```bash
mkdir -p build/repro_aie2p_bf16 && cp scripts/repro/aie2p_bf16/*.sh build/repro_aie2p_bf16/

git checkout 9b4f831a
./scripts/build/build.sh --target iree-compile iree-run-module
bash build/repro_aie2p_bf16/matmul.sh i8 4096    # 기대: RESULT: FAIL

git checkout aie2p-bf16-vectorized
./scripts/build/build.sh --target iree-compile iree-run-module iree-benchmark-module
```

## 확인 이력

- 2026-10-22, 커밋 2 이후 빌드로 컨테이너 안에서 빌드 명령과 R1~R5, R7(`layer1`)을 실행해 "커밋 2까지" 칸과 같은 판정(PASS,
  top-1 정답)을 확인했습니다. 단, 이때 빌드에는 이 문서에 없는 미커밋 수정이 들어 있어서 bf16
  corr 값은 표보다 조금 높게 나왔습니다(R4·R5 0.99997).
- "커밋 1까지" 칸은 이 스크립트가 아니라 이전 판 재현 스크립트로 측정한 값입니다([패치 노트](patch_aie2p_bf16_vectorized.md) 커밋 2 절의
  검증 표). "(선택) 커밋 1 상태로" 절차는 아직 실행해 보지 않았습니다.
