# iree-amd-aie 컨테이너 사용자 가이드

따라만 하면 개발/배포 환경이 구축되는 최소 명령 모음이다. 배경·이유·문제해결 등 자세한 내용은
[`HOST_PREREQUISITES.md`](HOST_PREREQUISITES.md) · [`DEV_CONTAINER.md`](DEV_CONTAINER.md) ·
[`RUN.md`](RUN.md) 참고.

- 대상: Ubuntu 24.04 + AMD NPU(amdxdna) 호스트
- 구조: 소스/빌드 산출물은 호스트, 컨테이너는 빌드/실행 환경만 제공(bind-mount)

---

## Part 1. Prerequisites (호스트 준비)

### 1-1. 확인 (다 있으면 Part 2로)
```bash
. /etc/os-release; echo "$PRETTY_NAME"   # Ubuntu 24.04 계열
uname -r                                  # 커널 (6.11+/6.14+ 권장)
lsmod | grep amdxdna                      # NPU 커널 드라이버 로드됨
modinfo amdxdna | grep ^version           # 로드된 KMD 버전 (SHIM과 호환 계열인지 판단용)
ls -l /dev/accel/                         # NPU 디바이스 노드(accelN) 존재
ls /usr/lib/firmware/amdnpu               # NPU 펌웨어 존재
docker info >/dev/null 2>&1 && echo "docker OK (sudo 불필요)"  # Docker 설치 + sudo 없이 데몬 접근 확인
```

### 1-2. 없을 때 설치
```bash
# Docker (한 줄 설치)
curl -fsSL https://get.docker.com | sudo sh
# sudo 없이 docker 사용 (docker 그룹 추가)
sudo usermod -aG docker "$USER"
newgrp docker                             # 현재 셸에 그룹 즉시 적용 (또는 완전히 로그아웃 후 재로그인)
docker info >/dev/null && echo "docker OK (sudo 불필요)"   # 권한오류가 나면 재로그인 후 다시

# NPU 드라이버/펌웨어 (amdxdna가 없거나 SHIM과 다른 버전 계열일 때 설치)
#   사용할 드라이버 커밋은 iree-amd-aie README의 'Dependencies > Driver'에서 확인한다.
#   그 커밋으로 KMD+펌웨어+XRT를 빌드·설치:
#     git clone https://github.com/amd/xdna-driver.git
#     cd xdna-driver && git checkout <README가 지정한 커밋> && git submodule update --init --recursive
#     # 이후 빌드·설치는 xdna-driver README 절차를 따름 (DKMS 설치 시 커널 업데이트마다 자동 재빌드)
```
참고: 설치 후 `source /opt/xilinx/xrt/setup.sh && xrt-smi examine` 로 NPU가 인식되는지 확인한다.

### 1-3. (대용량 모델) NPU dispatch 타임아웃 상향
기본 2초 TDR이 크고 느린 dispatch를 abort(`ert state 6`)한다. 대용량 모델 전 1회 설정(재부팅 유지).
```bash
echo "options amdxdna timeout_in_sec=60" | sudo tee /etc/modprobe.d/amdxdna.conf
sudo modprobe -r amdxdna && sudo modprobe amdxdna   # 즉시 반영("in use"면 컨테이너 종료 후)
cat /sys/module/amdxdna/parameters/timeout_in_sec   # 60 확인
```

---

## Part 2. 개발용 컨테이너 (빌드 + 검증)

### 2-1. 소스 준비 (호스트)
```bash
mkdir -p ~/Projects
# dev = 개발 내용이 merge되는 브랜치
git clone --recursive --branch dev \
  https://github.com/ace-knu/iree-amd-aie.git ~/Projects/iree-amd-aie
cd ~/Projects/iree-amd-aie
git remote add upstream https://github.com/nod-ai/iree-amd-aie.git   # 원본(upstream) 추적
bash build_tools/download_peano.sh          # Peano(llvm-aie) v22 → ./llvm-aie (fork release mirror)
```
참고: recursive clone이 중간에 끊기면 이어받지 말고 폴더를 지우고 처음부터 다시 받는다.

### 2-2. 이미지 빌드 & 컨테이너 기동 (호스트, repo 루트에서)
```bash
./scripts/docker/build-dev.sh     # dev 이미지 빌드
./scripts/docker/run-dev.sh       # dev 컨테이너 진입 (NPU 자동감지, 호스트 uid, repo를 /workspace로 mount)
```

### 2-3. 빌드 & 테스트 (컨테이너 내부)
```bash
./scripts/build/configure.sh      # cmake configure (frontend 전부 ON, assertions ON, python bindings ON)
./scripts/build/build.sh          # 빌드 (-j는 코어와 RAM에 맞춰 자동 축소, RAM 부족 먹통 방지. config.sh에서 조절 가능)
ctest --test-dir build -R amd-aie --output-on-failure
# 기대: 100% tests passed, 0 failed out of 221
```
참고: configure/build는 `build/` 외에 `build-trace/`(console tracing을 켠 런타임, `iree-run-module`만)도
함께 만든다. 2-4의 `--verbose`(dispatch별 시간)가 이것을 쓴다.

### 2-4. 모델 → NPU 실행

**ONNX 모델은 실행 스크립트를 쓴다.** 컴파일 옵션이 `scripts/onnx/options.py`에 모여 있어 모두 같은
옵션으로 실행하고, 결과는 CPU(llvm-cpu) 기준값과 비교해 정확도까지 출력한다. 상세는
[`scripts/onnx/README.md`](../../scripts/onnx/README.md).
```bash
# (호스트, repo 루트) 컨테이너를 띄워 컴파일 + 실행 + 정확도 확인. ONNX 파일은 repo 안에 있어야 한다
./scripts/docker/run-onnx.sh e2e models/mlp_2layer/mlp_2layer.onnx
# 출력: mlp_2layer: compile X s | run Y s (median of 3) | corr ..., maxabs ...

# 옵션 예: bf16 변환 + dispatch별 시간(NPU/CPU 할당, 각 실행 시간)
./scripts/docker/run-onnx.sh e2e models/vgg16/vgg16-12.onnx --numeric bf16 --verbose

# 이미 컴파일한 결과(build/onnx/<model>-<options>/)만 다시 실행
./scripts/docker/run-onnx.sh run build/onnx/mlp_2layer-ukernel_all-vectorize_off-numeric_f32
```
컨테이너 안에서는 `python3 scripts/onnx/{compile,run,e2e}.py`를 같은 인자로 실행한다(아래의
`source`/`export` 후).

**그 밖의 모델(PyTorch 등)은 직접 import → compile → run 한다** (컨테이너 내부):
```bash
# 모델 import 준비: venv 활성화(python=torch/onnx 포함) + iree.compiler 경로
source /opt/venv/bin/activate
export PYTHONPATH=/workspace/build/compiler/bindings/python

# 원본 모델 → MLIR (해당하는 것 하나)
# ONNX
python -m iree.compiler.tools.import_onnx model.onnx -o model.mlir
# PyTorch (model.pt2 = torch.export로 저장한 ExportedProgram)
python - <<'PY'
import torch
from iree.compiler.extras.fx_importer import FxImporter
ep = torch.export.load("model.pt2")
imp = FxImporter(); imp.import_frozen_program(ep)
open("model.mlir", "w").write(str(imp.module))
PY

# MLIR → 컴파일
build/tools/iree-compile --iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir=/workspace/llvm-aie --iree-amdaie-stack-size=2048 \
  model.mlir -o model.vmfb

# NPU에서 실행
build/tools/iree-run-module --device=amdxdna --module=model.vmfb \
  --function=main --input="128x128xf32=1"
```
참고:
- iree-amdaie-target-device는 NPU 세대에 맞춘다 (Phoenix=npu1_4col, Strix=npu4).
- iree-amd-aie-peano-install-dir는 필수다 (환경변수만으론 iree-compile이 Peano를 못 찾음).
- iree-amdaie-stack-size는 'insufficient stack' 에러가 나는 워크로드에서 늘린다 (예: 2048).
- --function/--input은 model.mlir의 `func.func @<이름>(%arg0: tensor<...>)`를 보고 맞춘다:
  - 함수명: @<이름> (ONNX는 graph명, PyTorch fx는 보통 main)
  - 입력: 각 인자의 shape·dtype을 인자 수만큼

<details>
<summary><b>번들 예시 모델로 셋업 확인</b> — <code>models/mlp_2layer</code></summary>

> 2-layer MLP: `MatMul(x,w1) → Relu → MatMul(_,w2)` (bias 없음, 가중치는 상수, f32 128×128).
> `mlp_2layer.onnx`는 repo에 있다(생성 스크립트: `models/mlp_2layer/export_mlp_2layer.py`).
>
> ```bash
> # (호스트, repo 루트)
> ./scripts/docker/run-onnx.sh e2e models/mlp_2layer/mlp_2layer.onnx
> # 기대: corr 1.00000 (기본 --numeric f32). --numeric bf16이면 corr 0.9999x
> ```
>
> 컴파일 각 단계의 MLIR을 덤프해 변환 과정을 추적하려면 `./scripts/docker/run-debug.sh`
> (도구 `scripts/debug/pipeline_dump.py`) — 상세는 [`DEBUG_PIPELINE.md`](DEBUG_PIPELINE.md).

</details>

### 2-5. 개발 (소스 수정 → 재빌드 → 실행)

소스코드는 **호스트(`~/Projects/iree-amd-aie`)와 컨테이너(`/workspace`)가 같은 파일**이다(bind-mount).
호스트 에디터로 고치든 컨테이너 안에서 고치든 동일하게 반영된다.

수정 후에는 **컨테이너 안에서 다시 빌드·실행**한다 (ninja 증분 빌드라 바뀐 부분만 재컴파일 → 빠름):
```bash
# (컨테이너 내부) 소스 수정 후
./scripts/build/build.sh                                # 증분 재빌드 (build/ + build-trace/)
ctest --test-dir build -R amd-aie --output-on-failure   # (선택) 테스트
# 필요하면 2-4의 모델 → NPU 실행을 다시 수행
```

---

## Part 3. 배포용 컨테이너 (실행 전용 이미지)

받는 쪽이 **모델을 컴파일·실행만** 하는 self-contained 이미지. 소스/툴체인 없음.

### 3-1. 이미지 빌드 & 배포 (제공자 측, repo 루트에서)
```bash
# 고정 커밋에서 빌드 → tar.gz 저장까지 한 번에
./scripts/docker/build-deploy.sh <배포할 커밋 SHA>
```
참고: 배포할 커밋 SHA는 fork에 push된 커밋이어야 한다 (`git ls-remote https://github.com/ace-knu/iree-amd-aie.git dev` 로 확인).

### 3-2. 이미지 실행 (받는 측) — 모델 컴파일 + 실행
dev와 같은 ONNX 스크립트(`scripts/onnx`)가 이미지에 들어 있다. 받는 측이 repo를 함께 받았다면
`run-deploy.sh`가 NPU·GPU 장치를 넘기고 현재 디렉터리를 `/work`로 mount한다 (출력은 `./onnx/`).
```bash
./scripts/docker/load-deploy.sh iree-amd-aie-deploy.tar.gz   # 전달받은 이미지 로드
cd <모델이 있는 디렉터리>
<repo>/scripts/docker/run-deploy.sh e2e vgg16-12.onnx --numeric bf16
<repo>/scripts/docker/run-deploy.sh e2e vgg16-12.onnx --numeric f32 --devices local-host,amdxdna-npu4,vulkan-gfx1150
<repo>/scripts/docker/run-deploy.sh            # 인자 없으면 bash
```
옵션·출력·요약 줄은 [`scripts/onnx/README.md`](../../scripts/onnx/README.md)와 같다. `--verbose`(dispatch별
시간)는 trace 런타임이 없어 deploy 이미지에서는 쓸 수 없다.

repo 없이 raw `docker run`으로 쓸 때 (root로 실행되어 `/work`에 생기는 파일은 root 소유):
```bash
gunzip -c iree-amd-aie-deploy.tar.gz | docker load
NPU=$(ls /dev/accel/ | head -1); GPU=$(ls /dev/dri/renderD* | head -1)
# --ulimit memlock=-1: 대용량 모델 큰 BO의 memlock 제한 회피
docker run --rm -it --device=/dev/accel/$NPU --ulimit memlock=-1 --device=$GPU \
  -v "$PWD:/work" iree-amd-aie:deploy \
  python3 /opt/iree-amd-aie/scripts/onnx/e2e.py vgg16-12.onnx --numeric bf16
```
참고:
- 실행하는 iree-compile 명령은 출력 디렉터리의 `compile_cmd.sh`에 그대로 남는다.
- import/compile/run 상세(타깃 디바이스, 함수·입력)는 2-4를 참고한다.
- 배포 이미지는 `PATH`·`PYTHONPATH`·`PEANO_INSTALL_DIR`가 이미 설정돼 있어 2-4의 `source`/`export` 단계는 필요 없다.
