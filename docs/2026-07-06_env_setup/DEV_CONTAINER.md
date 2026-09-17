# iree-amd-aie 개발 컨테이너 재현 가이드 (Phase 1 / dev)

이 문서는 다른 사람/기관이 **똑같이 따라 해서** iree-amd-aie 개발 컨테이너를 구축할 수 있도록,
실제로 실행할 명령과 그 과정의 판단(문제·수정 포함)을 기록한다.

- 구조: **소스는 호스트에 두고 컨테이너에 mount하는 방식(bind-mount)**. 소스+submodule+Peano는
  **호스트가 준비**하고, 컨테이너는 `/workspace`로 mount해서 **빌드/실행만** 한다(이미지에 소스를 굽지 않음).
- 사용자: 컨테이너는 **호스트 사용자와 같은 uid/gid로 실행**한다(`docker run --user "$(id -u):$(id -g)"`
  또는 VS Code가 자동 매핑). 이렇게 해야 mount로 생성되는 파일이 호스트에서 **본인 소유로 남는다**
  (root 소유가 되어 sudo가 필요해지는 문제 방지).
- 범위: **개발(dev) 컨테이너**. 배포(runtime) 이미지는 [`RUN.md`](RUN.md) 참조(따라하기 요약은 `USER_GUIDE.md`).

> **왜 venv가 아니라 Docker인가?** 소스·산출물이 호스트에 남는다는 점은 venv와 비슷하지만, venv는
> **python 패키지만** 격리한다. 이 빌드는 clang/lld·cmake·ninja 같은 **네이티브 C/C++ 툴체인**과
> 시스템 dev 라이브러리(`libudev-dev`, `uuid-dev`), glibc까지 버전에 민감한데 — Docker는 이 **OS
> 유저스페이스 전체를 고정**해 기관마다 다른 툴체인 차이를 없앤다(말하자면 "유저스페이스 전체용 venv").

---

## 0. 전제: 호스트 준비 완료

이 컨테이너는 **유저스페이스(빌드/실행 환경)만** 담는다. 커널 드라이버(amdxdna KMD)·NPU 펌웨어·Docker
등 **호스트 준비는 먼저 [`HOST_PREREQUISITES.md`](HOST_PREREQUISITES.md)를 완료**한다(확인 항목 +
없을 때 설치 방법 포함).

**호스트 준비 이후는 모든 호스트에서 동일**하다. 호스트-특정 항목(커널/KMD/디바이스/펌웨어/Docker)만
각자 맞추면, 그 다음(**동일 fork `git clone --recursive` → `download_peano.sh` v19 → `docker build` →
컨테이너에서 고정 플래그로 빌드**)은 **정해진 소스/버전**이라 동일하다. 특히:

- **유저스페이스 XRT SHIM(amdxdna HAL 드라이버)은 git에 포함되어 빌드된다** — repo의
  `runtime/src/iree-amd-aie/driver/amdxdna/`(SHIM 소스) + `third_party/XRT` submodule에서
  `-DIREE_EXTERNAL_HAL_DRIVERS=amdxdna`로 컴파일된다. **호스트의 `/opt/xilinx/xrt`를 쓰지 않으며**
  별도 설치도 필요 없다(이미지에 그 경로 없음). → 재현 가능. 단 이 SHIM은 **호스트 KMD와 ABI 호환**돼야
  실제 NPU 실행이 안전하다(`HOST_PREREQUISITES.md`의 버전 정합).
- 나머지 변동 요인(nightly/submodule 만료, apt·base image 드리프트)은 §7에 정리.

---

## 1. 호스트: 소스 준비 (recursive clone + Peano)

**소스와 모든 submodule은 호스트가 준비**한다. 컨테이너는 빌드/실행 환경만 담고, 마운트된
소스를 사용한다. 컨테이너 설정·문서·pin이 담긴 `dev` 브랜치 기준으로 진행한다.

```bash
mkdir -p ~/Projects
# 모든 submodule까지 한 번에 받는다 (recursive). 중간에 끊기면 재개가 아니라
# 처음부터 다시 받는 것이 안전하다(부분 clone은 'Unable to find current revision' 유발).
git clone --recursive --branch dev https://github.com/ace-knu/iree-amd-aie.git ~/Projects/iree-amd-aie
cd ~/Projects/iree-amd-aie
git remote add upstream https://github.com/nod-ai/iree-amd-aie.git

# Peano(llvm-aie) - AIE core 컴파일용 (aie2xclbin, e2e/실행 테스트에 필요).
# repo 루트에서 실행하면 프로젝트 안 './llvm-aie'에 설치된다(원샷). pin은 v19(5절 참조).
bash build_tools/download_peano.sh
```

> Peano는 repo 루트의 `llvm-aie/`(다운로드 아티팩트, `.gitignore`에 등록됨)에 설치된다. 컨테이너는
> 이를 mount로 보므로, Peano가 필요한 테스트/e2e는 `PEANO_INSTALL_DIR=/workspace/llvm-aie`로 지정한다.
> 예: `docker run ... -e PEANO_INSTALL_DIR=/workspace/llvm-aie ... bash -lc 'ctest --test-dir build -R amd-aie'`.
>
> 이 방식이라 컨테이너는 submodule/Peano를 받지 않는다(그래서 `devcontainer.json`에 postCreate가 없다).
> 호스트가 recursive clone + download_peano로 전부 준비하고, 컨테이너는 순수 빌드/실행 환경이다.

---

## 2. 컨테이너 파일 (repo 루트에 커밋)

- `Dockerfile` — 4스테이지: `base-deps`(공통 빌드 환경) + `dev`(non-root, 모델 importer torch/onnx 포함)
  + `builder`(배포 빌드) + `runtime`(배포 이미지).
- `.dockerignore` — dev 스테이지는 소스를 COPY하지 않으므로 빌드 컨텍스트 최소화.
- `.devcontainer/devcontainer.json` — dev 타깃 빌드 + `/workspace` mount + `--device=/dev/accel/accel0`
  + `remoteUser: ubuntu`. (submodule/Peano는 호스트가 준비하므로 postCreate 없음.)
- `.gitignore` — `.claude/`(Claude Code, git/컨테이너 미포함) + `/llvm-aie/`·`/llvm_aie-*.dist-info/`
  (download_peano 아티팩트, 실수 커밋 방지) 추가.
- `build_tools/peano_commit_linux.txt` — Peano pin을 **v19로 고정**(fork release 미러에서 받음).
  v21은 ctest는 통과하나 npu4 f32 codegen을 깨뜨려 되돌림(§5).
- `scripts/` — 긴 명령 래퍼: `docker/`(호스트: build/run dev·deploy) + `build/`(컨테이너: configure·build) + `config.sh`.
- `docs/2026-07-06_env_setup/` — 따라하기(`USER_GUIDE.md`) + 상세: `DEV_CONTAINER.md`·`HOST_PREREQUISITES.md`·`RUN.md`.

의존성 근거(소스 대조):
- cmake: IREE 요구 `3.26...3.29` ⊇ Ubuntu 24.04 apt cmake `3.28.3` → apt 사용(별도 install_cmake.sh 불필요).
- python: repo에 버전 강제 없음 → 24.04 기본 python3(3.12) 사용.
- **python3-numpy: `IREE_BUILD_PYTHON_BINDINGS=ON`에 필수**(FindPython3 NumPy 컴포넌트). apt 패키지로
  설치해 PEP 668(pip) 문제를 피한다.
- `libudev-dev`, `uuid-dev`: `third_party/XRT`(amdxdna SHIM) 빌드 의존성 → 별도 xdna-driver clone 불필요.

---

## 3. dev 컨테이너 빌드 & 기동

### 방법 A: VS Code Dev Containers
1절에서 **호스트가 recursive clone**으로 소스+submodule을 이미 준비했으므로, 폴더
(`~/Projects/iree-amd-aie`) 열기 → "Reopen in Container"만 하면 바로 빌드 가능한 환경이 된다.

> 검증 완료: `docker build --target dev`가 약 24초에 성공. 이미지 내 툴체인 —
> cmake `3.28.3`, ninja `1.11.1`, clang/lld `18.1.3`, ccache `4.9.1`, python `3.12.3`, git `2.43.0`.
> `--user "$(id -u):$(id -g)"`로 실행하면 컨테이너가 호스트 사용자 uid로 돌아 mount 쓰기가 정상이다.

### 방법 B: CLI
```bash
cd ~/Projects/iree-amd-aie
# dev 이미지 태그는 유저별로 붙인다(-$(id -un)). Docker 데몬은 호스트에서 공유되므로 고정 태그면
# 한 유저의 재빌드가 남의 iree-amd-aie:dev 태그를 덮어쓴다. scripts/config.sh 기본값과 동일.
IMG=iree-amd-aie:dev-$(id -un)
docker build --target dev -t "$IMG" .

# NPU 디바이스 노드는 인덱스 기반(accelN)이라 고정 보장이 아니다. 먼저 확인:
#   ls /dev/accel/        (단일 NPU면 보통 accel0; 여러 개면 다를 수 있음)
NPU=$(ls /dev/accel/ | head -1)   # 예: accel0
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  --device=/dev/accel/$NPU \
  -v ~/Projects/iree-amd-aie:/workspace \
  -e HOME=/workspace \
  -e PEANO_INSTALL_DIR=/workspace/llvm-aie \
  "$IMG" bash
# --user "$(id -u):$(id -g)": 호스트 사용자 uid/gid로 실행 → 생성 파일이 본인 소유(uid 1000 아니어도 OK).
#   (본인이 clone한 소스라 소유자와 uid가 일치해 쓰기 정상. -e HOME=/workspace 는 임의 uid에서 ccache 등 HOME 의존 도구 대비.)
# -v /etc/passwd:ro -v /etc/group:ro: uid/gid를 컨테이너 안에서 '이름'으로 해석 (uid≠1000일 때
#   "I have no name!"/"groups: cannot find name" 방지). 소유권과 무관, 표시만의 문제.
# 소스+submodule은 호스트가 준비했으므로 컨테이너에서 바로 4절 빌드로 진행한다.
# (빌드만 할 거면 --device / PEANO_INSTALL_DIR 없이도 되지만, ctest amd-aie 전체 통과엔 PEANO_INSTALL_DIR 필요.)
```
> 위 두 단계는 `./scripts/docker/build-dev.sh` + `./scripts/docker/run-dev.sh`와 동일하다(스크립트가 이 플래그들을 감싼다).

---

## 4. 첫 빌드 검증 (컨테이너 내부, frontend 전부 ON)

```bash
cmake -B build -S third_party/iree -G Ninja \
  -DIREE_CMAKE_PLUGIN_PATHS=$PWD -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_INPUT_TORCH=ON -DIREE_INPUT_STABLEHLO=ON -DIREE_INPUT_TOSA=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_EXTERNAL_HAL_DRIVERS=amdxdna \
  -DIREE_BUILD_TESTS=ON -DIREE_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"

# 빌드. LLVM 컴파일은 잡당 ~4GB라 RAM 초과 시 먹통/OOM되므로 코어·RAM에 맞춰 -j를 정한다.
# 아래 -j 6은 수동 실행용 고정 예시값(안전한 기본 6 안팎)이다 — RAM/코어가 적으면 더 낮춘다.
# 스크립트 build.sh는 이 -j를 min(6, nproc-2, RAM_GB/4)로 자동 계산한다.
cmake --build build -j 6

# 테스트. -j 없으면 순차(안전). 221개가 순차로도 수초라 -j 불필요.
ctest --test-dir build -R amd-aie --output-on-failure
# 기대: 100% tests passed, 0 failed out of 221
# (IREE_ENABLE_ASSERTIONS=ON + PEANO_INSTALL_DIR 필수 — 하나라도 빠지면 일부 테스트 실패)
```

> configure + build는 `./scripts/build/configure.sh` + `./scripts/build/build.sh`와 동일하다(스크립트가 위 플래그를 감싼다).

빌드가 통과하면 dev 기반이 확정된다(배포 이미지도 동일 base-deps를 공유하므로 재현된다).

> 검증 결과(구축 당시, `--cpus=6 -j 6`): **빌드 성공(rc=0, 약 25분)**. `ctest -R amd-aie`는
> Release(assertions OFF) 빌드에서 **214개 중 201개 통과(94%)**.
>
> 실패 13개는 **Peano와 무관**하다. 전부 `aie_cdo_gen_test`/`aie_elf_files_gen_test`/`iree-opt`를
> 실행하는 테스트이고, 기대하는 `XAIE API: ... with args:` 트레이스는 `iree_aie_runtime.h`의
> **`LLVM_DEBUG(...)`** 로 출력된다. `LLVM_DEBUG`는 **assertions 빌드에서만** 활성이므로, Release
> (`-DNDEBUG`) 빌드에서는 트레이스가 사라져 FileCheck가 실패한다(이 테스트들엔 `REQUIRES: asserts`
> 가드가 없어 skip이 아니라 fail). upstream CI도 `assertions:[ON,OFF]` 매트릭스로 빌드하며 이 트레이스
> 테스트들은 ON에서만 통과한다.
>
> → 전부 통과시키려면 **`-DIREE_ENABLE_ASSERTIONS=ON`으로 재빌드**한다(NDEBUG 제거 = LLVM 포함 전체
> 재컴파일). dev 컨테이너에는 assertions ON이 사실 더 적합한 기본값이다.
>
> **최종 검증**: assertions ON 재빌드(rc=0) 후 `ctest -R amd-aie`는 **211/214**로 개선(트레이스
> 10개 통과). 남은 3개(`ctrlpkt_gen`, `elf_pm_size`, `convert_device_to_control_packets`)는
> `aie_elf_files_gen_test`가 `PEANO_INSTALL_DIR`을 요구하는 진짜 Peano 의존 테스트였고, Peano v21
> (`21.0.0.2026070201`)을 `PEANO_INSTALL_DIR`로 지정하니 통과 → **214/214 전부 통과**. (Peano v21은
> 5절대로 pin에 반영됨.)

### 4-1. 모델 e2e 테스트 (ONNX/PyTorch → NPU) — dev도 배포와 동일하게 가능

dev 이미지는 배포 이미지의 모델 importer(torch/onnx)를 `/opt/venv`에 포함한다(배포 = dev의 부분집합).
빌드를 `IREE_BUILD_PYTHON_BINDINGS=ON`으로 한 뒤(위 4절 명령이 이미 그럼), 원본 모델을 NPU까지 돌릴 수 있다:

```bash
# venv 활성화(python=torch/onnx 포함) + iree.compiler 경로 (dev는 빌드 트리라 직접 지정)
source /opt/venv/bin/activate
export PYTHONPATH=/workspace/build/compiler/bindings/python

# 원본 모델 -> MLIR
python -m iree.compiler.tools.import_onnx model.onnx -o model.mlir   # ONNX
# PyTorch는 torch.export.load("model.pt2") + fx_importer (USER_GUIDE 2-4 / RUN.md 참조)

# MLIR -> NPU. iree-compile은 peano 경로를 '플래그로' 받아야 한다(환경변수만으론 부족)
build/tools/iree-compile --iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir=/workspace/llvm-aie --iree-amdaie-stack-size=2048 \
  model.mlir -o model.vmfb           # 일부 워크로드는 stack-size 증대 필요
build/tools/iree-run-module --device=amdxdna --module=model.vmfb \
  --function=main --input="1x3x224x224xf32=1"   # 함수명/입력은 model.mlir func.func 시그니처에 맞춤(예시)
```
> torch는 `2.12.1+cpu`로 pin됨(dev/배포 동일). 배포 실행 세부는 `RUN.md` 참조.

---

## 5. 알려진 이슈 / 주의

- **assertions 빌드 (ctest amd-aie 100% 통과 조건)**: 다수 Target/Transforms 테스트가 `LLVM_DEBUG`
  트레이스(`XAIE API: ...`)를 FileCheck한다. `LLVM_DEBUG`는 assertions 빌드에서만 활성이라, Release
  (`-DNDEBUG`) 빌드에서는 이 13개가 실패한다. `-DIREE_ENABLE_ASSERTIONS=ON`으로 빌드하면 통과한다.
  (dev 컨테이너 기본값으로 권장. upstream CI도 `assertions:[ON,OFF]` 매트릭스 사용.)
- **Peano pin을 v19로 고정 + fork release 미러 (해결)**: pin 이력이 두 번 바뀌었다.
  (1) 원래 pin `llvm_aie==19.0.0.2025052701+31d2aa6e`(v19)는 Xilinx/llvm-aie **nightly 자산 만료로 삭제**되어
  `download_peano.sh`로 받을 수 없었다. (2) 그래서 당시 공개돼 있던 `21.0.0.2026070201+4617c73e`(v21)로
  bump했고 ctest(위 3개 Peano 테스트 포함 amd-aie **214/214**)는 통과했다. **그러나 v21은 npu4 f32
  matmul을 수치적으로 깨뜨린다** — 동일 MLIR·플래그·NPU에서 v21은 쓰레기값(`2.95E20/INF/0`), v19는
  정답(128)임을 IREE/Peano 2×2 교차검증으로 확정했다(회귀는 IREE가 아니라 **Peano 코어 codegen**).
  (3) 따라서 **v19로 되돌렸다.** nightly는 v19를 이미 prune했으므로, v19를 **ace-knu fork의 GitHub
  Release 자산으로 미러링**(tag `peano-v19`, `peano-v19-linux.tar.gz`)하고 `download_peano.sh`가 그
  tarball을 받아 `./llvm-aie`에 풀도록 수정했다(§7 미러링 옵션 채택). release 자산은 nightly와 달리
  자동 prune되지 않아 재현 가능하다.
  주의: ctest 214/214는 f32 **e2e codegen 정확도**를 잡아내지 못한다(그래서 v21이 통과했었다). bf16 정확
  경로는 npu4 ukernel이 필요하며 별도 과제다. (`peano_commit_windows.txt`는 미변경 — linux만 대상.)
- **submodule은 무중단 recursive clone**: `git clone --recursive`를 중간에 끊으면 부분 clone이 남아
  재실행 시 `fatal: Unable to find current revision`가 난다. 끊겼으면 해당 submodule(또는 clone 전체)을
  삭제하고 처음부터 다시 받는다.
- **NumPy**: `IREE_BUILD_PYTHON_BINDINGS=ON`은 NumPy가 없으면 configure 단계에서 실패한다
  (`Could NOT find Python3 ... NumPy`). base-deps의 `python3-numpy`로 해결.
- **non-root 소유권**: 컨테이너를 root로 실행하면 mount로 생성된 build 산출물이 호스트에서
  `root:root`가 되어 호스트 사용자가 sudo 없이 다루지 못한다. 그래서 **호스트 사용자 uid/gid로 실행**한다
  — CLI는 `docker run --user "$(id -u):$(id -g)"`, VS Code devcontainer는 `updateRemoteUserUID`(기본 true)가
  이미지의 `ubuntu` 사용자 uid를 호스트 uid로 자동 remap. (이미지 기본 사용자는 `ubuntu`이지만 uid 1000에
  고정 의존하지 않는다.)
  단, uid/gid가 1000(이미지의 `ubuntu`)이 아니면 컨테이너 passwd에 이름이 없어 셸이 `I have no name!` /
  `groups: cannot find name`을 낸다(소유권은 정상, 표시만의 문제). run-dev.sh/run-debug.sh가
  `-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro`로 호스트 이름을 해석해 이를 없앤다.
- **frontend 전부 ON**: `IREE_INPUT_STABLEHLO/TORCH/TOSA=ON`은 관련 submodule(stablehlo, torch-mlir)이
  필요하다. recursive clone이 이를 포함한다.
- **제약 호스트(노트북 등)에서의 자원 제한**: 이 빌드는 LLVM 포함이라 전 코어를 오래 100%로 점유한다.
  코어가 적거나 발열/응답성이 걱정되는 호스트(예: 노트북)에서는 `-j`를 낮추고 컨테이너 CPU를 제한한다.
  예: `docker run --cpus=6 ... iree-amd-aie:dev-$(id -un) bash -lc 'nice -n 19 cmake --build build -j 6'`.
  이는 **실행 단위(per-run) 선택**일 뿐 이미지/`devcontainer.json`에는 넣지 않는다(빌드 서버는 풀 병렬 사용).
  (구축 당시 실측: `-j 24` 풀 병렬은 12코어/24스레드 노트북에서 CPU 기아 + RAM 초과로 먹통/강제종료 —
  `nproc`는 SMT 스레드(24)를 세므로 `nproc-2`도 과다다. LLVM 컴파일은 잡당 ~2-4GB라 RAM이 병목이다.
  그래서 config.sh는 `min(6, nproc-2, RAM_GB/4)`로 캡한다. `--cpus=6 -j 6`으로 응답성 유지하며 정상 진행.)
- **한 호스트를 여러 유저가 공유할 때**: rootful Docker는 데몬이 시스템 전역 하나라 이미지 태그
  네임스페이스를 전 유저가 공유한다. 그래서:
  - **dev 이미지 태그는 유저별**(`iree-amd-aie:dev-$(id -un)`, `config.sh` 기본값)이다. dev는 소스를
    굽지 않아 같은 브랜치면 이미지가 동일하지만, 유저마다 브랜치(Dockerfile)가 다르거나 한 유저가
    재빌드하면 고정 태그는 남의 `iree-amd-aie:dev`를 조용히 덮어쓴다. 유저별 태그로 이를 없앤다
    (`IMAGE_DEV=...`로 override 가능).
  - **deploy 이미지 태그는 공유(`iree-amd-aie:deploy`)로 둔다** — 연구실이 정한 커밋 하나로 빌드해
    넘기는 canonical 이미지라 머신당 하나가 의도다. 단 `build-deploy.sh`/`load-deploy.sh`는 그 공유
    태그를 덮어쓰므로, 서로 다른 커밋을 굽는 경우엔 `IMAGE_DEPLOY=...`로 구분한다.
  - **NPU는 물리 디바이스 1개**(`/dev/accel/accel0`)다. 두 유저가 `run-deploy.sh`/`run-debug.sh`로
    실제 NPU 실행을 동시에 하면 경합해 실패/행 가능 — **NPU 실행은 한 번에 한 유저**로 조율한다
    (빌드만이면 무관). 두 유저의 동시 LLVM 빌드는 `-j` 캡이 유저별 독립이라 합산 RAM 초과 위험이
    있으니 공유 시 `JOBS`를 낮춘다.
  - **위 두 조율은 이제 수동이 아니라 자동이다** (`scripts/lock/`): `/run/lock/iree-amd-aie/`
    아래 `npu.lock`·`build.lock` 두 개를 `flock`으로 관리하고, 이 디렉토리를 모든 컨테이너에
    같은 경로로 bind-mount해서 유저마다 컨테이너가 달라도 락을 공유한다. `build.sh`는 빌드를,
    `run-deploy.sh`/`run-debug.sh`는 세션 전체를 자동으로 이 락으로 감싼다(둘 다 반드시 실제
    NPU를 건드리는 흐름이라). `run-dev.sh` 안에서 `iree-run-module`을 손으로 돌릴 때는
    `./scripts/lock/with-npu-lock.sh <명령...>`로 감싼다. tmpfs 기반이라 재부팅하면 자동
    초기화되고, 락을 쥔 프로세스가 죽거나 Ctrl-C해도 OS가 즉시 풀어준다(수동 정리 불필요).
    실행 전 사용 여부만 안 기다리고 확인하려면:
    ```bash
    ./scripts/lock/status.sh
    # NPU: free            (또는 BUSY + user=... pid=... started=... cmd=...)
    # build: free
    ```

---

## 6. 배포 이미지 (Phase 2)

받는 기관이 **실행만** 하도록, builder에서 특정 커밋을 clone/build하고 runtime에 산출물만 `COPY`하는
self-contained 이미지를 만든다(모델 importer 포함). 빌드·tar.gz 배포·NPU 실행 절차는 [`RUN.md`](RUN.md),
따라하기 요약은 [`USER_GUIDE.md`](USER_GUIDE.md) Part 3 참조.

---

## 7. 재현성 견고성 / 만료 위험 (Peano 미러링은 적용됨; 나머지는 문서화만)

이 셋업은 clone 시 외부 소스 여러 곳 + nightly 바이너리 + 버전 미고정 apt에 의존한다. "몇 년 뒤 /
다른 기관에서 그대로 재현"을 보장하려면 아래 위험을 인지하고, 필요 시 미러링/스냅샷을 도입한다.

**만료·소멸 위험 (구축 당시 실측 기준):**

| 의존물 | 출처 | 위험 | 이유 |
|---|---|---|---|
| Peano (llvm-aie) | **ace-knu fork Release** (미러) | 낮음 | v19를 fork release 자산으로 미러(§5). release는 자동 prune 안 됨 |
| XRT, mlir-air | **nod-ai fork**, feature 브랜치 | 중간 | 브랜치 force-push/삭제 시 pin 커밋 orphan → `clone --recursive` 영구 실패 |
| openssl-cmake | viaduck (소규모 3rd-party) | 중간 | 소규모 저장소 소멸 가능성 |
| iree, iree-org/llvm-project | iree-org (fork) | 낮음~중간 | rc 커밋, fork |
| apt 패키지 (cmake 등), `ubuntu:24.04` | Ubuntu 24.04 archive (버전 미고정) | 낮음 | 24.04 내 드리프트 + EOL 후 old-releases 이동(수년) |

**개선 옵션 (도입 시):**
- **Peano 미러링 — 적용됨(§5)**: v19를 ace-knu fork의 **GitHub Release 자산**(tag `peano-v19`,
  `peano-v19-linux.tar.gz`)으로 올리고(release 자산은 nightly와 달리 자동 prune 안 됨) `download_peano.sh`가
  그 tarball을 받게 수정했다.
- **전체 소스트리 스냅샷(가장 견고)**: 완전히 채워진 트리(submodule + peano)를 tarball로 durable 보관 →
  live clone 실패 시 fallback. submodule orphan까지 대비. 수 GB.
- **base 이미지/apt 고정(선택)**: `ubuntu:24.04@sha256:...` 다이제스트 pin, 필요 시 apt 버전 pin.

> 현재는 v19 pin(fork release 미러)으로 재현 가능한 상태이며, 소스트리 스냅샷/base 이미지 고정은
> **환경을 개선·정리할 때 재검토**한다.
