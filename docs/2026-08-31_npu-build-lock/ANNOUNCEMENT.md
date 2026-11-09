# [공지] 빌드/NPU 사용 자동 조율 (`scripts/lock/`) 적용 — 모든 사용자 조치 필요

이 호스트를 여러 명이 공유하면서 생기던 두 가지 문제를 자동화된 락으로 막습니다. 기존
컨테이너를 쓰고 있다면 **재시작이 필요**하니 아래 체크리스트를 따라주세요.

> **이미 `/tmp/iree-amd-aie-npu-build-lock.tar.gz`를 받아서 적용했다면 다시 받아주세요.**
> 첫 배포본의 `scripts/lock/lock-common.sh`에 메타데이터 파일 권한 버그가 있었습니다
> (작업 도중 Ctrl-C/강제종료 시 남는 `*.lock.meta` 파일이 본인만 쓰기 가능한 권한으로
> 생성돼, 이후 다른 사용자가 락을 정상적으로 획득해도 "누가 쓰고 있는지" 표시 기능이
> 영구히 깨질 수 있었음 — 실제 상호배제 자체는 영향 없었습니다). 지금 파일은 수정판입니다.

## 왜

- **NPU는 물리 디바이스 1개**(`/dev/accel/accel0`)라, 두 사람이 동시에 실제 NPU 실행을
  하면 경합해서 실패하거나 행(hang)이 걸릴 수 있습니다.
- **동시 풀빌드는 RAM을 합산 초과**시켜 호스트 전체가 먹통이 되거나 강제종료될 수 있습니다.
  (`-j`는 유저별로 독립적으로 계산되기 때문에, 내 빌드만 보면 안전해 보여도 남의 빌드와
  합치면 RAM이 부족해집니다.)

지금까지는 "NPU 실행은 한 번에 한 유저로 조율" 이라고 문서(`DEV_CONTAINER.md` §5)에만
적혀 있고 사람이 알아서 조심해야 했습니다. 이제 자동으로 막습니다.

## 무엇이 바뀌었나

`/run/lock/iree-amd-aie/` 아래 `npu.lock` · `build.lock` 두 파일을 `flock`으로 관리하고,
이 디렉토리를 **모든 컨테이너에 같은 경로로 마운트**해서 사용자마다 컨테이너가 달라도
같은 락을 공유합니다.

| 스크립트 | 락 적용 방식 |
|---|---|
| `scripts/build/build.sh` | 빌드 전체를 `build.lock`으로 자동 감쌈 |
| `scripts/docker/run-deploy.sh` | 세션 전체를 `npu.lock`으로 자동 감쌈 |
| `scripts/docker/run-debug.sh` | 세션 전체를 `npu.lock`으로 자동 감쌈 |
| `scripts/docker/run-dev.sh` | 락 디렉토리만 마운트 (수동 조합용, 아래 참고) |

- tmpfs(`/run/lock`) 기반이라 **재부팅하면 자동 초기화**됩니다.
- 락을 쥔 프로세스가 죽거나 Ctrl-C해도 OS가 즉시 풀어줍니다 — **수동 정리 불필요**.
- 충돌 시 그냥 막지 않고 **누가/언제부터/무슨 명령으로 쓰고 있는지 보여주고 대기**하다가
  자동으로 이어서 실행됩니다.

## 사용자가 해야 할 일 (체크리스트)

이번 배포는 git이 아니라 **이 호스트에 이미 다들 계정이 있다는 점을 이용해, 파일을
직접 풀어 넣는 방식**입니다 (별도 브랜치/PR 없이 바로 적용).

### 1. 파일 받기 (이 호스트에서, 본인 체크아웃 루트에서)
```bash
cd <내 iree-amd-aie 체크아웃>       # 예: ~/Projects/iree-amd-aie-dev
tar -xzvf /tmp/iree-amd-aie-npu-build-lock.tar.gz
```
`/tmp/iree-amd-aie-npu-build-lock.tar.gz`는 이 호스트에서 누구나 읽을 수 있게 올려둔
파일입니다(다른 호스트에서는 접근 안 됨). 아래 10개 파일을 **체크아웃 안 같은 경로에
덮어씁니다** (실행권한 포함, 새로 추가되는 파일은 `scripts/lock/*`):
```
scripts/lock/lock-common.sh        (신규)
scripts/lock/status.sh             (신규)
scripts/lock/with-npu-lock.sh      (신규)
scripts/lock/check-containers.sh   (신규)
scripts/build/build.sh             (덮어씀)
scripts/docker/run-dev.sh          (덮어씀)
scripts/docker/run-deploy.sh       (덮어씀)
scripts/docker/run-debug.sh        (덮어씀)
docs/2026-07-06_env_setup/DEV_CONTAINER.md   (덮어씀)
docs/2026-07-06_env_setup/USER_GUIDE.md      (덮어씀)
```
> **주의**: 저 `build.sh`/`run-*.sh` 4개를 본인이 직접 손대서 커스터마이징했다면, 덮어쓰기
> 전에 먼저 diff를 떠서 본인 수정분이 날아가지 않는지 확인하세요:
> ```bash
> tar -xzvf /tmp/iree-amd-aie-npu-build-lock.tar.gz -O scripts/build/build.sh | diff - scripts/build/build.sh
> ```
> (다른 3개 파일도 경로만 바꿔 동일하게 확인 가능)

### 2. 내 컨테이너가 영향받는지 확인 (호스트에서)
```bash
./scripts/lock/check-containers.sh
```
`NEEDS RESTART`로 뜨는 컨테이너는 **`scripts/lock/` 이 생기기 전에 뜬 컨테이너**라서
새 락 디렉토리 마운트가 없습니다. Docker는 이미 뜬 컨테이너에 마운트를 나중에 추가할
방법이 없어서, 안에서 락 스크립트를 돌려도 "성공한 것처럼" 보이지만 실제로는 그
컨테이너 안에서만 갇힌 락이라 **다른 사람과 전혀 조율되지 않습니다.**
(안에서 직접 확인하고 싶다면 1단계로 파일을 받은 뒤 `./scripts/build/build.sh`나
`./scripts/lock/status.sh`를 실행했을 때 다음 경고가 뜨는지 보면 됩니다:
`WARNING: ... is NOT bind-mounted from the host in this container.`)

### 3. 필요하면 컨테이너 재시작
`NEEDS RESTART`인 컨테이너를 내리고, 같은 방식(`run-dev.sh`/`run-deploy.sh`/`run-debug.sh`
또는 동등한 커스텀 `docker run`에 `-v /run/lock/iree-amd-aie:/run/lock/iree-amd-aie` 추가)으로
다시 띄우면 끝입니다. 소스는 호스트 bind-mount라 컨테이너를 새로 띄워도 작업 중이던 파일은
그대로 남아 있습니다 (재시작해도 안전).

## 평소 사용법

**실행/빌드 전 사용 여부 확인 (대기 없이 즉시 조회):**
```bash
./scripts/lock/status.sh
# NPU: free            (또는 BUSY + user=... pid=... started=... cmd=...)
# build: free
```

**`run-dev.sh` 셸 안에서 직접 `iree-run-module` 등을 돌릴 때 (수동으로 감싸기):**
```bash
./scripts/lock/with-npu-lock.sh build/tools/iree-run-module --device=amdxdna ...
```

`build.sh`, `run-deploy.sh`, `run-debug.sh`는 이미 자동으로 감싸져 있어 별도 조치가
필요 없습니다.

## 참고

- 상세 설계/배경: [`../2026-07-06_env_setup/DEV_CONTAINER.md`](../2026-07-06_env_setup/DEV_CONTAINER.md) §5
- 따라하기 요약: [`../2026-07-06_env_setup/USER_GUIDE.md`](../2026-07-06_env_setup/USER_GUIDE.md)
- 구현: `scripts/lock/lock-common.sh`, `status.sh`, `with-npu-lock.sh`, `check-containers.sh`
