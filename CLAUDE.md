# CLAUDE.md

## Shared host — always use the NPU/build lock

This checkout lives on `ace-amd01`, a host shared by multiple users with **one physical
NPU** and one Docker daemon. Concurrent real-NPU execution across users can race/hang the
device; concurrent full builds can overcommit RAM and freeze the host. `scripts/lock/`
prevents this with cluster-wide `flock` coordination under `/run/lock/iree-amd-aie/`
(tmpfs — self-clears on reboot, auto-released if the holder dies or Ctrl-C's).

**Before running anything that touches the real NPU or does a full build, in every
session, unconditionally:**

1. Check who's using it first: `./scripts/lock/status.sh` (non-blocking; shows `NPU: free`
   / `BUSY` + holder's user/pid/start time/command for both the NPU and build locks).
2. `./scripts/lock/check-containers.sh` — audits every running `iree-amd-aie:*` container
   for whether it actually has the lock directory bind-mounted. A container started before
   `scripts/lock/` existed (or via a hand-rolled `docker run` skipping the mount) shows
   `NEEDS RESTART` and is **invisible to the lock** — `NPU: free` does NOT mean nobody else
   is using the device if such a container is running. Confirm with that person directly,
   don't just trust `status.sh`.
3. Wrap the actual command:
   - Ad-hoc real-NPU commands (`iree-run-module --device=amdxdna`, a raw XRT/mlir-aie host
     binary, etc.): `./scripts/lock/with-npu-lock.sh <command...>`
   - `scripts/build/build.sh`, `scripts/docker/run-deploy.sh`, `scripts/docker/run-debug.sh`
     already wrap themselves automatically — no extra action needed.
   - `scripts/docker/run-dev.sh` only mounts the lock directory in; commands run inside that
     shell still need to be wrapped manually per the ad-hoc case above.

Full design/background: `docs/2026-07-06_env_setup/DEV_CONTAINER.md` §5. Quick-start:
`docs/2026-07-06_env_setup/USER_GUIDE.md`.

## 완결된 수정사항은 `docs/COMPLETED_FIXES.md`

이 저장소가 무엇을 고쳐서 지금 상태가 되었는지(이종 CPU+NPU 배치, 배치 matmul
패딩/row-overflow, `aten.bmm` int8 양자화, batch-0 lock 신원 문제)는
`docs/COMPLETED_FIXES.md` 에 누적 정리돼 있다. **끝난 것만** 들어가고,
진행 중이거나 원인 미확정인 건은 날짜별 조사 문서(`docs/2026-*.md`)에 있다.
새로 완결된 항목이 생기면 그 파일에 추가할 것.
