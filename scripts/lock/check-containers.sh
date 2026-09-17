#!/usr/bin/env bash
# [host] Audit every currently-running iree-amd-aie container (any user, any image tag) for
# whether it actually has the shared lock directory bind-mounted. A container started before
# scripts/lock/ existed — or via a hand-rolled `docker run` that skips the mount — has its own
# container-local, isolated lock dir: it does NOT coordinate with anyone else, silently. Such
# containers must be stopped and recreated (via scripts/docker/run-*.sh, or by adding
# `-v $LOCK_ROOT:$LOCK_ROOT` to a custom `docker run`) before the lock covers them.
# Run this any time after rolling out scripts/lock/ to see who still needs to restart.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/lock-common.sh"

printf '%-32s %-26s %-10s %s\n' "CONTAINER" "IMAGE" "LOCK" "MOUNT-POINT-INSIDE-CONTAINER"
found=0
for id in $(docker ps -q); do
  image="$(docker inspect "$id" --format '{{.Config.Image}}')"
  case "$image" in
    iree-amd-aie:*) ;;
    *) continue ;;
  esac
  found=1
  name="$(docker inspect "$id" --format '{{.Name}}' | sed 's#^/##')"
  if docker inspect "$id" --format '{{range .Mounts}}{{.Destination}}{{println}}{{end}}' \
      | grep -qx "$LOCK_ROOT"; then
    status="ok"
  else
    status="NEEDS RESTART"
  fi
  printf '%-32s %-26s %-10s %s\n' "$name" "$image" "$status" "$LOCK_ROOT"
done

[ "$found" -eq 1 ] || echo "(no running iree-amd-aie:* containers found)"
