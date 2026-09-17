#!/usr/bin/env bash
# Cluster-wide flock-based coordination for the shared host (one physical NPU, one Docker
# daemon, multiple users). See docs/2026-07-06_env_setup/DEV_CONTAINER.md §5 "한 호스트를
# 여러 유저가 공유할 때" for the two risks this addresses:
#   - concurrent real NPU execution across users can race/hang the device
#   - concurrent full LLVM builds across users can overcommit RAM and freeze/crash the host
# Locks live under /run/lock (tmpfs): they self-clear on reboot, and an flock is released
# automatically if the holding process dies or is Ctrl-C'd, so there is no stale-lock
# cleanup to do by hand. Sourced by scripts/build/build.sh, scripts/docker/run-{dev,deploy,
# debug}.sh, and scripts/lock/{status,with-npu-lock}.sh. Not `set -e`/`set -u` here — this
# file is sourced into scripts that set their own options.

LOCK_ROOT="${LOCK_ROOT:-/run/lock/iree-amd-aie}"
NPU_LOCK_FILE="$LOCK_ROOT/npu.lock"
BUILD_LOCK_FILE="$LOCK_ROOT/build.lock"

# Sticky + world-writable: any user may create/hold a lock file here, none may delete
# another's. Call this (host side) before bind-mounting $LOCK_ROOT into a container, so
# Docker doesn't auto-create it as a root-owned, non-writable directory instead.
lock_ensure_dir() {
  mkdir -p "$LOCK_ROOT" 2>/dev/null || true
  chmod 1777 "$LOCK_ROOT" 2>/dev/null || true
  lock_warn_if_unshared
}

# A container started before scripts/lock/ existed (or via a hand-rolled `docker run` that
# doesn't bind-mount $LOCK_ROOT) has no way to receive that mount after the fact — Docker
# can't add mounts to a running container. `mkdir -p` above then silently creates a
# container-LOCAL, isolated $LOCK_ROOT: flock still "succeeds" but coordinates with no one,
# which is worse than an error (looks safe, isn't). Detect that case and say so loudly,
# instead of pretending the lock is doing its job. `/.dockerenv` is Docker's own marker for
# "we are inside a container"; a host shell never has it, so this never fires on the host
# (there, $LOCK_ROOT is correctly just a plain, non-mountpoint directory under host tmpfs).
lock_warn_if_unshared() {
  if [ -f /.dockerenv ] && ! mountpoint -q "$LOCK_ROOT" 2>/dev/null; then
    {
      echo "[lock] WARNING: $LOCK_ROOT is NOT bind-mounted from the host in this container."
      echo "[lock]          Locks here are container-local only — not shared with other"
      echo "[lock]          users or containers, so build/NPU conflicts are NOT prevented."
      echo "[lock]          This container was likely started before scripts/lock/ existed."
      echo "[lock]          Stop it and start a new one via scripts/docker/run-*.sh (or add"
      echo "[lock]          -v $LOCK_ROOT:$LOCK_ROOT to your own docker run) to fix this."
    } >&2
  fi
}

# lock_status <lock_file> <label> — non-blocking check; prints "<label>: free" or
# "<label>: BUSY" plus the holder's recorded metadata (user/host/pid/start time/command).
lock_status() {
  local file="$1" label="$2"
  lock_ensure_dir
  if [ ! -e "$file" ]; then
    echo "$label: free"
    return 0
  fi
  # Read-only fd on purpose — see the note above lock_run's `200<`.
  if ( flock -n 200 ) 200<"$file" 2>/dev/null; then
    echo "$label: free"
  else
    echo "$label: BUSY"
    [ -e "$file.meta" ] && sed 's/^/  /' "$file.meta"
  fi
}

# lock_run <lock_file> <label> -- <cmd...> — acquires the lock (blocking; prints the
# current holder and waits if busy), records holder metadata, runs the command, and always
# releases on exit (including Ctrl-C or a crash — the OS drops the flock when the
# subshell's fd closes, no matter how it closes).
lock_run() {
  local file="$1" label="$2"; shift 2
  [ "${1:-}" = "--" ] && shift
  lock_ensure_dir
  # Subshell + umask 000: same reasoning as inside lock_run's main subshell below — creates
  # the lock file itself already world-writable, instead of depending on a chmod that could
  # be skipped by a kill landing between the two commands.
  [ -e "$file" ] || ( umask 000; : > "$file" 2>/dev/null; chmod 666 "$file" 2>/dev/null )
  # The fd below is opened read-only, so unlike the old `200>` it will NOT create the file.
  # Fail loudly rather than let the redirect error out with a bare "No such file".
  if [ ! -e "$file" ]; then
    echo "[lock] ERROR: could not create lock file $file — refusing to run unlocked." >&2
    return 1
  fi
  (
    set +e  # local to this subshell: guarantees our own cleanup/exit-code logic below runs
             # regardless of "$@"'s exit status, without touching the caller's `set -e`.
    # Force the *.meta file to be created world-writable regardless of the caller's umask.
    # Without this, whoever's umask creates it first (typically 644) leaves it owned by
    # them; if THAT run is later killed (Ctrl-C/crash) before the `rm -f` cleanup below runs,
    # the leftover .meta is unwritable by anyone else — and undeletable too, since the
    # sticky bit on $LOCK_ROOT blocks non-owners from removing it. The flock itself would
    # still work (mutual exclusion is unaffected), but every future holder's metadata write
    # would fail with EPERM forever, silently breaking "who's holding it" visibility.
    umask 000
    # EXIT trap (not just a manual rm at the bottom): fires even if "$@" or this subshell is
    # killed by a signal (e.g. Ctrl-C during the actual command, not just the wait above), so
    # the .meta cleanup isn't skipped in exactly the cases where it matters most.
    trap 'rm -f "$file.meta" 2>/dev/null' EXIT
    if ! flock -n 200; then
      echo "[lock] $label busy:" >&2
      [ -e "$file.meta" ] && sed 's/^/  /' "$file.meta" >&2
      echo "[lock] waiting for $label (Ctrl-C to abort)..." >&2
      flock 200 || exit 1
    fi
    printf 'user=%s host=%s pid=%s started=%s cmd=%s\n' \
      "$(id -un)" "$(hostname)" "$$" "$(date -Iseconds)" "$*" > "$file.meta" 2>/dev/null
    "$@"
    exit $?
  # Open the lock file READ-ONLY. `flock` locks the open file description, not the file's
  # contents, so a read-only fd takes an exclusive lock just fine — and a write open would
  # be *rejected* here on any host with `fs.protected_regular` set (2 on ace-amd01):
  # $LOCK_ROOT is sticky + world-writable, so the kernel refuses to open a file for writing
  # when its owner is neither the caller nor the directory's owner. That is exactly the
  # cross-user case this lock exists for: whoever created the lock file first, every OTHER
  # user's write open got EACCES, so `flock` never even ran. `lock_status` then failed
  # closed and reported "BUSY" forever, and `lock_run` blocked forever on a lock nobody
  # held. (Read-only also avoids the pointless truncation that `200>` did.)
  ) 200<"$file"
}
