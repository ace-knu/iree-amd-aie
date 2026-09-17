#!/usr/bin/env bash
# [host] Who is loading this shared box right now, and is it safe to start a
# timing-sensitive NPU run?
#
# Why this exists: scripts/lock/ only serializes the two things that can break
# the host outright (the NPU device, and full builds). Plenty of other work eats
# cores without ever touching those locks -- an IDE's C++ indexer (cpptools) on
# an LLVM-sized tree routinely takes 8+ cores, and a hand-rolled `ninja` that
# skips scripts/build/build.sh takes the rest. Since the bug we chase is a
# timing race, that background load can change results, so check here first.
#
# Reads CPU from /proc/stat deltas (a real instantaneous sample), NOT ps's %cpu
# -- ps reports an average over each process's whole lifetime, so a spike that
# ended ten minutes ago still shows up as "busy" (measured 591% via ps vs 224%
# via a live sample on the same instant).
set -uo pipefail

# Sampling window. Keep it short: `watch` redraws only after the command exits,
# so a long window makes the display feel frozen. 0.3s is enough to separate
# "someone is indexing right now" from "idle".
INTERVAL="${1:-0.3}"

# --- per-user instantaneous CPU, from two /proc samples -----------------------
declare -A t0 name0
while read -r pid utime stime user; do
  t0["$pid"]=$((utime + stime)); name0["$pid"]=$user
done < <(ps -eo pid=,user= --no-headers | while read -r pid user; do
  read -r _ _ _ _ _ _ _ _ _ _ _ _ _ ut st _ < /proc/"$pid"/stat 2>/dev/null \
    && echo "$pid $ut $st $user"
done 2>/dev/null)

sleep "$INTERVAL"

declare -A cpu
while read -r pid utime stime user; do
  prev="${t0[$pid]:-}"
  [ -n "$prev" ] || continue
  d=$((utime + stime - prev))
  [ "$d" -gt 0 ] && cpu["$user"]=$(( ${cpu[$user]:-0} + d ))
done < <(ps -eo pid=,user= --no-headers | while read -r pid user; do
  read -r _ _ _ _ _ _ _ _ _ _ _ _ _ ut st _ < /proc/"$pid"/stat 2>/dev/null \
    && echo "$pid $ut $st $user"
done 2>/dev/null)

HZ=$(getconf CLK_TCK)
CORES=$(nproc)

printf '%-10s %9s %9s %7s  %s\n' USER "CPU(%)" "RAM(GB)" PROCS "top process"
printf -- '%.0s-' {1..64}; echo
for u in "${!cpu[@]}"; do
  pct=$(awk -v t="${cpu[$u]}" -v hz="$HZ" -v i="$INTERVAL" 'BEGIN{printf "%.1f", 100*t/hz/i}')
  awk -v p="$pct" 'BEGIN{exit !(p>1)}' || continue
  ram=$(ps -eo user=,rss= --no-headers | awk -v u="$u" '$1==u{s+=$2} END{printf "%.2f", s/1048576}')
  n=$(ps -eo user= --no-headers | grep -cx "$u")
  topp=$(ps -eo user=,pcpu=,comm= --no-headers --sort=-pcpu | awk -v u="$u" '$1==u{print $3; exit}')
  printf '%-10s %9s %9s %7s  %s\n' "$u" "$pct" "$ram" "$n" "${topp:-?}"
done | sort -k2 -rn

# --- verdict ------------------------------------------------------------------
read -r l1 _ < /proc/loadavg
mem_avail=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
echo
printf 'load(1m) %s / %s cores = %s%%   RAM available %sGB\n' \
  "$l1" "$CORES" "$(awk -v l="$l1" -v c="$CORES" 'BEGIN{printf "%.0f", 100*l/c}')" "$mem_avail"

# 판정은 방금 측정한 CPU 합계로 한다. load(1m) 은 지난 1분 평균이라 지금 시작된
# 부하에는 늦게 반응하고, 이미 끝난 스파이크는 계속 높게 남는다.
others=0
for u in "${!cpu[@]}"; do
  [ "$u" = "$(id -un)" ] && continue
  others=$(awk -v a="$others" -v t="${cpu[$u]}" -v hz="$HZ" -v i="$INTERVAL" \
           'BEGIN{printf "%.1f", a + 100*t/hz/i}')
done
printf '내 것 제외한 타 사용자 CPU 합계: %s%% (코어 %s개 = %s%%)\n' \
  "$others" "$CORES" "$((CORES*100))"

lowmem=$(awk -v m="$mem_avail" 'BEGIN{print (m < 8) ? 1 : 0}')
# 코어 25% 이상을 남이 쓰고 있으면 타이밍 실험에는 시끄럽다고 본다
noisy=$(awk -v o="$others" -v c="$CORES" 'BEGIN{print (o > c*25) ? 1 : 0}')
if [ "$lowmem" = 1 ]; then
  echo "=> BUSY: RAM 여유 ${mem_avail}GB (8GB 미만) — 빌드/실험 미룰 것"
elif [ "$noisy" = 1 ]; then
  echo "=> NOISY: 타 사용자가 코어를 많이 쓰는 중 — 타이밍에 민감한 NPU 실험은 미룰 것"
  echo "   (정확도만 보는 실험이나 컴파일은 진행해도 됨)"
else
  echo "=> OK: 조용함. NPU 실험 가능 (락은 scripts/lock/status.sh 로 별도 확인)"
fi
