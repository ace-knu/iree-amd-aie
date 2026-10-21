# v9 회귀 공용 함수 (run_regression.sh, ext/run_ext.sh가 source). v8 run_regression.sh 안에 있던 compile_flags를 그대로 옮겼다.
PEANO=${PEANO:-/workspace/llvm-aie}

# 컴파일 타깃 목록(쉼표 구분)에서 플래그를 만든다. CLAUDE.md 의 F3 와 같은 값.
# vulkan 이 있으면 --iree-vulkan-target=gfx1150 을 반드시 붙인다
# (빠지면 에러 없이 vp_android_baseline_2022 로 떨어진다).
compile_flags() {
  local targets=$1 f="" d
  IFS=, read -ra devs <<<"$targets"
  for d in "${devs[@]}"; do f+=" --iree-hal-target-device=$d"; done
  for d in "${devs[@]}"; do
    case $d in
      local)   f+=" --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=host" ;;
      amdxdna) f+=" --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=$PEANO --iree-amdaie-stack-size=2048" ;;
      vulkan)  f+=" --iree-vulkan-target=gfx1150" ;;
    esac
  done
  f+=" --iree-hal-external-resources-mappable=true"
  echo "$f"
}

# alloc_select 후보 장치 이름(cand 순서, '+'로 이음). stderr 의 [DBG] 로그 기준 (v8 run_case 와 같은 규칙).
alloc_cands() {
  local err=$1 cands="" p nm
  for p in $(grep -oE 'alloc_select cand\[[0-9]+\] dev=0x[0-9a-f]+' "$err" | sed 's/.*dev=//' | awk '!s[$0]++'); do
    nm=$(grep -m1 -oE "fence_create dev='[a-z-]+' ptr=$p\b" "$err" | sed -E "s/.*dev='([a-z-]+)'.*/\1/")
    cands+="${cands:++}${nm:-?}"
  done
  echo "$cands"
}
