#!/usr/bin/env bash
# 3-device (CPU + NPU + GPU) 회귀 테스트.
#
# 위치: docs/regression/run_regression.sh (v6 부터 여기서 관리, 현재 v8(기본 실행 = 수동+자동). 케이스 목록·버전 이력은
#       같은 폴더 README.md). v1~v6 의 이전 사본은 docs/2026-09-28/02_회귀테스트_기준선_재현/.
#
# 각 케이스: .mlir 을 현재 컴파일러로 새로 컴파일 -> iree-run-module 실행 ->
# 출력 첫 원소를 기대값과 비교 -> PASS/FAIL. 기존 vmfb 는 절대 재사용하지 않는다
# (바이너리가 바뀌었는데 옛 vmfb 로 정답이 나오는 착시를 막기 위해).
#
# 컨테이너(iree-aie-gpu:dev, /workspace = 저장소 루트) 안에서 실행한다.
#   bash /workspace/docs/regression/run_regression.sh              # 전체
#   bash /workspace/docs/regression/run_regression.sh 'mm_probe'   # 케이스 이름 정규식 필터
#
# 환경 변수
#   IREE_BUILD   빌드 디렉토리            (기본 /workspace/build-gpu)
#   MODEL_DIR    입력 .mlir 폴더           (기본 이 스크립트 옆 models/)
#   REG_ROOT     산출물 상위 폴더          (기본 /tmp)
#   REG_OUT      산출물 폴더 전체 경로     (기본 $REG_ROOT/regression_YYYYMMDD_HHMM)
#   REG_TIMEOUT  케이스별 컴파일/실행 제한 시간(초) (기본 600)
#   REG_INPUT_DEVICE  both(기본): manual 과 auto 를 차례로 둘 다 돌린다. 결과는
#                       $REG_OUT/manual/, $REG_OUT/auto/ 이고, 합친 요약은 $REG_OUT/summary.txt.
#                     manual: 케이스의 --input_device 를 붙인다.
#                     auto: --input_device 를 빼고 실행한다 (iree-run-module 이 컴파일러의
#                     reflection amdaie.input_devices 로 배치). lead 등 나머지 인자는 같다.
#
# 산출물($REG_OUT): <key>.vmfb, <key>.compile.log, <key>.stream.mlir(지표용),
#                   <key>.stream.log, <case>.stdout, <case>.stderr,
#                   summary.txt. 저장소 안에는 아무것도 쓰지 않는다.

set -u -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IREE_BUILD=${IREE_BUILD:-/workspace/build-gpu}
COMPILE=$IREE_BUILD/tools/iree-compile
RUN=$IREE_BUILD/tools/iree-run-module
MODEL_DIR=${MODEL_DIR:-$SCRIPT_DIR/models}
REG_ROOT=${REG_ROOT:-/tmp}
OUT=${REG_OUT:-$REG_ROOT/regression_$(date +%Y%m%d_%H%M)}
TIMEOUT=${REG_TIMEOUT:-600}
INPUT_MODE=${REG_INPUT_DEVICE:-both}
FILTER=${1:-}

# 기본(both): 이 스크립트를 모드별로 한 번씩 다시 부르고 요약을 합친다. 둘 중 하나라도 FAIL 이면 rc 1.
if [[ $INPUT_MODE == both ]]; then
  mkdir -p "$OUT" || exit 2
  rc_all=0
  for m in manual auto; do
    echo "===== input mode: $m ====="
    REG_INPUT_DEVICE=$m REG_OUT="$OUT/$m" bash "${BASH_SOURCE[0]}" "$@" || rc_all=1
  done
  {
    echo "regression run (both input modes): $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "out dir       : $OUT  (manual/, auto/)"
    for m in manual auto; do echo; echo "===== $m ====="; cat "$OUT/$m/summary.txt"; done
    echo
    echo "BOTH: manual $(grep -m1 '^PASS=' "$OUT/manual/summary.txt") / auto $(grep -m1 '^PASS=' "$OUT/auto/summary.txt")"
  } >"$OUT/summary.txt"
  tail -1 "$OUT/summary.txt"
  exit $rc_all
fi
PEANO=/workspace/llvm-aie

mkdir -p "$OUT" || exit 2
SUMMARY=$OUT/summary.txt

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

# 출력(stdout)에서 첫 번째 결과 텐서의 첫 원소를 뽑는다.
#   예) 128x128xf32=[32768 32768 ...][...]  ->  32768
first_elem() {
  grep -m1 -E '^[0-9x]+[a-z]+[0-9]+=' "$1" 2>/dev/null |
    sed -E 's/^[^=]*=//' | tr -d '[]' | awk '{print $1}'
}

# 상대 오차 1e-3 (절대 1e-5 하한) 이내면 일치.
matches() {
  [[ $1 =~ ^-?[0-9.]+([eE][-+]?[0-9]+)?$ ]] || return 1
  awk -v a="$1" -v e="$2" 'BEGIN{d=a-e; if(d<0)d=-d; t=(e<0?-e:e)*1e-3; if(t<1e-5)t=1e-5; exit !(d<=t)}'
}

declare -A COMPILED   # key -> 컴파일 rc (같은 실행 안에서 같은 key 는 한 번만 컴파일)
declare -A NCMDCOPY   # key -> 최종 stream IR 의 stream.cmd.copy 개수 (지표)
ROWS=()
NPASS=0; NFAIL=0

# run_case NAME KEY MLIR TARGETS RUN_DEVICES FUNC EXPECT INPUT_DEVICES LEAD INPUT...
#   TARGETS       컴파일 --iree-hal-target-device 순서 (쉼표 구분)
#   RUN_DEVICES   실행 --device 순서 (쉼표 구분, TARGETS 와 같은 순서여야 함)
#   INPUT_DEVICES --input_device 값 (쉼표 구분, 없으면 -)
#   LEAD          --device_lead_allocator 값 (없으면 -)
run_case() {
  local name=$1 key=$2 mlir=$3 targets=$4 rdevs=$5 func=$6 expect=$7 idevs=$8 lead=$9
  shift 9
  local inputs=("$@")
  if [[ -n $FILTER && ! $name =~ $FILTER ]]; then return; fi

  local vmfb=$OUT/$key.vmfb src=$mlir
  [[ $src == /* ]] || src=$MODEL_DIR/$mlir
  if [[ -z ${COMPILED[$key]+x} ]]; then
    local flags; flags=$(compile_flags "$targets")
    echo "[compile] $key <- $(basename "$mlir") ($targets)"
    {
      echo "# $COMPILE $src $flags -o $vmfb"
      # shellcheck disable=SC2086
      timeout "$TIMEOUT" "$COMPILE" "$src" $flags -o "$vmfb"
    } >"$OUT/$key.compile.log" 2>&1
    COMPILED[$key]=$?
    # 지표(판정에는 쓰지 않음): stream 단계까지만 따로 컴파일해서 최종 stream IR 의
    # stream.cmd.copy 개수를 센다. 위 본 컴파일 명령은 건드리지 않는다.
    # shellcheck disable=SC2086
    if timeout "$TIMEOUT" "$COMPILE" "$src" $flags --compile-to=stream \
         -o "$OUT/$key.stream.mlir" >"$OUT/$key.stream.log" 2>&1; then
      NCMDCOPY[$key]=$(grep -c 'stream\.cmd\.copy' "$OUT/$key.stream.mlir")
    else
      NCMDCOPY[$key]=err
    fi
  fi
  local crc=${COMPILED[$key]}

  local args=(--module="$vmfb" --function="$func")
  local d i
  IFS=, read -ra rd <<<"$rdevs";  for d in "${rd[@]}"; do args+=(--device="$d"); done
  if [[ $idevs != - && $INPUT_MODE != auto ]]; then
    IFS=, read -ra idv <<<"$idevs"; for i in "${idv[@]}"; do args+=(--input_device="$i"); done
  fi
  [[ $lead != - ]] && args+=(--device_lead_allocator="$lead")
  for i in "${inputs[@]}"; do args+=(--input="$i"); done

  local rrc=- actual=- nsel=- nxcopy=- placed=- status=FAIL
  if [[ $crc -eq 0 ]]; then
    echo "[run]     $name"
    echo "# $RUN ${args[*]}" >"$OUT/$name.cmd"
    timeout "$TIMEOUT" "$RUN" "${args[@]}" >"$OUT/$name.stdout" 2>"$OUT/$name.stderr"
    rrc=$?
    actual=$(first_elem "$OUT/$name.stdout"); actual=${actual:--}
    # 런타임 폴백(hal.allocator.select)이 실제로 후보를 고른 횟수.
    # module.c 의 [DBG] 로그 기준: "policy pairs=N -> K" / "no policy ... -> first".
    nsel=$(grep -cE '^\[DBG\] alloc_select: (policy|no policy)' "$OUT/$name.stderr")
    # 후보 장치 이름: "alloc_select cand[i] dev=PTR" 의 PTR 을
    # "fence_create dev='NAME' ptr=PTR" 로 역참조한다. amdxdna 가 후보에 있으면
    # NPU 가 끼는 경계가 컴파일 타임에 확정되지 않았다는 뜻이므로 FAIL 로 본다.
    # 표시 형식: 횟수(cand0+cand1->선택). 후보는 cand 인덱스 순서(첫 호출 기준).
    local cands="" p nm pick selnm=""
    for p in $(grep -oE 'alloc_select cand\[[0-9]+\] dev=0x[0-9a-f]+' "$OUT/$name.stderr" | sed 's/.*dev=//' | awk '!s[$0]++'); do
      nm=$(grep -m1 -oE "fence_create dev='[a-z-]+' ptr=$p\b" "$OUT/$name.stderr" | sed -E "s/.*dev='([a-z-]+)'.*/\1/")
      cands+="${cands:++}${nm:-?}"
    done
    pick=$(grep -m1 -oE 'alloc_select: policy pairs=[0-9]+ -> [0-9]+' "$OUT/$name.stderr" | sed 's/.*-> //')
    if [[ -n $pick ]]; then
      IFS=+ read -ra cn <<<"$cands"; selnm=${cn[$pick]:-?}
    fi
    [[ $nsel != 0 ]] && nsel="$nsel($cands${selnm:+->$selnm})"
    # 지표(판정에는 쓰지 않음): amdxdna copy_buffer 실행 횟수. 경계 복사와
    # 툴링의 출력 전송(lead 로 옮기기)이 모두 포함된다.
    nxcopy=$(grep -c '^\[BR\] amdxdna copy' "$OUT/$name.stderr")
    # 지표(판정에는 쓰지 않음): 입력 재배치 결과. i 번째 칸 = 옮긴 장치 인덱스(--device 순서),
    # L = 옮기지 않음(lead 에 둠). "[BR] input[i] -> device N" / "-> lead" 로그 기준.
    local ni=${#inputs[@]} k
    placed=""
    for ((k = 0; k < ni; ++k)); do
      local d; d=$(grep -m1 -oE "^\[BR\] input\[$k\] -> device [0-9]+" "$OUT/$name.stderr" | sed 's/.* //')
      placed+="${placed:+,}${d:-L}"
    done
    if [[ $rrc -eq 0 ]] && matches "$actual" "$expect" && [[ $cands != *amdxdna* ]]; then status=PASS; fi
  fi
  if [[ $status == PASS ]]; then NPASS=$((NPASS + 1)); else NFAIL=$((NFAIL + 1)); fi
  ROWS+=("$(printf '| %-26s | %-8s | %-10s | %-3s | %-3s | %-32s | %-8s | %-10s | %-9s | %-4s |' \
    "$name" "$expect" "$actual" "$crc" "$rrc" "$nsel" "${NCMDCOPY[$key]:--}" "$nxcopy" "$placed" "$status")")
}

# ---------------------------------------------------------------------------
# 케이스 정의 (근거: /workspace/.home-gpu/.bash_history 263, 318, 323-327, 331행)
# ---------------------------------------------------------------------------
T3=amdxdna,vulkan,local
R3=amdxdna,vulkan,local-task
T2=local,vulkan
R2=local-task,vulkan
X=128x128xf32

# chain2: matmul1 -> NPU(@__device_0), matmul2 -> GPU(@__device_1), 손으로 쓴 affinity.
#   (1x1)=128, (128x2)=256 x 128 = 32768
run_case chain2_3dev          chain2_3dev   chain2_npufirst.mlir $T3 $R3 chain2     32768 0,0,1 - "$X=1" "$X=1" "$X=2"
# mlp_reduce: matmul -> softmax(CPU) -> matmul, 자동 배정. softmax 균등 1/128 x 128 = 1
run_case mlp_reduce_3dev      mlp_reduce_3dev mlp_reduce.mlir    $T3 $R3 mlp_reduce 1     0,0,1 - "$X=1" "$X=1" "$X=1"
# cg_topo: mlp_reduce 를 CPU+GPU 2-device 로. 소유자 동률(CPU<->GPU) -> lead 가 고름. 1 x 2 = 2
# 수동 모드의 --input_device 는 lead 인덱스 그대로다 (입력을 lead 에 둠 = v7 까지의 플래그 없는 동작.
# lead=local-task 면 CPU 입력을 GPU 가 host 포인터로 import 하는 경로를 검증한다). 플래그가 없으면
# iree-run-module 이 reflection 으로 GPU 에 재배치하므로 명시한다 (docs/2026-09-29/2_…/02 5-6).
# 자동 모드(REG_INPUT_DEVICE=auto)에서는 이 값이 빠지고 reflection 배치를 쓴다.
run_case cg_topo_lead0        cg_topo       mlp_reduce.mlir      $T2 $R2 mlp_reduce 2     0,0,0 0 "$X=1" "$X=1" "$X=2"
run_case cg_topo_lead1        cg_topo       mlp_reduce.mlir      $T2 $R2 mlp_reduce 2     1,1,1 1 "$X=1" "$X=1" "$X=2"
# mm_probe: 단일 matmul, CPU+GPU. 1 x 2 x 128 = 256 (--input_device 는 위와 같은 이유로 lead 인덱스)
run_case mm_probe_lead0       mm_probe      mm_probe.mlir        $T2 $R2 mm         256   0,0   0 "$X=1" "$X=2"
run_case mm_probe_lead1       mm_probe      mm_probe.mlir        $T2 $R2 mm         256   1,1   1 "$X=1" "$X=2"

# 디바이스 선언 순서 뒤집기 (docs/2026-09-28/03). NPU 가 index 0 이 아니어도 NPU 경계
# 소유자가 topology 로 확정되는지 본다: 정답 + alloc_select 후보에 amdxdna 없음.
# 약어 a=amdxdna v=vulkan l=local. --input_device 는 각 순서의 stream.tensor.import 기준.
# (chain2_order_vla/val 은 NPU 코드젠 실패로 여기서는 제외했었다 -> docs/2026-09-28/03 README 6절.
#  docs/2026-09-28/06, 10 에서 해결되어 아래 placement 블록에 편입됨)
run_case order_chain2_lav     order_chain2_lav     chain2_order_lav.mlir local,amdxdna,vulkan local-task,amdxdna,vulkan chain2     32768 1,1,2 - "$X=1" "$X=1" "$X=2"
run_case order_chain2auto_avl order_chain2auto_avl chain2_auto.mlir      $T3                  $R3                        chain2     32768 0,0,1 - "$X=1" "$X=1" "$X=2"
run_case order_chain2auto_vla order_chain2auto_vla chain2_auto.mlir      vulkan,local,amdxdna vulkan,local-task,amdxdna chain2     32768 0,0,2 - "$X=1" "$X=1" "$X=2"
run_case order_chain2auto_val order_chain2auto_val chain2_auto.mlir      vulkan,amdxdna,local vulkan,amdxdna,local-task chain2     32768 0,0,1 - "$X=1" "$X=1" "$X=2"
run_case order_mlp_reduce_vla order_mlp_reduce_vla mlp_reduce.mlir       vulkan,local,amdxdna vulkan,local-task,amdxdna mlp_reduce 1     0,0,2 - "$X=1" "$X=1" "$X=1"
run_case order_mlp_reduce_lav order_mlp_reduce_lav mlp_reduce.mlir       local,amdxdna,vulkan local-task,amdxdna,vulkan mlp_reduce 1     1,1,2 - "$X=1" "$X=1" "$X=1"
run_case order_mlp_reduce_val order_mlp_reduce_val mlp_reduce.mlir       vulkan,amdxdna,local vulkan,amdxdna,local-task mlp_reduce 1     0,0,1 - "$X=1" "$X=1" "$X=1"

# 경계 버퍼 전 케이스 (docs/2026-09-28/05). 첫 결과 텐서의 첫 원소만 비교한다.
# 3후보 경계 optimal<[npu,gpu,cpu]> -> NPU 소유. CPU 생산자: s=softmax(a)@CPU, matmul(s,b)@NPU, matmul(s,c)@GPU -> 1
run_case fanout3cpu_avl       fanout3cpu_avl       fanout3_cpu.mlir      $T3                  $R3                        fanout3_cpu 1     2,0,1 - "$X=1" "$X=1" "$X=1"
# 3후보 경계, GPU 생산자: matmul(a,b)@GPU, matmul(m0,c)@NPU, softmax(m0)@CPU -> 16384.
# lead=1(amdxdna). lead=0(vulkan)은 docs/2026-09-28/11 이후 PASS (아래 fanout3_val_lead0)
run_case fanout3_val_lead1    fanout3_val          fanout3.mlir          vulkan,amdxdna,local vulkan,amdxdna,local-task fanout3     16384 0,0,1 1 "$X=1" "$X=1" "$X=1"
# NPU 생산자: 3후보가 안 생기고 [npu,gpu], [npu,cpu] 두 경계 + 복사 2회 (04 README 오판 ①) -> 16384
run_case fanout3_avl          fanout3_avl          fanout3.mlir          $T3                  $R3                        fanout3     16384 0,0,1 - "$X=1" "$X=1" "$X=1"
# 상수 가중치(비-splat): w1 -> NPU matmul, w2 -> GPU matmul, 각 상수가 쓰는 장치에 놓임 -> 16384
run_case mlp2const_avl        mlp2const_avl        mlp_2layer_const.mlir $T3                  $R3                        mlp_2layer_const 16384 0 - "$X=1"

# amdxdna 동기 할당 placement 기록 (docs/2026-09-28/10). 동기 할당 NPU 버퍼를 GPU 가 dma-buf 로 읽는다.
# 공유 상수: w1 == w2 -> 상수 하나(NPU)를 NPU matmul 과 GPU matmul 이 같이 씀 -> 16384
run_case mlp2constshared_avl  mlp2constshared_avl  mlp_2layer_const_shared.mlir $T3           $R3                        mlp_2layer_const 16384 0 - "$X=1"
# 손 affinity chain2, GPU 가 NPU 보다 앞: --input_device 로 NPU 에 재배치한 a, b 를 GPU matmul1 이 읽음 -> 32768
run_case order_chain2_vla     order_chain2_vla     chain2_order_vla.mlir vulkan,local,amdxdna vulkan,local-task,amdxdna chain2     32768 2,2,0 - "$X=1" "$X=1" "$X=2"
run_case order_chain2_val     order_chain2_val     chain2_order_val.mlir vulkan,amdxdna,local vulkan,amdxdna,local-task chain2     32768 1,1,0 - "$X=1" "$X=1" "$X=2"

# Vulkan copy_buffer BRIDGE (docs/2026-09-28/11). lead=vulkan 이면 툴링이 CPU 출력 s(local-task 버퍼)를
# Vulkan copy_buffer 로 옮긴다 (전에는 segfault). 첫 결과 m1 = 16384, 둘째 s = 0.0078125
run_case fanout3_val_lead0    fanout3_val          fanout3.mlir          vulkan,amdxdna,local vulkan,amdxdna,local-task fanout3     16384 0,0,1 0 "$X=1" "$X=1" "$X=1"
run_case fanout3_vla_lead0    fanout3_vla          fanout3.mlir          vulkan,local,amdxdna vulkan,local-task,amdxdna fanout3     16384 0,0,2 0 "$X=1" "$X=1" "$X=1"
run_case fanout3_avl_lead1    fanout3_avl          fanout3.mlir          $T3                  $R3                        fanout3     16384 0,0,1 1 "$X=1" "$X=1" "$X=1"

# BRIDGE subspan offset (docs/2026-09-29/1_구조전환_offset회귀_input_device준비/03). CPU 출력 2개(s2 = z*z, s1 = softmax(y))가
# 한 외부 할당(131072B)에 packing 되고 subview 로 export 된다. 첫 출력 s2 는 offset 65536 에 있다.
# lead=vulkan(1): 툴링이 두 출력을 Vulkan copy_buffer BRIDGE 로 옮기며 byte_offset=65536 import 가 일어난다 -> s2 = 4
# lead=local-task(2): BRIDGE 를 타지 않는 대조군 -> 4
run_case c2e_avl_lead1        c2e_avl              cpu2export.mlir       $T3                  $R3                        cpu2export  4     0,2,2 1 "$X=1" "$X=1" "$X=2"
run_case c2e_avl_lead2        c2e_avl              cpu2export.mlir       $T3                  $R3                        cpu2export  4     0,2,2 2 "$X=1" "$X=1" "$X=2"

# --input_device 자동화 (docs/2026-09-29/2_VGG16상수묶음확인_input_device자동화/02). 한 입력을 여러 장치가 import:
# arg0 -> {NPU, GPU}, arg2 -> {GPU, NPU}. 입력은 모든 importer 가 읽을 수 있는 NPU 에 있어야 한다 (0,0,0).
# m0 = x@y(NPU), m1 = x@z(GPU), m2 = y@z(NPU), r = m0@m2(GPU). x=1,y=1,z=2 -> r = 4194304
run_case n2o_avl              n2o_avl              npu2out.mlir          $T3                  $R3                        npu2out     4194304 0,0,0 - "$X=1" "$X=1" "$X=2"

# 추가 케이스 파일(실험용). run_case 줄만 담은 bash 파일을 source 한다.
if [[ -n ${REG_EXTRA_CASES:-} ]]; then
  # shellcheck disable=SC1090
  source "$REG_EXTRA_CASES"
fi

# ---------------------------------------------------------------------------
{
  echo "regression run: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "out dir       : $OUT"
  echo "iree-compile  : $(stat -c '%y' "$IREE_BUILD/lib/libIREECompiler.so" 2>/dev/null) (libIREECompiler.so)"
  echo "iree-run-module: $(stat -c '%y' "$RUN" 2>/dev/null)"
  echo "model dir     : $MODEL_DIR"
  echo "input mode    : $INPUT_MODE (REG_INPUT_DEVICE)"
  echo
  echo "| case                       | expected | actual     | crc | rrc | alloc_select(후보->선택)         | cmd.copy | xdna_copy  | placed    | res  |"
  echo "|----------------------------|----------|------------|-----|-----|----------------------------------|----------|------------|-----------|------|"
  printf '%s\n' "${ROWS[@]}"
  echo
  echo "PASS=$NPASS FAIL=$NFAIL"
  echo "(crc/rrc = compile/run exit code, alloc_select = 런타임 폴백 선택 횟수(cand 순서 후보->선택 장치). 후보에 amdxdna 가 있으면 FAIL)"
  echo "(cmd.copy = --compile-to=stream 결과의 stream.cmd.copy 개수, xdna_copy = stderr 의 '[BR] amdxdna copy' 줄 수(출력 전송 포함). 둘 다 지표이고 판정에 쓰지 않음)"
  echo "(placed = 입력 재배치 결과, i 번째 칸 = 옮긴 장치 인덱스(--device 순서), L = lead 에 둠. 지표이고 판정에 쓰지 않음)"
} | tee "$SUMMARY"

[[ $NFAIL -eq 0 ]]
