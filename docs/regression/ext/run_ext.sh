#!/usr/bin/env bash
# v9 회귀 새 시험 실행기 (run_regression.sh 가 quick/full 단계에서 부른다. 단독 실행도 된다).
#   bash ext/run_ext.sh <quick|full> <out_dir> [케이스 이름 정규식]
# 시험 (판정은 ext/judge.py, 입력은 ext/gen_inputs.py 로 실행 때 생성. 자세한 표는 ../README.md)
#   V1 vec_*   bf16 벡터화 matmul × rand·relu      → 흉내 V3.1(conv_even) 원소 100% 비트 일치
#   V2 rnd_*   --iree-amdaie-core-rounding-mode=floor → floor V3.1 100% + conv_even V3.1과 50% 미만
#   V3 i8_*    int8 벡터화 matmul                  → 정수 정답과 같음
#   V4 mlp_*   mlp_reduce 난수 입력(3장치·CPU+GPU)  → 출력 전체 float64 기준 rel ≤ 1e-5
#   V5 sem_*   NPU→GPU, GPU→NPU, CPU→NPU × 시드 3 × 2회 → 정확, 같은 시드 비트 동일, 로그 순서
#   V6 edge_*  fanout3 3경우 + NPU→GPU→CPU 직결     → 출력 전부 float64 기준
#   F1~F3 vgg_* (full) VGG16 2장치 끔/켬, f32 3장치 → 출력 sha256 (vgg16/expected_sha256.txt)
# 환경 변수
#   REG_FAULT   검출력 확인용(평소 비움): v1_floor(V1 컴파일에 floor 옵션), v1_v1model(V1 판정을 옛 누산 모델로),
#               v4_ref(V4 기준을 다른 시드 입력으로 계산)
#   REG_VGG_MLIR  (기본 /work/vgg16/vgg.mlir)   REG_VGG_INPUT (기본 /workspace/models/vgg16/input.npy)
#   REG_KEEP_VGG_VMFB=1  VGG vmfb 를 판정 뒤에도 남김(기본은 지움, 1.16 GB)
#   IREE_BUILD, REG_TIMEOUT, REG_PYTHON 은 run_regression.sh 와 같다.
set -u -o pipefail
EXT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REG_DIR=$(dirname "$EXT_DIR")
# shellcheck source=../lib.sh
source "$REG_DIR/lib.sh"
LEVEL=${1:?quick|full}; OUT=${2:?out_dir}; FILTER=${3:-}
IREE_BUILD=${IREE_BUILD:-/workspace/build-gpu}; COMPILE=$IREE_BUILD/tools/iree-compile; RUN=$IREE_BUILD/tools/iree-run-module
TIMEOUT=${REG_TIMEOUT:-600}; PY=${REG_PYTHON:-/opt/venv/bin/python3}; FAULT=${REG_FAULT:-}
MODELS=$REG_DIR/models; IN=$OUT/inputs
mkdir -p "$OUT" "$IN" || exit 2
ROWS=(); NP=0; NF=0; NS=0

# NPU + CPU 2장치(10/21 시험과 같은 구성). bf16 은 팀 demote, 벡터화는 명시적으로 켠다(기본값도 켬).
NC_FLAGS="--iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local --iree-hal-local-target-device-backends=llvm-cpu
 --iree-hal-default-device=npu --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=$PEANO --iree-amdaie-enable-vectorization-passes=true"

row() {  # 이름 판정 결과요약 crc rrc res
  ROWS+=("$(printf '| %-30s | %-9s | %-3s | %-3s | %-4s | %s |' "$1" "$2" "$4" "$5" "$6" "$3")")
  case $6 in PASS) NP=$((NP + 1)) ;; SKIP) NS=$((NS + 1)) ;; *) NF=$((NF + 1)) ;; esac
}
want() { [[ -z $FILTER || $1 =~ $FILTER ]]; }

declare -A CRC
compile() {  # key mlir flags...
  local key=$1 mlir=$2; shift 2
  if [[ -z ${CRC[$key]+x} ]]; then
    echo "[compile] $key <- $(basename "$mlir")"
    { echo "# $COMPILE $mlir $* -o $OUT/$key.vmfb"; timeout "$TIMEOUT" "$COMPILE" "$mlir" "$@" -o "$OUT/$key.vmfb"; } >"$OUT/$key.compile.log" 2>&1
    CRC[$key]=$?
  fi
  return "${CRC[$key]}"
}
run() {  # name vmfb func [iree-run-module 인자...]   (stderr 시각 도장, rc 반환)
  local name=$1 vmfb=$2 func=$3; shift 3
  echo "[run]     $name"
  echo "# $RUN --module=$vmfb --function=$func $*" >"$OUT/$name.cmd"
  timeout "$TIMEOUT" "$PY" "$EXT_DIR/stamp_run.py" "$OUT/$name.stderr" "$OUT/$name.stdout" -- "$RUN" --module="$vmfb" --function="$func" "$@"
}
judge() { "$PY" "$EXT_DIR/judge.py" "$@" 2>&1 | head -1; }
nocand() { [[ $(alloc_cands "$1") != *amdxdna* ]]; }   # v8 과 같은 규칙: alloc_select 후보에 amdxdna 가 있으면 FAIL

# ---------------- V1·V2 bf16 벡터화 matmul ----------------
V1_EXTRA=(); V1_MODEL=v31
[[ $FAULT == v1_floor ]] && V1_EXTRA=(--iree-amdaie-core-rounding-mode=floor)
[[ $FAULT == v1_v1model ]] && V1_MODEL=old
bf16_case() {  # name shape kind rnd(판정) extra_flag(-|플래그)
  local name=$1 S=$2 kind=$3 rnd=$4 xf=$5 M N K key crc rrc j res=FAIL fl=()
  want "$name" || return 0
  IFS=x read -r M N K <<<"$S"
  [[ $xf != - ]] && fl+=("$xf")
  [[ $name == vec_* ]] && fl+=("${V1_EXTRA[@]}")
  key=bf16_${S}${xf:+_$(echo "$xf" | sed 's/.*=//')}; [[ $name == vec_* && ${#V1_EXTRA[@]} -gt 0 ]] && key+=_fault
  # shellcheck disable=SC2086
  compile "$key" "$MODELS/vec/mm_f32_$S.mlir" $NC_FLAGS --iree-amdaie-demote-contraction-inputs-to-bf16 "${fl[@]}"; crc=$?
  "$PY" "$EXT_DIR/gen_inputs.py" bf16 "$IN/bf16_${S}_$kind" "$M" "$N" "$K" "$kind" || crc=98
  rrc=-; j=-
  if [[ $crc -eq 0 ]]; then
    run "$name" "$OUT/$key.vmfb" mm --device=amdxdna --device=local-task --input=@"$IN/bf16_${S}_$kind/a.npy" --input=@"$IN/bf16_${S}_$kind/b.npy" --output=@"$OUT/$name.npy"; rrc=$?
    if [[ $rrc -eq 0 ]]; then
      if [[ $name == rnd_* ]]; then
        j=$(judge bfp "$OUT/$name.npy" "$IN/bf16_${S}_$kind" floor)
        local jn; jn=$(judge bfp_neg "$OUT/$name.npy" "$IN/bf16_${S}_$kind" half_even 50)
        [[ $j == OK* && $jn == OK* ]] && res=PASS; j="$j ; $jn"
      else
        j=$(judge bfp "$OUT/$name.npy" "$IN/bf16_${S}_$kind" "$rnd" "$V1_MODEL"); [[ $j == OK* ]] && res=PASS
      fi
    fi
  fi
  row "$name" "bfp" "$j" "$crc" "$rrc" "$res"
}
for S in 256x64x32 3072x256x1152 768x512x2304; do for kind in rand relu; do bf16_case "vec_bf16_${S}_$kind" "$S" "$kind" half_even -; done; done
bf16_case rnd_floor_3072x256x1152_relu 3072x256x1152 relu floor --iree-amdaie-core-rounding-mode=floor

# ---------------- V3 int8 ----------------
if want i8_3136x64x576; then
  compile i8_3136x64x576 "$MODELS/vec/mm_i8_3136x64x576.mlir" $NC_FLAGS; crc=$?; rrc=-; j=-; res=FAIL
  "$PY" "$EXT_DIR/gen_inputs.py" i8 "$IN/i8_3136x64x576" 3136 64 576 || crc=98
  if [[ $crc -eq 0 ]]; then
    run i8_3136x64x576 "$OUT/i8_3136x64x576.vmfb" mm --device=amdxdna --device=local-task --input=@"$IN/i8_3136x64x576/a.npy" --input=@"$IN/i8_3136x64x576/b.npy" --output=@"$OUT/i8_3136x64x576.npy"; rrc=$?
    [[ $rrc -eq 0 ]] && { j=$(judge int "$OUT/i8_3136x64x576.npy" "$IN/i8_3136x64x576"); [[ $j == OK* ]] && res=PASS; }
  fi
  row i8_3136x64x576 int "$j" "$crc" "$rrc" "$res"
fi

# ---------------- V4~V6 128x128 3장치·2장치 (난수 입력, float64 기준) ----------------
devargs() { local o="" d; IFS=, read -ra x <<<"$1"; for d in "${x[@]}"; do o+=" --device=$d"; done; echo "$o"; }
idargs() { local o="" d; [[ $1 == - ]] && return; IFS=, read -ra x <<<"$1"; for d in "${x[@]}"; do o+=" --input_device=$d"; done; echo "$o"; }
for s in 0 1 2; do "$PY" "$EXT_DIR/gen_inputs.py" r128 "$IN/r128_s$s" "$s"; done
# ref_case 이름 모델 함수 식 컴파일순서 실행순서 input_device lead 시드 결과수
ref_case() {
  local name=$1 mlir=$2 func=$3 form=$4 tg=$5 rd=$6 idv=$7 lead=$8 s=$9 nout=${10} key crc rrc=- j=- res=FAIL o=() k
  want "$name" || return 0
  key=$(basename "$mlir" .mlir)_$(echo "$tg" | sed 's/amdxdna/a/; s/vulkan/v/; s/local/l/; s/,//g')
  # shellcheck disable=SC2046
  compile "$key" "$MODELS/$mlir" $(compile_flags "$tg"); crc=$?
  for ((k = 0; k < nout; ++k)); do o+=(--output=@"$OUT/$name.o$k.npy"); done
  if [[ $crc -eq 0 ]]; then
    # shellcheck disable=SC2046
    run "$name" "$OUT/$key.vmfb" "$func" $(devargs "$rd") $(idargs "$idv") $([[ $lead != - ]] && echo --device_lead_allocator=$lead) \
      --input=@"$IN/r128_s$s/a.npy" --input=@"$IN/r128_s$s/b.npy" --input=@"$IN/r128_s$s/c.npy" "${o[@]}"; rrc=$?
    if [[ $rrc -eq 0 ]]; then
      local outs=(); for ((k = 0; k < nout; ++k)); do outs+=("$OUT/$name.o$k.npy"); done
      if [[ $name == mlp_* && $FAULT == v4_ref ]]; then j=$(REG_JUDGE_REFDIR="$IN/r128_s1" judge ref "$form" "$IN/r128_s$s" 1e-5 "${outs[@]}")
      else j=$(judge ref "$form" "$IN/r128_s$s" 1e-5 "${outs[@]}"); fi
      [[ $j == OK* ]] && nocand "$OUT/$name.stderr" && res=PASS
      nocand "$OUT/$name.stderr" || j="$j ; alloc_select 후보에 amdxdna"
    fi
  fi
  row "$name" ref "$j" "$crc" "$rrc" "$res"
}
T3=amdxdna,vulkan,local; R3=amdxdna,vulkan,local-task
# V4 mlp_reduce 강화 (기존 mlp_reduce_3dev, cg_topo_lead0/1 과 같은 장치·인자)
ref_case mlp_reduce_3dev_rand   mlp_reduce.mlir mlp_reduce mlp_reduce $T3 $R3 0,0,1 - 0 1
ref_case mlp_cg_topo_lead0_rand mlp_reduce.mlir mlp_reduce mlp_reduce local,vulkan local-task,vulkan 0,0,0 0 0 1
ref_case mlp_cg_topo_lead1_rand mlp_reduce.mlir mlp_reduce mlp_reduce local,vulkan local-task,vulkan 1,1,1 1 0 1
# V6 미시험 경계 버퍼 (10/20 경계 버퍼 표 B7~B9, 미시험 2)
ref_case edge_fanout3cpu_avl    fanout3_cpu.mlir fanout3_cpu fanout3_cpu $T3 $R3 2,0,1 - 0 2
ref_case edge_fanout3_val_lead1 fanout3.mlir     fanout3     fanout3     vulkan,amdxdna,local vulkan,amdxdna,local-task 0,0,1 1 0 2
ref_case edge_fanout3_avl       fanout3.mlir     fanout3     fanout3     $T3 $R3 0,0,1 - 0 2
ref_case edge_chain3_ngc_avl    chain3_ngc.mlir  chain3      chain3      $T3 $R3 0,0,0 - 0 1

# V5 세마포어 대표 경우: 시드 0·1·2 × 2회. 정확(매회) + 같은 시드 비트 동일 + 로그 순서(매회)
sem_case() {  # name mlir func form 컴파일순서 실행순서 input_device order(n2g|g2n|-)
  local name=$1 mlir=$2 func=$3 form=$4 tg=$5 rd=$6 idv=$7 ord=$8 key crc s r j res=PASS detail="" nrun=0 nok=0 nord=0 nsame=0
  want "$name" || return 0
  key=$(basename "$mlir" .mlir)_$(echo "$tg" | sed 's/amdxdna/a/; s/vulkan/v/; s/local/l/; s/,//g')
  # shellcheck disable=SC2046
  compile "$key" "$MODELS/$mlir" $(compile_flags "$tg"); crc=$?
  [[ $crc -ne 0 ]] && { row "$name" sem - "$crc" - FAIL; return; }
  for s in 0 1 2; do
    for r in 1 2; do
      local n=${name}_s${s}r$r; nrun=$((nrun + 1))
      # shellcheck disable=SC2046
      run "$n" "$OUT/$key.vmfb" "$func" $(devargs "$rd") $(idargs "$idv") \
        --input=@"$IN/r128_s$s/a.npy" --input=@"$IN/r128_s$s/b.npy" --input=@"$IN/r128_s$s/c.npy" --output=@"$OUT/$n.npy" || { res=FAIL; detail+=" $n:rc"; continue; }
      j=$(judge ref "$form" "$IN/r128_s$s" 1e-5 "$OUT/$n.npy"); [[ $j == OK* ]] && nok=$((nok + 1)) || { res=FAIL; detail+=" $n:$j"; }
      nocand "$OUT/$n.stderr" || { res=FAIL; detail+=" $n:alloc_select 후보에 amdxdna"; }
      if [[ $ord != - ]]; then j=$(judge order "$OUT/$n.stderr" "$ord"); [[ $j == OK* ]] && nord=$((nord + 1)) || { res=FAIL; detail+=" $n:$j"; }; fi
    done
    j=$(judge same "$OUT/${name}_s${s}r1.npy" "$OUT/${name}_s${s}r2.npy"); [[ $j == OK* ]] && nsame=$((nsame + 1)) || { res=FAIL; detail+=" s$s:$j"; }
  done
  row "$name" sem "정확 $nok/$nrun, 같은 시드 비트 동일 $nsame/3$([[ $ord != - ]] && echo ", 순서($ord) $nord/$nrun")$detail" "$crc" 0 "$res"
}
sem_case sem_n2g_chain2auto_avl   chain2_auto.mlir chain2     chain2     $T3 $R3 0,0,1 n2g
sem_case sem_g2n_chain2auto_vla   chain2_auto.mlir chain2     chain2     vulkan,local,amdxdna vulkan,local-task,amdxdna 0,0,2 g2n
sem_case sem_c2n_mlpreduce_vla    mlp_reduce.mlir  mlp_reduce mlp_reduce vulkan,local,amdxdna vulkan,local-task,amdxdna 0,0,2 -

# ---------------- F1~F3 VGG16 (full) ----------------
if [[ $LEVEL == full ]]; then
  VM=${REG_VGG_MLIR:-/work/vgg16/vgg.mlir}; VI=${REG_VGG_INPUT:-/workspace/models/vgg16/input.npy}; V=$REG_DIR/vgg16
  vgg_case() {  # name flags_file map(-|파일) 실행장치
    local name=$1 ff=$2 map=$3 rd=$4 exp crc=- rrc=- j=- res=FAIL fl=()
    want "$name" || return 0
    exp=$(awk -v n="$name" '$1==n{print $2}' "$V/expected_sha256.txt")
    if [[ ! -f $VM || ! -f $VI ]]; then row "$name" sha "SKIP: VGG 입력 없음 ($VM, $VI)" - - SKIP; return; fi
    mapfile -t fl < "$V/$ff"
    [[ $map != - ]] && fl+=("--iree-amdaie-placement-map=@$V/$map")
    compile "$name" "$VM" "${fl[@]}"; crc=$?
    if [[ $crc -eq 0 ]]; then
      # shellcheck disable=SC2046
      run "$name" "$OUT/$name.vmfb" mxnet_converted_model $(devargs "$rd") --input=@"$VI" --output=@"$OUT/$name.npy"; rrc=$?
      [[ $rrc -eq 0 ]] && { j=$(judge sha "$OUT/$name.npy" "$exp"); [[ $j == OK* ]] && res=PASS; }
      j="$j | vmfb $(sha256sum "$OUT/$name.vmfb" | cut -c1-16)…(지표)"
    fi
    [[ ${REG_KEEP_VGG_VMFB:-0} == 1 ]] || rm -f "$OUT/$name.vmfb"
    row "$name" sha "$j" "$crc" "$rrc" "$res"
  }
  vgg_case vgg_2dev_novec    flags_b.txt                - amdxdna,local-task
  vgg_case vgg_2dev_vec      flags_b_vec.txt            - amdxdna,local-task
  vgg_case vgg_3dev_npu9     flags_nodemote_unroll0.txt npu9_gpu4.map amdxdna,vulkan,local-task
fi

{
  echo "ext run ($LEVEL): $(date '+%Y-%m-%d %H:%M:%S %Z')$([[ -n $FAULT ]] && echo "   REG_FAULT=$FAULT")"
  echo "out dir : $OUT"
  echo
  echo "| case                           | 판정      | crc | rrc | res  | 결과 요약 |"
  echo "|--------------------------------|-----------|-----|-----|------|-----------|"
  printf '%s\n' "${ROWS[@]}"
  echo
  echo "EXT PASS=$NP FAIL=$NF SKIP=$NS"
} | tee "$OUT/ext_summary.txt"
[[ $NF -eq 0 && $NS -eq 0 ]]
