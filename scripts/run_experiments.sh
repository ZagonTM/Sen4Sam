#!/usr/bin/env bash
# All experiments for the report, sized for a ~50 min budget on 1x 96 GB GPU / 32 vCPU.
# Run on the pod from the repo root:
#     nohup bash scripts/run_experiments.sh > run_all.log 2>&1 &
#     tail -f results/progress.log
# Jobs are queued most-important-first and run PARALLEL at a time:
#   1) untrained baseline on test: filters on, filters off
#   2) untrained grid-density sweep pps 16/32/64 on SWEEP_LIMIT test patches
#   3) trainings A/B/C/D (2x2) + A_loss_select, each followed by its test eval (normal + filters off)
# Every job logs to $OUT_ROOT/logs/<job>.log; one line per finished job goes to $OUT_ROOT/progress.log.
# Results: $OUT_ROOT/runs/<name>/ and $OUT_ROOT/diag/.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PYTHON:-python}
SPLIT="--split-file results/split.csv"
PARALLEL=${PARALLEL:-10}                # concurrent jobs (96 GB VRAM: ~5-6 GB per training)
EPOCHS=${EPOCHS:-6}                     # max epochs per training
PATIENCE=${PATIENCE:-2}                 # early-stopping patience
TRAIN_PATCHES=${TRAIN_PATCHES:-150}     # 0 = all 624 train patches
PARCELS=${PARCELS:-30}                  # parcels per train patch
VAL_PATCHES=${VAL_PATCHES:-20}          # val patches for the val loss
GEN_VAL_PATCHES=${GEN_VAL_PATCHES:-12}  # val patches for the generator metric (run every epoch)
SWEEP_LIMIT=${SWEEP_LIMIT:-40}          # test patches for the grid-density sweep (pps=64 is slow)
EXTRA_TRAIN=${EXTRA_TRAIN:-}            # e.g. "--epochs 1 --val-patches 2" for a quick test
EVAL_LIMIT=${EVAL_LIMIT:-0}             # 0 = all 132 test patches
LIM=""; if [ "$EVAL_LIMIT" -gt 0 ]; then LIM="--limit $EVAL_LIMIT"; fi
OUT_ROOT=${OUT_ROOT:-results}           # change for a dry run
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
THREADS=$(( NCPU / PARALLEL )); [ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS:-$THREADS}
mkdir -p "$OUT_ROOT/runs" "$OUT_ROOT/diag" "$OUT_ROOT/logs"
PROGRESS=$OUT_ROOT/progress.log
T0=$(date +%s)

log() { echo "[$(date +%T) +$(( ($(date +%s) - T0) / 60 ))m] $*" | tee -a "$PROGRESS"; }
log "START PARALLEL=$PARALLEL OMP_NUM_THREADS=$OMP_NUM_THREADS EPOCHS=$EPOCHS PATIENCE=$PATIENCE TRAIN_PATCHES=$TRAIN_PATCHES SWEEP_LIMIT=$SWEEP_LIMIT EVAL_LIMIT=$EVAL_LIMIT"

# ---- job bodies (stdout/stderr go to the job log) -----------------------------------------
evaluate() {   # tag, out-prefix, extra args...
  local tag=$1 prefix=$2; shift 2
  echo "=== eval $tag: $*"
  $PY src/evaluate_masks.py $SPLIT --split test "$@" \
      --out-patches "${prefix}_patches.csv" --out-summary "${prefix}_summary.csv"
}

job_base()      { evaluate base "$OUT_ROOT/diag/untrained" $LIM; }
job_base_nof()  { evaluate base_nofilter "$OUT_ROOT/diag/untrained_nofilter" $LIM --pred-iou 0 --stability 0; }
job_pps()       { evaluate "pps$1" "$OUT_ROOT/diag/untrained_pps$1" --limit "$SWEEP_LIMIT" --points-per-side "$1"; }

job_train() {  # name, train args... : train, then test eval (normal + filters off, in parallel)
  local name=$1; shift
  local out=$OUT_ROOT/runs/$name
  mkdir -p "$out"
  echo "=== train $name: $*"
  $PY src/train_decoder.py --out "$out" "$@" || { echo "TRAIN FAILED"; return 1; }
  test -f "$out/best_trainable.pt" || { echo "no best_trainable.pt"; return 1; }
  evaluate test "$out/test" --weights "$out/best_trainable.pt" $LIM > "$out/eval.log" 2>&1 &
  local p1=$!
  evaluate test_nofilter "$out/test_nofilter" --weights "$out/best_trainable.pt" $LIM --pred-iou 0 --stability 0 \
      > "$out/eval_nofilter.log" 2>&1
  local rc2=$?
  wait $p1; local rc1=$?
  cat "$out/eval.log" "$out/eval_nofilter.log"
  [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
}

# ---- queue (most important first) ---------------------------------------------------------
common="--epochs $EPOCHS --patience $PATIENCE --train-patches $TRAIN_PATCHES --parcels-per-patch $PARCELS \
--val-patches $VAL_PATCHES --gen-val-patches $GEN_VAL_PATCHES"
JOBS=(
  "base|job_base"
  "base_nofilter|job_base_nof"
  "pps16|job_pps 16"
  "pps32|job_pps 32"
  "pps64|job_pps 64"
  "A_centroid_tok0|job_train A_centroid_tok0 --prompt centroid --tokens 0 --select-by generator $common $EXTRA_TRAIN"
  "D_random_multi|job_train D_random_multi --prompt random --tokens multi --select-by generator $common $EXTRA_TRAIN"
  "E_grid_multi|job_train E_grid_multi --prompt grid --tokens multi --select-by generator $common $EXTRA_TRAIN"
  "B_centroid_multi|job_train B_centroid_multi --prompt centroid --tokens multi --select-by generator $common $EXTRA_TRAIN"
  "C_random_tok0|job_train C_random_tok0 --prompt random --tokens 0 --select-by generator $common $EXTRA_TRAIN"
  "A_loss_select|job_train A_loss_select --prompt centroid --tokens 0 --select-by loss $common $EXTRA_TRAIN"
)
if [ -n "${ONLY:-}" ]; then   # e.g. ONLY="base pps16" to run a subset
  sel=(); for e in "${JOBS[@]}"; do case " $ONLY " in *" ${e%%|*} "*) sel+=("$e");; esac; done
  JOBS=("${sel[@]}")
fi

run_job() {   # name, command... : runs in a background subshell, writes one progress line
  local name=$1; shift
  local t=$(date +%s)
  log "START $name"
  "$@" > "$OUT_ROOT/logs/$name.log" 2>&1
  local rc=$?
  local status=OK; [ $rc -ne 0 ] && status="FAIL(rc=$rc)"
  log "$status $name ($(( ($(date +%s) - t) / 60 ))m$(( ($(date +%s) - t) % 60 ))s) log=$OUT_ROOT/logs/$name.log"
}

n_running() { jobs -rp | wc -l | tr -d ' '; }

for entry in "${JOBS[@]}"; do
  while [ "$(n_running)" -ge "$PARALLEL" ]; do sleep 5; done   # portable replacement for wait -n
  name=${entry%%|*}; cmd=${entry#*|}
  run_job "$name" $cmd &
  sleep 2   # stagger model loading
done
wait
log "ALL DONE ($(( ($(date +%s) - T0) / 60 )) min). Summaries:"
for f in "$OUT_ROOT"/diag/*_summary.csv "$OUT_ROOT"/runs/*/test*_summary.csv; do
  [ -f "$f" ] && { echo "--- $f"; cat "$f"; } >> "$PROGRESS"
done
tail -n +1 "$PROGRESS" | grep -E "OK|FAIL|ALL DONE"
