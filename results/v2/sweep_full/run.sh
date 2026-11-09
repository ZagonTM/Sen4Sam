cd /workspace/Sen4Sam
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for p in 32 64; do for k in 0 1 2 3; do
  python src/evaluate_masks.py --split-file results/sweep_full/split_shard$k.csv --split test --points-per-side $p --out results/sweep_full/pps${p}_shard$k > results/sweep_full/pps${p}_shard$k.log 2>&1 &
done; done
wait; echo SWEEP_FULL_DONE
