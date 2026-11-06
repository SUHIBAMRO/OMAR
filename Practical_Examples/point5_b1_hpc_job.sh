#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omar
cd ~/OMAR/Practical_Examples

RUN_DIR=/scratch/gole3157/pfem_run_b1
CKPT_CANDIDATES_DIR=/scratch/gole3157/pfem_run_b1_ckpt_candidates
MATERIAL=neo_hookean
NTRAIN=800
NTEST=200

mkdir -p "$RUN_DIR"

echo "=== Resolving the real PI checkpoint by fingerprint (not a guessed path) ==="
python3 -c "
from omar_pfem.resolve_b1_checkpoint import resolve_b1_neo_hookean_checkpoint
path, fp = resolve_b1_neo_hookean_checkpoint('$CKPT_CANDIDATES_DIR')
print(path)
" | tee /tmp/resolve_pi_ckpt_$$.log
PI_CKPT=$(tail -n 1 /tmp/resolve_pi_ckpt_$$.log)
rm -f /tmp/resolve_pi_ckpt_$$.log
echo "Resolved PI checkpoint: $PI_CKPT"

for N in 13 33; do
  LABEL="B1_${MATERIAL}_dd_labelN${N}"
  H5_DIR="$RUN_DIR/fem_${LABEL}"
  NPZ_DIR="$RUN_DIR/npz_${LABEL}"
  NPZ_PATH="$NPZ_DIR/hyperelastic_training_data_q4.npz"
  OUT_DIR="$RUN_DIR/$LABEL"

  echo ""
  echo "=================================================================="
  echo "=== $LABEL: generating DD labels at N=$N ==="
  if [ ! -f "$NPZ_PATH" ]; then
    python -u -m omar_pfem.data.data_generate_B1 \
      --num_index 1 --num_samples $((NTRAIN + NTEST)) \
      --Nx "$N" --Ny "$N" --material "$MATERIAL" \
      --n_workers "$(nproc)" --out_dir "$H5_DIR"
    python -u -m omar_pfem.data.convert_B1_quad \
      --h5_dir "$H5_DIR" --out_dir "$NPZ_DIR"
  else
    echo "[$LABEL] dataset already exists, skipping generation."
  fi

  echo "=== $LABEL: training DD (AdamW+OneCycleLR, the best-DD recipe) ==="
  if [ ! -f "$OUT_DIR/model_best.pt" ]; then
    python -u -m omar_pfem.train_data_driven \
      --geometry B1 --material "$MATERIAL" \
      --path "$NPZ_PATH" --out_dir "$OUT_DIR" \
      --ntrain $NTRAIN --ntest $NTEST \
      --match_pi_optimizer 0
  else
    echo "[$LABEL] DD checkpoint already exists, skipping training."
  fi

  echo "=== $LABEL: evaluating PI vs. DD against the real N=2236 reference ==="
  python -u -m omar_pfem.evaluate_B1_pi_vs_dd_fine_reference \
    --pi_checkpoint "$PI_CKPT" --dd_checkpoint "$OUT_DIR/model_best.pt" \
    --resolutions 13 29 49 --fine_N 2236 --n_seeds 5 --material "$MATERIAL" \
    --out_json "$RUN_DIR/point5_${LABEL}_vs_N2236.json"
done

echo ""
echo "All done. Results in $RUN_DIR/point5_*_vs_N2236.json"
