#!/usr/bin/env bash
set -euo pipefail

# =========================
# TimeBridge Finance Stage A hyperparam search (local-friendly)
# Train: <= 2022-12-31
#  Val:  2023-01-01 ~ 2024-12-31
# Test:  >= 2025-01-01
#
# - Print to terminal (so you can see progress/errors)
# - Save full logs per run under logs/Finance/TimeBridge/
# - Summarize best validation loss per run -> summary_val_loss.tsv
# =========================

# ---- paths ----
ROOT_PATH="/Users/siweiwang/Documents/stocks/TimeBridge/data/spy_constituents_hist/dataset/sp500_2018_2025_logret/"
DATA_PATH="sp500_2018_2025_logret_raw_clean_spyfull_2018_2025.csv"

# ---- task fixed args ----
MODEL="TimeBridge"
DATA="custom"
FEATURES="M"
FREQ="d"
ENC_IN=359

SPLIT_MODE="date"
VAL_START="2023-01-01"
TEST_START="2025-01-01"

# ---- training fixed args ----
BATCH_SIZE=16
NUM_WORKERS=0
LABEL_LEN=50
PRED_LEN=1

# Safer seq_len for daily finance, also tends to avoid internal divisibility quirks
SEQ_LEN=240
PERIOD=24

TRAIN_EPOCHS=60
PATIENCE=10

# ---- search space (16 runs by default) ----
# Rule for heads: keep head_dim around 16
# d_model=128 -> heads=8, d_model=256 -> heads=16
D_MODELS=(128 256)
IA_LAYERS=(1 2)
CA_LAYERS=(0 1)         # once stable, you can extend to (0 1 2)
LRS=(0.0002 0.0005)
ALPHA=0.2               # after you find top2, you can micro-tune alpha: 0.1/0.2/0.35

# d_ff fixed to 2*d_model to reduce grid size; can later compare d_ff=d_model vs 2*d_model
heads_for_dmodel() {
  local dm="$1"
  if [[ "$dm" -eq 128 ]]; then echo 8; return; fi
  if [[ "$dm" -eq 256 ]]; then echo 16; return; fi
  echo $((dm/16))
}
dff_for_dmodel() {
  local dm="$1"
  echo $((dm*2))
}

# ---- logs ----
LOGDIR="/Users/siweiwang/Documents/stocks/TimeBridge/logs/Finance/TimeBridge"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary_val_loss.tsv"

echo "=== Finance Stage A grid search ==="
echo "root_path=$ROOT_PATH"
echo "data_path=$DATA_PATH"
echo "seq_len=$SEQ_LEN  period=$PERIOD  enc_in=$ENC_IN"
echo "val_start=$VAL_START  test_start=$TEST_START"
echo "logs=$LOGDIR"
echo ""

# count total
total=0
for dm in "${D_MODELS[@]}"; do
  for ia in "${IA_LAYERS[@]}"; do
    for ca in "${CA_LAYERS[@]}"; do
      for lr in "${LRS[@]}"; do
        total=$((total+1))
      done
    done
  done
done
echo "Grid size = $total runs"
echo ""

# run loop
k=0
for dm in "${D_MODELS[@]}"; do
  heads="$(heads_for_dmodel "$dm")"
  dff="$(dff_for_dmodel "$dm")"

  for ia in "${IA_LAYERS[@]}"; do
    for ca in "${CA_LAYERS[@]}"; do
      for lr in "${LRS[@]}"; do
        k=$((k+1))
        model_id="A_fin_sl${SEQ_LEN}_dm${dm}_h${heads}_ia${ia}_ca${ca}_lr${lr}_a${ALPHA}"
        logfile="${LOGDIR}/${model_id}.log"

        echo "================================================================"
        echo "[$k/$total] RUN ${model_id}"
        echo "log: ${logfile}"
        echo "================================================================"

        cmd=(
          python3.11 -u run.py
          --is_training 1
          --model_id "${model_id}"
          --model "${MODEL}"
          --data "${DATA}"
          --root_path "${ROOT_PATH}"
          --data_path "${DATA_PATH}"
          --features "${FEATURES}"
          --freq "${FREQ}"
          --seq_len "${SEQ_LEN}"
          --label_len "${LABEL_LEN}"
          --pred_len "${PRED_LEN}"
          --enc_in "${ENC_IN}"
          --split_mode "${SPLIT_MODE}"
          --val_start "${VAL_START}"
          --test_start "${TEST_START}"
          --batch_size "${BATCH_SIZE}"
          --num_workers "${NUM_WORKERS}"
          --period "${PERIOD}"
          --d_model "${dm}"
          --d_ff "${dff}"
          --n_heads "${heads}"
          --ia_layers "${ia}"
          --pd_layers 1
          --ca_layers "${ca}"
          --alpha "${ALPHA}"
          --learning_rate "${lr}"
          --train_epochs "${TRAIN_EPOCHS}"
          --patience "${PATIENCE}"
          --itr 1
        )

        # Show on terminal AND save to logfile
        set -o pipefail
        "${cmd[@]}" 2>&1 | tee "${logfile}"
        set +o pipefail

      done
    done
  done
done

echo ""
echo "=== All runs finished ==="
echo "Logs: $LOGDIR"
echo ""