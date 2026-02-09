import pandas as pd
csv = "/Users/siweiwang/Documents/stocks/TimeBridge/data/spy_constituents_hist/dataset/sp500_2018_2025_logret/sp500_2018_2025_logret_raw_clean_spyfull_2018_2025.csv"
df = pd.read_csv(csv, nrows=1)
print(len(df.columns) - 1)

# python3.11 eval_sharpe_rank_raw_datesplit.py \
#   --pred results/ORIGSTRUCT_TRAIN_TO_2024_TimeBridge_custom_bs16_ftM_sl250_ll50_pl1_dm64_nh8_ial3_pdl1_cal0_df64_ebtimeF_test_0/pred.npy \
#   --raw_csv /Users/siweiwang/Documents/stocks/TimeBridge/data/spy_constituents_hist/dataset/sp500_2018_2025_logret/sp500_2018_2025_logret_raw_clean_spyfull_2018_2025.csv \
#   --topk 10 \
#   --seq_len 250 \
#   --test_start 2025-01-01 \
#   --tc_bps 3


# python3.11 -u run.py \
#   --is_training 1 \
#   --model_id sp500_dateSplit_clean_250_1 \
#   --model TimeBridge \
#   --data custom \
#   --root_path /Users/siweiwang/Documents/stocks/TimeBridge/data/spy_constituents_hist/dataset/sp500_2018_2025_logret/ \
#   --data_path sp500_2018_2025_logret_raw_clean_spyfull_2018_2025.csv \
#   --features M \
#   --freq d \
#   --seq_len 250 \
#   --label_len 50 \
#   --pred_len 1 \
#   --enc_in 359 \
#   --split_mode date \
#   --val_start 2024-01-01 \
#   --test_start 2025-01-01 \
#   --batch_size 16 \
#   --num_workers 0 \
#   --d_model 64 \
#   --d_ff 64 \
#   --n_heads 8 \
#   --ia_layers 3 \
#   --pd_layers 1 \
#   --ca_layers 0 \
#   --alpha 0.35 \
#   --learning_rate 0.0002 \
#   --train_epochs 10 \
#   --patience 3 \
#   --itr 1


