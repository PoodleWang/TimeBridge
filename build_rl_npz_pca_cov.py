# -*- coding: utf-8 -*-
"""
Build RL dataset with:
- pred_t1 scores (T,C)
- pred_t3 scores (T,C) from horizon=3 (index=2)
- PCA relational embedding from rolling correlation (no-parameter)
- rewards aligned to "decision day" consistent with your eval scripts

This script aligns pred rows to raw_csv by the SAME indexing you used in eval_sharpe_rank_raw_datesplit(_h3).py

Key alignment:
- raw_csv: date + per-ticker LOG returns (logret), shape (N,C)
- test_start: first test date
- idx_test_start = first index with date >= test_start
- border1 = max(0, idx_test_start - seq_len)

For pred row i (0-indexed):
- decision day index (in df) : d_idx = border1 + i + seq_len - 1
  meaning the input window ends at day d_idx (inclusive),
  and reward uses future days after d_idx.

Rewards:
- reward_1d[i] uses logret at day d_idx+1
- reward_h[i]  uses sum logret over days [d_idx+1 .. d_idx+h] (inclusive), then expm1

Relational embedding:
- rel_full[t] computed from rolling window of logret that ends at day t (inclusive)
  window = logret[t-lookback+1 : t+1]
- rel[i] = rel_full[d_idx]

Outputs npz:
- dates_decision: (T_eff,) datetime64[ns]
- tickers: (C,)
- pred_t1: (T_eff,C)
- pred_t3: (T_eff,C)
- rel: (T_eff,C,rel_dim)
- reward_1d: (T_eff,C)
- reward_h: (T_eff,C)
- meta: seq_len, lookback, rel_dim, horizon_h, test_start, border1, idx_test_start
"""

import argparse
import numpy as np
import pandas as pd


def load_pred_scores(pred_path: str, horizon: int = 1) -> np.ndarray:
    """
    Return scores (T, C) for the given horizon (1-indexed).
    Supports shapes:
      (T, H, C)
      (T, H, 1, C)
      (T, 1, H, C)
      (T, 1, C)  (treated as H=1)
      (T, C)
    """
    x = np.load(pred_path)
    x = np.asarray(x)

    # squeeze singleton dims except keep (T, H, C) if possible
    while x.ndim > 3:
        squeezed = False
        for ax in range(1, x.ndim):
            if x.shape[ax] == 1:
                x = np.squeeze(x, axis=ax)
                squeezed = True
                break
        if not squeezed:
            break

    if x.ndim == 2:
        if horizon != 1:
            raise ValueError(f"pred is (T,C) but horizon={horizon}. Need pred_len>=horizon.")
        return x.astype(np.float32)

    if x.ndim != 3:
        raise ValueError(f"Unsupported pred shape after squeeze: {x.shape}")

    T, H, C = x.shape
    if horizon < 1 or horizon > H:
        raise ValueError(f"horizon={horizon} out of range for pred_len={H}")

    return x[:, horizon - 1, :].astype(np.float32)  # (T,C)


def rolling_corr_pca_embedding_from_logret(
    logret: np.ndarray,
    lookback: int = 64,
    rel_dim: int = 8,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute no-param relational embedding per date t using rolling correlation PCA.

    logret: (N,C) daily log returns, where row t is logret into day t (close[t-1]->close[t])
    For each t >= lookback-1:
      window = logret[t-lookback+1 : t+1]  (ends at t inclusive)
      standardize per stock within window => correlation structure
      do SVD on (lookback x C): X = U diag(s) Vt
      eigenvalues approx s^2/(n-1), eigenvectors columns of V
      embedding per stock = V_k * sqrt(lambda_k)

    Returns:
      rel_full: (N,C,rel_dim)  (t < lookback-1 -> zeros)
    """
    N, C = logret.shape
    rel_full = np.zeros((N, C, rel_dim), dtype=np.float32)

    for t in range(lookback - 1, N):
        W = logret[t - lookback + 1 : t + 1]  # (lookback, C)

        mu = W.mean(axis=0, keepdims=True)
        sd = W.std(axis=0, ddof=0, keepdims=True)
        sd = np.where(sd < eps, 1.0, sd)
        X = (W - mu) / sd  # (lookback, C)

        try:
            _, s, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        k = min(rel_dim, Vt.shape[0])  # rank <= lookback
        V_k = Vt[:k].T  # (C,k)

        n = X.shape[0]
        lam = (s[:k] ** 2) / max(1.0, (n - 1.0))
        lam = np.maximum(lam, 0.0)

        emb = V_k * np.sqrt(lam.reshape(1, -1))  # (C,k)

        # stabilize sign to reduce random flips day-to-day
        sign = np.sign(emb[0, :])
        sign[sign == 0] = 1.0
        emb = emb * sign.reshape(1, -1)

        if k < rel_dim:
            emb = np.concatenate([emb, np.zeros((C, rel_dim - k), dtype=emb.dtype)], axis=1)

        rel_full[t] = emb.astype(np.float32)

    return rel_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_t1", required=True)
    ap.add_argument("--pred_t3", required=True)
    ap.add_argument("--raw_csv", required=True, help="CSV with 'date' + per-ticker LOG returns (logret)")
    ap.add_argument("--seq_len", type=int, default=250)
    ap.add_argument("--test_start", type=str, default="2025-01-01")
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--rel_dim", type=int, default=8)
    ap.add_argument("--horizon_h", type=int, default=3, help="holding horizon for reward_h (e.g., 3)")
    ap.add_argument("--out_npz", type=str, default="rl_dataset_pca.npz")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # --- load preds (scores) ---
    pred1 = load_pred_scores(args.pred_t1, horizon=1)             # (T1,C)
    pred3 = load_pred_scores(args.pred_t3, horizon=3)             # (T3,C) pick step 3
    if pred1.shape[1] != pred3.shape[1]:
        raise ValueError(f"pred C mismatch: pred1 {pred1.shape}, pred3 {pred3.shape}")

    T1, C = pred1.shape
    T3, _ = pred3.shape

    # --- load raw logret csv ---
    df = pd.read_csv(args.raw_csv)
    if df.columns[0] != "date" and "date" not in df.columns:
        # allow first col be date
        df = df.rename(columns={df.columns[0]: "date"})
    if "date" not in df.columns:
        raise ValueError("raw_csv must contain a 'date' column (or first column is date).")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    tickers = [c for c in df.columns if c != "date"]
    if len(tickers) != C:
        raise ValueError(f"Channel mismatch: pred C={C}, raw cols={len(tickers)}")

    logret = df[tickers].astype(float).to_numpy(dtype=np.float32)  # (N,C)
    N = len(df)

    test_start = pd.to_datetime(args.test_start).normalize()
    idx_candidates = df.index[df["date"] >= test_start].to_list()
    if not idx_candidates:
        raise ValueError(f"test_start={args.test_start} is after last date in raw_csv.")
    idx_test_start = int(idx_candidates[0])

    seq_len = int(args.seq_len)
    border1 = max(0, idx_test_start - seq_len)

    # pred row i corresponds to decision day index:
    # d_idx = border1 + i + seq_len - 1
    # and rewards use future days after d_idx.
    def d_idx_for_i(i: int) -> int:
        return border1 + i + seq_len - 1

    # maximum i such that reward_h exists up to d_idx + horizon_h
    h = int(args.horizon_h)
    if h <= 0:
        raise ValueError("--horizon_h must be positive")

    max_i_by_data = (N - 1 - h) - (border1 + seq_len - 1)  # ensure d_idx+h <= N-1
    if max_i_by_data < 0:
        raise ValueError("Not enough data for given seq_len/test_start/horizon_h.")

    T_eff = min(T1, T3, max_i_by_data + 1)
    if T_eff <= 0:
        raise ValueError("T_eff <= 0, check inputs.")
    if args.debug:
        print(f"[DBG] pred1={pred1.shape} pred3={pred3.shape} raw N={N} C={C}")
        print(f"[DBG] test_start={test_start.date()} idx_test_start={idx_test_start} border1={border1} seq_len={seq_len}")
        print(f"[DBG] max_i_by_data={max_i_by_data} -> T_eff={T_eff} (min(T1,T3,data))")
        print(f"[DBG] first decision index d0={d_idx_for_i(0)} date={df['date'].iloc[d_idx_for_i(0)].date()}")
        print(f"[DBG] last  decision index dL={d_idx_for_i(T_eff-1)} date={df['date'].iloc[d_idx_for_i(T_eff-1)].date()}")

    pred1 = pred1[:T_eff]
    pred3 = pred3[:T_eff]

    # --- compute relational embedding for ALL dates, then gather at decision dates ---
    rel_full = rolling_corr_pca_embedding_from_logret(
        logret, lookback=int(args.lookback), rel_dim=int(args.rel_dim)
    )

    # --- build aligned arrays ---
    dates_decision = np.empty((T_eff,), dtype="datetime64[ns]")
    rel = np.zeros((T_eff, C, int(args.rel_dim)), dtype=np.float32)
    reward_1d = np.zeros((T_eff, C), dtype=np.float32)
    reward_h = np.zeros((T_eff, C), dtype=np.float32)

    for i in range(T_eff):
        d_idx = d_idx_for_i(i)
        dates_decision[i] = df["date"].iloc[d_idx].to_datetime64()

        # relational embedding at decision day
        rel[i] = rel_full[d_idx]

        # 1-day reward: day d_idx+1 logret -> simple ret
        r1 = np.expm1(logret[d_idx + 1]).astype(np.float32)
        reward_1d[i] = r1

        # h-day reward: sum logret over [d_idx+1 .. d_idx+h]
        logret_h = logret[d_idx + 1 : d_idx + h + 1].sum(axis=0)
        rh = np.expm1(logret_h).astype(np.float32)
        reward_h[i] = rh

    # quick sanity prints
    if args.debug:
        print(f"[DBG] rel_full shape={rel_full.shape}, rel gathered={rel.shape}")
        print(f"[DBG] reward_1d shape={reward_1d.shape}, reward_h shape={reward_h.shape}")

    np.savez_compressed(
        args.out_npz,
        dates_decision=dates_decision,
        tickers=np.array(tickers, dtype=object),
        pred_t1=pred1.astype(np.float32),
        pred_t3=pred3.astype(np.float32),
        rel=rel.astype(np.float32),
        reward_1d=reward_1d.astype(np.float32),
        reward_h=reward_h.astype(np.float32),
        # meta
        seq_len=np.int32(seq_len),
        lookback=np.int32(args.lookback),
        rel_dim=np.int32(args.rel_dim),
        horizon_h=np.int32(h),
        test_start=np.array(str(test_start.date()), dtype=object),
        idx_test_start=np.int32(idx_test_start),
        border1=np.int32(border1),
    )

    print("=" * 80)
    print(f"[SAVED] {args.out_npz}")
    print(f"T_eff={T_eff}  C={C}")
    print(f"decision date range: {pd.to_datetime(dates_decision[0]).date()} -> {pd.to_datetime(dates_decision[-1]).date()}")
    print(f"pred_t1: {pred1.shape}  pred_t3: {pred3.shape}  rel: {rel.shape}")
    print(f"reward_1d: {reward_1d.shape}  reward_h(T+{h}): {reward_h.shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()
