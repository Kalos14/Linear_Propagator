from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl

import priceprop.batch as pp_batch
import priceprop.propagator as prop


# -------------------------
# CONFIG
# -------------------------
CLEAN_ROOT = Path("clean data")
OUT_DIR = Path("results") / "plots" / "flashcrash_modeA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRASH_DAY = pd.Timestamp("2010-05-06").date()

N_HORIZON = 200

CRASH_TRUE_THRESH = -0.02
PRED_Z = 3.0


# -------------------------
# Helpers
# -------------------------
def load_ticker_tt(clean_root: Path, ticker: str) -> pd.DataFrame:
    tdir = clean_root / ticker
    if not tdir.exists():
        raise FileNotFoundError(f"Missing folder: {tdir}")
    files = sorted(tdir.glob(f"{ticker}_*_felix_tt_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"No cleaned parquets found in {tdir}")

    lf = pl.scan_parquet([f.as_posix() for f in files]).select(
        ["date", "ts_ms", "sign", "r1", "change"]
    )
    df = lf.collect().to_pandas()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["sign"] = df["sign"].astype(np.int8)
    df["change"] = df["change"].astype(bool)
    df["r1"] = df["r1"].astype(float)
    df = df.sort_values(["date", "ts_ms"]).reset_index(drop=True)
    return df


def rolling_sum_forward(x: np.ndarray, N: int) -> np.ndarray:
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    x0 = np.where(mask, x, 0.0)

    y = np.convolve(x0, np.ones(N, dtype=float), mode="valid")
    valid = np.convolve(mask.astype(float), np.ones(N, dtype=float), mode="valid") == N
    y[~valid] = np.nan
    return y


def align_valid(ts_ms: np.ndarray, arr_valid: np.ndarray) -> np.ndarray:
    return ts_ms[: len(arr_valid)]


def smooth_kernel(x: np.ndarray) -> np.ndarray:
    return prop.smooth_tail_rbf(np.asarray(x, float))


def delta_c_from_train(tt_train: pd.DataFrame) -> float:
    x = tt_train.loc[tt_train["change"], "r1"].abs()
    return float(x.mean()) if len(x) else np.nan


def to_time_of_day(ts_ns: np.ndarray) -> pd.DatetimeIndex:
    dt = pd.to_datetime(ts_ns, unit="ns", utc=True)
    return dt.tz_convert("America/New_York")


def first_crossing(x: np.ndarray, thresh: float) -> int | None:
    idx = np.where(np.isfinite(x) & (x < thresh))[0]
    return int(idx[0]) if len(idx) else None


def calibrate_on_train(tt_train: pd.DataFrame, models: list[str]) -> dict:
    """
    Calibrate kernels on training set.
    If HDIM2 is requested, prefer group=True for robustness/performance.
    """
    tt_train = tt_train.copy()
    pp_batch.complete_data_columns(tt_train, split_dates=False)

    use_group = ("hdim2" in models)  # faster + avoids nasty edge cases
    cal = pp_batch.calibrate_models(
        tt_train,
        nfft="pad",
        group=use_group,
        models=models,
    )
    return cal


def predict_on_df(tt_df: pd.DataFrame, cal: dict, models: list[str], delta_c: float) -> pd.DataFrame:
    """
    Compute per-trade predicted returns for requested models using calibrated kernels.
    Returns a copy with columns r_cim, r_tim1, r_tim2, r_hdim2 (if requested).
    """
    out = tt_df.copy()
    pp_batch.complete_data_columns(out, split_dates=False)

    s = out["sign"].values
    c = out["change"].values

    # CIM2 scaled by delta_c to convert to return units
    if "cim" in models:
        out["r_cim"] = (out["change"].astype(float) * out["sign"].astype(float)) * float(delta_c)

    # TIM1
    if "tim1" in models:
        g = smooth_kernel(cal["g"])
        out["r_tim1"] = prop.tim1(s, g)

    # TIM2
    if "tim2" in models:
        gn = smooth_kernel(cal["gn"])
        gc = smooth_kernel(cal["gc"])
        out["r_tim2"] = prop.tim2(s, c, gn, gc)

    # HDIM2
    if "hdim2" in models:
        kn = smooth_kernel(cal["kn"])
        kc = smooth_kernel(cal["kc"])
        out["r_hdim2"] = prop.hdim2(s, c, kn, kc)

    return out


# -------------------------
# Main Mode A routine
# -------------------------
def run_modeA_for_ticker(
    ticker: str,
    models: list[str] = ["cim", "tim1", "tim2", "hdim2"],
) -> None:
    tt = load_ticker_tt(CLEAN_ROOT, ticker)

    tt_train = tt[tt["date"] < CRASH_DAY].copy()
    tt_test = tt[tt["date"] == CRASH_DAY].copy()

    if tt_test.empty:
        raise RuntimeError(f"{ticker}: no data found for {CRASH_DAY} in cleaned files.")
    if len(tt_train) < 10_000:
        print(f"[warn] {ticker}: training set seems small ({len(tt_train)} events).")

    # delta_c from train only (no leakage)
    dc = delta_c_from_train(tt_train)
    if not np.isfinite(dc) or dc <= 0:
        raise RuntimeError(f"{ticker}: bad delta_c from train: {dc}")

    # calibrate on train
    cal = calibrate_on_train(tt_train, models=models)

    # predict on crash day + on train (for thresholds)
    pred_test = predict_on_df(tt_test, cal=cal, models=models, delta_c=dc)
    pred_train = predict_on_df(tt_train, cal=cal, models=models, delta_c=dc)

    # forward true on crash day
    r_true = rolling_sum_forward(pred_test["r1"].to_numpy(), N_HORIZON)
    ts_valid = align_valid(pred_test["ts_ms"].to_numpy(), r_true)
    dt_valid = to_time_of_day(ts_valid)

    crash_idx = first_crossing(r_true, CRASH_TRUE_THRESH)

    # choose alarm model (prefer HDIM2 > TIM2 > TIM1 > CIM)
    if "hdim2" in models and "r_hdim2" in pred_test.columns:
        alarm_model = "r_hdim2"
    elif "tim2" in models and "r_tim2" in pred_test.columns:
        alarm_model = "r_tim2"
    elif "tim1" in models and "r_tim1" in pred_test.columns:
        alarm_model = "r_tim1"
    else:
        alarm_model = "r_cim"

    # threshold on TRAIN predicted forward sums
    train_Rpred = rolling_sum_forward(pred_train[alarm_model].to_numpy(), N_HORIZON)
    mu = np.nanmean(train_Rpred)
    sd = np.nanstd(train_Rpred)
    pred_thresh = mu - PRED_Z * sd

    # forward predicted sums for each model on crash day
    series = {}
    for col in ["r_cim", "r_tim1", "r_tim2", "r_hdim2"]:
        if col in pred_test.columns:
            series[col] = rolling_sum_forward(pred_test[col].to_numpy(), N_HORIZON)

    Rpred_alarm = series[alarm_model]
    alarm_idx = first_crossing(Rpred_alarm, pred_thresh)

    resid = r_true - Rpred_alarm

    # intraday log-price proxy
    logp = np.nancumsum(pred_test["r1"].to_numpy())
    dt_day = to_time_of_day(pred_test["ts_ms"].to_numpy())

    # -------------------------
    # REPORT
    # -------------------------
    print(f"\n=== {ticker} Mode A (Flash Crash {CRASH_DAY}) ===")
    print(f"Train events: {len(tt_train):,}  Test-day events: {len(tt_test):,}")
    print(f"delta_c (train): {dc:.6g}")
    print(f"N_HORIZON: {N_HORIZON} trades")
    print(f"Crash threshold (true): {CRASH_TRUE_THRESH:.4f}  (log-return)")
    print(f"Alarm model: {alarm_model}  pred threshold = mean - {PRED_Z}*std = {pred_thresh:.4f}")

    if crash_idx is None:
        print("[info] No crash trigger on this ticker/day under your CRASH_TRUE_THRESH.")
    else:
        print(f"[crash] first trigger at {dt_valid[crash_idx]}")
        print(f"        R_true^{N_HORIZON} = {r_true[crash_idx]:.4f}")

    if alarm_idx is None:
        print("[alarm] No early-warning trigger from predicted return under pred_thresh.")
    else:
        print(f"[alarm] first trigger at {dt_valid[alarm_idx]}")
        print(f"        R_pred^{N_HORIZON} = {Rpred_alarm[alarm_idx]:.4f}")
        if crash_idx is not None:
            lead_trades = crash_idx - alarm_idx
            lead_time = dt_valid[crash_idx] - dt_valid[alarm_idx]
            print(f"        lead: {lead_trades} rolling-steps (~trades) ; lead time: {lead_time}")

    # -------------------------
    # PLOTS
    # -------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    axes[0].plot(dt_day, logp, linewidth=1.0)
    axes[0].set_title(f"{ticker} {CRASH_DAY} — intraday log-price proxy (cumsum r1)")
    axes[0].set_ylabel("log price (shifted)")

    axes[1].plot(dt_valid, r_true, label="R_true", linewidth=1.5)

    for k, v in series.items():
        axes[1].plot(dt_valid, v, label=k.replace("r_", "R_"), linewidth=1.2)

    axes[1].axhline(CRASH_TRUE_THRESH, linestyle="--", linewidth=1.0, label="crash thresh (true)")
    axes[1].axhline(pred_thresh, linestyle=":", linewidth=1.0, label="alarm thresh (pred)")

    axes[1].set_title(f"Forward N-trade returns (N={N_HORIZON})")
    axes[1].set_ylabel("forward log-return")
    axes[1].legend(loc="best")

    if crash_idx is not None:
        axes[1].axvline(dt_valid[crash_idx], linestyle="--", linewidth=1.0)
        axes[2].axvline(dt_valid[crash_idx], linestyle="--", linewidth=1.0)
    if alarm_idx is not None:
        axes[1].axvline(dt_valid[alarm_idx], linestyle=":", linewidth=1.0)
        axes[2].axvline(dt_valid[alarm_idx], linestyle=":", linewidth=1.0)

    axes[2].plot(dt_valid, resid, linewidth=1.2)
    axes[2].set_title(f"Residual e(t) = R_true - R_pred  (model: {alarm_model})")
    axes[2].set_ylabel("residual")
    axes[2].set_xlabel("NY time")

    plt.tight_layout()
    outpath = OUT_DIR / f"{ticker}_modeA_N{N_HORIZON}_{CRASH_DAY}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"[ok] saved plot: {outpath}")


if __name__ == "__main__":
    # Now includes HDIM2
    run_modeA_for_ticker("ORCL.OQ", models=["cim", "tim1", "tim2"])
