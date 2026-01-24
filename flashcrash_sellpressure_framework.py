from __future__ import annotations

"""
Flash-crash Mode A with sell-pressure conditioning (buy/sell framework).

High-level idea
- Build a causal sell-pressure proxy from past trade signs.
- Turn it into a soft gate w_t in [0, 1].
- Split signed flow into two channels:
    s_base   = sign * (1 - w)
    s_stress = sign * w
- Calibrate a two-kernel propagator (TIM2-style) on these channels:
    r_pred = propagate(s_base, g_base) + propagate(s_stress, g_stress)

Alarm rule (method 1: quantile threshold)
- Compute forward N-trade sums of predicted returns on the training set
- Set threshold as a low quantile of that training distribution (e.g., q=0.001)
- Trigger if crash-day predicted forward sum crosses below the threshold

Note
- Forward N-trade sums use future order flow within the day, so they are mainly a diagnostic.
  The goal here is to compare signals consistently with the previous Mode A script.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl

import scorr
import priceprop.propagator as prop


# -------------------------
# Defaults
# -------------------------
DEFAULT_CRASH_DAY = pd.Timestamp("2010-05-06").date()
DEFAULT_N_HORIZON = 200
DEFAULT_CRASH_TRUE_THRESH = -0.02

DEFAULT_NS_PRESSURE = 50
DEFAULT_TAU = 0.20
DEFAULT_ALPHA = 12.0

# Alarm method 1: quantile threshold on train forward sums
DEFAULT_ALARM_Q = 0.001  # 0.1% quantile


# -------------------------
# IO
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
    df["ts_ms"] = df["ts_ms"].astype(np.int64)   # stored as integer timestamp (ns units in your pipeline)
    df["sign"] = df["sign"].astype(np.int8)
    df["change"] = df["change"].astype(bool)
    df["r1"] = df["r1"].astype(float)

    df = df.sort_values(["date", "ts_ms"]).reset_index(drop=True)
    return df


# -------------------------
# Time / windows
# -------------------------
def to_time_of_day(ts_ns: np.ndarray) -> pd.DatetimeIndex:
    dt = pd.to_datetime(ts_ns, unit="ns", utc=True)
    return dt.tz_convert("America/New_York")


def rolling_sum_forward(x: np.ndarray, N: int) -> np.ndarray:
    """Forward sum: y[t] = sum_{j=0..N-1} x[t+j]. Length = len(x)-N+1."""
    x = np.asarray(x, float)
    if len(x) < N:
        return np.array([], dtype=float)

    mask = np.isfinite(x)
    x0 = np.where(mask, x, 0.0)

    y = np.convolve(x0, np.ones(N, dtype=float), mode="valid")
    valid = (np.convolve(mask.astype(float), np.ones(N, dtype=float), mode="valid") == N)
    y[~valid] = np.nan
    return y


def align_valid(ts_ns: np.ndarray, arr_valid: np.ndarray) -> np.ndarray:
    return ts_ns[: len(arr_valid)]


def first_crossing(x: np.ndarray, thresh: float) -> int | None:
    idx = np.where(np.isfinite(x) & (x < thresh))[0]
    return int(idx[0]) if len(idx) else None


def safe_quantile(x: np.ndarray, q: float) -> float:
    """Quantile on finite values only."""
    x = np.asarray(x, float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.nan
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(finite, q))


# -------------------------
# Sell-pressure features
# -------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_sell_pressure_features(
    tt: pd.DataFrame,
    ns_pressure: int = DEFAULT_NS_PRESSURE,
    tau: float = DEFAULT_TAU,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """
    Adds:
      - epsbar: rolling mean of sign over past ns_pressure trades (per day)
      - w:      stress gate in [0,1], high when epsbar is strongly negative
      - s_base:   sign*(1-w)
      - s_stress: sign*w
    """
    out = tt.copy()
    out["epsbar"] = np.nan
    out["w"] = 0.0

    # Per-day rolling mean (causal). Early trades don't have enough history:
    # set w=0 so the channels are still defined.
    for _, g in out.groupby("date", sort=True):
        s = g["sign"].astype(float)
        epsbar = s.rolling(window=ns_pressure, min_periods=ns_pressure).mean().to_numpy()
        out.loc[g.index, "epsbar"] = epsbar

        w = np.zeros_like(epsbar, dtype=float)
        finite = np.isfinite(epsbar)
        if np.any(finite):
            z = alpha * ((-epsbar[finite]) - float(tau))
            w[finite] = sigmoid(z)
        out.loc[g.index, "w"] = w

    out["s_base"] = out["sign"].astype(float) * (1.0 - out["w"].astype(float))
    out["s_stress"] = out["sign"].astype(float) * out["w"].astype(float)
    return out


# -------------------------
# Calibration utilities (TIM2-style on two continuous channels)
# -------------------------
def _return_response(ret: str, x: np.ndarray, maxlag: int):
    ret = ret.lower()
    res = []
    for i in ret:
        if i == "l":
            res.append(np.arange(-maxlag, maxlag + 1))
        elif i == "s":
            res.append(np.concatenate([x[-maxlag:], x[: maxlag + 1]]))
        elif i == "r":
            res.append(
                np.concatenate(
                    [
                        -np.cumsum(x[: -maxlag - 1 : -1])[::-1],
                        [0],
                        np.cumsum(x[:maxlag]),
                    ]
                )
            )
    return tuple(res) if len(res) > 1 else res[0]


def response_grouped_safe(
    df: pd.DataFrame,
    col_r: str,
    col_s: str,
    nfft: str = "pad",
    subtract_mean: bool = False,
    maxlag: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Day-by-day response using scorr, with an explicit maxlag cap."""
    x = scorr.xcorr_grouped_df(
        df,
        [col_r, col_s],
        by="date",
        nfft=nfft,
        funcs=(lambda x: x, lambda x: x),
        subtract_mean=subtract_mean,
        norm="cov",
        return_df=False,
    )[0]

    L = len(x) // 2
    L = int(L if maxlag is None else min(L, int(maxlag)))
    return _return_response("lsr", x, L)


def calibrate_stress_tim2(
    tt_train: pd.DataFrame,
    nfft: str = "pad",
    maxlag_cap: int = 256,
) -> dict:
    """Calibrate two kernels (base/stress) using the same TIM2 algebra."""
    df = tt_train.copy()
    df = df.loc[
        np.isfinite(df["r1"].values)
        & np.isfinite(df["s_base"].values)
        & np.isfinite(df["s_stress"].values)
    ]
    if len(df) == 0:
        raise ValueError("No valid rows in training set after NaN filtering.")

    nfft_opt, events_required = scorr.get_nfft(nfft, df.groupby("date")["r1"])
    maxlag = int(min(maxlag_cap, nfft_opt // 2, events_required))
    maxlag = max(1, maxlag)

    kwargs = {"subtract_mean": False, "norm": "cov"}

    cccorr = scorr.fftcrop(
        scorr.xcorr_grouped_df(df, ["s_stress", "s_stress"], nfft=nfft, return_df=False, **kwargs)[0],
        maxlag,
    )
    nncorr = scorr.fftcrop(
        scorr.xcorr_grouped_df(df, ["s_base", "s_base"], nfft=nfft, return_df=False, **kwargs)[0],
        maxlag,
    )
    cncorr = scorr.fftcrop(
        scorr.xcorr_grouped_df(df, ["s_stress", "s_base"], nfft=nfft, return_df=False, **kwargs)[0],
        maxlag,
    )
    nccorr = scorr.fftcrop(
        scorr.xcorr_grouped_df(df, ["s_base", "s_stress"], nfft=nfft, return_df=False, **kwargs)[0],
        maxlag,
    )

    signed_lags, Sl_base, _ = response_grouped_safe(df, "r1", "s_base", nfft=nfft, maxlag=maxlag)
    _, Sl_stress, _ = response_grouped_safe(df, "r1", "s_stress", nfft=nfft, maxlag=maxlag)

    gn, gc = prop.calibrate_tim2(
        nncorr, cccorr, cncorr, nccorr,
        Sl_base, Sl_stress,
        maxlag=maxlag
    )

    return {
        "maxlag": maxlag,
        "signed_lags": signed_lags,
        "gn": gn,
        "gc": gc,
    }


def propagate_weighted(x: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Propagate with identity sfunc (keep the channel weights)."""
    return prop.propagate(np.asarray(x, float), np.asarray(G, float), sfunc=lambda z: z)


def predict_stress_tim2_one_day(day_df: pd.DataFrame, cal: dict, smooth: bool = True) -> np.ndarray:
    s_base = day_df["s_base"].to_numpy(dtype=float)
    s_stress = day_df["s_stress"].to_numpy(dtype=float)

    # If anything non-finite slips through, treat it as zero contribution.
    if not np.all(np.isfinite(s_base)):
        s_base = np.where(np.isfinite(s_base), s_base, 0.0)
    if not np.all(np.isfinite(s_stress)):
        s_stress = np.where(np.isfinite(s_stress), s_stress, 0.0)

    gn = cal["gn"]
    gc = cal["gc"]
    if smooth:
        gn = prop.smooth_tail_rbf(np.asarray(gn, float))
        gc = prop.smooth_tail_rbf(np.asarray(gc, float))

    return propagate_weighted(s_base, gn) + propagate_weighted(s_stress, gc)


def predict_stress_by_day(tt: pd.DataFrame, cal: dict, smooth: bool = True) -> pd.Series:
    """Predict r_stress2 day-by-day (avoid cross-day memory)."""
    preds = pd.Series(np.nan, index=tt.index, dtype=float, name="r_stress2")
    for _, g in tt.groupby("date", sort=True):
        preds.loc[g.index] = predict_stress_tim2_one_day(g, cal, smooth=smooth)
    return preds


# -------------------------
# Main routine
# -------------------------
def delta_c_from_train(tt_train: pd.DataFrame) -> float:
    """Only used for the CIM baseline scaling."""
    x = tt_train.loc[tt_train["change"], "r1"].abs()
    return float(x.mean()) if len(x) else np.nan


def run_modeA_sellpressure(
    tt: pd.DataFrame,
    ticker: str = "TICKER",
    crash_day=DEFAULT_CRASH_DAY,
    out_dir: Path | None = None,
    n_horizon: int = DEFAULT_N_HORIZON,
    crash_true_thresh: float = DEFAULT_CRASH_TRUE_THRESH,
    ns_pressure: int = DEFAULT_NS_PRESSURE,
    tau: float = DEFAULT_TAU,
    alpha: float = DEFAULT_ALPHA,
    smooth_kernel: bool = True,
    alarm_q: float = DEFAULT_ALARM_Q,
) -> None:
    if out_dir is None:
        out_dir = Path("results") / "plots" / "flashcrash_sellpressure"
    out_dir.mkdir(parents=True, exist_ok=True)

    tt_train = tt[tt["date"] < crash_day].copy().reset_index(drop=True)
    tt_test = tt[tt["date"] == crash_day].copy().reset_index(drop=True)

    if tt_test.empty:
        raise RuntimeError(f"{ticker}: no data found for {crash_day} in cleaned files.")
    if len(tt_train) < 10_000:
        print(f"[warn] {ticker}: training set seems small ({len(tt_train)} events).")

    dc = delta_c_from_train(tt_train)
    if not np.isfinite(dc) or dc <= 0:
        dc = float(np.nanmean(np.abs(tt_train["r1"].to_numpy())))
        print(f"[warn] {ticker}: delta_c from change-trades unavailable; fallback dc={dc:.6g}")

    train_feat = add_sell_pressure_features(tt_train, ns_pressure=ns_pressure, tau=tau, alpha=alpha)
    test_feat = add_sell_pressure_features(tt_test, ns_pressure=ns_pressure, tau=tau, alpha=alpha)

    cal = calibrate_stress_tim2(train_feat, nfft="pad", maxlag_cap=256)

    test_feat["r_stress2"] = predict_stress_by_day(test_feat, cal, smooth=smooth_kernel).to_numpy()
    train_feat["r_stress2"] = predict_stress_by_day(train_feat, cal, smooth=smooth_kernel).to_numpy()

    # Baseline CIM (kept for comparison)
    test_feat["r_cim"] = (test_feat["change"].astype(float) * test_feat["sign"].astype(float)) * float(dc)
    train_feat["r_cim"] = (train_feat["change"].astype(float) * train_feat["sign"].astype(float)) * float(dc)

    # Forward true on crash day
    r_true = rolling_sum_forward(test_feat["r1"].to_numpy(), n_horizon)
    ts_valid = align_valid(test_feat["ts_ms"].to_numpy(), r_true)
    dt_valid = to_time_of_day(ts_valid)
    crash_idx = first_crossing(r_true, crash_true_thresh)

    # Forward predicted sums on crash day (diagnostic)
    Rpred_stress = rolling_sum_forward(test_feat["r_stress2"].to_numpy(), n_horizon)
    Rpred_cim = rolling_sum_forward(test_feat["r_cim"].to_numpy(), n_horizon)

    # Alarm threshold: quantile on TRAIN forward predicted sums
    train_Rpred = rolling_sum_forward(train_feat["r_stress2"].to_numpy(), n_horizon)
    pred_thresh = safe_quantile(train_Rpred, alarm_q)
    alarm_idx = first_crossing(Rpred_stress, pred_thresh) if np.isfinite(pred_thresh) else None

    resid = r_true - Rpred_stress

    logp = np.nancumsum(test_feat["r1"].to_numpy())
    dt_day = to_time_of_day(test_feat["ts_ms"].to_numpy())

    # -------------------------
    # REPORT
    # -------------------------
    print(f"\n=== {ticker} Mode A (SELL-PRESSURE) — Flash Crash {crash_day} ===")
    print(f"Train events: {len(tt_train):,}  Test-day events: {len(tt_test):,}")
    print(f"delta_c (train): {dc:.6g}")
    print(f"N_HORIZON: {n_horizon} trades")
    print(f"Sell-pressure window Ns: {ns_pressure}  tau: {tau:.3f}  alpha: {alpha:.3f}")
    print(f"Crash threshold (true): {crash_true_thresh:.4f}  (log-return)")

    if np.isfinite(pred_thresh):
        print(f"Alarm model: r_stress2  pred threshold = quantile(q={alarm_q:g}) = {pred_thresh:.4f}")
    else:
        print("[warn] Could not compute pred threshold (train_Rpred has insufficient finite values).")

    if crash_idx is None:
        print("[info] No crash trigger on this ticker/day under CRASH_TRUE_THRESH.")
    else:
        print(f"[crash] first trigger at {dt_valid[crash_idx]}")
        print(f"        R_true^{n_horizon} = {r_true[crash_idx]:.4f}")

    if alarm_idx is None:
        print("[alarm] No early-warning trigger from predicted return under pred_thresh.")
    else:
        print(f"[alarm] first trigger at {dt_valid[alarm_idx]}")
        print(f"        R_pred^{n_horizon} = {Rpred_stress[alarm_idx]:.4f}")
        if crash_idx is not None:
            lead_trades = crash_idx - alarm_idx
            lead_time = dt_valid[crash_idx] - dt_valid[alarm_idx]
            print(f"        lead: {lead_trades} rolling-steps (~trades) ; lead time: {lead_time}")

    # Diagnostics: MSE (on forward sums)
    if len(Rpred_stress) == len(r_true) and len(r_true) > 0:
        m = np.isfinite(r_true) & np.isfinite(Rpred_stress)
        if np.any(m):
            mse_stress = float(np.mean((r_true[m] - Rpred_stress[m]) ** 2))
            print(f"  MSE(R_stress2): {mse_stress:.6g}")
        else:
            print("  MSE(R_stress2): nan (insufficient finite data)")

        if len(Rpred_cim) == len(r_true):
            m2 = np.isfinite(r_true) & np.isfinite(Rpred_cim)
            if np.any(m2):
                mse_cim = float(np.mean((r_true[m2] - Rpred_cim[m2]) ** 2))
                print(f"  MSE(R_cim): {mse_cim:.6g}")
            else:
                print("  MSE(R_cim): nan (insufficient finite data)")

    # -------------------------
    # PLOTS
    # -------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    axes[0].plot(dt_day, logp, linewidth=1.0)
    axes[0].set_title(f"{ticker} {crash_day} — intraday log-price proxy (cumsum r1)")
    axes[0].set_ylabel("log price (shifted)")

    axes[1].plot(dt_valid, r_true, label="R_true", linewidth=1.5)
    if len(Rpred_stress) == len(r_true):
        axes[1].plot(dt_valid, Rpred_stress, label="R_stress2", linewidth=1.2)
    if len(Rpred_cim) == len(r_true):
        axes[1].plot(dt_valid, Rpred_cim, label="R_cim", linewidth=1.0)

    axes[1].axhline(crash_true_thresh, linestyle="--", linewidth=1.0, label="crash thresh (true)")
    if np.isfinite(pred_thresh):
        axes[1].axhline(pred_thresh, linestyle=":", linewidth=1.0, label=f"alarm thresh (q={alarm_q:g})")

    axes[1].set_title(f"Forward N-trade returns (N={n_horizon})")
    axes[1].set_ylabel("forward log-return")
    axes[1].legend(loc="best")

    if crash_idx is not None:
        axes[1].axvline(dt_valid[crash_idx], linestyle="--", linewidth=1.0)
        axes[2].axvline(dt_valid[crash_idx], linestyle="--", linewidth=1.0)
    if alarm_idx is not None:
        axes[1].axvline(dt_valid[alarm_idx], linestyle=":", linewidth=1.0)
        axes[2].axvline(dt_valid[alarm_idx], linestyle=":", linewidth=1.0)

    axes[2].plot(dt_valid, resid, linewidth=1.2)
    axes[2].set_title("Residual e(t) = R_true - R_pred  (model: r_stress2)")
    axes[2].set_ylabel("residual")
    axes[2].set_xlabel("NY time")

    plt.tight_layout()
    outpath = out_dir / f"{ticker}_sellpressure_N{n_horizon}_{crash_day}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"[ok] saved plot: {outpath}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Flash crash Mode A with sell-pressure conditioning (quantile alarm).")
    ap.add_argument("--clean-root", type=str, default="clean data",
                    help="Root folder containing cleaned parquet subfolders.")
    ap.add_argument("--ticker", type=str, required=True, help="Ticker folder name, e.g. ORCL.OQ")
    ap.add_argument("--out-dir", type=str, default=str(Path("results") / "plots" / "flashcrash_sellpressure"))
    ap.add_argument("--crash-day", type=str, default=str(DEFAULT_CRASH_DAY))
    ap.add_argument("--N", type=int, default=DEFAULT_N_HORIZON)
    ap.add_argument("--crash-thresh", type=float, default=DEFAULT_CRASH_TRUE_THRESH)

    ap.add_argument("--Ns", type=int, default=DEFAULT_NS_PRESSURE, help="Sell-pressure window in trades.")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Threshold on (-epsbar) for stress gating.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Sigmoid steepness for stress gating.")

    ap.add_argument("--alarm-q", type=float, default=DEFAULT_ALARM_Q,
                    help="Quantile for alarm threshold on train forward sums (e.g., 0.001).")

    ap.add_argument("--no-smooth", action="store_true", help="Disable kernel tail smoothing.")
    args = ap.parse_args()

    clean_root = Path(args.clean_root)
    out_dir = Path(args.out_dir)
    crash_day = pd.Timestamp(args.crash_day).date()

    tt = load_ticker_tt(clean_root, args.ticker)
    run_modeA_sellpressure(
        tt,
        ticker=args.ticker,
        crash_day=crash_day,
        out_dir=out_dir,
        n_horizon=args.N,
        crash_true_thresh=args.crash_thresh,
        ns_pressure=args.Ns,
        tau=args.tau,
        alpha=args.alpha,
        smooth_kernel=(not args.no_smooth),
        alarm_q=args.alarm_q,
    )


if __name__ == "__main__":
    main()
