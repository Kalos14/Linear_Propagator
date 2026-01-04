# run_model_vs_ticksize.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import polars as pl

import priceprop.batch as pp_batch  # package name from setup.py


# -------------------------
# config
# -------------------------
CLEAN_ROOT = Path("clean data")       # folder containing per-ticker cleaned parquets
RESULTS_ROOT = Path("results")
PLOTS_DIR = RESULTS_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_AGG = 50  # N-trade aggregation (paper uses N=50) 


# -------------------------
# helpers
# -------------------------
def load_ticker_tt(clean_root: Path, ticker: str) -> pd.DataFrame:
    """
    Load all daily cleaned Felix-style tt parquets for a ticker into one pandas DF.
    Expected columns: date, ts_ms, sign, r1, change
    """
    tdir = clean_root / ticker
    if not tdir.exists():
        raise FileNotFoundError(f"Missing folder: {tdir}")

    files = sorted(tdir.glob(f"{ticker}_*_felix_tt_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"No cleaned parquets found in {tdir}")

    # Polars -> pandas
    lf = pl.scan_parquet([f.as_posix() for f in files]).select(
        ["date", "ts_ms", "sign", "r1", "change"]
    )
    df = lf.collect().to_pandas()

    # enforce dtypes expected by priceprop
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["sign"] = df["sign"].astype(np.int8)
    df["change"] = df["change"].astype(bool)
    df["r1"] = df["r1"].astype(float)

    # important: sort like a single event-time series
    df = df.sort_values(["date", "ts_ms"]).reset_index(drop=True)
    return df


def microstructural_eta_from_tt(tt: pd.DataFrame) -> float:
    """
    η = Nc / (2 Na), computed from the direction of mid-price *changes*
    (continuations vs alternations). 
    """
    # only price-changing events carry direction
    rchg = tt.loc[tt["change"], "r1"].to_numpy()
    d = np.sign(rchg)
    d = d[d != 0]
    if len(d) < 2:
        return np.nan
    Nc = np.sum(d[1:] == d[:-1])
    Na = np.sum(d[1:] != d[:-1])
    if Na == 0:
        return np.nan
    return float(Nc / (2.0 * Na))


def delta_c_from_tt(tt: pd.DataFrame) -> float:
    """
    Δc = E[|r(t)| | π(t)=c] (eq. 7). 
    Your r1 is the one-trade log mid return.
    """
    x = tt.loc[tt["change"], "r1"].abs()
    return float(x.mean()) if len(x) else np.nan


def rolling_sum(x: np.ndarray, N: int) -> np.ndarray:
    """
    Rolling sum over N trades (valid mode), preserving NaNs:
    output length = len(x) - N + 1
    """
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    x0 = np.where(mask, x, 0.0)

    y = np.convolve(x0, np.ones(N, dtype=float), mode="valid")
    valid = np.convolve(mask.astype(float), np.ones(N, dtype=float), mode="valid") == N
    y[~valid] = np.nan
    return y


def xcorr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation on finite overlapping values.
    """
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


# -------------------------
# main experiment
# -------------------------
def evaluate_one_ticker(ticker: str) -> dict:
    tt = load_ticker_tt(CLEAN_ROOT, ticker)

    eta = microstructural_eta_from_tt(tt)
    dc = delta_c_from_tt(tt)

    # Run models out-of-sample via odd/even split ("sample" column is created internally)
    # This mirrors the paper's odd/even procedure. 
    dbc = {"tt": tt.copy()}
    pp_batch.calc_models(
        dbc,
        calibrate=True,
        split_by="sample",
        group=False,          
        models=["cim", "tim1", "tim2", "hdim2"],
        rshift=0,
        smooth_kernel=True,
    )


    tt_m = dbc["tt"]

    # Scale CIM2 by Δc (eq. 6–7); batch gives ±1/0, so we restore return units. 
    if np.isfinite(dc):
        tt_m["r_cim"] = tt_m["r_cim"].astype(float) * dc

    # Build N-trade aggregated returns and compute correlations (paper Fig.3 metric).
    r_true = rolling_sum(tt_m["r1"].to_numpy(), N_AGG)

    out = {"ticker": ticker, "eta": eta, "delta_c": dc}

    for col in ["r_tim1", "r_tim2", "r_hdim2", "r_cim"]:
        if col not in tt_m.columns:
            out[col + "_xcorr"] = np.nan
            continue
        r_pred = rolling_sum(tt_m[col].to_numpy(), N_AGG)
        out[col + "_xcorr"] = xcorr(r_true, r_pred)

    return out


def plot_xcorr_vs_eta(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(9, 5))
    for col, label, marker in [
        ("r_tim1_xcorr", "TIM1", "o"),
        ("r_tim2_xcorr", "TIM2", "s"),
        ("r_hdim2_xcorr", "HDIM2", "^"),
        ("r_cim_xcorr", "CIM2", "D"),
    ]:
        plt.scatter(df["eta"], df[col], label=label, marker=marker)

    plt.axvline(0.5, linestyle="--")  # η=0.5 split: small vs large tick
    plt.xlabel("microstructural parameter η")
    plt.ylabel(f"xcorr( R_N true , R_N model )  with N={N_AGG}")
    plt.title("Model short-term performance vs discretisation (η)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    # Put here the tickers you want to test (start small, then expand)
    tickers = [
        "AAPL.OQ"]
        # "MSFT.OQ",
        # "CSCO.OQ",
        # "AMZN.OQ",
        # "INTC.OQ",
        # "GOOG.OQ",
        # "ORCL.OQ",
        # ]
    

    rows = []
    for t in tickers:
        print(f"[run] {t}")
        rows.append(evaluate_one_ticker(t))

    res = pd.DataFrame(rows).sort_values("eta").reset_index(drop=True)

    # save table
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_ROOT / "xcorr_vs_eta.csv"
    res.to_csv(out_csv, index=False)
    print(f"[ok] saved: {out_csv}")

    # plot
    out_png = PLOTS_DIR / "xcorr_vs_eta.png"
    plot_xcorr_vs_eta(res, out_png)
    print(f"[ok] saved: {out_png}")

    # quick “who is best” summary per ticker
    score_cols = ["r_tim1_xcorr", "r_tim2_xcorr", "r_hdim2_xcorr", "r_cim_xcorr"]
    for _, r in res.iterrows():
        best = max(score_cols, key=lambda c: (r[c] if np.isfinite(r[c]) else -np.inf))
        print(f"{r['ticker']:10s}  eta={r['eta']:.3f}  best={best}  scores="
              f"TIM1 {r['r_tim1_xcorr']:.3f} | TIM2 {r['r_tim2_xcorr']:.3f} | "
              f"HDIM2 {r['r_hdim2_xcorr']:.3f} | CIM2 {r['r_cim_xcorr']:.3f}")


if __name__ == "__main__":
    main()
