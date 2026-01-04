from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from priceprop import batch  # contains calc_models

def rolling_mean(x: np.ndarray, N: int) -> np.ndarray:
    w = np.ones(N) / N
    return np.convolve(x, w, mode="valid")

def rolling_sum(x: np.ndarray, N: int) -> np.ndarray:
    w = np.ones(N)
    return np.convolve(x, w, mode="valid")

# -----------------------------
# IO helpers
# -----------------------------
def load_clean_tt(clean_dir: Path, ticker: str) -> pd.DataFrame:
    files = sorted((clean_dir / ticker).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {clean_dir/ticker}")

    tt = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    tt["date"] = pd.to_datetime(tt["date"]).dt.date
    tt = tt.sort_values(["date", "ts_ms"]).reset_index(drop=True)

    tt["change"] = tt["change"].astype(bool)
    tt["sign"] = tt["sign"].astype(np.int8)
    return tt


# -----------------------------
# Δc (missing ingredient)
# -----------------------------
def estimate_delta_c(tt: pd.DataFrame) -> float:
    """
    Δc = <|r(t)| | change=True> using your r1 (log mid-return).
    """
    mask = tt["change"].values
    if mask.sum() == 0:
        raise ValueError("No change=True events found, cannot estimate Δc.")
    return float(np.mean(np.abs(tt.loc[mask, "r1"].values)))


# -----------------------------
# block aggregation (N trades)
# -----------------------------
def block_aggregate(x: np.ndarray, N: int, how: str) -> np.ndarray:
    n_blocks = len(x) // N
    x = x[: n_blocks * N].reshape(n_blocks, N)
    if how == "mean":
        return x.mean(axis=1)
    if how == "sum":
        return x.sum(axis=1)
    raise ValueError("how must be 'mean' or 'sum'")


# -----------------------------
# figure maker (returns fig)
# -----------------------------
def make_excerpt_figure(
    tt: pd.DataFrame,
    title: str,
    N: int = 50,
    start_trade: int = 0,
    n_trades: int = 1000,
) -> plt.Figure:
    ex = tt.iloc[start_trade : start_trade + n_trades].copy().reset_index(drop=True)

    s = ex["sign"].values.astype(float)
    r_true = ex["r1"].values.astype(float)
    r_tim2 = ex["r_tim2"].values.astype(float)

    # FIX CIM2: multiply by Δc
    delta_c = estimate_delta_c(tt)
    r_cim2 = (ex["sign"].values * ex["change"].values.astype(int)).astype(float) * delta_c

    eps_bar = rolling_mean(s, N)
    R_true = rolling_sum(r_true, N) * 1e4
    R_tim2 = rolling_sum(r_tim2, N) * 1e4
    R_cim2 = rolling_sum(r_cim2, N) * 1e4

    x_blocks = np.arange(len(eps_bar))  # now hundreds of points

    # correlations shown in paper captions
    c_tim2 = np.corrcoef(R_true, R_tim2)[0, 1] if len(R_true) > 1 else np.nan
    c_cim2 = np.corrcoef(R_true, R_cim2)[0, 1] if len(R_true) > 1 else np.nan

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # Top panel: sign markers + binned eps_bar
    buys = np.where(ex["sign"].values == 1)[0]
    sells = np.where(ex["sign"].values == -1)[0]
    ax1.scatter(buys, np.ones_like(buys), s=10, marker="o", label="+1")
    ax1.scatter(sells, -np.ones_like(sells), s=10, marker="o", label="-1")

    y_step = np.r_[eps_bar, eps_bar[-1]] if len(eps_bar) > 0 else np.array([])
    x_step = np.r_[x_blocks, x_blocks[-1] + N] if len(x_blocks) > 0 else np.array([])
    if len(x_step) > 0:
        ax1.step(x_step, y_step, where="post", linewidth=2, label=f"ε̄ (bins of {N})")

    ax1.set_ylim(-1.2, 1.2)
    ax1.set_ylabel(f"ε̄ (n={N})")
    ax1.set_title(title)

    # Bottom panel: N-trade returns
    ax2.plot(
        x_blocks, R_true,
        marker="^",
        markevery=max(1, len(x_blocks)//20),
        linewidth=2,
        label="True"
    )
    ax2.plot(
        x_blocks, R_tim2,
        marker="D",
        markevery=max(1, len(x_blocks)//20),
        linewidth=2,
        label=f"TIM2 (corr={c_tim2:.2f})" if np.isfinite(c_tim2) else "TIM2"
    )
    ax2.plot(
        x_blocks, R_cim2,
        marker="*",
        markevery=max(1, len(x_blocks)//20),
        linewidth=2,
        label=f"CIM2 (corr={c_cim2:.2f})" if np.isfinite(c_cim2) else "CIM2"
    )

    ax2.set_ylabel(f"r (n={N}) [bp]")
    ax2.set_xlabel("Trades")

    ax1.legend(loc="upper right", frameon=False)
    ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    return fig


# -----------------------------
# main runner: calibrate TIM2 then save plot
# -----------------------------
def run_and_save_figure(
    clean_root: Path,
    results_root: Path,
    ticker: str,
    fig_tag: str,
    start_trade: int,
    n_trades: int,
    title: str,
    N: int = 50,
) -> Path:
    tt = load_clean_tt(clean_root, ticker)

    # Calibrate + simulate models using the whole dataset
    dbc = {"tt": tt}
    batch.calc_models(
        dbc,
        models=["tim2", "cim"],   # r_cim produced by batch is unscaled; we'll rescale in plotting
        calibrate=True,
        split_by=None,
        group=False,
        rshift=0,
        smooth_kernel=True
    )

    fig = make_excerpt_figure(
        tt=tt,
        title=title,
        N=N,
        start_trade=start_trade,
        n_trades=n_trades,
    )

    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_path = plots_dir / f"{fig_tag}_{ticker}_N{N}_start{start_trade}_len{n_trades}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    CLEAN_ROOT = Path("clean data")   # where your cleaned daily parquets live
    RESULTS_ROOT = Path("results")    # will create results/plots

    # Fig 1-like: small tick-ish example (AAPL)
    p1 = run_and_save_figure(
        clean_root=CLEAN_ROOT,
        results_root=RESULTS_ROOT,
        ticker="AAPL.OQ",                 # adapt to your folder name
        fig_tag="fig1_style",
        start_trade=0,
        n_trades=950,
        title="Fig 1-style excerpt (small tick example)",
        N=50,
    )
    print("Saved:", p1)

    # Fig 2-like: large tick-ish replacement (e.g. MSFT)
    p2 = run_and_save_figure(
        clean_root=CLEAN_ROOT,
        results_root=RESULTS_ROOT,
        ticker="MSFT.OQ",              # adapt; try MSFT.OQ/CSCO.OQ/INTC.OQ
        fig_tag="fig2_style",
        start_trade=0,
        n_trades=1600,
        title="Fig 2-style excerpt (large tick example)",
        N=50,
    )
    print("Saved:", p2)
