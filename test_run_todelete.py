from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
from priceprop.batch import complete_data_columns, calibrate_models, calc_models


def load_clean_day(path: Path) -> pd.DataFrame:
    # Read with polars, convert to pandas
    df = pl.read_parquet(path).to_pandas()

    # Make sure types match what priceprop expects
    df["date"] = pd.to_datetime(df["date"]).dt.date        # critical for complete_data_columns
    df["sign"] = df["sign"].astype(int)                    # +/-1
    df["change"] = df["change"].astype(bool)               # bool
    df["r1"] = df["r1"].astype(float)

    return df

def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)

    # Baselines
    mse0 = np.mean(y**2)                      # baseline predict 0
    mse = np.mean((y - yhat)**2)

    # Skill score (variance explained vs 0 predictor)
    skill = 1.0 - mse / mse0 if mse0 > 0 else np.nan

    # Classic R2 (same as above if mean(y) ~ 0; still fine)
    var_y = np.var(y)
    r2 = 1.0 - np.var(y - yhat) / var_y if var_y > 0 else np.nan

    # Correlation (sign of relationship)
    corr = np.corrcoef(y, yhat)[0, 1] if (np.std(y) > 0 and np.std(yhat) > 0) else np.nan

    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "skill_vs_0": skill,
        "r2": r2,
        "corr": corr,
    }

def evaluate_models(tt: pd.DataFrame, model_cols: list[str]) -> None:
    # Evaluate on all events + change==True events
    masks = {
        "ALL": np.ones(len(tt), dtype=bool),
        "CHANGE_ONLY": tt["change"].values.astype(bool),
    }

    for name, m in masks.items():
        y = tt.loc[m, "r1"].values
        print(f"\n=== Metrics on {name} (n={m.sum()}) ===")

        rows = []
        for col in model_cols:
            yhat = tt.loc[m, col].values
            out = metrics(y, yhat)
            out["model"] = col
            rows.append(out)

        res = pd.DataFrame(rows).set_index("model").sort_values("skill_vs_0", ascending=False)
        print(res)

    # Optional: per-day skill (useful to check stability and no leakage)
    print("\n=== Per-day skill_vs_0 on CHANGE_ONLY ===")
    day_rows = []
    for d, g in tt.groupby("date"):
        y = g.loc[g["change"], "r1"].values
        if len(y) < 100:  # avoid tiny days
            continue
        for col in model_cols:
            yhat = g.loc[g["change"], col].values
            out = metrics(y, yhat)
            day_rows.append({"date": d, "model": col, **out})

    day_df = pd.DataFrame(day_rows)
    if not day_df.empty:
        print(day_df.groupby("model")["skill_vs_0"].describe())
    else:
        print("No per-day rows (too few change events).")

def main():
    project_dir = Path(__file__).resolve().parent
    clean_dir = project_dir / "clean data" / "GOOG.OQ"

    files = [
        clean_dir / "GOOG.OQ_2010-05-04_felix_tt_clean.parquet",
        clean_dir / "GOOG.OQ_2010-05-05_felix_tt_clean.parquet",
        clean_dir / "GOOG.OQ_2010-05-06_felix_tt_clean.parquet",
        clean_dir / "GOOG.OQ_2010-05-07_felix_tt_clean.parquet",
    ]

    # --- load and stack 2 days ---
    dfs = [load_clean_day(p) for p in files]
    tt = pd.concat(dfs, ignore_index=True)

    print("Loaded rows:", len(tt))
    print("Days:", sorted(tt["date"].unique()))
    print(tt[["date", "sign", "change", "r1"]].head())

    # --- build internal columns sc/sn/sample (required by calibration) ---
    complete_data_columns(tt)

    # --- choose models (keep it light at first) ---
    models = ["cim", "tim1", "tim2", "hdim2_x2"]

    # --- simulate/predict model returns for each event ---
    calc_models(tt, group=False, models=models, split_by="sample", nfft=4096)
    col_models = ["r_cim", "r_tim1", "r_tim2", "r_hdim2_x2"]

    evaluate_models(tt, col_models)

    # --- what you should expect to see ---
    print("\nModel return columns created:")
    print([c for c in tt.columns if c.startswith("r_")])

    show_cols = ["date", "sign", "change", "r1", "r_tim1", "r_tim2", "r_hdim2_x2"]
    print("\nHead (truth vs predictions):")
    print(tt[show_cols].head(10))

    # sanity checks
    print("\nSanity checks:")
    print("r1 mean/std:", tt["r1"].mean(), tt["r1"].std())
    print("r_tim1 mean/std:", tt["r_tim1"].mean(), tt["r_tim1"].std())
    print("r_tim2 mean/std:", tt["r_tim2"].mean(), tt["r_tim2"].std())
    print("r_hdim2_x2 mean/std:", tt["r_hdim2_x2"].mean(), tt["r_hdim2_x2"].std())

    # HDIM2_x2 should be exactly zero on non-price-changing events:
    zero_frac = (tt.loc[~tt["change"], "r_hdim2_x2"] == 0).mean()
    print("Fraction of r_hdim2_x2 == 0 when change==False:", zero_frac)


if __name__ == "__main__":
    main()
