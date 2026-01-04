# %%
import polars as pl
from data_preprocessing import *
from pathlib import Path

# project_dir = Path(__file__).resolve().parent
# raw_root = project_dir / "US_flash_crash" / "US_flash_crash"
# ticker = "CSCO.OQ"
# date = "2010-12-28"

# trade_path, bbo_path = build_paths(raw_root, ticker, date)

# tmp = (
#     pl.scan_parquet(trade_path)
#     .select(["xltime"])
#     .with_columns(ts_ns=xltime_to_unix_ns(pl.col("xltime")))
#     .with_columns(ts_ny=unix_ns_to_ny_time(pl.col("ts_ns")))
#     .select(
#         pl.min("ts_ny").alias("min_ts_ny"),
#         pl.max("ts_ny").alias("max_ts_ny"),
#         pl.col("ts_ny").dt.hour().min().alias("min_hour_ny"),
#         pl.col("ts_ny").dt.hour().max().alias("max_hour_ny"),
#     )
#     .collect()
# )
# print(tmp)


# tmp2 = (
#     pl.scan_parquet(trade_path)
#     .select(["xltime"])
#     .with_columns(ts_ns=xltime_to_unix_ns(pl.col("xltime")))
#     .with_columns(ts_ny=unix_ns_to_ny_time(pl.col("ts_ns")))
#     .with_columns(h=pl.col("ts_ny").dt.hour(), m=pl.col("ts_ny").dt.minute())
#     .select(
#         ((pl.col("h")==9) & (pl.col("m")>=30)).sum().alias("count_after_open"),
#         ((pl.col("h")>=10) & (pl.col("h")<16)).sum().alias("count_around_10am"),
#     )
#     .collect()
# )
# print(tmp2)




from pathlib import Path
import polars as pl

from data_preprocessing import xltime_to_unix_ns, unix_ns_to_ny_time

def _in_session(ts_ny: pl.Expr, start=(10,0), end=(16,0)) -> pl.Expr:
    sh, sm = start
    eh, em = end
    after_start = (ts_ny.dt.hour() > sh) | ((ts_ny.dt.hour() == sh) & (ts_ny.dt.minute() >= sm))
    before_end  = (ts_ny.dt.hour() < eh) | ((ts_ny.dt.hour() == eh) & (ts_ny.dt.minute() < em))
    return after_start & before_end

def report_attrition(trade_path: Path, bbo_path: Path, start=(10,0), end=(16,0)) -> None:
    trade_path = Path(trade_path)
    bbo_path   = Path(bbo_path)

    # ---- raw trades (no filters) ----
    trades0 = (
        pl.scan_parquet(trade_path)
        .select(["xltime", "trade-price", "trade-stringflag", "trade-volume"])
        .with_columns(ts_ns=xltime_to_unix_ns(pl.col("xltime")))
        .with_columns(ts_ny=unix_ns_to_ny_time(pl.col("ts_ns")))
    )

    # ---- raw bbo (no filters) ----
    bbo0 = (
        pl.scan_parquet(bbo_path)
        .select(["xltime", "bid-price", "ask-price"])
        .with_columns(ts_ns=xltime_to_unix_ns(pl.col("xltime")))
        .with_columns(ts_ny=unix_ns_to_ny_time(pl.col("ts_ns")))
    )

    # session trims
    trades_sess = trades0.filter(_in_session(pl.col("ts_ny"), start, end))
    bbo_sess    = bbo0.filter(_in_session(pl.col("ts_ny"), start, end))

    # regular trades only (your current choice)
    trades_reg = trades_sess.filter(pl.col("trade-stringflag") == "uncategorized")

    # basic quote sanity
    bbo_clean = (
        bbo_sess
        .filter(
            pl.col("bid-price").is_finite()
            & pl.col("ask-price").is_finite()
            & (pl.col("bid-price") > 0)
            & (pl.col("ask-price") > 0)
            & (pl.col("ask-price") >= pl.col("bid-price"))
        )
        .select(["ts_ns", "bid-price", "ask-price"])
        .sort("ts_ns")
    )

    trades_reg_sorted = (
        trades_reg
        .filter(pl.col("trade-price").is_finite() & (pl.col("trade-price") > 0))
        .select(["ts_ns", "ts_ny", "trade-price", "trade-volume"])
        .sort("ts_ns")
    )

    # asof join
    joined = (
        trades_reg_sorted.join_asof(bbo_clean, on="ts_ns", strategy="backward")
        .with_columns(mid=(pl.col("bid-price") + pl.col("ask-price")) / 2.0)
    )

    # sign + mid-trades
    signed = (
        joined.with_columns(
            sign=pl.when(pl.col("trade-price") > pl.col("mid")).then(1)
                  .when(pl.col("trade-price") < pl.col("mid")).then(-1)
                  .otherwise(None)
                  .cast(pl.Int8)
        )
    )

    signed_kept = signed.filter(pl.col("mid").is_not_null() & pl.col("sign").is_not_null())

    # merge by (ms, sign)
    merged = (
        signed_kept
        .with_columns(ts_ms=(pl.col("ts_ns") // 1_000_000) * 1_000_000)
        .group_by(["ts_ms", "sign"])
        .agg(
            pl.len().alias("n_trades_in_bucket"),
            pl.first("mid").alias("mid"),
        )
    )

    bucket_sizes = (
    merged.select("n_trades_in_bucket")
    .describe()
    )
    print(bucket_sizes)

    top = (
        merged.select("n_trades_in_bucket")
        .group_by("n_trades_in_bucket")
        .agg(pl.len().alias("n_events"))
        .sort("n_trades_in_bucket")
    )
    print(top.tail(20))  # show largest bucket sizes

    out = pl.DataFrame({
        "stage": [
            "raw_trades_total",
            "raw_trades_in_session",
            "regular_trades_in_session",
            "raw_bbo_in_session",
            "clean_bbo_in_session",
            "after_asof_join (mid non-null)",
            "after_sign (drop mid-trades)",
            "after_merge (events = unique (ms,sign))",
        ],
        "count": [
            trades0.select(pl.len()).collect().item(),
            trades_sess.select(pl.len()).collect().item(),
            trades_reg.select(pl.len()).collect().item(),
            bbo0.select(pl.len()).collect().item(),
            bbo_clean.select(pl.len()).collect().item(),
            joined.filter(pl.col("mid").is_not_null()).select(pl.len()).collect().item(),
            signed_kept.select(pl.len()).collect().item(),
            merged.select(pl.len()).collect().item(),
        ],
    })

    # extra: how many were dropped because trade==mid vs because no quote
    extras = pl.DataFrame({
        "metric": ["frac_mid_trades_among_joined", "frac_missing_quote_among_regular_session", "avg_trades_per_event_after_merge"],
        "value": [
            signed.filter(pl.col("mid").is_not_null()).select((pl.col("sign").is_null()).mean()).collect().item(),
            joined.select((pl.col("mid").is_null()).mean()).collect().item(),
            (signed_kept.select(pl.len()).collect().item() / max(1, merged.select(pl.len()).collect().item())),
        ]
    })

    print(out)
    print("\n", extras)

if __name__ == "__main__":
    # Example: adapt these paths
    trade_path = Path(r"C:\Users\calan\Desktop\MA3\Financial_Big_Data\Project\US_flash_crash\US_flash_crash\trade\CSCO.OQ\2010-12-28-CSCO.OQ-trade.parquet")
    bbo_path   = Path(r"C:\Users\calan\Desktop\MA3\Financial_Big_Data\Project\US_flash_crash\US_flash_crash\bbo\CSCO.OQ\2010-12-28-CSCO.OQ-bbo.parquet")
    report_attrition(trade_path, bbo_path)

