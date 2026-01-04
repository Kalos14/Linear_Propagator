from __future__ import annotations

from pathlib import Path
import polars as pl

# -----------------------------
# constants / time conversion
# -----------------------------
EXCEL_UNIX_EPOCH_DAYS = 25569  # Excel serial day for 1970-01-01
NS_PER_DAY = 86_400_000_000_000


def xltime_to_unix_ns(x: pl.Expr) -> pl.Expr:
    """
    Convert Excel serial day float (xltime) -> Unix timestamp in nanoseconds (Int64).
    """
    return ((x - EXCEL_UNIX_EPOCH_DAYS) * NS_PER_DAY).round(0).cast(pl.Int64)


def unix_ns_to_ny_time(ts_ns: pl.Expr) -> pl.Expr:
    """
    Unix ns -> timezone-aware datetime in America/New_York.
    Assumes the underlying clock is UTC.
    """
    return (
        pl.from_epoch(ts_ns, time_unit="ns")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("America/New_York")
    )


# -----------------------------
# core preprocessing (unchanged)
# -----------------------------
def preprocess_felix_day_lazy(
    trade_parquet: str | Path,
    bbo_parquet: str | Path,
    tick_size: float = 0.01,
    keep_only_regular_trades: bool = True,
    ny_start_hhmm: tuple[int, int] = (10, 0),  # 30 min after 9:30 open
    ny_end_hhmm: tuple[int, int] = (16, 0),    # before close
) -> pl.LazyFrame:
    """
    Build Felix-style event-time dataset:
        date | ts_ms | sign | r1 | change
    """
    trade_parquet = str(trade_parquet)
    bbo_parquet = str(bbo_parquet)

    # --------- load & clean BBO ---------
    bbo = (
        pl.scan_parquet(bbo_parquet)
        .select(["xltime", "bid-price", "ask-price"])
        .with_columns(ts_ns=xltime_to_unix_ns(pl.col("xltime")))
        .filter(
            pl.col("bid-price").is_finite()
            & pl.col("ask-price").is_finite()
            & (pl.col("bid-price") > 0)
            & (pl.col("ask-price") > 0)
            & (pl.col("ask-price") >= pl.col("bid-price"))
        )
        .with_columns(ts_ny=unix_ns_to_ny_time(pl.col("ts_ns")))
        .filter(
            (pl.col("ts_ny").dt.hour() > ny_start_hhmm[0])
            | ((pl.col("ts_ny").dt.hour() == ny_start_hhmm[0]) & (pl.col("ts_ny").dt.minute() >= ny_start_hhmm[1]))
        )
        .filter(
            (pl.col("ts_ny").dt.hour() < ny_end_hhmm[0])
            | ((pl.col("ts_ny").dt.hour() == ny_end_hhmm[0]) & (pl.col("ts_ny").dt.minute() < ny_end_hhmm[1]))
        )
        .select(["ts_ns", "bid-price", "ask-price"])
        .sort("ts_ns")
    )

    # --------- load & clean trades ---------
    trades = (
        pl.scan_parquet(trade_parquet)
        .select(["xltime", "trade-price", "trade-volume", "trade-stringflag", "trade-rawflag"])
        .with_columns(ts_ns=xltime_to_unix_ns(pl.col("xltime")))
        .filter(pl.col("trade-price").is_finite() & (pl.col("trade-price") > 0))
    )

    if keep_only_regular_trades:
        trades = trades.filter(pl.col("trade-stringflag") == "uncategorized")

    trades = (
        trades
        .with_columns(ts_ny=unix_ns_to_ny_time(pl.col("ts_ns")))
        .filter(
            ((pl.col("ts_ny").dt.hour() > ny_start_hhmm[0])
             | ((pl.col("ts_ny").dt.hour() == ny_start_hhmm[0]) & (pl.col("ts_ny").dt.minute() >= ny_start_hhmm[1])))
            &
            ((pl.col("ts_ny").dt.hour() < ny_end_hhmm[0])
             | ((pl.col("ts_ny").dt.hour() == ny_end_hhmm[0]) & (pl.col("ts_ny").dt.minute() < ny_end_hhmm[1])))
        )
        .select(["ts_ns", "ts_ny", "trade-price", "trade-volume"])
        .sort("ts_ns")
    )

    # --------- asof join: attach prevailing quote just before each trade ---------
    tt = (
        trades.join_asof(bbo, on="ts_ns", strategy="backward")
        .with_columns(
            mid=((pl.col("bid-price") + pl.col("ask-price")) / 2.0),

            # half-tick mid index (integer, in units of tick/2)
            bid_tick=(pl.col("bid-price") / tick_size).round(0).cast(pl.Int64),
            ask_tick=(pl.col("ask-price") / tick_size).round(0).cast(pl.Int64),
        )
        .with_columns(
            mid_half_tick=pl.col("bid_tick") + pl.col("ask_tick")
        )
        .filter(pl.col("mid").is_not_null())
        .with_columns(
            sign=pl.when(pl.col("trade-price") > pl.col("mid")).then(1)
                .when(pl.col("trade-price") < pl.col("mid")).then(-1)
                .otherwise(None)
                .cast(pl.Int8)
        )
        .filter(pl.col("sign").is_not_null())
    )


    # --------- merge same sign + millisecond timestamp ---------
    tt = (
        tt.with_columns(ts_ms=((pl.col("ts_ns") // 1_000_000) * 1_000_000))
        .group_by(["ts_ms", "sign"])
        .agg(
            pl.first("ts_ny").alias("ts_ny"),
            pl.first("mid").alias("mid"),
            pl.first("mid_half_tick").alias("mid_half_tick"),  # <-- keep it
            pl.sum("trade-volume").alias("volume"),
        )
        .sort("ts_ms")
    )


    # --------- build r1 + change (FIXED: half-tick mid) ---------
    tt = (
        tt.with_columns(
            date=pl.col("ts_ny").dt.date(),
            mid_next=pl.col("mid").shift(-1),
            mid_half_tick_next=pl.col("mid_half_tick").shift(-1),
        )
        .with_columns(
            change=(pl.col("mid_half_tick_next") != pl.col("mid_half_tick")),
            r1=pl.when(pl.col("mid_next").is_not_null() & (pl.col("mid_half_tick_next") != pl.col("mid_half_tick")))
                .then(pl.col("mid_next").log() - pl.col("mid").log())
                .when(pl.col("mid_next").is_not_null())
                .then(0.0)
                .otherwise(None)
        )
        .filter(pl.col("r1").is_not_null())
        .select(["date", "ts_ms", "sign", "r1", "change"])
    )



    return tt


# -----------------------------
# path helpers + runner
# -----------------------------
def build_paths(raw_root: Path, ticker: str, date: str) -> tuple[Path, Path]:
    """
    raw_root points to: .../US_flash_crash/US_flash_crash
    Files are:
      bbo/<ticker>/<date>-<ticker>-bbo.parquet
      trade/<ticker>/<date>-<ticker>-trade.parquet
    """
    bbo_path = raw_root / "bbo" / ticker / f"{date}-{ticker}-bbo.parquet"
    trade_path = raw_root / "trade" / ticker / f"{date}-{ticker}-trade.parquet"
    return trade_path, bbo_path

def debug_one_day(trade_path: Path, bbo_path: Path, tick_size: float = 0.01) -> None:
    # raw counts
    raw_trades = pl.scan_parquet(trade_path).select(pl.len().alias("raw_trades")).collect()
    raw_bbo = pl.scan_parquet(bbo_path).select(pl.len().alias("raw_bbo")).collect()
    print("RAW:", raw_trades, raw_bbo, sep="\n")

    # what are the flags?
    flags = (
        pl.scan_parquet(trade_path)
        .select(pl.col("trade-stringflag"))
        .group_by("trade-stringflag")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect()
    )
    print("trade-stringflag counts:\n", flags)

    # timestamp sanity BEFORE NY session filter (check timezone assumption)
    tmp = (
        pl.scan_parquet(trade_path)
        .select(["xltime"])
        .with_columns(ts_ns=xltime_to_unix_ns(pl.col("xltime")))
        .with_columns(dt_utc=pl.from_epoch(pl.col("ts_ns"), time_unit="ns"))
        .select(
            pl.min("dt_utc").alias("min_dt_utc"),
            pl.max("dt_utc").alias("max_dt_utc"),
            pl.col("dt_utc").dt.hour().min().alias("min_hour_utc"),
            pl.col("dt_utc").dt.hour().max().alias("max_hour_utc"),
        )
        .collect()
    )
    print("UTC time coverage (from xltime->epoch):\n", tmp)

#     # final output count
    tmp2 = (
        pl.scan_parquet(trade_path)
        .select(["xltime"])
        .with_columns(ts_ns=xltime_to_unix_ns(pl.col("xltime")))
        .with_columns(ts_ny=unix_ns_to_ny_time(pl.col("ts_ns")))
        .with_columns(h=pl.col("ts_ny").dt.hour(), m=pl.col("ts_ny").dt.minute())
        .select(
            ((pl.col("h")==9) & (pl.col("m")>=30)).sum().alias("count_after_open"),
            ((pl.col("h")==10) & (pl.col("m")<5)).sum().alias("count_around_10am"),
        )
        .collect()
    )
    print(tmp2)


def list_available_dates(raw_root: Path, ticker: str) -> list[str]:
    """
    Discover dates for which both trade and bbo parquet exist for a given ticker.

    Expected filenames:
      trade/<ticker>/YYYY-MM-DD-<ticker>-trade.parquet
      bbo/<ticker>/YYYY-MM-DD-<ticker>-bbo.parquet
    """
    trade_dir = raw_root / "trade" / ticker
    bbo_dir = raw_root / "bbo" / ticker

    if not trade_dir.exists():
        raise FileNotFoundError(f"Missing trade dir: {trade_dir}")
    if not bbo_dir.exists():
        raise FileNotFoundError(f"Missing bbo dir: {bbo_dir}")

    trade_dates = set()
    for p in trade_dir.glob(f"????-??-??-{ticker}-trade.parquet"):
        trade_dates.add(p.name[:10])  # YYYY-MM-DD

    bbo_dates = set()
    for p in bbo_dir.glob(f"????-??-??-{ticker}-bbo.parquet"):
        bbo_dates.add(p.name[:10])

    dates = sorted(trade_dates & bbo_dates)
    return dates


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    raw_root = project_dir / "US_flash_crash" / "US_flash_crash"
    out_root = project_dir / "clean data"

    ticker = "AMZN.OQ"
    tick_size = 0.01

    # Safety: first run only a subset. Then set to None to run all.
    MAX_DAYS = None

    out_dir = out_root / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = list_available_dates(raw_root, ticker)
    if not dates:
        raise RuntimeError(f"No dates found for {ticker} under {raw_root}")

    if MAX_DAYS is not None:
        dates = dates[:MAX_DAYS]

    print(f"[info] {ticker}: found {len(list_available_dates(raw_root, ticker))} dates total")
    print(f"[info] processing {len(dates)} dates")

    for date in dates:
        trade_path, bbo_path = build_paths(raw_root, ticker, date)

        out_path = out_dir / f"{ticker}_{date}_felix_tt_clean.parquet"
        if out_path.exists():
            print(f"[skip] exists: {out_path.name}")
            continue

        print(f"[run] {ticker} {date}")

        lf = preprocess_felix_day_lazy(
            trade_parquet=trade_path,
            bbo_parquet=bbo_path,
            tick_size=tick_size,
        )

        # quick sanity summary (streaming)
        summary = (
            lf.select([
                pl.len().alias("n_events"),
                pl.col("change").mean().alias("P_c"),          # same as frac_change
                pl.col("r1").std().alias("std_r1"),
                pl.col("r1").abs().mean().alias("E_abs_r1"),
                pl.col("r1").filter(pl.col("change")).abs().mean().alias("Delta_c"),
            ])
            .collect(engine="streaming")
        )
        print(summary)


        lf.sink_parquet(out_path.as_posix(), compression="zstd")
        print(f"[ok] saved: {out_path.name}")

    print("[done]")


if __name__ == "__main__":
    main()
