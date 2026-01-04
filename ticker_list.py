from pathlib import Path

def count_dates(dir_path: Path, suffix: str) -> int:
    # files like YYYY-MM-DD-TICKER-trade.parquet or ...-bbo.parquet
    return sum(1 for _ in dir_path.glob(f"????-??-??-*-{suffix}.parquet"))

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    raw_root = project_dir / "US_flash_crash" / "US_flash_crash"

    trade_root = raw_root / "trade"
    bbo_root = raw_root / "bbo"

    tickers = sorted({p.name for p in trade_root.iterdir() if p.is_dir()}
                     & {p.name for p in bbo_root.iterdir() if p.is_dir()})

    rows = []
    for t in tickers:
        trade_dir = trade_root / t
        bbo_dir = bbo_root / t
        n_trade = count_dates(trade_dir, "trade")
        n_bbo = count_dates(bbo_dir, "bbo")
        rows.append((t, n_trade, n_bbo, min(n_trade, n_bbo)))

    rows.sort(key=lambda x: x[3], reverse=True)

    print("ticker,n_trade_days,n_bbo_days,n_common_days")
    for t, nt, nb, nc in rows:
        print(f"{t},{nt},{nb},{nc}")
