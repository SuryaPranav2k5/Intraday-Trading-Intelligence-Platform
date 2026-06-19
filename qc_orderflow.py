"""Quick quality check on all orderflow parquet files."""
import pandas as pd
from pathlib import Path

HARVEST_DIR = Path("market_data/orderflow")
files = sorted(HARVEST_DIR.glob("*.parquet"))
print(f"Found {len(files)} parquet files\n")

EXPECTED_COLS = [
    "timestamp", "last_traded_price", "total_buy_qty", "total_sell_qty",
    "imbalance_ratio", "top_bid_price", "top_ask_price", "volume",
    "bid_ask_spread", "depth_imbalance", "bid_depth_total", "ask_depth_total",
    "weighted_mid_price", "price_impact", "tick_direction",
    "bid_price_1", "bid_qty_1", "ask_price_1", "ask_qty_1",
    "bid_price_2", "bid_qty_2", "ask_price_2", "ask_qty_2",
    "bid_price_3", "bid_qty_3", "ask_price_3", "ask_qty_3",
    "bid_price_4", "bid_qty_4", "ask_price_4", "ask_qty_4",
    "bid_price_5", "bid_qty_5", "ask_price_5", "ask_qty_5",
]

total_issues = 0

for f in files:
    df = pd.read_parquet(f)
    sym = f.name.split("_orderflow_")[0]
    date_str = f.name.split("_orderflow_")[1].replace(".parquet", "")
    issues = []

    # 1. Column check
    missing = set(EXPECTED_COLS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLS)
    if missing:
        issues.append(f"MISSING COLS: {missing}")
    if extra:
        issues.append(f"EXTRA COLS: {extra}")

    # 2. Null check
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        issues.append(f"NULLS: {dict(null_cols)}")

    # 3. Zero price check
    zero_ltp = (df["last_traded_price"] == 0).sum()
    if zero_ltp > 0:
        issues.append(f"ZERO LTP: {zero_ltp} rows")

    # 4. Zero bid/ask depth
    zero_bid1 = (df["bid_price_1"] == 0).sum()
    zero_ask1 = (df["ask_price_1"] == 0).sum()
    if zero_bid1 > 0:
        issues.append(f"ZERO bid_price_1: {zero_bid1} rows")
    if zero_ask1 > 0:
        issues.append(f"ZERO ask_price_1: {zero_ask1} rows")

    # 5. Negative spread
    neg_spread = (df["bid_ask_spread"] < 0).sum()
    if neg_spread > 0:
        issues.append(f"NEGATIVE SPREAD: {neg_spread} rows")

    # 6. Crossed market (bid > ask)
    crossed = (df["bid_price_1"] > df["ask_price_1"]).sum()
    if crossed > 0:
        issues.append(f"CROSSED MARKET (bid>ask): {crossed} rows")

    # 7. Timestamp ordering
    ts_sorted = df["timestamp"].is_monotonic_increasing
    if not ts_sorted:
        issues.append("TIMESTAMPS NOT SORTED")

    # 8. Duplicate timestamps
    dupes = df["timestamp"].duplicated().sum()
    if dupes > 0:
        issues.append(f"DUPLICATE TIMESTAMPS: {dupes}")

    # 9. Market hours check (09:15 to 15:30)
    times = pd.to_datetime(df["timestamp"])
    minutes = times.dt.hour * 60 + times.dt.minute
    before_open = (minutes < 9 * 60 + 15).sum()
    after_close = (minutes > 15 * 60 + 30).sum()
    if before_open > 0:
        issues.append(f"BEFORE MARKET OPEN: {before_open} rows")
    if after_close > 0:
        issues.append(f"AFTER MARKET CLOSE: {after_close} rows")

    # 10. TBQ/TSQ always zero (known Mode 3 limitation)
    tbq_always_zero = (df["total_buy_qty"] == 0).all()
    tbq_note = "YES (Mode3)" if tbq_always_zero else "NO"

    # 11. Price consistency (weighted_mid between bid and ask)
    mid_ok = (
        (df["weighted_mid_price"] >= df["bid_price_1"])
        & (df["weighted_mid_price"] <= df["ask_price_1"])
    ).sum()
    mid_bad = len(df) - mid_ok
    if mid_bad > 0:
        issues.append(f"MID PRICE OUTSIDE BID/ASK: {mid_bad} rows")

    # 12. Depth level ordering (bid1 > bid2, ask1 < ask2)
    bid_order_bad = (
        (df["bid_price_1"] < df["bid_price_2"]) & (df["bid_price_2"] > 0)
    ).sum()
    ask_order_bad = (
        (df["ask_price_1"] > df["ask_price_2"]) & (df["ask_price_2"] > 0)
    ).sum()
    if bid_order_bad > 0:
        issues.append(f"BID DEPTH MISORDERED: {bid_order_bad} rows")
    if ask_order_bad > 0:
        issues.append(f"ASK DEPTH MISORDERED: {ask_order_bad} rows")

    # 13. tick_direction values (should be -1, 0, or 1 only)
    valid_dirs = df["tick_direction"].isin([-1, 0, 1]).all()
    if not valid_dirs:
        issues.append("INVALID tick_direction values")

    # 14. depth_imbalance sanity
    depth_zero_but_data = (
        (df["depth_imbalance"] == 0)
        & (df["bid_depth_total"] > 0)
        & (df["ask_depth_total"] > 0)
    ).sum()
    if depth_zero_but_data > 0:
        issues.append(f"DEPTH_IMBALANCE=0 with valid depths: {depth_zero_but_data}")

    total_issues += len(issues)

    # Print
    status = "PASS" if len(issues) == 0 else "WARN"
    ltp_min = df["last_traded_price"].min() / 100
    ltp_max = df["last_traded_price"].max() / 100
    spread_avg = df["bid_ask_spread"].mean() / 100
    time_range = f"{times.min().strftime('%H:%M:%S')}-{times.max().strftime('%H:%M:%S')}"

    print(f"[{status}] {sym:12s} | {date_str} | {len(df):>5,} rows | LTP: {ltp_min:>8.2f}-{ltp_max:>8.2f} | Spread: {spread_avg:.2f} | Time: {time_range} | TBQ/TSQ=0: {tbq_note} | Dupes: {dupes}")
    if issues:
        for iss in issues:
            print(f"       >> {iss}")
    print()

print("=" * 90)
if total_issues == 0:
    print("QUALITY CHECK RESULT: ALL FILES PASSED")
else:
    print(f"QUALITY CHECK RESULT: {total_issues} warning(s) found")
