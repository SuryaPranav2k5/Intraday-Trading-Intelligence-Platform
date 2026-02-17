"""
Exit Data Cleaning Pipeline

Transforms raw trade_states.parquet into ML-ready trade_states_clean.parquet

This script:
1. Validates structure
2. Removes broken/useless trades
3. Cleans numerical pathologies
4. Clips extreme values
5. Verifies distributions
6. Saves frozen clean dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "simulation_results/trade_states.parquet"
OUTPUT_FILE = "simulation_results/trade_states_clean.parquet"
TRADES_INPUT = "simulation_results/trades.csv"
TRADES_OUTPUT = "simulation_results/trades_clean.csv"

CLIP_LIMIT = 8.0  # ±8 ATR is the max we allow
MAX_STEP = 120    # Max trade duration (2 hours in 1-min data)

# ============================================================
# STEP 1 — LOAD & BASIC VALIDATION
# ============================================================

print("=" * 60)
print("EXIT DATA CLEANING PIPELINE")
print("=" * 60)

print("\n📁 Loading raw data...")
df = pd.read_parquet(INPUT_FILE)

print(f"\n📊 Initial Statistics:")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")
print(f"\n   Column names: {df.columns.tolist()}")

# Validate required columns exist
required_cols = ['trade_id', 'symbol', 'step', 'price_from_entry_atr', 
                 'mfe_atr', 'mae_atr', 'unrealized_pnl_atr']

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"❌ CRITICAL: Missing required columns: {missing}")
else:
    print(f"\n✅ All required columns present")

# ============================================================
# STEP 2 — REMOVE INVALID ROWS
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: REMOVING INVALID ROWS")
print("=" * 60)

initial_rows = len(df)

# 2.1 Drop NaN / Inf
print("\n2.1 Removing NaN/Inf values...")
df = df.replace([np.inf, -np.inf], np.nan)
nan_count = df.isna().sum().sum()
df = df.dropna()
print(f"   Removed {initial_rows - len(df):,} rows with NaN/Inf")
print(f"   Remaining: {len(df):,} rows")

# 2.2 Remove trades with < 2 states
print("\n2.2 Removing trades with < 2 states...")
trade_counts = df.groupby("trade_id")["step"].count()
short_trades = trade_counts[trade_counts < 2].index
valid_trades = trade_counts[trade_counts >= 2].index

rows_before = len(df)
df = df[df["trade_id"].isin(valid_trades)]
print(f"   Removed {len(short_trades):,} short trades ({rows_before - len(df):,} rows)")
print(f"   Remaining: {len(df):,} rows")

# 2.3 Remove ATR = 0 or corrupted
print("\n2.3 Checking ATR sanity...")

# Check initial_atr > 0
if 'initial_atr' in df.columns:
    rows_before = len(df)
    df = df[df["initial_atr"] > 0]
    print(f"   Removed {rows_before - len(df):,} rows with initial_atr <= 0")

# Check current_atr > 0
if 'current_atr' in df.columns:
    rows_before = len(df)
    df = df[(df["current_atr"] > 0)]
    print(f"   Removed {rows_before - len(df):,} rows with current_atr <= 0")
else:
    print(f"   ⚠️ 'current_atr' column not found, skipping ATR filter")

print(f"   Remaining: {len(df):,} rows")

# ============================================================
# STEP 3 — CLIP EXTREME VALUES (CRITICAL)
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: CLIPPING EXTREME VALUES")
print("=" * 60)

clip_cols = [
    "price_from_entry_atr",
    "vwap_from_entry_atr",
    "mfe_atr",
    "mae_atr",
    "unrealized_pnl_atr",
    "pullback_depth"
]

for col in clip_cols:
    if col in df.columns:
        before_min, before_max = df[col].min(), df[col].max()
        df[col] = df[col].clip(-CLIP_LIMIT, CLIP_LIMIT)
        after_min, after_max = df[col].min(), df[col].max()
        
        clipped = (before_min < -CLIP_LIMIT) or (before_max > CLIP_LIMIT)
        status = "✅ CLIPPED" if clipped else "✓ OK"
        print(f"   {col:30s} [{before_min:+.2f}, {before_max:+.2f}] → [{after_min:+.2f}, {after_max:+.2f}] {status}")
    else:
        print(f"   {col:30s} ⚠️ Column not found")

# Also clip momentum_decay (should already be clipped but double-check)
if 'momentum_decay' in df.columns:
    df['momentum_decay'] = df['momentum_decay'].clip(-3.0, 3.0)
    print(f"   {'momentum_decay':30s} Clipped to [-3, 3]")

# ============================================================
# STEP 4 — NORMALIZE LIFECYCLE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: NORMALIZING LIFECYCLE FEATURES")
print("=" * 60)

# 4.1 Step sanity - remove negative steps
rows_before = len(df)
df = df[df["step"] >= 0]
print(f"   Removed {rows_before - len(df):,} rows with negative step")

# 4.2 Cap extreme trade durations (very long trades bias attention)
rows_before = len(df)
df = df[df["step"] <= MAX_STEP]
print(f"   Removed {rows_before - len(df):,} rows with step > {MAX_STEP} (extreme duration)")

# 4.3 Add log-transformed step (helps with long trades)
df["step_log"] = np.log1p(df["step"])
print(f"   Added 'step_log' feature: log1p(step)")

# 4.4 Normalize volatility expansion ratio
if 'volatility_expansion_ratio' in df.columns:
    df['volatility_expansion_ratio'] = df['volatility_expansion_ratio'].clip(0.1, 5.0)
    print(f"   Clipped 'volatility_expansion_ratio' to [0.1, 5.0]")

print(f"\n   Final rows: {len(df):,}")

# ============================================================
# STEP 5 — DISTRIBUTION CHECKS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: DISTRIBUTION VERIFICATION")
print("=" * 60)

print("\n📊 Step distribution:")
print(df["step"].describe().to_string())

print("\n📊 Unrealized P&L (ATR) distribution:")
print(df["unrealized_pnl_atr"].describe().to_string())

print("\n📊 MFE (ATR) distribution:")
print(df["mfe_atr"].describe().to_string())

print("\n📊 MAE (ATR) distribution:")
print(df["mae_atr"].describe().to_string())

print("\n📊 States per symbol:")
symbol_counts = df.groupby("symbol").size()
for sym, count in symbol_counts.items():
    pct = count / len(df) * 100
    print(f"   {sym:12s}: {count:>10,} ({pct:5.1f}%)")

print("\n📊 Exit reasons distribution:")
if 'exit_reason' in df.columns:
    exit_counts = df.groupby("exit_reason").size()
    for reason, count in exit_counts.items():
        pct = count / len(df) * 100
        print(f"   {reason:15s}: {count:>10,} ({pct:5.1f}%)")

print("\n📊 Direction distribution:")
if 'direction' in df.columns:
    dir_counts = df.groupby("direction").size()
    for direction, count in dir_counts.items():
        pct = count / len(df) * 100
        print(f"   {direction:5s}: {count:>10,} ({pct:5.1f}%)")

# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)

checks_passed = True

# Check 1: Reasonable step mean
step_mean = df["step"].mean()
if 1 < step_mean < 20:
    print(f"✅ Step mean: {step_mean:.2f} (expected 2-15)")
else:
    print(f"⚠️  Step mean: {step_mean:.2f} (unusual, but may be OK)")

# Check 2: Balanced symbols
min_sym = symbol_counts.min()
max_sym = symbol_counts.max()
ratio = max_sym / min_sym
if ratio < 2.0:
    print(f"✅ Symbol balance: ratio {ratio:.2f} (good)")
else:
    print(f"⚠️  Symbol balance: ratio {ratio:.2f} (some imbalance)")

# Check 3: No remaining NaN
nan_count = df.isna().sum().sum()
if nan_count == 0:
    print(f"✅ No NaN values remaining")
else:
    print(f"❌ FAIL: {nan_count} NaN values still present!")
    checks_passed = False

# Check 4: Reasonable PnL range
pnl_min, pnl_max = df["unrealized_pnl_atr"].min(), df["unrealized_pnl_atr"].max()
if pnl_min >= -CLIP_LIMIT and pnl_max <= CLIP_LIMIT:
    print(f"✅ PnL range: [{pnl_min:.2f}, {pnl_max:.2f}] (within bounds)")
else:
    print(f"❌ FAIL: PnL range [{pnl_min:.2f}, {pnl_max:.2f}] exceeds clip limits!")
    checks_passed = False

# ============================================================
# STEP 6 — SAVE CLEAN DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: SAVING CLEAN DATASET")
print("=" * 60)

if checks_passed:
    # Save clean states
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n✅ CLEAN EXIT DATASET SAVED")
    print(f"   File: {OUTPUT_FILE}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # Calculate file size
    file_size = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)
    print(f"   Size: {file_size:.1f} MB")
    
    # Also clean and save trades summary
    print(f"\n📁 Cleaning trades summary...")
    trades_df = pd.read_csv(TRADES_INPUT)
    initial_trades = len(trades_df)
    
    # Keep only trades that are in clean states
    clean_trade_ids = df["trade_id"].unique()
    trades_df = trades_df[trades_df["trade_id"].isin(clean_trade_ids)]
    
    trades_df.to_csv(TRADES_OUTPUT, index=False)
    print(f"   Trades: {initial_trades:,} → {len(trades_df):,}")
    print(f"   File: {TRADES_OUTPUT}")
    
else:
    print(f"\n❌ VALIDATION FAILED - Dataset NOT saved")
    print(f"   Please fix the issues above before proceeding")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
📊 Dataset Statistics:
   - Original rows:     919,906
   - Clean rows:        {len(df):,}
   - Rows removed:      {919906 - len(df):,} ({(919906 - len(df))/919906*100:.1f}%)
   
   - Unique trades:     {df['trade_id'].nunique():,}
   - Unique symbols:    {df['symbol'].nunique()}
   - Columns:           {len(df.columns)}
   
📁 Output Files:
   - {OUTPUT_FILE}
   - {TRADES_OUTPUT}

🔒 FROM THIS POINT:
   ❌ Do NOT regenerate
   ❌ Do NOT re-clean  
   ❌ Do NOT mix raw + clean
   
   This file is now GROUND TRUTH for exit model training.
""")

print("=" * 60)
print("🎯 READY FOR EXIT MODEL TRAINING")
print("=" * 60)
