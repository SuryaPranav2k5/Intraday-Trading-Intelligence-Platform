"""
Exit Label Generation Pipeline

Generates training labels for exit models from clean trade states.

LABELS CREATED:
1. Future Path Labels (Transformer targets):
   - future_max_mfe_atr: Best upside still available
   - future_max_mae_atr: Worst downside still possible
   - future_return_atr: Close-to-close future return

2. Decision Labels (policy supervision):
   - hit_tp_before_sl: TP reached before SL
   - hit_sl_before_tp: SL reached before TP

3. Continuation Score (most important):
   - continuation_score: future_mfe / |future_mae|

LEAKAGE RULES:
- NO entry candle used
- NO crossing trade boundaries
- NO looking beyond trade exit
- NO mixing symbols
- All labels computed inside each trade_id
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE  = "simulation_results/trade_states_clean.parquet"
OUTPUT_FILE = "simulation_results/trade_states_labeled.parquet"

HORIZON = 10          # 10 bars ahead (10 minutes in 1-min data)
TP_ATR  = 0.6         # Take profit threshold
SL_ATR  = -0.4        # Stop loss threshold
EPS     = 1e-6        # Epsilon for division safety

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("EXIT LABEL GENERATION PIPELINE")
print("=" * 60)

print(f"\n📁 Loading clean exit states from: {INPUT_FILE}")
df = pd.read_parquet(INPUT_FILE)

print(f"   Rows: {len(df):,}")
print(f"   Unique trades: {df['trade_id'].nunique():,}")

# Sort for safety
df = df.sort_values(["trade_id", "step"]).reset_index(drop=True)

# ============================================================
# ALLOCATE LABEL COLUMNS
# ============================================================

print(f"\n📊 Allocating label columns...")

df["future_max_mfe_atr"] = np.nan
df["future_max_mae_atr"] = np.nan
df["future_return_atr"]  = np.nan
df["hit_tp_before_sl"]   = 0
df["hit_sl_before_tp"]   = 0
df["continuation_score"] = np.nan

# ============================================================
# LABEL GENERATION (PER TRADE - NO LEAKAGE)
# ============================================================

print(f"\n🔄 Generating labels with HORIZON={HORIZON}...")
print(f"   TP threshold: +{TP_ATR} ATR")
print(f"   SL threshold: {SL_ATR} ATR")

# Group by trade_id for no-leakage processing
trade_groups = df.groupby("trade_id")
n_trades = len(trade_groups)

# Process each trade
labeled_count = 0
skipped_count = 0

for trade_id, g in tqdm(trade_groups, total=n_trades, desc="Processing trades"):
    prices = g["price_from_entry_atr"].values
    idxs   = g.index.values
    n      = len(prices)
    
    for i in range(n):
        start = i + 1
        end   = min(i + 1 + HORIZON, n)
        
        # Skip if no future data available
        if start >= end:
            skipped_count += 1
            continue
        
        future_path = prices[start:end]
        
        # Future path labels
        max_mfe = np.max(future_path)
        min_mae = np.min(future_path)
        
        df.loc[idxs[i], "future_max_mfe_atr"] = max_mfe
        df.loc[idxs[i], "future_max_mae_atr"] = min_mae
        df.loc[idxs[i], "future_return_atr"]  = future_path[-1]
        
        # TP / SL ordering (decision labels)
        tp_hit = np.where(future_path >= TP_ATR)[0]
        sl_hit = np.where(future_path <= SL_ATR)[0]
        
        if len(tp_hit) > 0:
            if len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]:
                df.loc[idxs[i], "hit_tp_before_sl"] = 1
        
        if len(sl_hit) > 0:
            if len(tp_hit) == 0 or sl_hit[0] < tp_hit[0]:
                df.loc[idxs[i], "hit_sl_before_tp"] = 1
        
        # Continuation score (the gold metric)
        df.loc[idxs[i], "continuation_score"] = max_mfe / (abs(min_mae) + EPS)
        
        labeled_count += 1

print(f"\n✅ Label generation complete")
print(f"   Labeled: {labeled_count:,} states")
print(f"   Skipped (no future): {skipped_count:,} states")

# ============================================================
# DROP INCOMPLETE LABELS
# ============================================================

print(f"\n🧹 Dropping rows with incomplete labels...")

rows_before = len(df)
df = df.dropna(
    subset=[
        "future_max_mfe_atr",
        "future_max_mae_atr",
        "future_return_atr",
        "continuation_score",
    ]
).copy()  # Create explicit copy to avoid SettingWithCopyWarning
rows_dropped = rows_before - len(df)

print(f"   Dropped: {rows_dropped:,} rows")
print(f"   Remaining: {len(df):,} rows")

# ============================================================
# DISTRIBUTION CHECKS
# ============================================================

print(f"\n" + "=" * 60)
print("LABEL DISTRIBUTION CHECKS")
print("=" * 60)

print(f"\n📊 Future Max MFE (ATR):")
print(df["future_max_mfe_atr"].describe().to_string())

print(f"\n📊 Future Max MAE (ATR):")
print(df["future_max_mae_atr"].describe().to_string())

print(f"\n📊 Future Return (ATR):")
print(df["future_return_atr"].describe().to_string())

print(f"\n📊 Continuation Score:")
print(df["continuation_score"].describe().to_string())

print(f"\n📊 Decision Labels:")
tp_count = df["hit_tp_before_sl"].sum()
sl_count = df["hit_sl_before_tp"].sum()
neither  = len(df) - tp_count - sl_count + (df["hit_tp_before_sl"] & df["hit_sl_before_tp"]).sum()

print(f"   TP before SL: {tp_count:,} ({tp_count/len(df)*100:.1f}%)")
print(f"   SL before TP: {sl_count:,} ({sl_count/len(df)*100:.1f}%)")

# ============================================================
# CLIP LABELS FOR STABILITY
# ============================================================

print(f"\n📐 Clipping labels for stability...")

# Clip continuation_score
df["continuation_score"] = df["continuation_score"].clip(0, 10)
print(f"   continuation_score clipped to [0, 10]")

# Clip future_return_atr (extreme tails don't help)
df["future_return_atr"] = df["future_return_atr"].clip(-4, 4)
print(f"   future_return_atr clipped to [-4, 4]")

# Create compact exit_outcome for easier policy learning
# -1 = SL hit first, 0 = neither, 1 = TP hit first
df["exit_outcome"] = 0
df.loc[df["hit_tp_before_sl"] == 1, "exit_outcome"] = 1
df.loc[df["hit_sl_before_tp"] == 1, "exit_outcome"] = -1
print(f"   exit_outcome created: -1=SL, 0=neither, 1=TP")

# ============================================================
# SAVE LABELED DATASET
# ============================================================

print(f"\n" + "=" * 60)
print("SAVING LABELED DATASET")
print("=" * 60)

df.to_parquet(OUTPUT_FILE, index=False)

print(f"\n✅ EXIT LABELS GENERATED SUCCESSFULLY")
print(f"   File: {OUTPUT_FILE}")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

# Show new columns
new_cols = [
    "future_max_mfe_atr",
    "future_max_mae_atr", 
    "future_return_atr",
    "hit_tp_before_sl",
    "hit_sl_before_tp",
    "exit_outcome",
    "continuation_score"
]
print(f"\n   New label columns:")
for col in new_cols:
    print(f"      - {col}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
📊 Label Statistics:
   - Total labeled states: {len(df):,}
   - Unique trades:        {df['trade_id'].nunique():,}
   - Horizon:              {HORIZON} bars
   - TP threshold:         +{TP_ATR} ATR
   - SL threshold:         {SL_ATR} ATR

📁 Output File:
   {OUTPUT_FILE}

🎯 Labels Created:
   - future_max_mfe_atr    (Transformer target)
   - future_max_mae_atr    (Transformer target)
   - future_return_atr     (Transformer target)
   - hit_tp_before_sl      (LightGBM decision)
   - hit_sl_before_tp      (LightGBM decision)
   - continuation_score    (Exit control signal)

🔒 RULES FROM NOW ON:
   ❌ Do NOT recompute labels unless changing HORIZON
   ❌ If changing horizon, CREATE NEW COLUMNS
   ❌ Do NOT overwrite labeled data
   
   This is GROUND TRUTH for exit model training.
""")

print("=" * 60)
print("🎯 READY FOR EXIT MODEL TRAINING")
print("=" * 60)
