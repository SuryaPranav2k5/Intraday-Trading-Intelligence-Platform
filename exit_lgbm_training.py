"""
Exit LightGBM Policy Training

PURPOSE:
Train the EXIT vs HOLD decision model using:
- Transformer predictions (intelligence)
- Current trade state (reality)
- Risk & regime features (safety)

This is the ACTION layer of the exit system.
Transformer understands → LightGBM decides.

INPUT:  simulation_results/trade_states_with_tf.parquet
OUTPUT: exit_lgbm_policy.pkl
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "simulation_results/trade_states_with_tf.parquet"
MODEL_FILE = "exit_lgbm_policy.pkl"

RANDOM_STATE = 42

# ============================================================
# FEATURE DEFINITIONS (20 FEATURES - LOCKED)
# ============================================================

FEATURES = [
    # === TRANSFORMER INTELLIGENCE (4 features) ===
    "tf_pred_future_max_mfe_atr",
    "tf_pred_future_max_mae_atr",
    "tf_pred_future_return_atr",
    "tf_pred_continuation_score",
    
    # === CURRENT TRADE STATE (5 features) ===
    "unrealized_pnl_atr",
    "price_from_entry_atr",
    "mfe_atr",
    "mae_atr",
    "step_log",
    
    # === RISK & REGIME (4 features) ===
    "volatility_expansion_ratio",
    "momentum_decay",
    "pullback_depth",
    "vwap_from_entry_atr",
    
    # === EXTENDED STATE (3 features) ===
    "step",                    # Raw step count
    "current_atr",             # Current volatility
    "initial_atr",             # Entry volatility
    
    # === ENTRY CONTEXT (4 features) ===
    "entry_confidence",        # From entry model (if available)
    "direction",               # +1 for LONG, -1 for SHORT
    "symbol_id",               # 0-5 encoded
    "direction_x_pnl",         # Interaction: direction * unrealized_pnl
]

# Total: 20 features (optimal for LightGBM)

# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("=" * 60)
    print("EXIT LightGBM POLICY TRAINING")
    print("=" * 60)
    
    # ---------- Load Data ----------
    print(f"\n📁 Loading data from: {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)
    
    print(f"   Rows: {len(df):,}")
    print(f"   Trades: {df['trade_id'].nunique():,}")
    
    # ---------- Create Target ----------
    print(f"\n🎯 Creating target variable...")
    
    # Binary target: EXIT (1) or HOLD (0)
    # exit_outcome: -1 = SL hit, 0 = neither, 1 = TP hit
    # We want to exit if outcome is non-zero
    df["exit_now"] = (df["exit_outcome"] != 0).astype(int)
    
    print(f"   HOLD (0): {(df['exit_now'] == 0).sum():,} ({(df['exit_now'] == 0).mean()*100:.1f}%)")
    print(f"   EXIT (1): {(df['exit_now'] == 1).sum():,} ({(df['exit_now'] == 1).mean()*100:.1f}%)")
    
    # ---------- Encode Direction ----------
    print(f"\n🔄 Encoding direction...")
    df["direction"] = df["direction"].map({"LONG": 1, "SHORT": -1})
    
    # ---------- Encode Symbol ----------
    print(f"🔄 Encoding symbol...")
    symbol_map = {"LT": 0, "RELIANCE": 1, "SIEMENS": 2, "TATAELXSI": 3, "TITAN": 4, "TVSMOTOR": 5}
    df["symbol_id"] = df["symbol"].map(symbol_map)
    
    # ---------- Create Derived Features ----------
    print(f"🔄 Creating derived features...")
    
    # Interaction feature: direction * unrealized_pnl
    df["direction_x_pnl"] = df["direction"] * df["unrealized_pnl_atr"]
    
    # Entry confidence (placeholder if not available)
    if "entry_confidence" not in df.columns:
        # Use a reasonable default or derive from other features
        df["entry_confidence"] = 0.5  # Neutral confidence
        print(f"   ⚠️ entry_confidence not available, using default 0.5")
    
    # Ensure initial_atr exists
    if "initial_atr" not in df.columns:
        if "current_atr" in df.columns:
            df["initial_atr"] = df.groupby("trade_id")["current_atr"].transform("first")
        else:
            df["initial_atr"] = 1.0
        print(f"   ℹ️ initial_atr derived from current_atr")
    
    # Ensure current_atr exists
    if "current_atr" not in df.columns:
        df["current_atr"] = df["initial_atr"]
        print(f"   ⚠️ current_atr not available, using initial_atr")
    
    # ---------- Check Features Exist ----------
    print(f"\n📊 Checking features...")
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"   ❌ Missing features: {missing}")
        print(f"   Available columns: {df.columns.tolist()}")
        return
    else:
        print(f"   ✅ All {len(FEATURES)} features present")
    
    # ---------- Prepare Data ----------
    X = df[FEATURES].copy()
    y = df["exit_now"].copy()
    groups = df["trade_id"].copy()
    
    # Handle any NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️ Filling {nan_count} NaN values with 0")
        X = X.fillna(0)
    
    # ---------- Train/Validation Split by Trade ----------
    print(f"\n📊 Splitting by trade_id...")
    
    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=0.7,
        random_state=RANDOM_STATE
    )
    
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    
    # ---------- Handle Class Imbalance ----------
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    print(f"\n⚖️ Class imbalance: pos_weight = {pos_weight:.2f}")
    
    # ---------- Create LightGBM Datasets ----------
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # ---------- Model Parameters ----------
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        
        "min_data_in_leaf": 300,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        
        "scale_pos_weight": pos_weight,
        
        "verbosity": -1,
        "seed": RANDOM_STATE,
    }
    
    # ---------- Train ----------
    print(f"\n🚀 Training Exit LightGBM...")
    print(f"   Parameters:")
    print(f"      learning_rate: {params['learning_rate']}")
    print(f"      num_leaves: {params['num_leaves']}")
    print(f"      max_depth: {params['max_depth']}")
    print(f"      min_data_in_leaf: {params['min_data_in_leaf']}")
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=50)
    ]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=callbacks
    )
    
    # ---------- Evaluation ----------
    print(f"\n📊 Evaluation:")
    
    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred)
    print(f"   Validation AUC: {auc:.4f}")
    
    # Classification report at threshold 0.5
    y_pred_binary = (val_pred > 0.5).astype(int)
    print(f"\n   Classification Report (threshold=0.5):")
    print(classification_report(y_val, y_pred_binary, target_names=["HOLD", "EXIT"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred_binary)
    print(f"   Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  HOLD    EXIT")
    print(f"   Actual HOLD   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"   Actual EXIT   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # ---------- Feature Importance ----------
    print(f"\n📊 Feature Importance (top 10):")
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
    
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["importance"] / importance["importance"].max() * 20)
        print(f"   {row['feature']:35s} {bar} {row['importance']:.0f}")
    
    # ---------- Save Model ----------
    print(f"\n💾 Saving model to: {MODEL_FILE}")
    
    model_package = {
        "model": model,
        "features": FEATURES,
        "threshold": 0.5,
        "auc": auc,
        "params": params,
    }
    
    joblib.dump(model_package, MODEL_FILE)
    
    # ---------- Summary ----------
    print(f"\n" + "=" * 60)
    print("EXIT LightGBM TRAINING COMPLETE")
    print("=" * 60)
    print(f"""
📊 Model Statistics:
   - Validation AUC: {auc:.4f}
   - Features: {len(FEATURES)}
   - Best iteration: {model.best_iteration}
   - Threshold: 0.5

📁 Output:
   {MODEL_FILE}

🔒 FREEZE RULES:
   ❌ Do NOT retrain unless:
      - New symbols added
      - Entry logic changed
      - Exit horizon changed

🎯 NEXT: Run live_exit_engine.py for real-time exits
""")


if __name__ == "__main__":
    main()
