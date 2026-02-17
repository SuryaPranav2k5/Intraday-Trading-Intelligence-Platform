import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
import sys
import os

# Add current directory to path so we can import inference
sys.path.append(os.path.dirname(__file__))
from inference import InferenceEngine, get_inference_engine

# Configuration
SYMBOL = "TVSMOTOR"  # Default symbol to backtest
DATA_PATH = Path("Dataset/TVSMOTOR_2years_1min_fixed.parquet")
INITIAL_CAPITAL = 100000
QUANTITY = 25  # Fixed quantity

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    
    # Filter market hours (same as training/inference)
    df = df.between_time(time(9, 15), time(15, 30))
    return df

def run_backtest(symbol, df):
    print(f"Initializing Inference Engine for {symbol}...")
    engine = InferenceEngine()
    
    if symbol not in engine.models:
        print(f"❌ Model for {symbol} not found in model_artifacts!")
        return
        
    print("Generating Features (Exact Inference Pipeline)...")
    # We use the internal _compute_features method with return_full_df=True
    # This ensures 100% logic parity with live execution
    df_features = engine._compute_features(df, return_full_df=True)
    
    if df_features.empty:
        print("❌ Feature engineering failed.")
        return

    # Add symbol_id (needed for models)
    # Using the same map from inference.py
    SYMBOL_ID_MAP = {
        "TVSMOTOR": 5, "RELIANCE": 1, "LT": 2, 
        "TITAN": 3, "SIEMENS": 4, "TATAELXSI": 6
    }
    df_features['symbol_id'] = SYMBOL_ID_MAP.get(symbol, 0)
    
    print("Running Predictions...")
    # Get models and thresholds
    thresholds = engine.thresholds.get(symbol, {})
    
    # Ensemble weights/thresholds
    w_long = thresholds.get("ensemble_long_weight_lgb", 0.5)
    w_short = thresholds.get("ensemble_short_weight_lgb", 0.5)
    thresh_long = thresholds.get("ensemble_long_thresh", 0.5)
    thresh_short = thresholds.get("ensemble_short_thresh", 0.5)
    
    print(f"  Long Config:  LGB Weight={w_long:.2f}, Threshold={thresh_long:.3f}")
    print(f"  Short Config: LGB Weight={w_short:.2f}, Threshold={thresh_short:.3f}")

    # Prepare input features
    feature_cols = engine.models[symbol].get("feature_cols", engine.feature_cols)
    available_cols = [c for c in feature_cols if c in df_features.columns]
    X = df_features[available_cols].fillna(0).astype(np.float64)
    
    # 1. LightGBM Predictions
    lgb_long = engine.models[symbol]["lgb_long"].predict(X)
    lgb_short = engine.models[symbol]["lgb_short"].predict(X)
    
    # 2. XGBoost Predictions
    import xgboost as xgb
    dmatrix = xgb.DMatrix(X)
    xgb_long = engine.models[symbol]["xgb_long"].predict(dmatrix)
    xgb_short = engine.models[symbol]["xgb_short"].predict(dmatrix)
    
    # 3. Ensemble
    probs_long = (w_long * lgb_long) + ((1 - w_long) * xgb_long)
    probs_short = (w_short * lgb_short) + ((1 - w_short) * xgb_short)
    
    # 4. Generate Signals
    df_features['prob_long'] = probs_long
    df_features['prob_short'] = probs_short
    
    # Apply Logic
    df_features['signal_long'] = (probs_long >= thresh_long).astype(int)
    df_features['signal_short'] = (probs_short >= thresh_short).astype(int)
    
    # Mutual exclusivity (simple version for backtest)
    # If both active, take stronger relative to threshold
    conflict = (df_features['signal_long'] == 1) & (df_features['signal_short'] == 1)
    
    df_features.loc[conflict & ((probs_long/thresh_long) > (probs_short/thresh_short)), 'signal_short'] = 0
    df_features.loc[conflict & ((probs_long/thresh_long) <= (probs_short/thresh_short)), 'signal_long'] = 0
    
    # Filter out signals based on "Should Not Trade" session quality
    # This matches inference.py logic exactly
    df_features['signal_long'] = df_features['signal_long'] * (1 - df_features['should_not_trade'])
    df_features['signal_short'] = df_features['signal_short'] * (1 - df_features['should_not_trade'])

    # === SIMULATION LOOP ===
    print("Simulating Trades (Max 1 per day, Risk-Based Sizing)...")
    trades = []
    in_trade = False
    entry_price = 0
    entry_time = None
    direction = 0 # 1 long, -1 short
    qty = 0
    
    current_date = None
    trades_today = 0
    MAX_TRADES_PER_DAY = 1
    RISK_PER_TRADE = 2000  # Risk ₹2000 per trade
    
    # Simple exit logic for backtest (Target/Stop using ATR)
    # We use the same 'entry_atr' logic as app.py
    
    for i in range(len(df_features)):
        row = df_features.iloc[i]
        ts = row.name
        
        # Reset daily counter on new date
        if current_date != ts.date():
            current_date = ts.date()
            trades_today = 0
            
        price = row['close']
        atr = row['ATR_14']
        
        if not in_trade:
            # ENTRY LOGIC
            # Time gate: 10:15 - 15:15 (From inference.py)
            # AND Trade limit check
            if time(10, 15) <= ts.time() <= time(15, 15) and trades_today < MAX_TRADES_PER_DAY:
                
                # Dynamic Sizing: Risk ₹2000 / (0.5 * ATR)
                # Stop is 0.5 * ATR. Total risk = qty * 0.5 * ATR.
                sl_distance = 0.5 * atr
                if sl_distance > 0:
                    qty = int(RISK_PER_TRADE / sl_distance)
                else:
                    qty = 0
                    
                if qty > 0:
                    if row['signal_long']:
                        in_trade = True
                        trades_today += 1 
                        entry_price = price
                        entry_time = ts
                        direction = 1
                        tp = price + (1.5 * atr) 
                        sl = price - (0.5 * atr) 
                    elif row['signal_short']:
                        in_trade = True
                        trades_today += 1 
                        entry_price = price
                        entry_time = ts
                        direction = -1
                        tp = price - (1.5 * atr)
                        sl = price + (0.5 * atr)
        else:
            # EXIT LOGIC
            exit_reason = None
            
            # Hit API
            if direction == 1:
                if row['high'] >= tp: exit_reason = "Target"
                elif row['low'] <= sl: exit_reason = "Stop"
            else:
                if row['low'] <= tp: exit_reason = "Target"
                elif row['high'] >= sl: exit_reason = "Stop"
                
            # End of Day Exit
            if ts.time() >= time(15, 15):
                exit_reason = "EOD"
                
            if exit_reason:
                exit_price = tp if exit_reason == "Target" else (sl if exit_reason == "Stop" else price)
                pnl = (exit_price - entry_price) * direction * qty
                
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': ts,
                    'Type': 'Long' if direction == 1 else 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Qty': qty,
                    'PnL': pnl,
                    'Reason': exit_reason
                })
                in_trade = False
    
    # === REPORTING ===
    if not trades:
        print("No trades generated.")
        return

    trades_df = pd.DataFrame(trades)
    print("\n" + "="*40)
    print("BACKTEST RESULTS (INFERENCE LOGIC)")
    print("="*40)
    print(f"Total Trades:      {len(trades_df)}")
    print(f"Win Rate:          {len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100:.1f}%")
    print(f"Total PnL:         ₹{trades_df['PnL'].sum():.2f}")
    print(f"Avg PnL per Trade: ₹{trades_df['PnL'].mean():.2f}")
    print(f"Profit Factor:     {trades_df[trades_df['PnL']>0]['PnL'].sum() / abs(trades_df[trades_df['PnL']<0]['PnL'].sum()):.2f}")
    print("-" * 40)
    print(trades_df.tail(10).to_string())
    
    # Save to CSV
    trades_df.to_csv(f"backtest_results_{symbol}.csv")
    print(f"\nSaved detailed trade log to backtest_results_{symbol}.csv")

if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
    else:
        df = load_data(DATA_PATH)
        run_backtest(SYMBOL, df)
