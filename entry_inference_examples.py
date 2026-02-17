"""
Example: How to use the frozen Entry Inference Engine

This demonstrates the correct way to use entry_inference.py 
for exit simulation and live trading.
"""

from entry_inference import EntryInferenceEngine
from datetime import datetime
import pandas as pd

# ============================================================
# EXAMPLE 1: Single Symbol Setup
# ============================================================

def example_single_symbol():
    """Load engine for a single symbol and make predictions"""
    
    print("=" * 60)
    print("EXAMPLE 1: Single Symbol Entry Inference")
    print("=" * 60)
    
    # Initialize the engine for TITAN
    engine = EntryInferenceEngine("TITAN/model_artifacts")
    
    print(f"✅ Loaded TITAN entry engine")
    print(f"   Features required: {len(engine.feature_list)}")
    print(f"   Market hours: {engine.market_open} - {engine.market_close}")
    
    # Example feature dictionary (you would compute these from live/historical data)
    feature_dict = {
        "log_return_1m": 0.002,
        "log_return_3m": 0.005,
        "log_return_5m": 0.008,
        "candle_body_pct": 0.6,
        "upper_wick_pct": 0.2,
        "lower_wick_pct": 0.2,
        "range_pct": 0.015,
        "return_5m": 0.008,
        "return_15m": 0.012,
        "return_30m": 0.015,
        "price_momentum_3": 1.5,
        "price_momentum_5": 2.0,
        "price_momentum_10": 2.5,
        "EMA_9": 2500.0,
        "EMA_21": 2480.0,
        "EMA_50": 2450.0,
        "EMA_slope_9": 0.002,
        "EMA_slope_21": 0.001,
        "EMA_9_vs_21_distance": 0.008,
        "EMA_21_vs_50_distance": 0.012,
        "price_vs_EMA9": 0.004,
        "price_vs_EMA21": 0.008,
        "price_vs_EMA50": 0.02,
        "higher_high": 1.0,
        "lower_low": 0.0,
        "dist_from_swing_high": 0.02,
        "dist_from_swing_low": 0.05,
        "trend_strength_score": 0.7,
        "price_position_in_range": 0.65,
        "RSI_14": 58.0,
        "RSI_slope": 0.5,
        "RSI_5m": 60.0,
        "Stoch_K": 65.0,
        "Stoch_D": 60.0,
        "MACD_hist": 2.5,
        "momentum_consistency": 0.8,
        "ATR_14": 25.0,
        "ATR_5m": 20.0,
        "ATR_ratio": 1.25,
        "Bollinger_width": 0.04,
        "BB_position": 0.6,
        "volatility_percentile": 0.5,
        "true_range_expansion": 1.2,
        "volatility_regime": 1.0,
        "volume_zscore": 1.5,
        "volume_ratio_1m_vs_20m": 1.3,
        "VWAP": 2495.0,
        "VWAP_distance": 0.002,
        "VWAP_slope": 0.001,
        "OBV_delta": 1000.0,
        "volume_spike_flag": 0.0,
        "vol_price_corr": 0.6,
        "minute_of_day_sin": 0.5,
        "minute_of_day_cos": 0.866,
        "day_of_week_sin": 0.0,
        "day_of_week_cos": 1.0,
        "is_open_range": 0.0,
        "is_lunch_time": 0.0,
        "is_close_window": 0.0,
        "minutes_since_open": 75,
        "minutes_to_close": 285,
        "symbol_id": 4,
        "EMA_15m": 2490.0,
        "trend_15m": 1.0,
        "price_above_15m_ema": 1.0,
        "dist_prev_day_high": 0.01,
        "dist_prev_day_low": 0.05,
        "gap_up_flag": 0.0,
        "gap_down_flag": 0.0,
        "inside_day": 0.0,
        "outside_day": 0.0
    }
    
    # Make prediction
    timestamp = datetime(2026, 1, 17, 10, 30, 0)
    decision = engine.predict(feature_dict, "TITAN", timestamp)
    
    print(f"\n📊 Prediction Result:")
    print(f"   Timestamp: {timestamp}")
    print(f"   Decision: {decision}")
    
    return decision


# ============================================================
# EXAMPLE 2: Multi-Symbol Setup
# ============================================================

def example_multi_symbol():
    """Load engines for multiple symbols"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-Symbol Entry Inference")
    print("=" * 60)
    
    symbols = ['LT', 'RELIANCE', 'TITAN']
    engines = {}
    
    # Load all engines
    for symbol in symbols:
        engines[symbol] = EntryInferenceEngine(f"{symbol}/model_artifacts")
        print(f"✅ Loaded {symbol} engine")
    
    print(f"\n📦 Total engines loaded: {len(engines)}")
    
    return engines


# ============================================================
# EXAMPLE 3: Live Trading Integration Pattern
# ============================================================

def example_live_trading_pattern():
    """
    Pattern for integrating with live trading system
    This shows how you would use it in app.py or inference.py
    """
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Live Trading Integration Pattern")
    print("=" * 60)
    
    # Pseudo-code showing integration pattern
    code = '''
# In your app.py or inference.py:

from entry_inference import EntryInferenceEngine
import pandas as pd
from datetime import datetime

# Initialize engines at startup (once)
ENGINES = {
    'LT': EntryInferenceEngine('LT/model_artifacts'),
    'RELIANCE': EntryInferenceEngine('RELIANCE/model_artifacts'),
    'TITAN': EntryInferenceEngine('TITAN/model_artifacts'),
    # ... other symbols
}

def process_new_candle(symbol: str, df_history: pd.DataFrame):
    """
    Called when a new 1-minute candle closes
    
    df_history: DataFrame with at least 300 recent 1-min candles
                containing all raw OHLCV data
    """
    
    # Step 1: Compute features from df_history
    features = compute_features(df_history)  # Your existing feature engineering
    
    # Step 2: Extract latest feature row as dict
    feature_dict = features.iloc[-1].to_dict()
    
    # Step 3: Get current timestamp
    timestamp = datetime.now()
    
    # Step 4: Get entry decision
    decision = ENGINES[symbol].predict(feature_dict, symbol, timestamp)
    
    # Step 5: Act on decision
    if decision == "LONG":
        print(f"🟢 {symbol}: LONG entry signal at {timestamp}")
        # Place long order / Start Phase 2 monitoring
        
    elif decision == "SHORT":
        print(f"🔴 {symbol}: SHORT entry signal at {timestamp}")
        # Place short order / Start Phase 2 monitoring
        
    else:
        print(f"⚪ {symbol}: No entry signal")
        # Do nothing, wait for next candle
    
    return decision
'''
    
    print(code)


# ============================================================
# EXAMPLE 4: Exit Simulation Pattern
# ============================================================

def example_exit_simulation_pattern():
    """Pattern for exit simulation using frozen entry engine"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Exit Simulation Pattern")
    print("=" * 60)
    
    code = '''
# Exit simulation workflow:

from entry_inference import EntryInferenceEngine
import pandas as pd

# 1. Load entry engine
entry_engine = EntryInferenceEngine('TITAN/model_artifacts')

# 2. Load historical data
df = pd.read_parquet('Dataset/TITAN_2years_1min_fixed.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 3. Compute features for ALL bars
features_df = compute_all_features(df)  # Your feature engineering

# 4. Find entry signals
entry_signals = []

for idx in range(300, len(features_df)):  # Skip first 300 for feature warmup
    row = features_df.iloc[idx]
    feature_dict = row.to_dict()
    timestamp = row['timestamp']
    
    decision = entry_engine.predict(feature_dict, 'TITAN', timestamp)
    
    if decision in ['LONG', 'SHORT']:
        entry_signals.append({
            'timestamp': timestamp,
            'direction': decision,
            'entry_price': df.iloc[idx]['close'],
            'entry_idx': idx
        })

print(f"Found {len(entry_signals)} entry signals")

# 5. For each entry, simulate exit using your Phase 2 logic
trades = []

for entry in entry_signals:
    # Run your exit engine from entry_idx forward
    exit_result = simulate_exit(
        df=df,
        start_idx=entry['entry_idx'],
        direction=entry['direction'],
        entry_price=entry['entry_price']
    )
    
    trades.append({
        **entry,
        **exit_result
    })

# 6. Analyze results
trades_df = pd.DataFrame(trades)
print(f"Total P&L: {trades_df['pnl'].sum()}")
print(f"Win rate: {(trades_df['pnl'] > 0).mean():.2%}")
'''
    
    print(code)


# ============================================================
# RUN ALL EXAMPLES
# ============================================================

if __name__ == "__main__":
    
    # Example 1: Single symbol
    example_single_symbol()
    
    # Example 2: Multi-symbol
    example_multi_symbol()
    
    # Example 3: Live trading pattern
    example_live_trading_pattern()
    
    # Example 4: Exit simulation pattern
    example_exit_simulation_pattern()
    
    print("\n" + "=" * 60)
    print("✅ ALL EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Load engine once at startup")
    print("  2. Call .predict() for each new candle")
    print("  3. Feature dict must have exact 71 features")
    print("  4. Returns: 'LONG', 'SHORT', or None")
    print("  5. Entry engine is now a black box that just works")
