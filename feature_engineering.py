"""
Feature Engineering Module

Extracted from train_setup_quality.py
Reusable for both training and simulation
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FEATURE GROUP 1 — PRICE & RETURN STRUCTURE
# ============================================================

def compute_price_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Core price action features"""
    df = df.copy()
    
    # Log returns (better for trees)
    df['log_return_1m'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_3m'] = np.log(df['close'] / df['close'].shift(3))
    df['log_return_5m'] = np.log(df['close'] / df['close'].shift(5))
    
    # Candle structure
    df['candle_body_pct'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Multi-timeframe returns
    df['return_5m'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['return_15m'] = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    df['return_30m'] = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Price momentum
    df['price_momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    return df


# ============================================================
# FEATURE GROUP 2 — TREND & MARKET STRUCTURE
# ============================================================

def compute_trend_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trend features - directional edge"""
    df = df.copy()
    
    # EMAs
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # EMA slopes (normalized)
    df['EMA_slope_9'] = (df['EMA_9'] - df['EMA_9'].shift(3)) / df['EMA_9'].shift(3)
    df['EMA_slope_21'] = (df['EMA_21'] - df['EMA_21'].shift(5)) / df['EMA_21'].shift(5)
    
    # EMA distances
    df['EMA_9_vs_21_distance'] = (df['EMA_9'] - df['EMA_21']) / df['close']
    df['EMA_21_vs_50_distance'] = (df['EMA_21'] - df['EMA_50']) / df['close']
    
    # Price vs EMAs
    df['price_vs_EMA9'] = (df['close'] - df['EMA_9']) / df['close']
    df['price_vs_EMA21'] = (df['close'] - df['EMA_21']) / df['close']
    df['price_vs_EMA50'] = (df['close'] - df['EMA_50']) / df['close']
    
    # Swing detection
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))).astype(int)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))).astype(int)
    
    swing_high = df['high'].rolling(20).max()
    swing_low = df['low'].rolling(20).min()
    df['dist_from_swing_high'] = (swing_high - df['close']) / df['close']
    df['dist_from_swing_low'] = (df['close'] - swing_low) / df['close']
    
    # Trend strength
    df['trend_strength_score'] = (df['EMA_9'] - df['EMA_50']) / df['close']
    df['price_position_in_range'] = (df['close'] - swing_low) / (swing_high - swing_low + 1e-9)
    
    return df


# ============================================================
# FEATURE GROUP 3 — MOMENTUM INDICATORS
# ============================================================

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum indicators"""
    df = df.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_slope'] = df['RSI_14'] - df['RSI_14'].shift(3)
    df['RSI_5m'] = df['close'].diff().rolling(5).apply(lambda x: 100 - 100/(1 + (x[x>0].mean() / (-x[x<0].mean() + 1e-9) + 1e-9)))
    
    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd_line - signal_line
    
    # Momentum consistency
    df['momentum_consistency'] = (df['close'] > df['close'].shift(1)).rolling(5).mean()
    
    return df


# ============================================================
# FEATURE GROUP 4 — VOLATILITY CONTEXT
# ============================================================

def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility context"""
    df = df.copy()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()
    df['ATR_5m'] = true_range.rolling(5).mean()
    df['ATR_ratio'] = df['ATR_5m'] / (df['ATR_14'] + 1e-9)
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    df['Bollinger_width'] = (bb_upper - bb_lower) / df['close']
    df['BB_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # Volatility percentile
    df['volatility_percentile'] = df['ATR_14'].rolling(100).apply(lambda x: (x.iloc[-1] <= x).sum() / len(x), raw=False)
    
    # True range expansion
    df['true_range_expansion'] = true_range / (true_range.rolling(20).mean() + 1e-9)
    
    # Volatility regime
    df['volatility_regime'] = (df['ATR_14'] > df['ATR_14'].rolling(50).mean()).astype(int)
    
    return df


# ============================================================
# FEATURE GROUP 5 — VOLUME & PARTICIPATION
# ============================================================

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume and participation features"""
    df = df.copy()
    
    # Volume z-score
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)
    
    # Volume ratio
    df['volume_ratio_1m_vs_20m'] = df['volume'] / (vol_mean + 1e-9)
    
    # VWAP
    df['VWAP'] = (df['close'] * df['volume']).rolling(60).sum() / (df['volume'].rolling(60).sum() + 1e-9)
    df['VWAP_distance'] = (df['close'] - df['VWAP']) / df['close']
    df['VWAP_slope'] = (df['VWAP'] - df['VWAP'].shift(5)) / df['VWAP'].shift(5)
    
    # OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_delta'] = obv - obv.shift(5)
    
    # Volume spike
    df['volume_spike_flag'] = (df['volume'] > vol_mean + 2 * vol_std).astype(int)
    
    # Volume-price correlation
    df['vol_price_corr'] = df['volume'].rolling(20).corr(df['close'])
    
    return df


# ============================================================
# FEATURE GROUP 6 — SESSION & TIME CONTEXT
# ============================================================

def compute_session_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Session and time context"""
    df = df.copy()
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Cyclic encoding
    minute_of_day = df['hour'] * 60 + df['minute']
    df['minute_of_day_sin'] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    df['minute_of_day_cos'] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Session flags
    df['is_open_range'] = ((df['hour'] == 9) & (df['minute'] < 30)).astype(int)
    df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] < 14)).astype(int)
    df['is_close_window'] = ((df['hour'] == 15) & (df['minute'] >= 15)).astype(int)
    
    # Minutes since open/to close
    # Use timezone-naive calculation to avoid tz conflicts
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['minute_of_hour'] = df['timestamp'].dt.minute
    
    # Calculate minutes since 9:15 AM
    df['minutes_since_open'] = (df['hour_of_day'] - 9) * 60 + (df['minute_of_hour'] - 15)
    
    # Calculate minutes to 3:30 PM close
    df['minutes_to_close'] = (15 - df['hour_of_day']) * 60 + (30 - df['minute_of_hour'])
    
    return df


# ============================================================
# FEATURE GROUP 7 — HIGHER-TIMEFRAME BIAS
# ============================================================

def compute_htf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Higher-timeframe features"""
    df = df.copy()
    
    # 15-minute EMA
    df['EMA_15m'] = df['close'].ewm(span=15*15, adjust=False).mean()  # ~15 min worth of bars
    df['trend_15m'] = (df['EMA_15m'] > df['EMA_15m'].shift(15)).astype(int)
    df['price_above_15m_ema'] = (df['close'] > df['EMA_15m']).astype(int)
    
    # Daily metrics (approximated)
    daily_high = df['high'].rolling(390).max()  # ~1 day of 1-min bars
    daily_low = df['low'].rolling(390).min()
    prev_day_high = daily_high.shift(390)
    prev_day_low = daily_low.shift(390)
    
    df['dist_prev_day_high'] = (prev_day_high - df['close']) / df['close']
    df['dist_prev_day_low'] = (df['close'] - prev_day_low) / df['close']
    
    # Gap detection
    prev_day_close = df['close'].shift(390)
    df['gap_up_flag'] = ((df['open'] > prev_day_close * 1.002)).astype(int)
    df['gap_down_flag'] = ((df['open'] < prev_day_close * 0.998)).astype(int)
    
    # Inside/Outside day
    df['inside_day'] = ((df['high'] < daily_high.shift(390)) & (df['low'] > daily_low.shift(390))).astype(int)
    df['outside_day'] = ((df['high'] > daily_high.shift(390)) & (df['low'] < daily_low.shift(390))).astype(int)
    
    return df


# ============================================================
# MAIN FEATURE COMPUTATION
# ============================================================

def compute_all_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Compute all entry features for simulation
    
    Args:
        df: DataFrame with OHLCV columns
        symbol: Symbol name (optional, for symbol_id encoding)
    
    Returns:
        DataFrame with all features
    """
    print("Computing all features...")
    
    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Apply all feature groups
    df = compute_price_return_features(df)
    df = compute_trend_structure_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_session_time_features(df)
    df = compute_htf_features(df)
    
    # Symbol ID (one-hot encoded in simple form)
    if symbol:
        symbol_map = {'LT': 0, 'RELIANCE': 1, 'SIEMENS': 2, 'TATAELXSI': 3, 'TITAN': 4, 'TVSMOTOR': 5}
        df['symbol_id'] = symbol_map.get(symbol, 0)
    else:
        df['symbol_id'] = 0
    
    # Drop rows with NaN (from rolling calculations)
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    print(f"✅ Features computed")
    print(f"   Dropped {dropped} rows due to NaN (warmup)")
    print(f"   Remaining: {len(df):,} bars")
    
    return df
