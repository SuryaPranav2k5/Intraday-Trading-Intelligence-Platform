# ============================================================
# Entry Model Training - LightGBM + XGBoost
# Symbol: TVSMOTOR
# Author: Surya
# Purpose: Pure Entry Signal Generation (NO EXIT LOGIC)
# ============================================================

"""
GOLDEN RULE FOR ENTRY:
Entry sees a snapshot of the market, NOT a sequence.
- No memory
- No trade-relative info
- No exit-style features
- No future-looking features

Entry doesn't need to be perfect. It needs to be CONSISTENT.
Exit intelligence is where money is made.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from datetime import time
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
import json
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL CONFIG
# ============================================================

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
RANDOM_SEED = 42

# ATR-based labeling thresholds
PROFIT_TARGET_ATR = 0.7  # If +0.7 ATR before -0.4 ATR → LONG
STOP_LOSS_ATR = 0.4

# ============================================================
# 1. DATA LOADING & PREPARATION
# ============================================================

def load_raw_data(parquet_path: Path) -> pd.DataFrame:
    """Load 1-minute OHLCV data from parquet."""
    df = pd.read_parquet(parquet_path)
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    assert list(df.columns) == expected_cols, f"Unexpected columns: {df.columns}"
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only NSE regular session candles."""
    df = df.between_time(MARKET_OPEN, MARKET_CLOSE)
    return df


# ============================================================
# 2. FEATURE GROUP 1 — PRICE & RETURN STRUCTURE
# ============================================================

def compute_price_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core price action features - what price is doing RIGHT NOW.
    1-minute and multi-timeframe returns.
    """
    df = df.copy()
    eps = 1e-9
    
    # --- 1-MINUTE TIMEFRAME ---
    df['log_return_1m'] = np.log(df['close'] / (df['close'].shift(1) + eps))
    df['log_return_3m'] = np.log(df['close'] / (df['close'].shift(3) + eps))
    df['log_return_5m'] = np.log(df['close'] / (df['close'].shift(5) + eps))
    
    # Candle body and wicks (normalized by range)
    candle_range = df['high'] - df['low'] + eps
    df['candle_body_pct'] = np.abs(df['close'] - df['open']) / candle_range
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range
    df['range_pct'] = candle_range / (df['close'] + eps)
    
    # --- MULTI-TIMEFRAME RETURNS (CRITICAL) ---
    # Note: return_5m removed (redundant with log_return_5m)
    df['return_15m'] = (df['close'] - df['close'].shift(15)) / (df['close'].shift(15) + eps)
    df['return_30m'] = (df['close'] - df['close'].shift(30)) / (df['close'].shift(30) + eps)
    
    # Price momentum
    df['price_momentum_3'] = df['close'].diff(3)
    df['price_momentum_5'] = df['close'].diff(5)
    df['price_momentum_10'] = df['close'].diff(10)
    
    return df


# ============================================================
# 3. FEATURE GROUP 2 — TREND & MARKET STRUCTURE
# ============================================================

def compute_trend_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend features - is there a directional edge?
    Trees LOVE relative distances.
    """
    df = df.copy()
    eps = 1e-9
    
    # --- MOVING AVERAGES (1m timeframe) ---
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # EMA slopes
    df['EMA_slope_9'] = df['EMA_9'].diff(5) / 5
    df['EMA_slope_21'] = df['EMA_21'].diff(5) / 5
    
    # EMA distances (relative to close)
    df['EMA_9_vs_21_distance'] = (df['EMA_9'] - df['EMA_21']) / (df['close'] + eps)
    df['EMA_21_vs_50_distance'] = (df['EMA_21'] - df['EMA_50']) / (df['close'] + eps)
    df['price_vs_EMA9'] = (df['close'] - df['EMA_9']) / (df['close'] + eps)
    df['price_vs_EMA21'] = (df['close'] - df['EMA_21']) / (df['close'] + eps)
    df['price_vs_EMA50'] = (df['close'] - df['EMA_50']) / (df['close'] + eps)
    
    # --- STRUCTURE (HH/HL/LH/LL) ---
    lookback = 10
    df['recent_high'] = df['high'].rolling(lookback).max()
    df['recent_low'] = df['low'].rolling(lookback).min()
    
    # Higher high / Lower low flags
    df['higher_high'] = (df['high'] > df['recent_high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['recent_low'].shift(1)).astype(int)
    
    # Distance from swing points
    df['dist_from_swing_high'] = (df['recent_high'] - df['close']) / (df['close'] + eps)
    df['dist_from_swing_low'] = (df['close'] - df['recent_low']) / (df['close'] + eps)
    
    # Trend strength score (EMA alignment)
    ema_aligned_bull = ((df['EMA_9'] > df['EMA_21']) & (df['EMA_21'] > df['EMA_50'])).astype(int)
    ema_aligned_bear = ((df['EMA_9'] < df['EMA_21']) & (df['EMA_21'] < df['EMA_50'])).astype(int)
    df['trend_strength_score'] = ema_aligned_bull - ema_aligned_bear
    
    # Price position in range
    range_20 = df['high'].rolling(20).max() - df['low'].rolling(20).min() + eps
    df['price_position_in_range'] = (df['close'] - df['low'].rolling(20).min()) / range_20
    
    return df


# ============================================================
# 4. FEATURE GROUP 3 — MOMENTUM INDICATORS (COMPRESSED)
# ============================================================

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum indicators - distilled forms only.
    Avoid redundancy. Trees punish correlated noise.
    """
    df = df.copy()
    eps = 1e-9
    
    # --- RSI 14 (1m) ---
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + eps)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_slope'] = df['RSI_14'].diff(3)
    # Note: RSI_5m removed (redundant - just rolling mean of RSI_14)
    
    # --- STOCHASTIC ---
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + eps)
    # Note: Stoch_D removed (redundant - just rolling mean of Stoch_K)
    
    # --- MACD HISTOGRAM (only) ---
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd_line - signal_line
    
    # Momentum consistency
    returns = df['close'].pct_change()
    df['momentum_consistency'] = returns.rolling(10).apply(
        lambda x: (np.sign(x) == np.sign(x.mean())).mean() if len(x) > 0 else 0.5,
        raw=False
    )
    
    return df


# ============================================================
# 5. FEATURE GROUP 4 — VOLATILITY CONTEXT (CRITICAL)
# ============================================================

def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility context - entry without this is BLIND.
    This decides position sizing implicitly.
    """
    df = df.copy()
    eps = 1e-9
    
    # --- COMPUTE ATR (True Range basis) ---
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    
    df['ATR_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['ATR_5m'] = tr.ewm(alpha=1/5, adjust=False).mean()
    
    # ATR ratio (short vs longer)
    df['ATR_ratio'] = df['ATR_14'] / (df['ATR_5m'].rolling(5).mean() + eps)
    
    # --- BOLLINGER BANDS ---
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['BB_upper'] = sma_20 + 2 * std_20
    df['BB_lower'] = sma_20 - 2 * std_20
    df['Bollinger_width'] = (df['BB_upper'] - df['BB_lower']) / (sma_20 + eps)
    df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + eps)
    
    # --- VOLATILITY PERCENTILE (rolling 20-30 days = ~6000-9000 bars) ---
    df['volatility_percentile'] = df['ATR_14'].rolling(6000, min_periods=100).rank(pct=True)
    
    # True range expansion flag
    df['true_range_expansion'] = (tr > tr.rolling(20).mean() * 1.5).astype(int)
    
    # Volatility regime
    vol_pct = df['ATR_14'].rolling(200, min_periods=50).rank(pct=True)
    df['volatility_regime'] = pd.cut(
        vol_pct,
        bins=[0, 0.33, 0.67, 1.0],
        labels=[0, 1, 2],  # LOW, NORMAL, HIGH
        include_lowest=True
    ).astype(float).fillna(1)
    
    return df


# ============================================================
# 6. FEATURE GROUP 5 — VOLUME & PARTICIPATION
# ============================================================

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume and participation features.
    Trees are VERY good at volume logic.
    """
    df = df.copy()
    eps = 1e-9
    
    # Volume z-score
    vol_mean = df['volume'].rolling(30).mean()
    vol_std = df['volume'].rolling(30).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + eps)
    
    # Volume ratio
    df['volume_ratio_1m_vs_20m'] = df['volume'] / (df['volume'].rolling(20).mean() + eps)
    
    # --- VWAP ---
    df['date'] = df.index.date
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    vwap_series = df.groupby('date').apply(
        lambda g: (g['typical_price'] * g['volume']).cumsum() / (g['volume'].cumsum() + eps),
        include_groups=False
    )
    df['VWAP'] = vwap_series.droplevel(0).reindex(df.index).ffill()
    
    # VWAP distance and slope
    df['VWAP_distance'] = (df['close'] - df['VWAP']) / (df['close'] + eps)
    df['VWAP_slope'] = df['VWAP'].diff(5)
    
    # --- OBV (On-Balance Volume) ---
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_delta'] = obv.diff(5)
    
    # Volume spike flag
    df['volume_spike_flag'] = (df['volume'] > vol_mean + 2 * vol_std).astype(int)
    
    # Volume-price correlation
    df['vol_price_corr'] = df['volume'].rolling(20).corr(df['close'])
    
    df = df.drop(columns=['date', 'typical_price'])
    
    return df


# ============================================================
# 7. FEATURE GROUP 6 — SESSION & TIME CONTEXT
# ============================================================

def compute_session_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session and time context - often ignored, BIG MISTAKE.
    Entry probability is time-dependent.
    """
    df = df.copy()
    
    # Time encoding (sin/cos for cyclical)
    minutes_of_day = df.index.hour * 60 + df.index.minute
    minutes_normalized = minutes_of_day / (24 * 60)
    
    df['minute_of_day_sin'] = np.sin(2 * np.pi * minutes_normalized)
    df['minute_of_day_cos'] = np.cos(2 * np.pi * minutes_normalized)
    
    # Day of week encoding
    day_of_week = df.index.dayofweek / 7
    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week)
    
    # Session flags
    open_min = 9 * 60 + 15
    close_min = 15 * 60 + 30
    
    df['is_open_range'] = (minutes_of_day <= open_min + 15).astype(int)
    df['is_lunch_time'] = ((minutes_of_day >= 12 * 60) & (minutes_of_day <= 13 * 60 + 30)).astype(int)
    df['is_close_window'] = (minutes_of_day >= close_min - 30).astype(int)
    
    # Minutes since open / to close
    df['minutes_since_open'] = minutes_of_day - open_min
    df['minutes_to_close'] = close_min - minutes_of_day
    
    # Symbol ID (for multi-symbol future compatibility)
    # TVSMOTOR = 5
    df['symbol_id'] = 5
    
    return df


# ============================================================
# 8. FEATURE GROUP 7 — HIGHER-TIMEFRAME BIAS (EDGE)
# ============================================================

def compute_htf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Higher-timeframe features - this is where win rate jumps.
    15m trend direction, gaps, inside/outside days.
    """
    df = df.copy()
    eps = 1e-9
    
    # --- 15-MINUTE TREND ---
    df['EMA_15m'] = df['close'].ewm(span=15, adjust=False).mean()  # Approx 15m
    df['trend_15m'] = np.sign(df['EMA_15m'].diff(5))
    
    # 15m EMA alignment
    df['price_above_15m_ema'] = (df['close'] > df['EMA_15m']).astype(int)
    
    # --- PREVIOUS DAY HIGH/LOW ---
    df['date'] = df.index.date
    daily_high = df.groupby('date')['high'].transform('max')
    daily_low = df.groupby('date')['low'].transform('min')
    
    prev_day_high = daily_high.shift(1).ffill()
    prev_day_low = daily_low.shift(1).ffill()
    
    df['dist_prev_day_high'] = (prev_day_high - df['close']) / (df['close'] + eps)
    df['dist_prev_day_low'] = (df['close'] - prev_day_low) / (df['close'] + eps)
    
    # Gap detection
    prev_close = df.groupby('date')['close'].transform('last').shift(1).ffill()
    first_open = df.groupby('date')['open'].transform('first')
    gap = (first_open - prev_close) / (prev_close + eps)
    
    df['gap_up_flag'] = (gap > 0.005).astype(int)  # 0.5% gap
    df['gap_down_flag'] = (gap < -0.005).astype(int)
    
    # Inside/Outside day
    prev_day_range = (prev_day_high - prev_day_low).fillna(0)
    current_range = df['high'] - df['low']
    
    df['inside_day'] = (current_range < prev_day_range * 0.8).astype(int)
    df['outside_day'] = (current_range > prev_day_range * 1.2).astype(int)
    
    df = df.drop(columns=['date'])
    
    return df


# ============================================================
# 9. LABEL GENERATION (ATR-BASED)
# ============================================================

def generate_atr_labels(df: pd.DataFrame, lookahead: int = 45) -> pd.DataFrame:
    """
    ATR-based labeling for entries.
    
    Labels:
        LONG = 1:  If +0.7 ATR reached before -0.4 ATR
        SHORT = 1: If -0.7 ATR reached before +0.4 ATR
        NO_TRADE = 0: Otherwise
    
    This matches how exits will behave later.
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atr = df['ATR_14'].values
    
    n = len(df)
    labels_long = np.zeros(n, dtype=np.int8)
    labels_short = np.zeros(n, dtype=np.int8)
    
    for i in range(n - lookahead):
        entry = closes[i]
        atr_i = atr[i]
        
        if np.isnan(atr_i) or atr_i <= 0:
            continue
        
        target_long = entry + PROFIT_TARGET_ATR * atr_i
        stop_long = entry - STOP_LOSS_ATR * atr_i
        
        target_short = entry - PROFIT_TARGET_ATR * atr_i
        stop_short = entry + STOP_LOSS_ATR * atr_i
        
        # Check lookahead window
        for j in range(i + 1, min(i + lookahead + 1, n)):
            # LONG check
            if highs[j] >= target_long:
                labels_long[i] = 1
                break
            if lows[j] <= stop_long:
                break  # Stop hit first
        
        # SHORT check
        for j in range(i + 1, min(i + lookahead + 1, n)):
            if lows[j] <= target_short:
                labels_short[i] = 1
                break
            if highs[j] >= stop_short:
                break  # Stop hit first
    
    labels_df = pd.DataFrame({
        'label_long': labels_long,
        'label_short': labels_short,
    }, index=df.index)
    
    return labels_df


# ============================================================
# 10. FINAL CLEANING & PREPARATION
# ============================================================

def clean_training_data(df: pd.DataFrame, labels_df: pd.DataFrame):
    """
    Final cleaning before training.
    Remove NaNs, clip extremes, ensure no lookahead.
    """
    df = df.copy()
    labels_df = labels_df.copy()
    
    # Drop OHLCV (not features)
    cols_to_drop = ['open', 'high', 'low', 'close', 'volume']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Drop intermediate calculation columns
    intermediate_cols = ['recent_high', 'recent_low', 'BB_upper', 'BB_lower', 
                         'EMA_9', 'EMA_21', 'EMA_50', 'VWAP', 'EMA_15m']
    for col in intermediate_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Convert all to float64
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                df = df.drop(columns=[col])
        else:
            df[col] = df[col].astype(np.float64)
    
    # Drop NaNs/Infs
    valid_mask = (
        df.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & labels_df.notna().all(axis=1)
    )
    
    df = df.loc[valid_mask]
    labels_df = labels_df.loc[valid_mask]
    
    # Using full market hours dataset (no time-window filtering)
    
    # Clip extremes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(
        lower=df[numeric_cols].quantile(0.001),
        upper=df[numeric_cols].quantile(0.999),
        axis=1
    )
    
    df, labels_df = df.align(labels_df, join="inner", axis=0)
    df = df.astype(np.float64)
    
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"LONG signals: {labels_df['label_long'].sum():,} ({labels_df['label_long'].mean()*100:.2f}%)")
    print(f"SHORT signals: {labels_df['label_short'].sum():,} ({labels_df['label_short'].mean()*100:.2f}%)")
    print(f"Feature list (first 20):")
    for i, col in enumerate(df.columns[:20], 1):
        print(f"  {i}. {col}")
    if len(df.columns) > 20:
        print(f"  ... and {len(df.columns) - 20} more features")
    
    return df, labels_df


# ============================================================
# 11. LIGHTGBM TRAINING (PRODUCTION SAFE)
# ============================================================

def train_lightgbm(X: pd.DataFrame, y: pd.Series, name: str):
    """
    Train LightGBM with Optuna Hyperparameter Optimization.
    Uses Walk-Forward Validation (TimeSeriesSplit) to avoid overfitting.
    """
    # TimeSeriesSplit
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"\n{'='*60}")
    print(f"Training LightGBM - {name} (Walk-Forward Validation)")
    print(f"{'='*60}")
    
    # Class imbalance handling (estimate from full dataset for stability)
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / max(pos, 1)

    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": RANDOM_SEED,
            "scale_pos_weight": scale_pos_weight,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "n_jobs": -1
        }

        fold_aucs = []
        # Walk-forward validation loop
        for train_index, val_index in tscv.split(X):
            X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
            y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
            
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            model = lgb.train(
                param,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            preds = model.predict(X_val)
            if y_val.nunique() > 1:
                auc = roc_auc_score(y_val, preds)
                fold_aucs.append(auc)
            else:
                fold_aucs.append(0.5)
                
        return np.mean(fold_aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print(f"\nBest params for {name}:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Retrain with best params using the LAST split (Max Data)
    splits = list(tscv.split(X))
    train_index, val_index = splits[-1]
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    print(f"Final Retraining Split - Train samples: {len(X_train):,} | Val samples: {len(X_val):,}")

    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": RANDOM_SEED,
        "scale_pos_weight": scale_pos_weight,
    })
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=500)
        ]
    )
    
    # Evaluate
    val_preds = model.predict(X_val)
    auc = roc_auc_score(y_val, val_preds)
    
    # Find optimal threshold
    best_threshold = 0.5
    best_score = 0
    for threshold in np.linspace(0.3, 0.7, 41):
        preds_binary = (val_preds >= threshold).astype(int)
        if preds_binary.sum() < 10:
            continue
        prec = precision_score(y_val, preds_binary, zero_division=0)
        trade_frac = preds_binary.mean()
        
        if prec >= 0.55 and trade_frac >= 0.05:
            score = prec * trade_frac
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    print(f"\nValidation AUC (Last Fold): {auc:.4f}")
    print(f"Optimal threshold: {best_threshold:.3f}")
    
    return model, best_threshold, auc, val_preds, val_index


# ============================================================
# 12. XGBOOST TRAINING (COMPLEMENTARY)
# ============================================================

def train_xgboost(X: pd.DataFrame, y: pd.Series, name: str):
    """
    Train XGBoost with Optuna Hyperparameter Optimization.
    Uses Walk-Forward Validation (TimeSeriesSplit).
    """
    # TimeSeriesSplit
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"\n{'='*60}")
    print(f"Training XGBoost - {name} (Walk-Forward Validation)")
    print(f"{'='*60}")
    
    # Class imbalance
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / max(pos, 1)
    
    def objective(trial):
        param = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": RANDOM_SEED,
            "verbosity": 0,
            "scale_pos_weight": scale_pos_weight,
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        }
        
        fold_aucs = []
        for train_index, val_index in tscv.split(X):
            X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
            y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                param,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            val_preds = model.predict(dval)
            if y_val.nunique() > 1:
                auc = roc_auc_score(y_val, val_preds)
                fold_aucs.append(auc)
            else:
                fold_aucs.append(0.5)
                
        return np.mean(fold_aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    print(f"\nBest params for {name}:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Retrain with best params using LAST split
    splits = list(tscv.split(X))
    train_index, val_index = splits[-1]
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    print(f"Final Retraining Split - Train samples: {len(X_train):,} | Val samples: {len(X_val):,}")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "scale_pos_weight": scale_pos_weight,
    })
    
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=500
    )
    
    # Evaluate
    val_preds = model.predict(dval)
    auc = roc_auc_score(y_val, val_preds)
    
    # Find optimal threshold
    best_threshold = 0.5
    best_score = 0
    for threshold in np.linspace(0.3, 0.7, 41):
        preds_binary = (val_preds >= threshold).astype(int)
        if preds_binary.sum() < 10:
            continue
        prec = precision_score(y_val, preds_binary, zero_division=0)
        trade_frac = preds_binary.mean()
        
        if prec >= 0.55 and trade_frac >= 0.05:
            score = prec * trade_frac
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    print(f"\nValidation AUC (Last Fold): {auc:.4f}")
    print(f"Optimal threshold: {best_threshold:.3f}")
    
    return model, best_threshold, auc, val_preds, val_index


# ============================================================
# 13. ENSEMBLE OPTIMIZATION
# ============================================================

def optimize_ensemble(y_val, preds_lgb, preds_xgb, name):
    """
    Find optimal weights for LGB + XGB ensemble.
    Returns: best_weight_lgb, best_threshold, best_auc
    Ensemble = w * LGB + (1-w) * XGB
    """
    print(f"\n{'-'*60}")
    print(f"Optimizing Ensemble Weights - {name}")
    print(f"{'-'*60}")
    
    best_auc = 0
    best_w = 0.5
    
    # 1. Optimize AUC based on Weight
    for w in np.linspace(0, 1, 21):
        combined = w * preds_lgb + (1 - w) * preds_xgb
        try:
            auc = roc_auc_score(y_val, combined)
        except:
            auc = 0
        
        if auc > best_auc:
            best_auc = auc
            best_w = w
            
    print(f"  Best Weight (LGB): {best_w:.2f} (AUC: {best_auc:.4f})")
    
    # 2. Optimize Threshold for this Weight
    final_preds = best_w * preds_lgb + (1 - best_w) * preds_xgb
    best_threshold = 0.5
    best_score = 0
    
    for threshold in np.linspace(0.3, 0.7, 41):
        preds_binary = (final_preds >= threshold).astype(int)
        if preds_binary.sum() < 10:
            continue
        prec = precision_score(y_val, preds_binary, zero_division=0)
        trade_frac = preds_binary.mean()
        
        if prec >= 0.55 and trade_frac >= 0.05:
            score = prec * trade_frac
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
    print(f"  Best Threshold: {best_threshold:.3f}")
    return best_w, best_threshold, best_auc


# ============================================================
# 14. MAIN TRAINING PIPELINE
# ============================================================

def main():
    """
    Main training pipeline.
    1. Load data
    2. Generate all 7 feature groups
    3. Generate ATR-based labels
    4. Train LightGBM + XGBoost for LONG and SHORT
    5. Save models and thresholds
    """
    
    data_path = Path("../Dataset/TVSMOTOR_2years_1min.parquet")
    
    print(f"\n{'='*60}")
    print("ENTRY MODEL TRAINING - LightGBM + XGBoost")
    print(f"{'='*60}")
    print(f"Data source: {data_path}")
    
    # ========== LOAD DATA ==========
    print("\n[1/9] Loading data...")
    df = load_raw_data(data_path)
    df = filter_market_hours(df)
    print(f"  Loaded {len(df):,} bars")
    
    # ========== FEATURE ENGINEERING ==========
    print("\n[2/9] Generating Feature Group 1 - Price & Return Structure...")
    df = compute_price_return_features(df)
    
    print("[3/9] Generating Feature Group 2 - Trend & Market Structure...")
    df = compute_trend_structure_features(df)
    
    print("[4/9] Generating Feature Group 3 - Momentum Indicators...")
    df = compute_momentum_features(df)
    
    print("[5/9] Generating Feature Group 4 - Volatility Context...")
    df = compute_volatility_features(df)
    
    print("[6/9] Generating Feature Group 5 - Volume & Participation...")
    df = compute_volume_features(df)
    
    print("[7/9] Generating Feature Group 6 - Session & Time Context...")
    df = compute_session_time_features(df)
    
    print("[8/9] Generating Feature Group 7 - Higher-Timeframe Bias...")
    df = compute_htf_features(df)
    
    # ========== LABEL GENERATION ==========
    print("\n[9/9] Generating ATR-based labels...")
    labels_df = generate_atr_labels(df)
    
    # ========== CLEANING ==========
    print("\nCleaning and preparing training data...")
    X, labels_df = clean_training_data(df, labels_df)
    
    # ========== TRAINING ==========
    y_long = labels_df['label_long']
    y_short = labels_df['label_short']
    
    # NOTE: At inference time, enforce mutual exclusivity:
    #   - Don't allow both LONG & SHORT signals on same bar
    #   - Policy: Take stronger probability OR skip if conflict
    #   - This is handled at inference/policy layer, not here
    
    # Train 4 models: LightGBM LONG/SHORT, XGBoost LONG/SHORT
    lgb_long_model, lgb_long_thresh, lgb_long_auc, lgb_long_preds, lgb_val_idx = train_lightgbm(X, y_long, "LONG")
    lgb_short_model, lgb_short_thresh, lgb_short_auc, lgb_short_preds, _ = train_lightgbm(X, y_short, "SHORT")
    
    xgb_long_model, xgb_long_thresh, xgb_long_auc, xgb_long_preds, xgb_val_idx = train_xgboost(X, y_long, "LONG")
    xgb_short_model, xgb_short_thresh, xgb_short_auc, xgb_short_preds, _ = train_xgboost(X, y_short, "SHORT")
    
    # ========== ENSEMBLE OPTIMIZATION ==========
    # Validate alignment
    if not np.array_equal(lgb_val_idx, xgb_val_idx):
        raise ValueError("Critical Mismatch: LightGBM and XGBoost validation splits are not aligned!")

    y_long_val = y_long.iloc[lgb_val_idx]
    y_short_val = y_short.iloc[lgb_val_idx]
    
    w_long, thresh_long, auc_long = optimize_ensemble(y_long_val, lgb_long_preds, xgb_long_preds, "LONG")
    w_short, thresh_short, auc_short = optimize_ensemble(y_short_val, lgb_short_preds, xgb_short_preds, "SHORT")

    # ========== SAVE MODELS ==========
    model_dir = Path("model_artifacts")
    model_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print(f"{'='*60}")
    
    lgb_long_model.save_model(str(model_dir / "lgb_long.txt"))
    lgb_short_model.save_model(str(model_dir / "lgb_short.txt"))
    xgb_long_model.save_model(str(model_dir / "xgb_long.json"))
    xgb_short_model.save_model(str(model_dir / "xgb_short.json"))
    
    # Save thresholds and weights
    thresholds = {
        "lgb_long": float(lgb_long_thresh),
        "lgb_short": float(lgb_short_thresh),
        "xgb_long": float(xgb_long_thresh),
        "xgb_short": float(xgb_short_thresh),
        "ensemble_long_weight_lgb": float(w_long),
        "ensemble_long_thresh": float(thresh_long),
        "ensemble_short_weight_lgb": float(w_short),
        "ensemble_short_thresh": float(thresh_short)
    }
    
    with open(model_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    
    # Save metrics
    metrics = {
        "lgb_long_auc": float(lgb_long_auc),
        "lgb_short_auc": float(lgb_short_auc),
        "xgb_long_auc": float(xgb_long_auc),
        "xgb_short_auc": float(xgb_short_auc),
        "ensemble_long_auc": float(auc_long),
        "ensemble_short_auc": float(auc_short),
        "total_features": len(X.columns),
        "feature_names": list(X.columns)
    }
    
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] Saved LightGBM LONG model: {model_dir / 'lgb_long.txt'}")
    print(f"[OK] Saved LightGBM SHORT model: {model_dir / 'lgb_short.txt'}")
    print(f"[OK] Saved XGBoost LONG model: {model_dir / 'xgb_long.json'}")
    print(f"[OK] Saved XGBoost SHORT model: {model_dir / 'xgb_short.json'}")
    print(f"[OK] Saved thresholds: {model_dir / 'thresholds.json'}")
    print(f"[OK] Saved metrics: {model_dir / 'metrics.json'}")
    
    # ========== SUMMARY ==========
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*60}")
    print(f"\nEnsemble Performance:")
    print(f"  LONG  - AUC: {auc_long:.4f}, Weight (LGB): {w_long:.2f}, Threshold: {thresh_long:.3f}")
    print(f"  SHORT - AUC: {auc_short:.4f}, Weight (LGB): {w_short:.2f}, Threshold: {thresh_short:.3f}")
    print(f"\nLightGBM Performance:")
    print(f"  LONG  - AUC: {lgb_long_auc:.4f}, Threshold: {lgb_long_thresh:.3f}")
    print(f"  SHORT - AUC: {lgb_short_auc:.4f}, Threshold: {lgb_short_thresh:.3f}")
    print(f"\nXGBoost Performance:")
    print(f"  LONG  - AUC: {xgb_long_auc:.4f}, Threshold: {xgb_long_thresh:.3f}")
    print(f"  SHORT - AUC: {xgb_short_auc:.4f}, Threshold: {xgb_short_thresh:.3f}")
    
    print(f"\nTotal Features: {len(X.columns)}")
    print(f"Target Range: 150-300 (LightGBM), 100-250 (XGBoost)")
    print(f"Status: {'[OK] GOOD' if 150 <= len(X.columns) <= 300 else '[WARNING] REVIEW'}")
    
    print(f"\n{'='*60}")
    print("HOW TO USE THESE MODELS")
    print(f"{'='*60}")
    print("""
1. Entry sees a SNAPSHOT of the market (NO sequence, NO memory)
2. Use LightGBM for smooth decision surfaces
3. Use XGBoost for sharp thresholds and edge cases
4. Ensemble: Take signal if BOTH models agree OR if one has high confidence
5. Entry doesn't need to be perfect - it needs to be CONSISTENT
6. Exit intelligence is where money is made (Phase 2)

Example inference:
    lgb_prob_long = lgb_long_model.predict(features)
    xgb_prob_long = xgb_long_model.predict(features)
    
    signal = (lgb_prob_long > lgb_long_thresh) & (xgb_prob_long > xgb_long_thresh)
""")
    
    print("\n[SUCCESS] DONE. Models ready for deployment.\n")


if __name__ == "__main__":
    main()