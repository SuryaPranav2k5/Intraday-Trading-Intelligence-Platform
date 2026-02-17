"""
Phase-1 Inference Engine (Institutional-Grade)

Produces Trade Context Object for Phase-2 consumption.

Models loaded per symbol:
- setup_quality_lgbm.txt (entry probability)
- mfe_model_lgbm.txt (expected MFE)
- mae_model_lgbm.txt (expected MAE)
- directional_model_lgbm.txt (directional bias)
- risk_model_lgbm.txt (risk multiplier)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import time, datetime
import logging
import json
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InferenceEngine")

# Add SmartApi to path for symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'SmartApi'))
from symbol_utils import SYMBOL_MAPPING


class InferenceEngine:
    """
    Symbol-aware inference engine that produces Trade Context Objects.
    
    ENTRY: Symbol-specific models (LT/, RELIANCE/, etc.)
    EXIT: Shared engine across all symbols (Transformer + LightGBM)
    """
    
    def __init__(self):
        # === ENTRY MODELS (per-symbol) ===
        self.models = {}           # symbol -> entry model
        self.mfe_models = {}       # symbol -> MFE regression model
        self.mae_models = {}       # symbol -> MAE regression model
        self.dir_models = {}       # symbol -> directional model
        self.risk_models = {}      # symbol -> risk model
        self.thresholds = {}       # symbol -> default threshold
        self.regime_thresholds = {} # symbol -> regime-specific thresholds
        self.session_configs = {}  # symbol -> session config
        self.feature_cols = None
        self.symbols = list(SYMBOL_MAPPING.keys())
        
        # === EXIT ENGINE (shared) ===
        self.exit_engine = None
        self.exit_trade_states = {}  # symbol -> state buffer
        
        # Load all models
        self.load_models()
        self._load_exit_engine()

    def load_models(self):
        """Load all Phase-1 models for all configured symbols"""
        import yaml
        import xgboost as xgb
        
        for symbol in self.symbols:
            base_path = Path(__file__).parent / symbol / "model_artifacts"
            
            if not base_path.exists():
                logger.warning(f"[WARN] model_artifacts not found for {symbol}")
                continue
            
            # === LightGBM Models ===
            lgb_long_path = base_path / "lgb_long.txt"
            lgb_short_path = base_path / "lgb_short.txt"
            
            if lgb_long_path.exists() and lgb_short_path.exists():
                self.models[symbol] = {
                    "lgb_long": lgb.Booster(model_file=str(lgb_long_path)),
                    "lgb_short": lgb.Booster(model_file=str(lgb_short_path)),
                }
                logger.info(f"[OK] Loaded LGB models for {symbol}")
            else:
                logger.warning(f"[WARN] LGB models not found for {symbol}")
                continue
            
            # === XGBoost Models ===
            xgb_long_path = base_path / "xgb_long.json"
            xgb_short_path = base_path / "xgb_short.json"
            
            if xgb_long_path.exists() and xgb_short_path.exists():
                xgb_long = xgb.Booster()
                xgb_long.load_model(str(xgb_long_path))
                xgb_short = xgb.Booster()
                xgb_short.load_model(str(xgb_short_path))
                self.models[symbol]["xgb_long"] = xgb_long
                self.models[symbol]["xgb_short"] = xgb_short
                logger.info(f"[OK] Loaded XGB models for {symbol}")
            
            # === Feature List ===
            feature_path = base_path / "feature_list.json"
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    feature_list = json.load(f)
                    if self.feature_cols is None:
                        self.feature_cols = feature_list
                    self.models[symbol]["feature_cols"] = feature_list
            
            # === Thresholds ===
            threshold_path = base_path / "thresholds.json"
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    self.thresholds[symbol] = json.load(f)
                logger.info(f"[OK] Loaded thresholds for {symbol}")
            
            # === Entry Config ===
            config_path = base_path / "entry_config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.session_configs[symbol] = yaml.safe_load(f)
                logger.info(f"[OK] Loaded entry_config for {symbol}")
            
            # === Metrics (optional) ===
            metrics_path = base_path / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.models[symbol]["metrics"] = json.load(f)
    
    def _load_exit_engine(self):
        """Load shared Exit Engine (Transformer + LightGBM)"""
        try:
            from live_exit_engine import LiveExitEngine
            self.exit_engine = LiveExitEngine()
            logger.info("[OK] Exit Engine loaded (shared across all symbols)")
        except Exception as e:
            logger.warning(f"[WARN] Exit Engine not loaded: {e}")
            self.exit_engine = None
    
    def reset_exit_trade(self, symbol: str):
        """Reset exit state for a new trade"""
        if self.exit_engine:
            self.exit_trade_states[symbol] = []
            self.exit_engine.reset_trade()
    
    def get_exit_decision(self, symbol: str, state_features: dict, entry_features: dict) -> dict:
        """
        Get exit decision for an active trade.
        
        Args:
            symbol: Trading symbol
            state_features: Current trade state (unrealized_pnl_atr, mfe_atr, etc.)
            entry_features: Entry context (direction, initial_atr)
        
        Returns:
            {"action": "HOLD"|"EXIT", "exit_probability": float}
        """
        if not self.exit_engine:
            return {"action": "HOLD", "error": "Exit engine not loaded"}
        
        # Add symbol to entry features
        entry_features["symbol"] = symbol
        
        decision = self.exit_engine.update_and_decide(state_features, entry_features)
        
        return {
            "action": decision,
            "symbol": symbol,
        }

    def _compute_features(self, df_1m: pd.DataFrame, return_full_df: bool = False) -> pd.DataFrame:
        """
        Full feature engineering pipeline EXACTLY matching train_setup_quality.py.
        All 7 feature groups with identical names and calculations.
        """
        if df_1m.empty or len(df_1m) < 60:
            return pd.DataFrame()
        
        eps = 1e-9
        df = df_1m.copy()
        
        # ============================================================
        # FEATURE GROUP 1 — PRICE & RETURN STRUCTURE
        # ============================================================
        df['log_return_1m'] = np.log(df['close'] / (df['close'].shift(1) + eps))
        df['log_return_3m'] = np.log(df['close'] / (df['close'].shift(3) + eps))
        df['log_return_5m'] = np.log(df['close'] / (df['close'].shift(5) + eps))
        
        # Candle body and wicks (normalized by range)
        candle_range = df['high'] - df['low'] + eps
        df['candle_body_pct'] = np.abs(df['close'] - df['open']) / candle_range
        df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
        df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range
        df['range_pct'] = candle_range / (df['close'] + eps)
        
        # Multi-timeframe returns
        # Note: return_5m removed (redundant with log_return_5m)
        df['return_15m'] = (df['close'] - df['close'].shift(15)) / (df['close'].shift(15) + eps)
        df['return_30m'] = (df['close'] - df['close'].shift(30)) / (df['close'].shift(30) + eps)
        
        # Price momentum
        df['price_momentum_3'] = df['close'].diff(3)
        df['price_momentum_5'] = df['close'].diff(5)
        df['price_momentum_10'] = df['close'].diff(10)
        
        # ============================================================
        # FEATURE GROUP 2 — TREND & MARKET STRUCTURE
        # ============================================================
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
        
        # Structure (HH/HL/LH/LL)
        lookback = 10
        df['recent_high'] = df['high'].rolling(lookback).max()
        df['recent_low'] = df['low'].rolling(lookback).min()
        
        df['higher_high'] = (df['high'] > df['recent_high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['recent_low'].shift(1)).astype(int)
        
        df['dist_from_swing_high'] = (df['recent_high'] - df['close']) / (df['close'] + eps)
        df['dist_from_swing_low'] = (df['close'] - df['recent_low']) / (df['close'] + eps)
        
        # Trend strength score (EMA alignment)
        ema_aligned_bull = ((df['EMA_9'] > df['EMA_21']) & (df['EMA_21'] > df['EMA_50'])).astype(int)
        ema_aligned_bear = ((df['EMA_9'] < df['EMA_21']) & (df['EMA_21'] < df['EMA_50'])).astype(int)
        df['trend_strength_score'] = ema_aligned_bull - ema_aligned_bear
        
        # Price position in range
        range_20 = df['high'].rolling(20).max() - df['low'].rolling(20).min() + eps
        df['price_position_in_range'] = (df['close'] - df['low'].rolling(20).min()) / range_20
        
        # ============================================================
        # FEATURE GROUP 3 — MOMENTUM INDICATORS (COMPRESSED)
        # ============================================================
        # RSI 14 (1m)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + eps)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['RSI_slope'] = df['RSI_14'].diff(3)
        # Note: RSI_5m removed (redundant - just rolling mean of RSI_14)
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['Stoch_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + eps)
        # Note: Stoch_D removed (redundant - just rolling mean of Stoch_K)
        
        # MACD histogram (only)
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
        
        # ============================================================
        # FEATURE GROUP 4 — VOLATILITY CONTEXT (CRITICAL)
        # ============================================================
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
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        BB_upper = sma_20 + 2 * std_20
        BB_lower = sma_20 - 2 * std_20
        df['Bollinger_width'] = (BB_upper - BB_lower) / (sma_20 + eps)
        df['BB_position'] = (df['close'] - BB_lower) / (BB_upper - BB_lower + eps)
        
        # Volatility percentile (rolling ~30 days = ~6000 bars, min 100)
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
        
        # ============================================================
        # FEATURE GROUP 5 — VOLUME & PARTICIPATION
        # ============================================================
        vol_mean = df['volume'].rolling(30).mean()
        vol_std = df['volume'].rolling(30).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + eps)
        
        # Volume ratio
        df['volume_ratio_1m_vs_20m'] = df['volume'] / (df['volume'].rolling(20).mean() + eps)
        
        # VWAP
        df['date'] = df.index.date
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP per day - handle both MultiIndex and single Index
        try:
            vwap_series = df.groupby('date').apply(
                lambda g: (g['typical_price'] * g['volume']).cumsum() / (g['volume'].cumsum() + eps),
                include_groups=False
            )
            # Check if result has MultiIndex
            if isinstance(vwap_series.index, pd.MultiIndex):
                df['VWAP'] = vwap_series.droplevel(0).reindex(df.index).ffill()
            else:
                df['VWAP'] = vwap_series.reindex(df.index).ffill()
        except Exception as e:
            # Fallback: simple VWAP without groupby
            df['VWAP'] = (df['typical_price'] * df['volume']).cumsum() / (df['volume'].cumsum() + eps)
        
        # VWAP distance and slope
        df['VWAP_distance'] = (df['close'] - df['VWAP']) / (df['close'] + eps)
        df['VWAP_slope'] = df['VWAP'].diff(5)
        
        # OBV (On-Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['OBV_delta'] = obv.diff(5)
        
        # Volume spike flag
        df['volume_spike_flag'] = (df['volume'] > vol_mean + 2 * vol_std).astype(int)
        
        # Volume-price correlation
        df['vol_price_corr'] = df['volume'].rolling(20).corr(df['close'])
        
        # ============================================================
        # FEATURE GROUP 6 — SESSION & TIME CONTEXT
        # ============================================================
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
        
        df['minutes_since_open'] = minutes_of_day - open_min
        df['minutes_to_close'] = close_min - minutes_of_day
        
        # Symbol ID - will be set per-symbol later
        df['symbol_id'] = 0  # Default, overridden by predict()
        
        # ============================================================
        # FEATURE GROUP 7 — HIGHER-TIMEFRAME BIAS (EDGE)
        # ============================================================
        df['EMA_15m'] = df['close'].ewm(span=15, adjust=False).mean()
        df['trend_15m'] = np.sign(df['EMA_15m'].diff(5))
        
        df['price_above_15m_ema'] = (df['close'] > df['EMA_15m']).astype(int)
        
        # Previous day high/low
        daily_high = df.groupby('date')['high'].transform('max')
        daily_low = df.groupby('date')['low'].transform('min')
        
        prev_day_high = daily_high.shift(1).ffill()
        prev_day_low = daily_low.shift(1).ffill()
        
        df['dist_prev_day_high'] = (prev_day_high - df['close']) / (df['close'] + eps)
        df['dist_prev_day_low'] = (df['close'] - prev_day_low) / (df['close'] + eps)
        
        # Gap detection
        prev_close_day = df.groupby('date')['close'].transform('last').shift(1).ffill()
        first_open = df.groupby('date')['open'].transform('first')
        gap = (first_open - prev_close_day) / (prev_close_day + eps)
        
        df['gap_up_flag'] = (gap > 0.005).astype(int)  # 0.5% gap
        df['gap_down_flag'] = (gap < -0.005).astype(int)
        
        # Inside/Outside day
        prev_day_range = (prev_day_high - prev_day_low).fillna(0)
        current_range = df['high'] - df['low']
        
        df['inside_day'] = (current_range < prev_day_range * 0.8).astype(int)
        df['outside_day'] = (current_range > prev_day_range * 1.2).astype(int)
        
        # ============================================================
        # ADDITIONAL COMPUTED FIELDS FOR UI (not model features)
        # ============================================================
        # Session high/low for UI display
        df["session_high"] = df.groupby('date')['high'].transform(lambda x: x.expanding().max())
        df["session_low"] = df.groupby('date')['low'].transform(lambda x: x.expanding().min())
        
        # Should not trade flag
        atr_pctl = df["ATR_14"].rolling(120, min_periods=30).rank(pct=True)
        df["session_quality"] = np.where(
            (atr_pctl >= 0.25) & (atr_pctl <= 0.85), 0.7, 0.4
        )
        df["should_not_trade"] = (df["session_quality"] < 0.3).astype(int)
        
        # Clean up temp columns
        df = df.drop(columns=['date', 'typical_price', 'recent_high', 'recent_low'], errors='ignore')
        
        if return_full_df:
            return df
        
        # Return only model features
        if self.feature_cols:
            available_cols = [c for c in self.feature_cols if c in df.columns]
            return df[available_cols].tail(1)
        return df.tail(1)

    def predict(self, symbol: str, df_1m: pd.DataFrame) -> dict:
        """
        Generate Trade Context Object for Phase-2 consumption.
        """
        if symbol not in self.models:
            return {"error": f"Model for {symbol} not loaded"}
        
        if len(df_1m) < 20:
            return {"error": "Insufficient data", "status": "warming_up"}
        
        try:
            df_features = self._compute_features(df_1m, return_full_df=True)
            if df_features.empty:
                return {"error": "Feature engineering failed"}
            
            # Set symbol_id (matches training)
            SYMBOL_ID_MAP = {
                "TVSMOTOR": 5, "RELIANCE": 1, "LT": 2, 
                "TITAN": 3, "SIEMENS": 4, "TATAELXSI": 6
            }
            df_features['symbol_id'] = SYMBOL_ID_MAP.get(symbol, 0)
            
            latest = df_features.iloc[-1]
            
            # === TIME & DATA INTEGRITY CHECKS ===
            
            # 1. Market Hour Gating (Strict Match to Training)
            # Training drops < 10:15 and > 15:15. Inference MUST do the same.
            now_time = datetime.now().time()
            start_gate = time(10, 15)
            end_gate = time(15, 15)
            
            # Allow UI updates (indicators), but GATE the signal logic
            is_outside_model_hours = (now_time < start_gate) or (now_time > end_gate)
            
            # 2. Get model features (Safe Filtering)
            # Ensures we NEVER pass dropped columns (EMA_9, VWAP, etc.) to the model
            feature_cols = self.models[symbol].get("feature_cols", self.feature_cols)
            available_cols = [c for c in feature_cols if c in df_features.columns]
            
            if len(available_cols) < 10:
                logger.warning(f"[WARN] Only {len(available_cols)} features found for {symbol}. Expected ~71.")
            
            model_input = df_features[available_cols].tail(1)
            
            # 3. Robust Data Cleaning (Fix Clipping Mismatch)
            # - Fill NaNs with 0
            # - Replace Infinity with NaNs then 0
            # - Clip to prevents floats overflow or extreme outliers
            model_input = model_input.replace([np.inf, -np.inf], np.nan).fillna(0)
            model_input = model_input.clip(lower=-1e9, upper=1e9) # Sanity clip
            
            # Ensure float64
            model_input = model_input.astype(np.float64)
            
            # === RUN ALL MODELS ===
            import xgboost as xgb
            
            # Get thresholds for this symbol
            thresholds = self.thresholds.get(symbol, {})
            lgb_long_thresh = thresholds.get("lgb_long", 0.5)
            lgb_short_thresh = thresholds.get("lgb_short", 0.5)
            xgb_long_thresh = thresholds.get("xgb_long", 0.5)
            xgb_short_thresh = thresholds.get("xgb_short", 0.5)
            
            # Get Learned Ensemble Weights & Thresholds
            # (These are now trained per-direction, not generic config)
            w_long_lgb = thresholds.get("ensemble_long_weight_lgb", 0.5)
            w_short_lgb = thresholds.get("ensemble_short_weight_lgb", 0.5)
            thresh_long = thresholds.get("ensemble_long_thresh", 0.5)
            thresh_short = thresholds.get("ensemble_short_thresh", 0.5)
            
            # LightGBM predictions
            lgb_long_prob = float(self.models[symbol]["lgb_long"].predict(model_input)[0])
            lgb_short_prob = float(self.models[symbol]["lgb_short"].predict(model_input)[0])
            
            # XGBoost predictions (if available)
            xgb_long_prob = 0.0
            xgb_short_prob = 0.0
            if "xgb_long" in self.models[symbol]:
                dmatrix = xgb.DMatrix(model_input)
                xgb_long_prob = float(self.models[symbol]["xgb_long"].predict(dmatrix)[0])
                xgb_short_prob = float(self.models[symbol]["xgb_short"].predict(dmatrix)[0])
            
            # Calculate Learned Ensemble Probabilities
            # w * LGB + (1-w) * XGB
            long_prob = (w_long_lgb * lgb_long_prob) + ((1 - w_long_lgb) * xgb_long_prob)
            short_prob = (w_short_lgb * lgb_short_prob) + ((1 - w_short_lgb) * xgb_short_prob)
            
            # Determine direction based on strength relative to its own threshold
            # Normalize to "ratio of threshold" for fair comparison?
            # Or just take the one with higher raw probability if above threshold?
            
            long_signal = long_prob >= thresh_long
            short_signal = short_prob >= thresh_short
            
            if long_signal and not short_signal:
                directional_bias = 1
                entry_prob = long_prob
                threshold = thresh_long
            elif short_signal and not long_signal:
                directional_bias = -1
                entry_prob = short_prob
                threshold = thresh_short
            elif long_signal and short_signal:
                # Conflict: Pick the one with higher margin above threshold
                long_margin = long_prob - thresh_long
                short_margin = short_prob - thresh_short
                if long_margin > short_margin:
                    directional_bias = 1
                    entry_prob = long_prob
                    threshold = thresh_long
                else:
                    directional_bias = -1
                    entry_prob = short_prob
                    threshold = thresh_short
            else:
                # No signal: Return the "stronger" of the two weak signals for monitoring
                if long_prob > short_prob:
                    directional_bias = 1
                    entry_prob = long_prob
                    threshold = thresh_long
                else:
                    directional_bias = -1
                    entry_prob = short_prob
                    threshold = thresh_short
            
            # MFE/MAE defaults (no separate models trained)
            expected_mfe = 1.5
            expected_mae = 0.5
            
            # Risk multiplier (default, no separate model)
            risk_multiplier = 0.5
            
            # Get regime values (volatility_regime is from features, others computed locally)
            volatility_regime = int(latest.get('volatility_regime', 1))
            session_quality = float(latest.get('session_quality', 0.5))
            should_not_trade = bool(latest.get('should_not_trade', False))
            
            # === BUILD TRADE CONTEXT OBJECT ===
            trade_context = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "status": "active",
                
                # Phase-1 outputs
                "entry_prob": entry_prob,
                "expected_mfe_atr": expected_mfe,
                "expected_mae_atr": expected_mae,
                "expected_time_to_resolution": 20,  # Default, could be another model
                "risk_multiplier": risk_multiplier,
                "directional_bias": directional_bias,
                "session_quality": session_quality,
                "should_not_trade": should_not_trade,
                
                # Regime info
                "regime": {
                    "volatility": volatility_regime,
                },
                
                # Signal (backward compatible)
                # Strict: Must exceed threshold AND be within trading hours AND quality checked
                "threshold": threshold,
                "signal": bool(entry_prob >= threshold and not should_not_trade and not is_outside_model_hours),
                
                # Key indicators (for UI) - using correct column names from training
                "vwap": float(latest.get('VWAP', 0)),
                "ema20": float(latest.get('EMA_21', 0)),  # Using EMA_21 as closest to EMA20
                "atr": float(latest.get('ATR_14', 0)),
                
                # Level info (for Phase-2)
                "session_high": float(latest.get('session_high', 0)),
                "session_low": float(latest.get('session_low', 0)),
                "htf_high": float(latest.get('session_high', 0)),  # Use session high as HTF reference
                "htf_low": float(latest.get('session_low', 0)),    # Use session low as HTF reference
            }
            
            return trade_context
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}", exc_info=True)
            return {"error": str(e)}


# Singleton instance
engine = InferenceEngine()

def get_inference_engine():
    return engine
