"""
Complete Backtest System (OPTIMIZED)
=====================================
Pre-computes features ONCE, then runs fast bar-by-bar simulation.

Uses EXACT Phase-1 + Phase-2 setup as live trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path
import sys
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backtest")

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'SmartApi'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Web', 'backend'))

from inference import InferenceEngine
from phase2_engine import Phase2Engine, TradeContext, TradeState, Phase2Config, LevelState
from smartapi_client import get_smartapi_client
from symbol_utils import get_smartapi_token, SYMBOL_MAPPING


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BacktestConfig:
    """Backtest configuration"""
    lookback_days: int = 90
    warmup_bars: int = 300
    initial_capital: float = 100000.0
    position_size_pct: float = 0.10
    min_entry_prob: float = 0.5
    respect_should_not_trade: bool = True
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    no_new_trades_after: time = time(14, 30)
    output_dir: str = "backtest_results"


# ============================================================
# TRADE RECORD
# ============================================================

@dataclass
class TradeRecord:
    """Record of a completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int
    quantity: int
    entry_prob: float
    expected_mfe_atr: float
    expected_mae_atr: float
    risk_multiplier: float
    directional_bias: int
    session_quality: float
    volatility_regime: int
    trend_regime: int
    market_state: int
    pnl: float = 0.0
    pnl_atr: float = 0.0
    actual_mfe: float = 0.0
    actual_mae: float = 0.0
    actual_mfe_atr: float = 0.0
    actual_mae_atr: float = 0.0
    holding_minutes: int = 0
    exit_reason: str = ""


# ============================================================
# DATA FETCHER
# ============================================================

class SmartAPIDataFetcher:
    """Fetch historical data from SmartAPI"""
    
    def __init__(self):
        self.client = get_smartapi_client()
        # Ensure client is connected before fetching data
        if not self.client.connected:
            logger.info("Connecting to SmartAPI...")
            self.client.connect()
    
    def fetch_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch historical 1-minute candles in chunks"""
        logger.info(f"Fetching {days} days of data for {symbol}...")
        
        token = get_smartapi_token(symbol)
        if not token:
            return pd.DataFrame()
        
        all_candles = []
        end_date = datetime.now()
        chunk_days = 25
        remaining_days = days
        
        while remaining_days > 0:
            fetch_days = min(chunk_days, remaining_days)
            start_date = end_date - timedelta(days=fetch_days)
            
            from_str = start_date.strftime('%Y-%m-%d 09:15')
            to_str = end_date.strftime('%Y-%m-%d 15:30')
            
            try:
                candles = self.client.get_historical_candles(
                    exchange="NSE",
                    symbol_token=token,
                    interval="ONE_MINUTE",
                    from_date=from_str,
                    to_date=to_str
                )
                
                if candles:
                    all_candles.extend(candles)
                    logger.info(f"  Fetched {len(candles)} candles ({start_date.date()} to {end_date.date()})")
                
            except Exception as e:
                logger.error(f"  Error: {e}")
            
            end_date = start_date - timedelta(days=1)
            remaining_days -= fetch_days
            
            import time as t
            t.sleep(0.5)
        
        if not all_candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"  Total: {len(df)} candles")
        return df[['open', 'high', 'low', 'close', 'volume']]


# ============================================================
# FEATURE PRE-COMPUTATION (OPTIMIZED)
# ============================================================

class FeaturePrecomputer:
    """
    Pre-computes ALL features for the entire dataset ONCE.
    This is the KEY optimization - instead of computing features bar-by-bar.
    """
    
    def __init__(self, inference: InferenceEngine):
        self.inference = inference
        self.feature_cols = inference.feature_cols
    
    def compute_all_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 65 features for entire dataset at once.
        Uses the EXACT same logic as inference._compute_features().
        """
        if df_1m.empty:
            return pd.DataFrame()
        
        eps = 1e-9
        
        # 1. Resample to 5m
        df_5m = df_1m.resample("5min", label="right", closed="right").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        if df_5m.empty:
            return pd.DataFrame()
        
        # 2. 5-min indicators
        high, low, close = df_5m["high"], df_5m["low"], df_5m["close"]
        prev_close = close.shift(1)
        
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        df_5m["atr_5m"] = tr.ewm(alpha=1/5, adjust=False).mean()
        df_5m["atr_30m"] = tr.ewm(alpha=1/30, adjust=False).mean()
        df_5m["ema20_5m"] = close.ewm(span=20, adjust=False).mean()
        
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + eps)
        df_5m["rsi_5m"] = 100 - (100 / (1 + rs))
        
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_5m.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_5m.index)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / (df_5m["atr_5m"] + eps))
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / (df_5m["atr_5m"] + eps))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + eps)) * 100
        df_5m["adx_5m"] = dx.ewm(alpha=1/14, adjust=False).mean()
        df_5m["plus_di_5m"] = plus_di
        df_5m["minus_di_5m"] = minus_di
        
        # 3. Merge back to 1m
        indicator_cols = ['atr_5m', 'atr_30m', 'ema20_5m', 'rsi_5m', 'adx_5m', 'plus_di_5m', 'minus_di_5m']
        df = pd.merge_asof(
            df_1m.sort_index(),
            df_5m[indicator_cols].sort_index(),
            left_index=True, right_index=True, direction="backward"
        )
        
        # 4. All other features (same as inference.py)
        df['date'] = df.index.date
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        cum_pv = (df['typical_price'] * df['volume']).groupby(df['date']).cumsum()
        cum_vol = df.groupby('date')['volume'].cumsum()
        df['vwap'] = cum_pv / (cum_vol + eps)
        
        atr_percentile = df["atr_5m"].rolling(200, min_periods=50).rank(pct=True)
        df["volatility_regime"] = pd.cut(atr_percentile, bins=[0, 0.33, 0.67, 1.0], labels=[0, 1, 2], include_lowest=True).astype(float).fillna(1)
        df["trend_regime"] = pd.cut(df["adx_5m"], bins=[0, 20, 35, 100], labels=[0, 1, 2], include_lowest=True).astype(float).fillna(0)
        df["trend_direction"] = np.where(df["plus_di_5m"] > df["minus_di_5m"], 1, np.where(df["plus_di_5m"] < df["minus_di_5m"], -1, 0))
        atr_ratio = df["atr_5m"] / (df["atr_5m"].rolling(60).mean() + eps)
        df["market_state"] = pd.cut(atr_ratio, bins=[0, 0.7, 1.3, np.inf], labels=[0, 1, 2], include_lowest=True).astype(float).fillna(1)
        df["volatility_expanding"] = (df["atr_5m"] > df["atr_5m"].shift(5)).astype(int)
        
        window = 20
        df["swing_high_20"] = df["high"].rolling(window).max()
        df["swing_low_20"] = df["low"].rolling(window).min()
        df["dist_swing_high_atr"] = (df["swing_high_20"] - df["close"]) / (df["atr_5m"] + eps)
        df["dist_swing_low_atr"] = (df["close"] - df["swing_low_20"]) / (df["atr_5m"] + eps)
        df["htf_high_60"] = df["high"].rolling(60).max()
        df["htf_low_60"] = df["low"].rolling(60).min()
        df["dist_htf_high_atr"] = (df["htf_high_60"] - df["close"]) / (df["atr_5m"] + eps)
        df["dist_htf_low_atr"] = (df["close"] - df["htf_low_60"]) / (df["atr_5m"] + eps)
        df["vwap_slope"] = df["vwap"].diff(5) / (df["atr_5m"] + eps)
        df["vwap_slope_direction"] = np.sign(df["vwap_slope"])
        df["above_vwap"] = (df["close"] > df["vwap"]).astype(int)
        df["above_ema20"] = (df["close"] > df["ema20_5m"]).astype(int)
        
        df["momentum_1m"] = df["close"].diff(1)
        df["momentum_1m_sign"] = np.sign(df["momentum_1m"])
        df["momentum_5m"] = df["close"].diff(5)
        df["momentum_5m_sign"] = np.sign(df["momentum_5m"])
        df["tf_alignment"] = (df["momentum_1m_sign"] == df["momentum_5m_sign"]).astype(int)
        df["tf_alignment_strength"] = df["tf_alignment"].rolling(10).mean()
        df["trend_impulse_conflict"] = ((df["trend_direction"] * df["momentum_1m_sign"]) < 0).astype(int)
        df["ema_acceptance_score"] = np.where(df["above_ema20"] == 1, (df["close"] - df["ema20_5m"]) / (df["atr_5m"] + eps), -(df["ema20_5m"] - df["close"]) / (df["atr_5m"] + eps))
        
        # FIX: Use expanding window within each day (no lookahead)
        # cummax/cummin only uses data up to current bar, not future bars
        df["session_high"] = df.groupby('date')['high'].transform(lambda x: x.expanding().max())
        df["session_low"] = df.groupby('date')['low'].transform(lambda x: x.expanding().min())
        df["atr_ratio_5_30"] = df["atr_5m"] / (df["atr_30m"] + eps)
        df["atr_5m_slope"] = df["atr_5m"].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)==10 else 0, raw=False)
        
        tr_1m = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
        df["tr_percentile_120"] = tr_1m.rolling(120).rank(pct=True)
        df["volatility_compression"] = df["atr_5m"] / (df["atr_5m"].rolling(60).max() + eps)
        atr_1m = tr_1m.ewm(alpha=1/14, adjust=False).mean()
        df["atr_ratio_1_5"] = atr_1m / (df["atr_5m"] + eps)
        df["range_expansion_ratio"] = (df["high"]-df["low"]) / ((df["high"]-df["low"]).rolling(30).mean() + eps)
        
        df["vwap_dist_atr"] = (df["close"] - df["vwap"]).abs() / (df["atr_5m"] + eps)
        df["ema20_dist_atr"] = (df["close"] - df["ema20_5m"]).abs() / (df["atr_5m"] + eps)
        
        low_60 = df["low"].rolling(60).min()
        high_60 = df["high"].rolling(60).max()
        df["range_position"] = (df["close"] - low_60) / (high_60 - low_60 + eps)
        
        overlap = (df["high"].shift(1).clip(upper=df["high"]) - df["low"].shift(1).clip(lower=df["low"])).clip(lower=0)
        df["candle_overlap_ratio"] = overlap.rolling(20).mean() / ((df["high"] - df["low"]).rolling(20).mean() + eps)
        df["dist_session_high_atr"] = (df["session_high"] - df["close"]).abs() / (df["atr_5m"] + eps)
        df["dist_session_low_atr"] = (df["close"] - df["session_low"]).abs() / (df["atr_5m"] + eps)
        
        df["rsi_strength"] = (df["rsi_5m"] - 50).abs()
        df["momentum_burst"] = (df["close"] - df["close"].shift(1)).abs() / (atr_1m + eps)
        returns = df["close"].pct_change()
        df["momentum_consistency"] = returns.rolling(10).apply(lambda x: (np.sign(x) == np.sign(x.mean())).mean() if len(x)==10 else 0, raw=False)
        df["body_ratio"] = ((df["close"] - df["open"]).abs() / ((df["high"] - df["low"]) + eps)).rolling(20).mean()
        
        df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + eps)
        df["volume_range_product"] = df["volume"] * (df["high"] - df["low"])
        vol_mean = df["volume"].rolling(20).mean()
        vol_std = df["volume"].rolling(20).std()
        df["volume_spike"] = (df["volume"] > (vol_mean + 2 * vol_std)).astype(int)
        df["volume_zscore"] = (df["volume"] - df["volume"].rolling(30).mean()) / (df["volume"].rolling(30).std() + eps)
        df["price_impact"] = (df["close"] - df["close"].shift(1)).abs() / (df["volume"] + eps)
        
        minutes = df.index.hour * 60 + df.index.minute
        df["minutes_since_open"] = minutes - (9*60 + 15)
        df["minutes_to_close"] = (15*60 + 30) - minutes
        df["lunch_flag"] = ((minutes >= 12*60) & (minutes <= 13*60 + 30)).astype(int)
        df["opening_range_flag"] = (df["minutes_since_open"] <= 30).astype(int)
        df["power_hour_flag"] = (df["minutes_to_close"] <= 60).astype(int)
        
        efficiency = (df["close"] - df["close"].shift(1)).abs() / (df["volume"] + eps)
        df["efficiency_score"] = efficiency
        df["efficiency_mean"] = efficiency.rolling(30).mean()
        df["efficiency_vol"] = efficiency.rolling(30).std()
        
        atr_pctl = df["atr_5m"].rolling(120, min_periods=30).rank(pct=True)
        df["session_quality"] = np.where((atr_pctl >= 0.25) & (atr_pctl <= 0.85), 0.7, 0.4)
        df["should_not_trade"] = (df["session_quality"] < 0.3).astype(int)
        
        return df


# ============================================================
# OPTIMIZED BACKTEST ENGINE
# ============================================================

class BacktestEngine:
    """
    Optimized backtest engine.
    Pre-computes features once, then runs fast bar-by-bar simulation.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        
        logger.info("Loading Phase-1 models...")
        self.inference = InferenceEngine()
        
        logger.info("Loading Phase-2 engine...")
        self.phase2 = Phase2Engine(Phase2Config())
        
        self.data: Dict[str, pd.DataFrame] = {}
        self.features: Dict[str, pd.DataFrame] = {}  # Pre-computed features
        
        self.active_trades: Dict[str, TradeState] = {}
        self.completed_trades: List[TradeRecord] = []
        
        self.capital = self.config.initial_capital
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        self.symbols = list(SYMBOL_MAPPING.keys())
        
        self.precomputer = FeaturePrecomputer(self.inference)
    
    def fetch_data(self):
        """Fetch all historical data from SmartAPI"""
        fetcher = SmartAPIDataFetcher()
        
        for symbol in self.symbols:
            df = fetcher.fetch_historical_data(symbol, self.config.lookback_days)
            if not df.empty:
                self.data[symbol] = df
                logger.info(f"Loaded {symbol}: {len(df)} bars")
    
    def precompute_features(self):
        """Pre-compute ALL features for ALL symbols at once"""
        logger.info("Pre-computing features for all symbols...")
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
            
            logger.info(f"  Computing features for {symbol}...")
            df = self.data[symbol]
            features_df = self.precomputer.compute_all_features(df)
            
            if not features_df.empty:
                self.features[symbol] = features_df
                logger.info(f"  {symbol}: {len(features_df)} rows with features")
        
        logger.info("Feature pre-computation complete!")
    
    def _run_models(self, symbol: str, row: pd.Series) -> dict:
        """Run all Phase-1 models on pre-computed features (FAST)"""
        if self.inference.feature_cols is None:
            return {"error": "No feature columns"}
        
        # Get features for this bar
        available_cols = [c for c in self.inference.feature_cols if c in row.index]
        if len(available_cols) < 50:
            return {"error": "Insufficient features"}
        
        features = row[available_cols].values.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0)
        
        try:
            # Entry probability
            entry_prob = float(self.inference.models[symbol].predict(features)[0])
            
            # MFE
            expected_mfe = 1.5
            if symbol in self.inference.mfe_models:
                expected_mfe = float(self.inference.mfe_models[symbol].predict(features)[0])
            
            # MAE
            expected_mae = 0.5
            if symbol in self.inference.mae_models:
                expected_mae = float(self.inference.mae_models[symbol].predict(features)[0])
            
            # Directional bias
            directional_bias = 0
            if symbol in self.inference.dir_models:
                dir_probs = self.inference.dir_models[symbol].predict(features)[0]
                directional_bias = int(np.argmax(dir_probs) - 1)
            
            # Risk multiplier
            risk_multiplier = 0.5
            if symbol in self.inference.risk_models:
                risk_multiplier = float(self.inference.risk_models[symbol].predict(features)[0])
                risk_multiplier = max(0, min(1, risk_multiplier))
            
            threshold = self.inference.thresholds.get(symbol, 0.5)
            
            return {
                "entry_prob": entry_prob,
                "probability": entry_prob,
                "expected_mfe_atr": expected_mfe,
                "expected_mae_atr": expected_mae,
                "directional_bias": directional_bias,
                "risk_multiplier": risk_multiplier,
                "threshold": threshold,
                "signal": entry_prob >= threshold,
                "vwap": row.get('vwap', 0),
                "atr": row.get('atr_5m', 0),
                "session_quality": row.get('session_quality', 0.5),
                "should_not_trade": bool(row.get('should_not_trade', False)),
                "session_high": row.get('session_high', 0),
                "session_low": row.get('session_low', 0),
                "htf_high": row.get('htf_high_60', 0),
                "htf_low": row.get('htf_low_60', 0),
                "regime": {
                    "volatility": int(row.get('volatility_regime', 1)),
                    "trend": int(row.get('trend_regime', 0)),
                    "market_state": int(row.get('market_state', 1))
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _create_trade_context(self, result: dict) -> TradeContext:
        regime = result.get('regime', {})
        return TradeContext(
            entry_prob=result.get('entry_prob', 0.5),
            expected_mfe_atr=result.get('expected_mfe_atr', 1.5),
            expected_mae_atr=result.get('expected_mae_atr', 0.5),
            expected_time_to_resolution=20,
            risk_multiplier=result.get('risk_multiplier', 0.5),
            directional_bias=result.get('directional_bias', 0),
            session_quality=result.get('session_quality', 0.5),
            volatility_regime=regime.get('volatility', 1),
            trend_regime=regime.get('trend', 0),
            market_state=regime.get('market_state', 1),
        )
    
    def _enter_trade(self, symbol: str, bar_time: datetime, row: pd.Series, result: dict, context: TradeContext):
        entry_price = row['close']
        atr = result.get('atr', entry_price * 0.01)
        # User trades 1 lot only
        quantity = 1
        
        direction = 1
        if context.directional_bias == -1:
            direction = -1
        
        trade = self.phase2.create_trade(
            symbol=symbol, entry_price=entry_price, entry_time=bar_time,
            entry_atr=atr, context=context, direction=direction
        )
        trade.quantity = quantity
        self.active_trades[symbol] = trade
        
        logger.debug(f"ENTRY: {symbol} @ {entry_price:.2f}")
    
    def _exit_trade(self, symbol: str, bar_time: datetime, row: pd.Series, trade: TradeState, reason: str):
        exit_price = row['close']
        pnl = (exit_price - trade.entry_price) * trade.direction * trade.quantity
        pnl_atr = (exit_price - trade.entry_price) * trade.direction / trade.entry_atr
        
        record = TradeRecord(
            symbol=symbol, entry_time=trade.entry_time, exit_time=bar_time,
            entry_price=trade.entry_price, exit_price=exit_price,
            direction=trade.direction, quantity=trade.quantity,
            entry_prob=trade.context.entry_prob,
            expected_mfe_atr=trade.context.expected_mfe_atr,
            expected_mae_atr=trade.context.expected_mae_atr,
            risk_multiplier=trade.context.risk_multiplier,
            directional_bias=trade.context.directional_bias,
            session_quality=trade.context.session_quality,
            volatility_regime=trade.context.volatility_regime,
            trend_regime=trade.context.trend_regime,
            market_state=trade.context.market_state,
            pnl=pnl, pnl_atr=pnl_atr,
            actual_mfe=trade.mfe, actual_mae=trade.mae,
            actual_mfe_atr=trade.mfe_atr, actual_mae_atr=trade.mae_atr,
            holding_minutes=trade.minutes_in_trade, exit_reason=reason
        )
        
        self.completed_trades.append(record)
        self.capital += pnl
        self.phase2.record_completed_trade(pnl_atr)
        del self.active_trades[symbol]
        
        logger.debug(f"EXIT: {symbol} @ {exit_price:.2f} | PnL={pnl:.2f}")
    
    def _evaluate_trade(self, symbol: str, bar_time: datetime, row: pd.Series, result: dict):
        if symbol not in self.active_trades:
            return
        
        trade = self.active_trades[symbol]
        regime = result.get('regime', {})
        
        self.phase2.update_levels(
            symbol=symbol,
            session_high=result.get('session_high', row['high']),
            session_low=result.get('session_low', row['low']),
            vwap=result.get('vwap', row['close']),
            htf_high=result.get('htf_high', row['high']),
            htf_low=result.get('htf_low', row['low'])
        )
        
        candle = {'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'], 'volume': row['volume']}
        
        updated = self.phase2.evaluate(
            trade=trade, candle=candle, vwap=result.get('vwap', row['close']),
            now=bar_time, current_market_state=regime.get('market_state', 1)
        )
        
        if updated.trade_status == "EXIT_RECOMMENDED":
            self._exit_trade(symbol, bar_time, row, updated, updated.exit_reason)
        else:
            self.active_trades[symbol] = updated
    
    def run(self):
        """Main backtest loop"""
        logger.info("="*60)
        logger.info("STARTING OPTIMIZED BACKTEST")
        logger.info("="*60)
        
        if not self.data:
            self.fetch_data()
        
        if not self.data:
            logger.error("No data. Aborting.")
            return
        
        # KEY OPTIMIZATION: Pre-compute all features ONCE
        if not self.features:
            self.precompute_features()
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in self.features.values():
            all_timestamps.update(df.index.tolist())
        
        all_timestamps = sorted(all_timestamps)
        
        # Skip warmup period
        all_timestamps = all_timestamps[self.config.warmup_bars:]
        
        logger.info(f"Processing {len(all_timestamps)} bars...")
        
        current_date = None
        total_bars = len(all_timestamps)
        progress_interval = max(1, total_bars // 20)
        
        for i, bar_time in enumerate(all_timestamps):
            if i % progress_interval == 0:
                pct = (i / total_bars) * 100
                logger.info(f"Progress: {pct:.0f}% ({i}/{total_bars} bars) | Trades: {len(self.completed_trades)}")
            
            # Reset session on new day
            if current_date != bar_time.date():
                current_date = bar_time.date()
                self.phase2.reset_session()
            
            # Market hours check
            if not (self.config.market_open <= bar_time.time() <= self.config.market_close):
                continue
            
            self.equity_curve.append((bar_time, self.capital))
            
            # Process each symbol
            for symbol in self.symbols:
                if symbol not in self.features:
                    continue
                
                features_df = self.features[symbol]
                if bar_time not in features_df.index:
                    continue
                
                row = features_df.loc[bar_time]
                
                # Run models (FAST - just prediction, no feature computation)
                result = self._run_models(symbol, row)
                if 'error' in result:
                    continue
                
                # Evaluate active trade
                if symbol in self.active_trades:
                    self._evaluate_trade(symbol, bar_time, row, result)
                
                # Check for entry
                elif bar_time.time() < self.config.no_new_trades_after:
                    if result.get('signal') and not result.get('should_not_trade'):
                        if result.get('entry_prob', 0) >= self.config.min_entry_prob:
                            context = self._create_trade_context(result)
                            should_trade, _ = self.phase2.should_take_trade(context)
                            if should_trade:
                                self._enter_trade(symbol, bar_time, row, result, context)
        
        # Close remaining
        logger.info("Closing remaining positions...")
        for symbol in list(self.active_trades.keys()):
            df = self.features[symbol]
            last_row = df.iloc[-1]
            trade = self.active_trades[symbol]
            self._exit_trade(symbol, df.index[-1], last_row, trade, "Backtest End")
        
        logger.info("Backtest complete!")
        self._generate_report()
    
    def _generate_report(self):
        """Generate report"""
        if not self.completed_trades:
            logger.warning("No trades completed!")
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        trades_df = pd.DataFrame([vars(t) for t in self.completed_trades])
        
        total = len(trades_df)
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winners) / total if total > 0 else 0
        profit_factor = abs(winners['pnl'].sum() / losers['pnl'].sum()) if losers['pnl'].sum() != 0 else np.inf
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades:     {total}")
        print(f"Win Rate:         {win_rate:.1%}")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"Total PnL:        Rs {trades_df['pnl'].sum():,.2f}")
        print(f"Avg PnL/Trade:    Rs {trades_df['pnl'].mean():,.2f}")
        print(f"Avg PnL (ATR):    {trades_df['pnl_atr'].mean():.3f}")
        
        if len(self.equity_curve) > 0:
            equity = pd.Series([e[1] for e in self.equity_curve])
            max_dd = ((equity - equity.expanding().max()) / equity.expanding().max()).min() * 100
            print(f"Max Drawdown:     {max_dd:.1f}%")
        
        print(f"\nExit Reasons:")
        print(trades_df.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean']).round(2))
        
        trades_df.to_csv(output_dir / "trades.csv", index=False)
        pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity']).to_csv(output_dir / "equity_curve.csv", index=False)
        
        print(f"\nSaved to {output_dir}")
        print("="*60)


def main():
    config = BacktestConfig(
        lookback_days=90,
        initial_capital=100000,
        position_size_pct=0.10,
    )
    
    engine = BacktestEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
