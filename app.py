# CRITICAL: Eventlet monkey patching MUST be at the very top before any other imports
# This ensures Flask-SocketIO can emit from background threads
import eventlet
eventlet.monkey_patch()

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import threading
import time
import logging
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduced from INFO to WARNING
logger = logging.getLogger("SmartApiServer")

# Add SmartApi to path so it can find its local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'SmartApi')))
# Add Web/backend to path for Phase2Engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Web', 'backend')))

from realtime_data_hub import get_realtime_data_hub
from symbol_utils import get_verified_smartapi_watchlist, get_smartapi_token
from inference import get_inference_engine
from smartapi_client import get_smartapi_client

# Simple TradeState data class (no external dependencies)
class TradeState:
    """Holds all state for an active supervised trade."""
    def __init__(self, **kwargs):
        # Set defaults
        self.symbol = ""
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_atr = 1.0
        self.direction = 1
        self.trade_status = "IN_TRADE"
        self.exit_reason = None
        self.mfe = 0.0
        self.mae = 0.0
        self.mfe_atr = 0.0
        self.mae_atr = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_atr = 0.0
        self.minutes_in_trade = 0
        self.step = 0
        self.profit_floor = None
        self.context = None
        # Override with provided values
        for k, v in kwargs.items():
            setattr(self, k, v)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartapi_secret_key'
CORS(app)

# Initialize SocketIO with eventlet for high-performance WebSockets
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=False, engineio_logger=False)

# Initialize Core Services
hub = get_realtime_data_hub()
engine = get_inference_engine()  # Includes Exit Engine (Transformer + LightGBM)

# ============================================================
# INFERENCE DATA STORE
# Stores the latest inference results per symbol
# Updated every 60 seconds from 1-minute REST candles
# ============================================================
inference_results = {}  # {symbol: {vwap, ema20, probability, threshold, signal, status, timestamp}}
inference_lock = threading.RLock()

# Configuration for inference
INFERENCE_INTERVAL_SECONDS = 60  # Run inference every 60 seconds
MIN_CANDLES_REQUIRED = 150  # Minimum candles needed for indicator warm-up
TARGET_CANDLES = 300  # Target number of candles to fetch at startup
SAFETY_OFFSET_SECONDS = 5  # Wait N seconds after minute for candle to finalize
STAGGER_DELAY_SECONDS = 0.3  # Delay between symbol fetches to avoid rate limits (reduced from 1s)

# ============================================================
# CANDLE CACHE
# Stores historical 1-min candles per symbol (DataFrame)
# Initialized at startup with ~300 candles, then incrementally updated
# ============================================================
candle_cache = {}  # {symbol: pd.DataFrame with OHLCV}
candle_cache_lock = threading.RLock()
cache_initialized = False  # Flag to track if initial fetch is complete

# ============================================================
# EXIT ENGINE: TRADE SUPERVISION STATE
# Stores active supervised trades per symbol
# Exit decisions made by Transformer + LightGBM
# ============================================================
active_supervised_trades: Dict[str, TradeState] = {}  # {symbol: TradeState}
phase2_lock = threading.RLock()  # Keep lock name for compatibility
higher_low_tracker = {}  # {symbol: float} Tracks the swing low for trend following exits


def fetch_full_history(symbol: str, client) -> pd.DataFrame:
    """
    Fetch ~300 historical 1-minute candles for initial cache population.
    Called ONCE at startup per symbol, NOT every minute!
    
    Returns a DataFrame with timestamp index and OHLCV columns.
    """
    try:
        token = get_smartapi_token(symbol)
        if not token:
            logger.warning(f"No token found for {symbol}")
            return pd.DataFrame()
        
        end_date = datetime.now()
        
        # Calculate previous trading day (skip weekends)
        def get_previous_trading_day(date):
            previous_day = date - timedelta(days=1)
            if date.weekday() == 0:  # Monday -> Friday
                previous_day = date - timedelta(days=3)
            elif date.weekday() == 6:  # Sunday -> Friday
                previous_day = date - timedelta(days=2)
            return previous_day
        
        # Get previous trading day for sufficient history
        previous_trading_day = get_previous_trading_day(end_date)
        start_date = previous_trading_day.replace(hour=9, minute=15, second=0, microsecond=0)
        
        from_str = start_date.strftime('%Y-%m-%d 09:15')
        to_str = end_date.strftime('%Y-%m-%d %H:%M') if 9 <= end_date.hour < 16 else end_date.strftime('%Y-%m-%d 15:30')
        
        # Fetch historical candles
        candles = client.get_historical_candles(
            exchange="NSE",
            symbol_token=token,
            interval="ONE_MINUTE",
            from_date=from_str,
            to_date=to_str
        )
        
        if not candles:
            logger.warning(f"No candles returned for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Ensure correct columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing columns in {symbol} data: {df.columns.tolist()}")
            return pd.DataFrame()
        
        return df[required_cols]
        
    except Exception as e:
        logger.error(f"Error fetching full history for {symbol}: {e}")
        return pd.DataFrame()


def fetch_latest_candle(symbol: str, client) -> pd.DataFrame:
    """
    Fetch ONLY the latest 1-minute candle for incremental update.
    Called every minute AFTER cache is initialized.
    
    This is ~300x more efficient than fetching full history each time!
    """
    try:
        token = get_smartapi_token(symbol)
        if not token:
            return pd.DataFrame()
        
        now = datetime.now()
        
        # Fetch just the last 5 minutes to ensure we get the latest closed candle
        start_time = now - timedelta(minutes=5)
        from_str = start_time.strftime('%Y-%m-%d %H:%M')
        to_str = now.strftime('%Y-%m-%d %H:%M')
        
        candles = client.get_historical_candles(
            exchange="NSE",
            symbol_token=token,
            interval="ONE_MINUTE",
            from_date=from_str,
            to_date=to_str
        )
        
        if not candles:
            return pd.DataFrame()
        
        # Convert to DataFrame and get only the latest candle
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        
        # Return only the last (most recent) candle
        return df[required_cols].tail(1)
        
    except Exception as e:
        logger.error(f"Error fetching latest candle for {symbol}: {e}")
        return pd.DataFrame()


def get_cached_candles(symbol: str) -> pd.DataFrame:
    """
    Get candles from cache for a symbol.
    Returns empty DataFrame if not cached.
    """
    with candle_cache_lock:
        return candle_cache.get(symbol, pd.DataFrame()).copy()


def update_candle_cache(symbol: str, new_candle_df: pd.DataFrame):
    """
    Append a new candle to the cache and drop the oldest to maintain buffer size.
    
    This is the KEY optimization:
    - Append new candle to cached DataFrame
    - Drop oldest row to maintain ~300 candle buffer
    - Inference uses this cached data (no refetch!)
    """
    if new_candle_df.empty:
        return
    
    with candle_cache_lock:
        if symbol not in candle_cache or candle_cache[symbol].empty:
            # No cache yet - this shouldn't happen after initialization
            candle_cache[symbol] = new_candle_df
            return
        
        existing_df = candle_cache[symbol]
        
        # Check if we already have this candle (avoid duplicates)
        new_ts = new_candle_df.index[0]
        if new_ts in existing_df.index:
            # Already have this candle, skip
            return
        
        # Append new candle
        combined = pd.concat([existing_df, new_candle_df])
        
        # Drop oldest rows to maintain target buffer size
        if len(combined) > TARGET_CANDLES:
            combined = combined.tail(TARGET_CANDLES)
        
        candle_cache[symbol] = combined


def run_inference_for_symbol(symbol: str, df_1m: pd.DataFrame) -> dict:
    """
    Run model inference for a symbol using 1-minute candles.
    Returns dict with: vwap, ema20, probability, threshold, signal, status
    
    CRITICAL: This uses ONLY closed 1-minute candles, never ticks!
    """
    try:
        if df_1m.empty or len(df_1m) < MIN_CANDLES_REQUIRED:
            return {
                "vwap": 0,
                "ema20": 0,
                "probability": 0,
                "threshold": 0.65,
                "signal": False,
                "status": "insufficient_data",
                "candle_count": len(df_1m)
            }
        
        # Run model prediction (this computes all features including VWAP, EMA-20)
        result = engine.predict(symbol, df_1m)
        
        if "error" in result:
            return {
                "vwap": 0,
                "ema20": 0,
                "probability": 0,
                "threshold": 0.65,
                "signal": False,
                "status": f"error: {result['error']}",
                "candle_count": len(df_1m)
            }
        
        # (Context is available from inference_results if needed)
        
        return {
            "vwap": result.get("vwap", 0),
            "ema20": result.get("ema20", 0),
            "atr": result.get("atr", 0),
            "probability": result.get("entry_prob", result.get("probability", 0)),
            "threshold": result.get("threshold", 0.65),
            "signal": result.get("signal", False),
            "status": result.get("status", "active"),
            "candle_count": len(df_1m),
            # New Phase-1 outputs for Phase-2
            "expected_mfe_atr": result.get("expected_mfe_atr", 1.5),
            "expected_mae_atr": result.get("expected_mae_atr", 0.5),
            "risk_multiplier": result.get("risk_multiplier", 0.5),
            "directional_bias": result.get("directional_bias", 0),
            "session_quality": result.get("session_quality", 0.5),
            "should_not_trade": result.get("should_not_trade", False),
            "regime": result.get("regime", {"volatility": 1, "trend": 0, "market_state": 1}),
        }
        
    except Exception as e:
        logger.error(f"Inference error for {symbol}: {e}", exc_info=True)
        return {
            "vwap": 0,
            "ema20": 0,
            "probability": 0,
            "threshold": 0.65,
            "signal": False,
            "status": f"exception: {str(e)}",
            "candle_count": 0
        }


def inference_background_worker():
    """
    Background thread that runs model inference on each candle close.
    
    ============================================================
    THREE KEY OPTIMIZATIONS (as per user guidance):
    ============================================================
    
    RULE 1: Fetch once at startup, then incrementally update
    - At startup: Fetch ~300 historical candles per symbol
    - Every minute: Fetch only 1 new candle, append to cache, drop oldest
    - Result: ~90% reduction in API calls!
    
    RULE 2: Safety offset after minute
    - Run inference at :05 seconds after the minute (not exactly :00)
    - This ensures SmartAPI has finalized the candle (1-2s lag)
    - Zero trading impact for intraday setups
    
    RULE 3: Stagger symbol requests
    - Instead of hitting all symbols at same second
    - Add 1 second delay between each symbol
    - Prevents rate limit bursts
    
    CRITICAL: Model inference uses ONLY closed 1-minute candles, NEVER ticks!
    """
    global cache_initialized
    
    logger.warning("🚀 Inference background worker started")
    
    # Wait for SmartAPI connection to be established
    time.sleep(10)
    
    symbols = get_verified_smartapi_watchlist()
    client = get_smartapi_client()
    
    # ============================================================
    # RULE 1: STARTUP - Fetch full history for all symbols ONCE
    # ============================================================
    logger.warning(f"📦 Initializing candle cache for {len(symbols)} symbols (~{TARGET_CANDLES} candles each)...")
    init_start = time.time()
    
    for i, symbol in enumerate(symbols):
        try:
            logger.warning(f"  [{i+1}/{len(symbols)}] Fetching history for {symbol}...")
            df_full = fetch_full_history(symbol, client)
            
            if not df_full.empty:
                with candle_cache_lock:
                    candle_cache[symbol] = df_full
                logger.warning(f"  ✅ {symbol}: Cached {len(df_full)} candles (oldest: {df_full.index[0]}, newest: {df_full.index[-1]})")
            else:
                logger.warning(f"  ⚠️ {symbol}: No historical data received")
            
            # RULE 3: Stagger initial fetches to avoid rate limits
            time.sleep(STAGGER_DELAY_SECONDS)
            
        except Exception as e:
            logger.error(f"Error fetching initial history for {symbol}: {e}")
            time.sleep(STAGGER_DELAY_SECONDS)
    
    cache_initialized = True
    init_time = time.time() - init_start
    logger.warning(f"✅ Candle cache initialized in {init_time:.1f}s")
    
    # Run initial inference with cached data
    logger.warning("📊 Running initial inference cycle...")
    for symbol in symbols:
        try:
            df_1m = get_cached_candles(symbol)
            if not df_1m.empty:
                result = run_inference_for_symbol(symbol, df_1m)
                result["timestamp"] = datetime.now().isoformat()
                result["last_candle_time"] = df_1m.index[-1].isoformat()
                with inference_lock:
                    inference_results[symbol] = result
        except Exception as e:
            logger.error(f"Initial inference error for {symbol}: {e}")
    
    logger.warning("✅ Initial inference complete - entering main loop")
    
    # ============================================================
    # MAIN LOOP: Incremental updates + inference every minute
    # Track last processed minute to prevent duplicates
    # ============================================================
    
    # Track the last processed minute as (hour, minute) tuple
    # This ensures we NEVER process the same minute twice
    # Example: (9, 20) means we processed the 9:20 candle
    last_processed_minute = None
    
    while True:
        try:
            now = datetime.now()
            current_minute_tuple = (now.hour, now.minute)
            
            # ============================================================
            # DUPLICATE CHECK: Skip if already processed this minute
            # ============================================================
            if current_minute_tuple == last_processed_minute:
                time.sleep(1)
                continue
            
            # ============================================================
            # SAFETY OFFSET: Wait until 5+ seconds past the minute
            # This ensures SmartAPI has finalized the candle
            # ============================================================
            if now.second < SAFETY_OFFSET_SECONDS:
                wait_time = SAFETY_OFFSET_SECONDS - now.second
                time.sleep(wait_time)
                now = datetime.now()
            
            # ============================================================
            # MARKET HOURS CHECK (9:15 AM - 3:30 PM IST)
            # ============================================================
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if not (market_open <= now <= market_close):
                # Outside market hours - mark as processed and wait
                last_processed_minute = current_minute_tuple
                
                # If within 5 minutes of market open, check more frequently
                time_to_open = (market_open - now).total_seconds()
                if 0 < time_to_open <= 300:  # Within 5 minutes of open
                    mins_to_open = int(time_to_open // 60)
                    secs_to_open = int(time_to_open % 60)
                    logger.warning(f"⏳ Market opens in {mins_to_open}m {secs_to_open}s - standing by...")
                    time.sleep(5)  # Check every 5 seconds
                else:
                    time.sleep(30)  # Normal wait
                continue
            
            # ============================================================
            # PROCESS THIS MINUTE'S CANDLE
            # ============================================================
            candle_time_str = f"{now.hour:02d}:{now.minute:02d}"
            logger.warning(f"📊 Processing candle {candle_time_str} for {len(symbols)} symbols...")
            cycle_start = time.time()
            
            for i, symbol in enumerate(symbols):
                try:
                    # Fetch ONLY the latest candle (incremental update)
                    new_candle = fetch_latest_candle(symbol, client)
                    if not new_candle.empty:
                        update_candle_cache(symbol, new_candle)
                    
                    # Get cached candles for inference
                    df_1m = get_cached_candles(symbol)
                    
                    if df_1m.empty:
                        continue
                    
                    # Run inference on cached data
                    result = run_inference_for_symbol(symbol, df_1m)
                    result["timestamp"] = datetime.now().isoformat()
                    result["last_candle_time"] = df_1m.index[-1].isoformat()
                    
                    # Store results
                    with inference_lock:
                        inference_results[symbol] = result
                    
                    # ============================================================
                    # PHASE 2: Evaluate supervised trade (if active)
                    # ============================================================
                    with phase2_lock:
                        if symbol in active_supervised_trades:
                            # Get VWAP from inference result
                            vwap = result.get('vwap', 0)
                            
                            # Get the last closed candle for evaluation
                            last_candle = {
                                'open': df_1m.iloc[-1]['open'],
                                'high': df_1m.iloc[-1]['high'],
                                'low': df_1m.iloc[-1]['low'],
                                'close': df_1m.iloc[-1]['close'],
                                'volume': df_1m.iloc[-1]['volume']
                            }
                            
                            # Update higher low tracker for Trend trades
                            df_5min = get_5min_candles_from_1min(df_1m)
                            if not df_5min.empty:
                                update_higher_low(symbol, df_5min)
                            
                            # Evaluate the trade
                            evaluate_supervised_trade(symbol, last_candle, vwap)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                
                # Stagger requests between symbols
                time.sleep(STAGGER_DELAY_SECONDS)
            
            cycle_time = time.time() - cycle_start
            logger.warning(f"✅ Candle {candle_time_str} processed in {cycle_time:.1f}s")
            
            # ============================================================
            # MARK AS PROCESSED - Prevents duplicate processing
            # ============================================================
            last_processed_minute = current_minute_tuple
            
            # ============================================================
            # WAIT FOR NEXT MINUTE
            # ============================================================
            now = datetime.now()
            # Seconds until next minute starts (when second = 0)
            seconds_until_next_minute = 60 - now.second
            # Add safety offset
            total_wait = seconds_until_next_minute + SAFETY_OFFSET_SECONDS
            
            # Sleep until next processing time
            time.sleep(total_wait)
            
        except Exception as e:
            logger.error(f"Error in inference worker: {e}", exc_info=True)
            time.sleep(10)


def get_inference_result(symbol: str) -> dict:
    """
    Get the latest inference result for a symbol.
    Returns cached result from the last inference cycle.
    """
    with inference_lock:
        return inference_results.get(symbol, {
            "vwap": 0,
            "ema20": 0,
            "probability": 0,
            "threshold": 0.65,
            "signal": False,
            "status": "no_data"
        })


def background_hub_initializer():
    """Start SmartAPI connection in background"""
    try:
        logger.warning("Starting SmartAPI connection in background...")
        hub.start_smartapi_connection()
        logger.warning("SmartAPI Data Hub connected")
    except Exception as e:
        logger.error(f"Failed to initialize Data Hub connection: {e}")


def handle_new_tick(symbol, tick_data):
    """
    Callback triggered whenever SmartAPI sends a new tick.
    Broadcasts the tick data to all connected frontend clients via WebSocket.
    
    DATA FLOW:
    - Ticks are for UI display only (LTP, volume, change)
    - VWAP, EMA-20, and predictions come from inference results (1-min candle based)
    - This ensures training-inference parity
    """
    try:
        # Create payload with real-time data (ticks)
        tick_dict = tick_data.to_dict()
        
        # Get inference results (computed from 1-min candles, not ticks)
        inference = get_inference_result(symbol)
        
        # Determine trade status based on model signal
        # WAITING = signal active, ready for trade entry
        # NO_TRADE = no signal
        trade_status = 'NO_TRADE'
        if inference.get('signal', False):
            trade_status = 'WAITING'  # Signal active - use WAITING (frontend-compatible value)
        
        # Enhanced data: ticks for display + inference for indicators/signals
        enhanced_data = {
            **tick_dict,
            # Real VWAP and EMA-20 from 1-min candle inference (or LTP if not available)
            'vwap': inference.get('vwap') or tick_dict.get('ltp', 0),
            'ema20': inference.get('ema20') or tick_dict.get('ltp', 0),
            # Model predictions
            'setup_quality': inference.get('probability', 0),
            'threshold': inference.get('threshold', 0.65),
            'signal': inference.get('signal', False),
            'trade_status': trade_status,
            'inference_status': inference.get('status', 'no_data'),
            # Phase-1 MFE/MAE predictions (for display)
            'expected_mfe_atr': inference.get('expected_mfe_atr', 0),
            'expected_mae_atr': inference.get('expected_mae_atr', 0),
            'risk_multiplier': inference.get('risk_multiplier', 0),
            'directional_bias': inference.get('directional_bias', 0),
            'session_quality': inference.get('session_quality', 0),
            'should_not_trade': inference.get('should_not_trade', False),
            # Regime info
            'regime': inference.get('regime', {'volatility': 1, 'trend': 0, 'market_state': 1}),
        }
        
        # Emit to all connected website clients
        socketio.emit('market_data_update', {
            'type': 'market_data_update',
            'data': {
                'symbol': symbol,
                'data': enhanced_data
            },
            'timestamp': datetime.now().isoformat()
        })
            
    except Exception as e:
        logger.error(f"Error handling new tick for {symbol}: {e}", exc_info=True)


# Register the immediate callback
hub.on_tick_callbacks.append(handle_new_tick)

# Start background services
init_thread = threading.Thread(target=background_hub_initializer, daemon=True)
init_thread.start()

# Start inference background worker (fetches 1-min candles every 60s and runs model)
inference_thread = threading.Thread(target=inference_background_worker, daemon=True)
inference_thread.start()


def ensure_naive(dt):
    """
    Ensure datetime is naive (no timezone) for comparison with datetime.now().
    If the input is timezone-aware, convert to LOCAL time first, then strip timezone.
    This ensures UTC timestamps from frontend are properly converted to local time.
    """
    if dt is None: 
        return None
    if isinstance(dt, str):
        try: 
            # Handle ISO format with Z suffix (UTC)
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError: 
            return None
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # Convert to local time FIRST, then strip timezone
        # This fixes the bug where UTC time was compared with local time
        local_dt = dt.astimezone()  # Convert to system local timezone
        return local_dt.replace(tzinfo=None)
    return dt


# ============================================================
# PHASE 2: HELPER FUNCTIONS
# ============================================================

def update_higher_low(symbol: str, df_5min: pd.DataFrame):
    """
    Update the higher low tracker for Trend-type trades.
    Called with 5-minute candles to identify swing structure.
    
    Simple approach: Track the lowest low of the last 3 complete 5-min candles.
    If a new low forms above this level, we have a higher low.
    """
    global higher_low_tracker
    
    if df_5min.empty or len(df_5min) < 3:
        return
    
    # Get the last 3 completed 5-min candles
    recent_candles = df_5min.tail(3)
    
    # The higher low is the lowest of the recent swing
    # For simplicity, use the minimum low of last 3 candles as reference
    current_higher_low = recent_candles['low'].min()
    
    higher_low_tracker[symbol] = current_higher_low


def get_5min_candles_from_1min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute candles to 5-minute candles for higher low tracking.
    """
    if df_1min.empty:
        return pd.DataFrame()
    
    try:
        df_5min = df_1min.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return df_5min
    except Exception as e:
        logger.error(f"Error resampling to 5min: {e}")
        return pd.DataFrame()


def emit_supervision_state(symbol: str, trade: TradeState):
    """
    Emit Phase 2 supervision state to frontend via WebSocket.
    Called after each candle evaluation.
    """
    try:
        # Get Phase-1 context if available
        context = trade.context if hasattr(trade, 'context') and trade.context else None
        
        supervision_data = {
            'symbol': symbol,
            'status': trade.trade_status,
            'exit_reason': trade.exit_reason,
            'entry': {
                'symbol': trade.symbol,
                'entry_type': 'VWAP',  # Default for now - could be stored in TradeState
                'direction': trade.direction if hasattr(trade, 'direction') else 1,
                'entry_price': trade.entry_price,
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
            },
            'metrics': {
                'minutes_in_trade': trade.minutes_in_trade,
                'unrealized_pnl': trade.unrealized_pnl,
                'unrealized_pnl_atr': trade.unrealized_pnl_atr if hasattr(trade, 'unrealized_pnl_atr') else 0,
                'unrealized_pnl_percent': (trade.unrealized_pnl / trade.entry_price * 100) if trade.entry_price > 0 else 0,
                'max_favorable_excursion': trade.mfe,
                'mfe_atr': trade.mfe_atr if hasattr(trade, 'mfe_atr') else 0,
                'max_adverse_excursion': trade.mae,
                'mae_atr': trade.mae_atr if hasattr(trade, 'mae_atr') else 0,
                'entry_atr': trade.entry_atr,
                'profit_floor': trade.profit_floor,
                'profit_floor_active': trade.profit_floor is not None,
            },
            # Include Phase-1 context if available
            'context': {
                'expected_mfe_atr': context.get('expected_mfe_atr', 0) if isinstance(context, dict) else 0,
                'expected_mae_atr': context.get('expected_mae_atr', 0) if isinstance(context, dict) else 0,
                'risk_multiplier': context.get('risk_multiplier', 0.5) if isinstance(context, dict) else 0.5,
                'directional_bias': context.get('directional_bias', 0) if isinstance(context, dict) else 0,
                'session_quality': context.get('session_quality', 0.5) if isinstance(context, dict) else 0.5,
            } if context else None
        }
        
        socketio.emit('phase2_supervision_update', {
            'type': 'phase2_supervision_update',
            'data': supervision_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error emitting supervision state: {e}")


def evaluate_supervised_trade(symbol: str, candle: dict, vwap: float):
    """
    Evaluate a supervised trade after a new 1-min candle closes.
    Called only if there's an active supervised trade for the symbol.
    
    Uses ONLY Exit Engine (Transformer + LightGBM) for ML-based exit decisions.
    """
    global active_supervised_trades
    
    with phase2_lock:
        if symbol not in active_supervised_trades:
            return
        
        trade = active_supervised_trades[symbol]
        
        # Safety check: Ensure entry_time is naive
        if trade.entry_time and hasattr(trade.entry_time, 'tzinfo') and trade.entry_time.tzinfo is not None:
            trade.entry_time = trade.entry_time.replace(tzinfo=None)
        
        # Get current market state from latest inference
        inference = get_inference_result(symbol)
        regime = inference.get('regime', {'volatility': 1, 'trend': 0, 'market_state': 1})
        
        # Update trade state
        atr = trade.entry_atr if hasattr(trade, 'entry_atr') and trade.entry_atr > 0 else 1.0
        current_price = candle['close']
        direction = getattr(trade, 'direction', 1)
        unrealized_pnl = (current_price - trade.entry_price) * direction
        
        # Update trade metrics
        trade.unrealized_pnl = unrealized_pnl
        trade.unrealized_pnl_atr = unrealized_pnl / atr if atr > 0 else 0
        
        # Update MFE/MAE
        if unrealized_pnl > trade.mfe:
            trade.mfe = unrealized_pnl
            trade.mfe_atr = unrealized_pnl / atr if atr > 0 else 0
        if unrealized_pnl < trade.mae:
            trade.mae = unrealized_pnl
            trade.mae_atr = unrealized_pnl / atr if atr > 0 else 0
        
        # FIX Issue 2: Track step as CANDLE COUNT (not wall-clock minutes)
        # This keeps train-live parity
        if not hasattr(trade, 'step'):
            trade.step = 0
        trade.step += 1  # Increment on each candle
        
        # Also track minutes for UI
        now = datetime.now()
        if trade.entry_time:
            trade.minutes_in_trade = int((now - trade.entry_time).total_seconds() / 60)
        
        # ============================================================
        # EXIT ENGINE: Transformer + LightGBM for exit decision
        # ============================================================
        exit_decision = "HOLD"
        exit_reason = None
        
        # FIX Issue 3: Compute momentum_decay (how much profit given back from MFE)
        mfe_atr = trade.mfe_atr if hasattr(trade, 'mfe_atr') else 0
        pnl_atr = trade.unrealized_pnl_atr if hasattr(trade, 'unrealized_pnl_atr') else 0
        momentum_decay = (mfe_atr - pnl_atr) if mfe_atr > 0 else 0
        
        # Symbol ID for embedding (numeric)
        SYMBOL_ID_MAP = {"LT": 0, "RELIANCE": 1, "SIEMENS": 2, "TATAELXSI": 3, "TITAN": 4, "TVSMOTOR": 5}
        symbol_id = SYMBOL_ID_MAP.get(symbol, 0)
        
        # ============================================================
        # GUARDRAIL 1: Skip exit check on first candle
        # Let trade breathe before checking exit
        # ============================================================
        if trade.step <= 1:
            logger.info(f"[EXIT ENGINE] Skipping exit check for {symbol} (step={trade.step}, trade just started)")
            # Update trade status but don't check exit
            trade.trade_status = "IN_TRADE"
            trade.exit_reason = None
            active_supervised_trades[symbol] = trade
            emit_supervision_state(symbol, trade)
            return
        
        if engine.exit_engine:
            try:
                state_features = {
                    "step": trade.step,  # FIX Issue 2: candle count, not minutes
                    "price_from_entry_atr": (current_price - trade.entry_price) / atr,
                    "unrealized_pnl_atr": pnl_atr,
                    "vwap_from_entry_atr": (vwap - trade.entry_price) / atr if vwap > 0 else 0,
                    "mfe_atr": mfe_atr,
                    "mae_atr": trade.mae_atr if hasattr(trade, 'mae_atr') else 0,
                    "volatility_expansion_ratio": regime.get('volatility', 1),
                    "pullback_depth": (trade.mfe - unrealized_pnl) / atr if trade.mfe > 0 else 0,
                    "momentum_decay": momentum_decay,  # FIX Issue 3: actual value
                    "current_atr": atr,
                }
                
                # FIX Issue 1: Pass NUMERIC values matching training
                entry_features = {
                    "direction": direction,  # +1/-1 (numeric)
                    "symbol_id": symbol_id,  # 0-5 (numeric)
                    "initial_atr": atr,
                    "entry_confidence": inference.get('probability', 0.5),
                }
                
                # Get exit decision from ML engine
                exit_result = engine.get_exit_decision(symbol, state_features, entry_features)
                exit_decision = exit_result.get("action", "HOLD")
                
                if exit_decision == "EXIT":
                    exit_reason = "Exit Engine (Transformer + LightGBM)"
                
            except Exception as e:
                logger.error(f"Exit Engine error for {symbol}: {e}")
        
        # Update trade status
        if exit_decision == "EXIT":
            trade.trade_status = "EXIT_RECOMMENDED"
            trade.exit_reason = exit_reason
        else:
            trade.trade_status = "IN_TRADE"
            trade.exit_reason = None
        
        # Store updated state
        active_supervised_trades[symbol] = trade
        
        # Emit to frontend
        emit_supervision_state(symbol, trade)
        
        # Log exit recommendation
        if trade.trade_status == "EXIT_RECOMMENDED":
            logger.warning(f"[EXIT ENGINE] EXIT for {symbol}: {trade.exit_reason}")


# ============================================================
# PHASE 2: SOCKET.IO EVENT HANDLERS
# ============================================================

@socketio.on('phase2_register_entry')
def handle_phase2_register_entry(data):
    """
    Handle trade entry registration from frontend.
    Creates a new TradeState with Phase-1 context and starts monitoring.
    
    Expected data:
    {
        'symbol': 'RELIANCE',
        'entry_type': 'VWAP' | 'TREND',
        'entry_price': 2450.50,
        'entry_time': '2026-01-09T12:30:00',
        'direction': 1  # 1=long, -1=short (optional, defaults to 1)
    }
    """
    global active_supervised_trades
    
    try:
        symbol = data.get('symbol')
        entry_type = data.get('entry_type', 'VWAP').upper()
        entry_price = float(data.get('entry_price', 0))
        entry_time_str = data.get('entry_time')
        direction = int(data.get('direction', 1))  # 1=long, -1=short
        
        if not symbol or entry_price <= 0:
            socketio.emit('phase2_error', {
                'error': 'Invalid entry data',
                'message': 'Symbol and valid entry price are required'
            })
            return
        
        # ============================================================
        # GUARDRAIL 2: Block new entry if trade already exists
        # Prevents double positions and overlapping exit states
        # ============================================================
        with phase2_lock:
            if symbol in active_supervised_trades:
                existing_trade = active_supervised_trades[symbol]
                socketio.emit('phase2_error', {
                    'error': 'Trade already exists',
                    'message': f'You already have an active trade for {symbol} at ₹{existing_trade.entry_price:.2f}. Close it first.'
                })
                logger.warning(f"[GUARDRAIL] Blocked duplicate entry for {symbol} - trade already exists")
                return
        
        # Parse entry time
        try:
            entry_time = datetime.fromisoformat(entry_time_str) if entry_time_str else datetime.now()
            entry_time = ensure_naive(entry_time)
        except:
            entry_time = datetime.now()
        
        # Get Phase-1 inference results
        inference = get_inference_result(symbol)
        entry_atr = inference.get('atr', 0)
        
        # Fallback ATR
        if entry_atr <= 0:
            entry_atr = entry_price * 0.005
            logger.warning(f"ATR not available for {symbol}, using fallback: {entry_atr:.2f}")
        
        # Create new TradeState directly (no phase2_engine)
        trade = TradeState(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            entry_atr=entry_atr,
            direction=direction,
            trade_status="IN_TRADE",
            exit_reason=None,
            mfe=0.0,
            mae=0.0,
            mfe_atr=0.0,
            mae_atr=0.0,
            unrealized_pnl=0.0,
            unrealized_pnl_atr=0.0,
            minutes_in_trade=0,
            profit_floor=None,
        )
        
        # Store context for reference
        trade.context = {
            "entry_prob": inference.get('probability', 0.5),
            "expected_mfe_atr": inference.get('expected_mfe_atr', 1.5),
            "expected_mae_atr": inference.get('expected_mae_atr', 0.5),
            "risk_multiplier": inference.get('risk_multiplier', 0.5),
            "directional_bias": inference.get('directional_bias', 0),
            "session_quality": inference.get('session_quality', 0.5),
        }
        
        # Store trade
        with phase2_lock:
            active_supervised_trades[symbol] = trade
        
        # Reset Exit Engine for new trade
        if engine.exit_engine:
            engine.reset_exit_trade(symbol)
        
        logger.warning(f"[EXIT ENGINE] Entry Registered: {symbol} @ {entry_price:.2f} ({entry_type}) | Direction={'LONG' if direction == 1 else 'SHORT'}")
        
        # Emit confirmation
        emit_supervision_state(symbol, trade)
        
    except Exception as e:
        logger.error(f"Error registering Phase2 entry: {e}")
        socketio.emit('phase2_error', {
            'error': 'Registration failed',
            'message': str(e)
        })


@socketio.on('phase2_clear_trade')
def handle_phase2_clear_trade(data):
    """
    Handle trade clearing from frontend (user manually exited).
    Removes the trade from supervision.
    """
    global active_supervised_trades
    
    try:
        symbol = data.get('symbol')
        
        if not symbol:
            return
        
        with phase2_lock:
            if symbol in active_supervised_trades:
                del active_supervised_trades[symbol]
                logger.warning(f"🔴 Phase2 Trade Cleared: {symbol}")
        
        # Emit cleared state
        socketio.emit('phase2_supervision_update', {
            'type': 'phase2_supervision_update',
            'data': {
                'symbol': symbol,
                'status': 'NO_ACTIVE_TRADE',
                'exit_reason': None,
                'entry': None,
                'metrics': None
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error clearing Phase2 trade: {e}")


@socketio.on('phase2_get_state')
def handle_phase2_get_state(data):
    """
    Get current supervision state for a symbol.
    Called when frontend loads or switches symbols.
    """
    try:
        symbol = data.get('symbol')
        
        if not symbol:
            return
        
        with phase2_lock:
            if symbol in active_supervised_trades:
                trade = active_supervised_trades[symbol]
                emit_supervision_state(symbol, trade)
            else:
                # No active trade
                socketio.emit('phase2_supervision_update', {
                    'type': 'phase2_supervision_update',
                    'data': {
                        'symbol': symbol,
                        'status': 'NO_ACTIVE_TRADE',
                        'exit_reason': None,
                        'entry': None,
                        'metrics': None
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
    except Exception as e:
        logger.error(f"Error getting Phase2 state: {e}")




@app.route('/test/emit')
def test_emit():
    """Test endpoint to manually emit WebSocket data"""
    try:
        test_data = {
            'type': 'market_data_update',
            'data': {
                'symbol': 'TEST',
                'data': {
                    'symbol': 'TEST',
                    'ltp': 1000.0,
                    'volume': 12345,
                    'timestamp': datetime.now().isoformat(),
                    'change': 10.0,
                    'change_percent': 1.0,
                    'high': 1010.0,
                    'low': 990.0,
                    'open': 995.0,
                    'close': 990.0,
                    'vwap': 1000.0,
                    'ema20': 1000.0,
                    'setup_quality': 0.75,
                    'threshold': 0.65,
                    'trade_status': 'IN_TRADE_LONG'
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"📡 Test emission: {test_data}")
        socketio.emit('market_data_update', test_data)
        
        return jsonify({
            "status": "success",
            "message": "Test data emitted",
            "data": test_data
        })
    except Exception as e:
        logger.error(f"Error in test emit: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/')
def health_check():
    return jsonify({
        "server": "online",
        "websocket_active": True,
        "hub_status": hub.get_connection_status() if hub else "initializing"
    })


@app.route('/api/watchlist')
def get_watchlist():
    return jsonify({"symbols": get_verified_smartapi_watchlist()})


@app.route('/api/history/<symbol>')
def get_history(symbol):
    """
    Fetch chart history for a symbol - CURRENT DAY ONLY.
    Returns 1-minute candles from today's market open (9:15 AM) to now.
    
    TEMPORARILY DISABLED - Returns empty array to reduce API load.
    """
    # ============================================================
    # TEMPORARILY DISABLED - Uncomment below to re-enable
    # ============================================================
    return jsonify([])
    
    # --- DISABLED CODE BELOW ---
    try:
        client = get_smartapi_client()
        token = get_smartapi_token(symbol)
        if not token: return jsonify([])

        now = datetime.now()
        
        # Current day only - start from market open (9:15 AM)
        start_date = now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        from_str = start_date.strftime('%Y-%m-%d 09:15')
        # Use current time if within market hours, else market close
        if 9 <= now.hour < 16:
            to_str = now.strftime('%Y-%m-%d %H:%M')
        else:
            to_str = now.strftime('%Y-%m-%d 15:30')

        logger.info(f"📊 API History request for {symbol}: {from_str} to {to_str} (current day only)")

        historical_data = client.get_historical_candles(
            exchange="NSE", symbol_token=token, interval="ONE_MINUTE",
            from_date=from_str, to_date=to_str
        )
        
        history = []
        if historical_data:
            for item in historical_data:
                history.append({
                    "timestamp": item['timestamp'],
                    "open": item['open'], "high": item['high'], "low": item['low'],
                    "close": item['close'], "volume": item['volume']
                })
        
        # Merge with local ticks (UI-only, NOT for inference)
        df_local = hub.get_market_data_history(symbol)
        if not df_local.empty:
            last_ts = ensure_naive(history[-1]['timestamp']) if history else ensure_naive(start_date)
            for ts, row in df_local.iterrows():
                if ensure_naive(ts) > last_ts:
                    history.append({
                        "timestamp": ts.isoformat(),
                        "open": row['ltp'], "high": row['ltp'], "low": row['ltp'],
                        "close": row['ltp'], "volume": row['volume']
                    })
        return jsonify(history)
    except Exception as e:
        logger.error(f"History Error: {e}")
        return jsonify([])


@app.route('/api/prediction/<symbol>')
def get_prediction(symbol):
    """
    Return prediction data for a symbol.
    Uses real inference results computed from 1-minute candles.
    """
    current_market = hub.get_market_data(symbol)
    tick = current_market.get(symbol, {})
    
    # Get real inference results (computed from 1-min candles)
    inference = get_inference_result(symbol)
    
    # Determine direction based on price vs VWAP
    direction = "NEUTRAL"
    ltp = tick.get('ltp', 0) or 0
    vwap = inference.get('vwap', 0)
    if vwap > 0 and ltp > 0:
        if ltp > vwap * 1.001:  # 0.1% above VWAP
            direction = "BULLISH"
        elif ltp < vwap * 0.999:  # 0.1% below VWAP
            direction = "BEARISH"
    
    return jsonify({
        "prediction": {
            "last_price": ltp,
            "change": tick.get('change', 0) or 0,
            "change_percent": tick.get('change_percent', 0) or 0,
            # Real values from inference
            "setup_quality": inference.get('probability', 0),
            "threshold": inference.get('threshold', 0.65),
            "vwap": vwap or ltp,
            "ema20": inference.get('ema20', 0) or ltp,
            "signal": inference.get('signal', False),
            "probability": inference.get('probability', 0),
            "direction": direction,
            # Phase-1 model outputs
            "expected_mfe_atr": inference.get('expected_mfe_atr', 0),
            "expected_mae_atr": inference.get('expected_mae_atr', 0),
            "risk_multiplier": inference.get('risk_multiplier', 0),
            "directional_bias": inference.get('directional_bias', 0),
            "session_quality": inference.get('session_quality', 0),
            "should_not_trade": inference.get('should_not_trade', False),
            "regime": inference.get('regime', {'volatility': 1, 'trend': 0, 'market_state': 1}),
            # Additional inference metadata
            "inference_status": inference.get('status', 'no_data'),
            "candle_count": inference.get('candle_count', 0),
            "last_inference_time": inference.get('timestamp', None)
        }
    })


@app.route('/api/inference/status')
def get_inference_status():
    """
    Debug endpoint to check inference status for all symbols.
    """
    with inference_lock:
        status = {
            "symbols": {},
            "total_symbols": len(inference_results),
            "current_time": datetime.now().isoformat()
        }
        for symbol, result in inference_results.items():
            status["symbols"][symbol] = {
                "vwap": result.get('vwap', 0),
                "ema20": result.get('ema20', 0),
                "probability": result.get('probability', 0),
                "signal": result.get('signal', False),
                "status": result.get('status', 'unknown'),
                "candle_count": result.get('candle_count', 0),
                "timestamp": result.get('timestamp', None)
            }
    return jsonify(status)


@app.route('/api/cache/status')
def get_cache_status():
    """
    Debug endpoint to check candle cache status.
    Shows if incremental update strategy is working correctly.
    """
    with candle_cache_lock:
        status = {
            "cache_initialized": cache_initialized,
            "total_symbols": len(candle_cache),
            "target_candles_per_symbol": TARGET_CANDLES,
            "safety_offset_seconds": SAFETY_OFFSET_SECONDS,
            "stagger_delay_seconds": STAGGER_DELAY_SECONDS,
            "current_time": datetime.now().isoformat(),
            "symbols": {}
        }
        for symbol, df in candle_cache.items():
            if not df.empty:
                status["symbols"][symbol] = {
                    "candle_count": len(df),
                    "oldest_candle": df.index[0].isoformat(),
                    "newest_candle": df.index[-1].isoformat(),
                    "memory_kb": df.memory_usage(deep=True).sum() / 1024
                }
            else:
                status["symbols"][symbol] = {
                    "candle_count": 0,
                    "oldest_candle": None,
                    "newest_candle": None,
                    "memory_kb": 0
                }
    return jsonify(status)


if __name__ == '__main__':
    print("Starting Trading Terminal...")
    print("Backend: http://localhost:5000")
    print("Frontend: http://localhost:5173")
    print("Real-time data will flow during market hours (9:15 AM - 3:30 PM IST)")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
