"""
Real-Time Data Hub Core Infrastructure
Centralizes SmartAPI WebSocket connections and manages real-time market data distribution
"""
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, Any, List
from dataclasses import dataclass, asdict
import websocket
from collections import defaultdict
import pandas as pd

# Import existing SmartAPI client
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from smartapi_client import get_smartapi_client

# Import symbol utilities
from symbol_utils import get_verified_smartapi_watchlist, get_smartapi_token, get_smartapi_trading_symbol

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealTimeDataHub")

# Configure logging level based on environment variable
# Set RTDH_LOG_LEVEL=INFO or RTDH_LOG_LEVEL=DEBUG for verbose logging
# Default is WARNING to reduce tick data noise
log_level = os.getenv('RTDH_LOG_LEVEL', 'WARNING').upper()
logger.setLevel(getattr(logging, log_level, logging.WARNING))

@dataclass
class MarketDataPoint:
    """Market data model for consistent data structure"""
    symbol: str
    ltp: float
    volume: int
    timestamp: datetime
    change: float
    change_percent: float
    high: float
    low: float
    open: float
    close: float  # Previous day's closing price
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'ltp': self.ltp,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat(),
            'change': self.change,
            'change_percent': self.change_percent,
            'high': self.high,
            'low': self.low,
            'open': self.open,
            'close': self.close
        }

@dataclass
class WebSocketMessage:
    """WebSocket message model for consistent communication"""
    type: str  # 'market_data_update', 'connection_status', 'error'
    data: dict
    timestamp: datetime
    
    def to_json(self) -> str:
        return json.dumps({
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        })

@dataclass
class RealTimeConfig:
    """Configuration for real-time data hub"""
    symbols: List[str]
    update_interval_ms: int = 50
    max_websocket_clients: int = 100
    reconnect_attempts: int = 5
    memory_limit_mb: int = 100

class RealTimeDataHub:
    """
    Central component that manages SmartAPI WebSocket connection and broadcasts market data
    Implements thread-safe in-memory data storage and WebSocket client management
    Task 8: Includes automatic reconnection logic and error recovery
    """
    
    def __init__(self):
        # SmartAPI client integration
        self.smartapi_client = get_smartapi_client()
        
        # Thread-safe in-memory data storage
        self.market_data: Dict[str, MarketDataPoint] = {}
        self.data_lock = threading.RLock()
        
        # WebSocket client management for frontend connections
        self.websocket_clients: Set[Any] = set()
        self.client_lock = threading.RLock()
        
        # Connection state management
        self.connected = False
        self.websocket_active = False
        self.connection_lock = threading.RLock()
        
        # Task 8: Auto-reconnection configuration
        self.reconnect_thread = None
        self.reconnect_active = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # Start with 5 seconds
        self.max_reconnect_delay = 300  # Max 5 minutes
        self.reconnect_backoff_multiplier = 1.5  # Exponential backoff
        
        # Configuration - focus only on verified equity symbols
        watchlist_symbols = get_verified_smartapi_watchlist()
        
        # User requested to remove index symbols
        all_symbols = watchlist_symbols
        
        self.config = RealTimeConfig(
            symbols=all_symbols
        )
        
        logger.info(f"📊 Real-Time Data Hub: {len(watchlist_symbols)} equity symbols")
        
        # Data retention for memory management
        self.max_data_points_per_symbol = 1000
        self.data_history: Dict[str, List[MarketDataPoint]] = defaultdict(list)
        self.rest_history: Dict[str, pd.DataFrame] = {} # Store 1m candles for model
        self.history_lock = threading.RLock()
        
        # Task 8: Memory management configuration
        self.memory_cleanup_interval = 300  # Clean up every 5 minutes
        self.last_cleanup_time = datetime.now()
        self.max_memory_percent = 80  # Alert if memory usage exceeds 80%
        
        # Performance tracking
        self.last_update_time = None
        self.update_count = 0
        self.connection_errors = 0
        self.error_recovery_count = 0
        self.on_tick_callbacks: List[callable] = []
        
        logger.info(f"RealTimeDataHub initialized for {len(self.config.symbols)} symbols")
    
    def start_smartapi_connection(self) -> bool:
        """
        Initialize and connect SmartAPI WebSocket
        Returns True if connection successful, False otherwise
        """
        with self.connection_lock:
            try:
                # Connect to SmartAPI if not already connected
                if not self.smartapi_client.connected:
                    logger.info("Connecting to SmartAPI...")
                    if not self.smartapi_client.connect():
                        logger.error("Failed to connect to SmartAPI")
                        return False
                
                # Initialize WebSocket with custom callbacks
                logger.info("Initializing SmartAPI WebSocket...")
                success = self.smartapi_client.init_websocket(
                    on_message_callback=self._on_smartapi_message,
                    on_open_callback=self._on_smartapi_open,
                    on_error_callback=self._on_smartapi_error,
                    on_close_callback=self._on_smartapi_close,
                    max_retry_attempt=self.config.reconnect_attempts
                )
                
                if not success:
                    logger.error("Failed to initialize SmartAPI WebSocket")
                    return False
                
                # Connect WebSocket
                logger.info("Connecting SmartAPI WebSocket...")
                if not self.smartapi_client.connect_websocket():
                    logger.error("Failed to connect SmartAPI WebSocket")
                    return False
                
                # Subscribe to market data for configured symbols
                self._subscribe_to_symbols()
                
                # Wait a moment for subscription to take effect
                import time
                time.sleep(2)
                
                # Verify WebSocket is actually receiving data
                logger.info("🔍 Verifying WebSocket data reception...")
                logger.info("   If no tick data appears within 30 seconds, check:")
                logger.info("   1. Market hours (9:15 AM - 3:30 PM IST)")
                logger.info("   2. Token format matches SmartAPI expectations")
                logger.info("   3. Subscription mode is correct")
                logger.info("   4. WebSocket callbacks are properly registered")
                
                self.connected = True
                logger.info("✅ SmartAPI connection established successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error starting SmartAPI connection: {e}", exc_info=True)
                return False
    
    def _subscribe_to_symbols(self):
        """Subscribe to market data for configured symbols"""
        try:
            # Prepare token list for subscription
            token_list = []
            valid_tokens = []
            symbol_token_map = {}
            
            logger.info(f"🔍 Preparing subscription for {len(self.config.symbols)} symbols...")
            
            for symbol in self.config.symbols:
                token = get_smartapi_token(symbol)
                trading_symbol = get_smartapi_trading_symbol(symbol)
                
                if token and trading_symbol:
                    valid_tokens.append(token)
                    symbol_token_map[token] = symbol
                    logger.info(f"   ✅ {symbol} -> Token: {token} | Trading: {trading_symbol}")
                else:
                    logger.warning(f"   ⚠️ No token found for symbol: {symbol} (Index handling pending implementation)")
            
            if valid_tokens:
                # Create token list in SmartAPI format
                token_list = [{
                    "exchangeType": 1,  # NSE_CM
                    "tokens": valid_tokens
                }]
                
                # Subscribe with QUOTE mode (mode 2) for basic quote data
                logger.info(f"📡 Subscribing to {len(valid_tokens)} symbols in QUOTE mode...")
                logger.info(f"   Token list: {valid_tokens[:10]}..." if len(valid_tokens) > 10 else f"   Token list: {valid_tokens}")
                
                success = self.smartapi_client.subscribe(
                    token_list=token_list,
                    mode=2,  # QUOTE mode for LTP + volume + OHLC
                    correlation_id="RTDH001"
                )
                
                if success:
                    logger.info(f"✅ Successfully subscribed to {len(valid_tokens)} symbols")
                    logger.info(f"   Waiting for tick data...")
                    logger.info(f"   Symbol-Token mapping: {list(symbol_token_map.items())[:5]}...")
                else:
                    logger.error("❌ Failed to subscribe to symbols")
                    logger.error("   Check WebSocket connection status")
            else:
                logger.error("❌ No valid tokens found for subscription")
                logger.error("   Check symbol_mappings.json and symbol_utils.py")
                
        except Exception as e:
            logger.error(f"❌ Error subscribing to symbols: {e}", exc_info=True)
    
    def _on_smartapi_message(self, wsapp, message):
        """Process incoming SmartAPI market data with enhanced logging"""
        try:
            # DIAGNOSTIC: Log every message received (DEBUG level to reduce noise)
            logger.debug(f"🔔 RAW MESSAGE RECEIVED - Type: {type(message)}")
            
            # Handle both dict and object message types from SmartAPI
            if isinstance(message, dict):
                logger.debug(f"   ✅ Dict message with keys: {list(message.keys())}")
                logger.debug(f"   Full message: {message}")
                self.on_market_data_received(message)
            elif hasattr(message, '__dict__'):
                # Convert object to dict if needed
                logger.debug(f"   ✅ Object message, converting to dict")
                msg_dict = vars(message)
                logger.debug(f"   Converted: {msg_dict}")
                self.on_market_data_received(msg_dict)
            elif isinstance(message, (str, bytes)):
                logger.debug(f"   ⚠️ String/bytes message, attempting parse")
                try:
                    if isinstance(message, bytes):
                        message = message.decode('utf-8')
                    import json
                    msg_dict = json.loads(message)
                    logger.debug(f"   ✅ Parsed JSON successfully")
                    self.on_market_data_received(msg_dict)
                except Exception as parse_error:
                    logger.error(f"   ❌ Failed to parse: {parse_error}")
            else:
                logger.warning(f"❌ Unexpected message type: {type(message)}")
                logger.warning(f"   Message content: {str(message)[:200]}")
                logger.warning(f"   Dir: {dir(message)}")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Error processing SmartAPI message: {e}", exc_info=True)
    
    def _on_smartapi_open(self, wsapp):
        """Handle SmartAPI WebSocket open"""
        logger.info("SmartAPI WebSocket opened")
        self.websocket_active = True
        self._broadcast_connection_status("connected")
    
    def _on_smartapi_error(self, wsapp, error):
        """
        Handle SmartAPI WebSocket errors
        Task 8: Initiates automatic reconnection on error
        """
        logger.error(f"SmartAPI WebSocket error: {error}")
        self.websocket_active = False
        self.connection_errors += 1
        self._broadcast_connection_status("error", str(error))
        
        # Task 8: Initiate auto-reconnection
        self._initiate_reconnection()
    
    def _on_smartapi_close(self, wsapp):
        """
        Handle SmartAPI WebSocket close
        Task 8: Initiates automatic reconnection on close
        """
        logger.info("SmartAPI WebSocket closed")
        self.websocket_active = False
        self._broadcast_connection_status("disconnected")
        
        # Task 8: Initiate auto-reconnection
        self._initiate_reconnection()
    
    def _initiate_reconnection(self):
        """
        Task 8: Initiate automatic reconnection with exponential backoff
        Starts a background thread to attempt reconnection
        """
        if self.reconnect_active:
            logger.debug("Reconnection already in progress")
            return
        
        self.reconnect_active = True
        self.reconnect_thread = threading.Thread(
            target=self._reconnection_loop,
            daemon=True
        )
        self.reconnect_thread.start()
        logger.info("Started automatic reconnection loop")
    
    def _reconnection_loop(self):
        """
        Task 8: Background thread for automatic reconnection with exponential backoff
        Attempts to reconnect to SmartAPI with exponential delay
        """
        current_delay = self.reconnect_delay
        
        while self.reconnect_active and self.reconnect_attempts < self.max_reconnect_attempts:
            if self.connected and self.websocket_active:
                logger.info("✅ SmartAPI reconnected successfully")
                self.reconnect_attempts = 0
                self.error_recovery_count += 1
                self.reconnect_active = False
                return
            
            self.reconnect_attempts += 1
            logger.info(
                f"🔄 Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} "
                f"(waiting {current_delay}s)"
            )
            
            time.sleep(current_delay)
            
            try:
                # Attempt to reconnect
                if self.start_smartapi_connection():
                    logger.info("✅ Successfully reconnected to SmartAPI")
                    self.error_recovery_count += 1
                    self.reconnect_active = False
                    return
                else:
                    logger.warning("Reconnection attempt failed, will retry...")
                    # Calculate next delay with exponential backoff
                    current_delay = min(
                        current_delay * self.reconnect_backoff_multiplier,
                        self.max_reconnect_delay
                    )
            
            except Exception as e:
                logger.error(f"Error during reconnection attempt: {e}")
                current_delay = min(
                    current_delay * self.reconnect_backoff_multiplier,
                    self.max_reconnect_delay
                )
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("❌ Max reconnection attempts reached. Manual intervention required.")
            self._broadcast_connection_status(
                "error",
                "Failed to reconnect after maximum attempts"
            )
        
        self.reconnect_active = False
    
    def on_market_data_received(self, data):
        """
        Process incoming SmartAPI data, update in-memory store, and broadcast to clients
        Thread-safe implementation with proper locking
        Handles SmartAPI WebSocket V2 message format with comprehensive field mapping
        Task 8: Includes periodic memory cleanup
        """
        try:
            if not isinstance(data, dict):
                logger.warning(f"⚠️ on_market_data_received: data is not dict, type={type(data)}")
                return
            
            # DIAGNOSTIC: Log data processing attempt
            logger.debug(f"📊 Processing market data with {len(data)} fields")
            logger.debug(f"   Available keys: {list(data.keys())}")
            
            # Task 8: Periodic memory cleanup
            self._check_and_cleanup_memory()
            
            # Extract market data from SmartAPI message
            # SmartAPI WebSocket V2 sends various field names, handle all variants
            token = data.get('token') or data.get('tradingSymbol') or data.get('t')
            
            if not token:
                logger.warning(f"⚠️ No token found in message. Keys: {list(data.keys())}")
                logger.warning(f"   Full data: {data}")
                return
            
            logger.debug(f"   Token extracted: {token}")
            
            # Find symbol for this token
            symbol = self._get_symbol_from_token(str(token))
            if not symbol:
                logger.warning(f"⚠️ Unknown token: {token} - not in symbol mappings")
                logger.warning(f"   Subscribed symbols: {self.config.symbols[:5]}...")
                return
            
            logger.debug(f"   ✅ Token {token} mapped to symbol: {symbol}")
            
            # Extract price and volume data - handle multiple field name variants
            # SmartAPI sends prices in paise (multiply by 100)
            ltp = data.get('last_traded_price') or data.get('ltp') or data.get('lastPrice') or 0
            volume = data.get('volume_trade_for_the_day') or data.get('volume_traded') or data.get('volume') or data.get('v') or 0
            high = data.get('high_price_of_the_day') or data.get('high') or data.get('h') or ltp
            low = data.get('low_price_of_the_day') or data.get('low') or data.get('l') or ltp
            open_price = data.get('open_price_of_the_day') or data.get('open') or data.get('o') or ltp
            close_price = data.get('closed_price') or data.get('close') or data.get('c') or ltp
            
            # Convert from paise to rupees if needed
            # SmartAPI sends prices in paise (1 rupee = 100 paise)
            if ltp > 1000:  # Likely in paise
                ltp = round(ltp / 100.0, 2)
                high = round(high / 100.0, 2)
                low = round(low / 100.0, 2)
                open_price = round(open_price / 100.0, 2)
                close_price = round(close_price / 100.0, 2)
            
            # Validate data
            if ltp <= 0 or volume < 0:
                logger.warning(f"⚠️ Invalid data for {symbol}: ltp={ltp}, volume={volume}")
                logger.warning(f"   Raw data: {data}")
                return
            
            # Log at DEBUG level to reduce console noise (change to INFO if debugging)
            logger.debug(f"✅ Valid tick data: {symbol} @ ₹{ltp:.2f} | Vol: {volume}")
            
            # Calculate change and change percentage from previous day's close
            change = 0.0
            change_percent = 0.0
            
            # Calculate day's gain/loss: LTP - Previous Close
            if close_price > 0:
                change = ltp - close_price
                change_percent = (change / close_price) * 100
            
            with self.data_lock:
                # Create market data point
                market_data_point = MarketDataPoint(
                    symbol=symbol,
                    ltp=ltp,
                    volume=int(volume),
                    timestamp=datetime.now(),
                    change=round(change, 2),
                    change_percent=round(change_percent, 2),
                    high=high,
                    low=low,
                    open=open_price,
                    close=close_price
                )
                
                # Update in-memory store
                self.market_data[symbol] = market_data_point
                
                # Add to history with memory management
                self.data_history[symbol].append(market_data_point)
                if len(self.data_history[symbol]) > self.max_data_points_per_symbol:
                    self.data_history[symbol] = self.data_history[symbol][-self.max_data_points_per_symbol:]
                
                # Update performance tracking
                self.last_update_time = datetime.now()
                self.update_count += 1
            
            # Broadcast to internal callbacks immediately (app.py handles the emit)
            logger.debug(f"📤 Broadcasting {symbol} tick to {len(self.on_tick_callbacks)} callbacks")
            for callback in self.on_tick_callbacks:
                try:
                    callback(symbol, market_data_point)
                    logger.debug(f"   ✅ Callback executed successfully for {symbol}")
                except Exception as e:
                    logger.error(f"   ❌ Error in on_tick_callback: {e}", exc_info=True)

            # Broadcast to WebSocket clients
            self.broadcast_to_clients({
                'type': 'market_data_update',
                'symbol': symbol,
                'data': market_data_point.to_dict()
            })
            
            logger.debug(f"Updated {symbol}: ₹{ltp:.2f} | Change: {change:+.2f} ({change_percent:+.2f}%) | Vol: {volume}")
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}", exc_info=True)
    
    def _get_symbol_from_token(self, token: str) -> Optional[str]:
        """Get symbol name from SmartAPI token"""
        try:
            # Import here to avoid circular imports
            from symbol_utils import SMARTAPI_SYMBOL_MAPPINGS
            
            for symbol, mapping in SMARTAPI_SYMBOL_MAPPINGS.items():
                if mapping.get('token') == str(token):
                    return symbol
            return None
        except Exception as e:
            logger.error(f"Error getting symbol from token {token}: {e}")
            return None

    def get_market_data_history(self, symbol: str) -> pd.DataFrame:
        """
        Get historical market data for a symbol as a pandas DataFrame
        Returns empty DataFrame if symbol not found or no data
        """
        with self.data_lock:
            if symbol in self.data_history and self.data_history[symbol]:
                # Convert list of MarketDataPoint to list of dicts
                data_list = [asdict(p) for p in self.data_history[symbol]]
                df = pd.DataFrame(data_list)
                
                # Set timestamp as index and sort
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                
                return df
            else:
                return pd.DataFrame()

    def update_rest_history(self, symbol: str):
        """Fetch 1m historical candles via REST API for model inference - CURRENT + PREVIOUS TRADING DAY ONLY"""
        try:
            token = get_smartapi_token(symbol)
            if not token:
                return False

            end_date = datetime.now()
            
            # Calculate previous trading day (skip weekends) - SAME LOGIC AS FRONTEND
            def get_previous_trading_day(date):
                previous_day = date - timedelta(days=1)
                
                # If current day is Monday (0), go back to Friday
                if date.weekday() == 0:  # Monday
                    previous_day = date - timedelta(days=3)
                # If current day is Sunday (6), go back to Friday  
                elif date.weekday() == 6:  # Sunday
                    previous_day = date - timedelta(days=2)
                
                return previous_day
            
            # Get previous trading day
            previous_trading_day = get_previous_trading_day(end_date)
            
            # Set start date to previous trading day at market open
            start_date = previous_trading_day.replace(hour=9, minute=15, second=0, microsecond=0)
            
            from_str = start_date.strftime('%Y-%m-%d 09:15')
            to_str = end_date.strftime('%Y-%m-%d 15:30')
            if 9 <= end_date.hour < 16:
                to_str = end_date.strftime('%Y-%m-%d %H:%M')

            logger.info(f"📊 Fetching {symbol} data from {from_str} to {to_str} (current + previous trading day)")

            candles = self.smartapi_client.get_historical_candles(
                exchange="NSE", symbol_token=token, interval="ONE_MINUTE",
                from_date=from_str, to_date=to_str
            )

            if candles:
                df = pd.DataFrame(candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                with self.history_lock:
                    self.rest_history[symbol] = df
                
                logger.info(f"✅ REST History Updated for {symbol}: {len(df)} candles (current + previous trading day)")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating REST history for {symbol}: {e}")
            return False

    def get_rest_history(self, symbol: str) -> pd.DataFrame:
        """Get cached REST historical data for a symbol"""
        with self.history_lock:
            return self.rest_history.get(symbol, pd.DataFrame())
    
    def get_market_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        API endpoint for Analysis.py - get market data with thread-safe access
        Returns all data if symbol is None, specific symbol data otherwise
        """
        with self.data_lock:
            if symbol:
                # Return specific symbol data
                if symbol in self.market_data:
                    return {
                        symbol: self.market_data[symbol].to_dict()
                    }
                else:
                    return {}
            else:
                # Return all market data
                return {
                    symbol: data_point.to_dict() 
                    for symbol, data_point in self.market_data.items()
                }
    
    def add_websocket_client(self, client):
        """Add a WebSocket client for real-time updates"""
        with self.client_lock:
            self.websocket_clients.add(client)
            logger.info(f"Added WebSocket client. Total clients: {len(self.websocket_clients)}")
    
    def remove_websocket_client(self, client):
        """Remove a WebSocket client"""
        with self.client_lock:
            self.websocket_clients.discard(client)
            logger.info(f"Removed WebSocket client. Total clients: {len(self.websocket_clients)}")
    
    def broadcast_to_clients(self, data: Dict[str, Any]):
        """
        Send data to all connected WebSocket clients
        Thread-safe implementation with error handling
        """
        if not self.websocket_clients:
            return
        
        message = WebSocketMessage(
            type=data.get('type', 'market_data_update'),
            data=data,
            timestamp=datetime.now()
        )
        
        # Create list of clients to avoid modification during iteration
        with self.client_lock:
            clients_to_notify = list(self.websocket_clients)
        
        # Broadcast to all clients
        failed_clients = []
        for client in clients_to_notify:
            try:
                # This will be implemented when integrating with Flask-SocketIO
                # For now, just store the message for later broadcasting
                if hasattr(client, 'emit'):
                    client.emit('market_data_update', message.to_json())
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                failed_clients.append(client)
        
        # Remove failed clients
        if failed_clients:
            with self.client_lock:
                for client in failed_clients:
                    self.websocket_clients.discard(client)
    
    def _broadcast_connection_status(self, status: str, message: str = ""):
        """Broadcast connection status to all clients"""
        self.broadcast_to_clients({
            'type': 'connection_status',
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def _check_and_cleanup_memory(self):
        """
        Task 8: Check memory usage and perform cleanup if needed
        Runs periodically to prevent memory bloat
        """
        try:
            now = datetime.now()
            # Only cleanup every memory_cleanup_interval seconds
            if (now - self.last_cleanup_time).total_seconds() < self.memory_cleanup_interval:
                return
            
            import psutil
            process = psutil.Process()
            memory_percent = process.memory_percent()
            
            # Log memory usage
            logger.debug(f"Memory usage: {memory_percent:.2f}%")
            
            # Alert if memory usage is high
            if memory_percent > self.max_memory_percent:
                logger.warning(f"⚠️  High memory usage: {memory_percent:.2f}%")
                self._aggressive_memory_cleanup()
            else:
                # Perform regular cleanup
                self._perform_memory_cleanup()
            
            self.last_cleanup_time = now
            
        except Exception as e:
            logger.debug(f"Error checking memory: {e}")
    
    def _perform_memory_cleanup(self):
        """
        Task 8: Perform regular memory cleanup
        Removes old data from history to maintain memory limits
        """
        with self.data_lock:
            try:
                total_points_before = sum(len(hist) for hist in self.data_history.values())
                
                # Keep only the last N points per symbol
                for symbol in self.data_history:
                    if len(self.data_history[symbol]) > self.max_data_points_per_symbol:
                        removed = len(self.data_history[symbol]) - self.max_data_points_per_symbol
                        self.data_history[symbol] = self.data_history[symbol][-self.max_data_points_per_symbol:]
                        logger.debug(f"Removed {removed} old data points for {symbol}")
                
                total_points_after = sum(len(hist) for hist in self.data_history.values())
                if total_points_before > total_points_after:
                    logger.info(f"Memory cleanup: {total_points_before} → {total_points_after} data points")
            
            except Exception as e:
                logger.error(f"Error during memory cleanup: {e}")
    
    def _aggressive_memory_cleanup(self):
        """
        Task 8: Perform aggressive memory cleanup when memory usage is high
        Reduces data retention to critical minimum
        """
        with self.data_lock:
            try:
                logger.warning("Performing aggressive memory cleanup...")
                
                # Reduce history to last 100 points per symbol (from 1000)
                aggressive_limit = 100
                total_points_before = sum(len(hist) for hist in self.data_history.values())
                
                for symbol in self.data_history:
                    if len(self.data_history[symbol]) > aggressive_limit:
                        self.data_history[symbol] = self.data_history[symbol][-aggressive_limit:]
                
                # Clear WebSocket clients that have errors
                dead_clients = []
                with self.client_lock:
                    for client in self.websocket_clients:
                        try:
                            if hasattr(client, 'connected') and not client.connected():
                                dead_clients.append(client)
                        except:
                            dead_clients.append(client)
                    
                    for client in dead_clients:
                        self.websocket_clients.discard(client)
                
                total_points_after = sum(len(hist) for hist in self.data_history.values())
                logger.info(
                    f"Aggressive cleanup complete: "
                    f"{total_points_before} → {total_points_after} data points, "
                    f"removed {len(dead_clients)} dead clients"
                )
            
            except Exception as e:
                logger.error(f"Error during aggressive memory cleanup: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status
        Task 8: Includes error recovery metrics
        """
        return {
            'smartapi_connected': self.connected,
            'websocket_active': self.websocket_active,
            'client_count': len(self.websocket_clients),
            'symbols_subscribed': len(self.config.symbols),
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'update_count': self.update_count,
            'connection_errors': self.connection_errors,
            'error_recovery_count': self.error_recovery_count,
            'reconnect_attempts': self.reconnect_attempts,
            'is_reconnecting': self.reconnect_active
        }
    
    def stop(self):
        """
        Stop the data hub and clean up resources
        Task 8: Properly stops reconnection attempts
        """
        logger.info("Stopping Real-Time Data Hub...")
        
        # Stop reconnection attempts
        self.reconnect_active = False
        
        with self.connection_lock:
            try:
                # Disconnect SmartAPI WebSocket
                if self.smartapi_client and self.smartapi_client.websocket:
                    self.smartapi_client.disconnect()
                
                # Clear client connections
                with self.client_lock:
                    self.websocket_clients.clear()
                
                # Clear data
                with self.data_lock:
                    self.market_data.clear()
                    self.data_history.clear()
                
                self.connected = False
                self.websocket_active = False
                
                # Wait for reconnection thread to finish if active
                if self.reconnect_thread and self.reconnect_thread.is_alive():
                    logger.info("Waiting for reconnection thread to stop...")
                    self.reconnect_thread.join(timeout=5)
                
                logger.info("Real-Time Data Hub stopped successfully")
                
            except Exception as e:
                logger.error(f"Error stopping data hub: {e}", exc_info=True)

# Global singleton instance
_data_hub_instance = None
_data_hub_lock = threading.RLock()

def get_realtime_data_hub() -> RealTimeDataHub:
    """Get the global Real-Time Data Hub singleton instance"""
    global _data_hub_instance
    
    with _data_hub_lock:
        if _data_hub_instance is None:
            _data_hub_instance = RealTimeDataHub()
        return _data_hub_instance

# Utility functions for easy access
def start_realtime_data_hub() -> RealTimeDataHub:
    """Start the real-time data hub and return the hub instance"""
    hub = get_realtime_data_hub()
    hub.start_smartapi_connection()
    return hub

def get_realtime_market_data(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Get real-time market data"""
    hub = get_realtime_data_hub()
    return hub.get_market_data(symbol)

def get_hub_status() -> Dict[str, Any]:
    """Get data hub connection status"""
    hub = get_realtime_data_hub()
    return hub.get_connection_status()

if __name__ == "__main__":
    # Test the Real-Time Data Hub
    print("🚀 Starting Real-Time Data Hub test...")
    
    hub = get_realtime_data_hub()
    
    if hub.start_smartapi_connection():
        print("✅ Real-Time Data Hub started successfully")
        print(f"📊 Monitoring {len(hub.config.symbols)} symbols")
        
        # Run for a short time to test data reception
        try:
            time.sleep(10)
            
            # Check received data
            data = hub.get_market_data()
            print(f"📈 Received data for {len(data)} symbols")
            
            for symbol, market_data in data.items():
                if isinstance(market_data, dict) and 'ltp' in market_data:
                    print(f"   {symbol}: ₹{market_data['ltp']:.2f}")
            
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        finally:
            hub.stop()
            print("🛑 Real-Time Data Hub stopped")
    else:
        print("❌ Failed to start Real-Time Data Hub")