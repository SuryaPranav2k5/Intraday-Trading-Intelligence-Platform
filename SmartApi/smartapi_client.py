"""
SmartAPI Client for live market data streaming
"""
import os
import logging
import pyotp
from SmartApi import SmartConnect as SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from dotenv import load_dotenv
import time
import json
import threading
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union

# Load environment variables from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartAPIClient")

class SmartAPIClient:
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SmartAPIClient, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        # Prevent re-initialization of the singleton
        if self._initialized:
            return
            
        self.api: Optional[SmartConnect] = None
        self.websocket: Optional[SmartWebSocketV2] = None
        self.connected: bool = False
        self._feed_opened: bool = False
        self.jwt_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.feed_token: Optional[str] = None
        # Use absolute path to backend directory to avoid duplicate token files
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        self.token_file: str = os.path.join(backend_dir, "smartapi_daily_token.json")
        
        # Thread safety lock for feed_opened flag and other shared state
        self._state_lock = threading.RLock()
        
        # Connection management lock to prevent concurrent connections
        self._connection_lock = threading.RLock()
        
        # WebSocket management lock
        self._websocket_lock = threading.RLock()
        
        self._initialized = True
        
        # Get credentials from environment variables with proper type handling
        self.api_key: str | None = os.getenv("API_KEY")
        self.client_id: str | None = os.getenv("CLIENT_ID")
        self.password: str | None = os.getenv("PASSWORD")
        self.totp_secret: str | None = os.getenv("TOTP_SECRET")
        
        # Check for placeholder values
        placeholder_values = ["your_angel_api_key", "your_client_id", "your_password", "YOUR_TOTP_SECRET_KEY_FROM_ANGEL_ONE"]
        
        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            logger.warning("SmartAPI credentials not found in environment variables")
        elif any(cred in placeholder_values for cred in [self.api_key, self.client_id, self.password, self.totp_secret]):
            logger.error("Placeholder values detected in .env file. Please update with your actual SmartAPI credentials.")
        else:
            # Try to load existing token on initialization
            self._load_daily_token()
    
    @property
    def feed_opened(self):
        """Thread-safe getter for feed_opened flag"""
        with self._state_lock:
            return self._feed_opened
    
    @feed_opened.setter
    def feed_opened(self, value):
        """Thread-safe setter for feed_opened flag"""
        with self._state_lock:
            self._feed_opened = value
    
    def _load_daily_token(self) -> bool:
        """Load token from file if it's from today"""
        try:
            if not os.path.exists(self.token_file):
                return False
                
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            # Check if token is from today
            token_date = token_data.get('date')
            today = str(date.today())
            
            if token_date == today:
                self.jwt_token = token_data.get('jwt_token')
                self.refresh_token = token_data.get('refresh_token')
                self.feed_token = token_data.get('feed_token')
                
                # Create SmartConnect object with existing token
                if self.api_key is not None:
                    self.api = SmartConnect(api_key=self.api_key)
                else:
                    logger.error("API key is None, cannot create SmartConnect object")
                    return False
                
                # CRITICAL: Remove "Bearer " prefix if present before calling setAccessToken
                # SmartConnect.setAccessToken() adds "Bearer " prefix automatically
                token_to_set = self.jwt_token.replace("Bearer ", "") if self.jwt_token and self.jwt_token.startswith("Bearer ") else self.jwt_token
                
                # Set the access token in the API object for all requests
                self.api.setSessionExpiryHook(self._session_expired_hook)
                self.api.setAccessToken(token_to_set)
                
                logger.info(f"✅ Loaded existing token from {token_date} and set access token")
                self.connected = True
                return True
            else:
                logger.info(f"Token expired (from {token_date}), generating new token")
                # Remove old token file
                os.remove(self.token_file)
                return False
                
        except Exception as e:
            logger.error(f"Error loading daily token: {e}")
            return False
    
    def _save_daily_token(self) -> bool:
        """Save token to file with today's date
        Note: Strips 'Bearer ' prefix before saving to avoid double-prefix issues
        """
        try:
            # Strip "Bearer " prefix if present before saving
            # This prevents double "Bearer Bearer " when loading and using the token
            jwt_token_clean = self.jwt_token.replace("Bearer ", "") if self.jwt_token and self.jwt_token.startswith("Bearer ") else self.jwt_token
            
            token_data = {
                'date': str(date.today()),
                'jwt_token': jwt_token_clean,
                'refresh_token': self.refresh_token,
                'feed_token': self.feed_token,
                'timestamp': time.time()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            logger.info(f"Token saved for {token_data['date']} (without Bearer prefix)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving daily token: {e}")
            return False
    
    def connect(self) -> bool:
        """Establish connection with SmartAPI using daily token persistence"""
        with self._connection_lock:
            # Check if already connected
            if self.connected:
                logger.info("SmartAPI already connected")
                return True
                
            try:
                # Always generate fresh token for reliable connection - NO MOCK DATA
                logger.info("Generating fresh SmartAPI token for REAL market data...")
                
                if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
                    logger.error("Missing SmartAPI credentials")
                    return False
                
                # Check for placeholder values
                placeholder_values = ["your_angel_api_key", "your_client_id", "your_password", "YOUR_TOTP_SECRET_KEY_FROM_ANGEL_ONE"]
                if any(cred in placeholder_values for cred in [self.api_key, self.client_id, self.password, self.totp_secret]):
                    logger.error("Placeholder values detected in .env file. Please update with your actual SmartAPI credentials.")
                    return False
                    
                # Generate TOTP
                try:
                    if self.totp_secret is None:
                        logger.error("TOTP_SECRET is None")
                        return False
                    totp = pyotp.TOTP(self.totp_secret).now()
                    logger.info("Generated TOTP for authentication")
                except Exception as totp_error:
                    logger.error(f"Invalid TOTP_SECRET format: {totp_error}")
                    logger.error("TOTP_SECRET must be a valid base32 encoded string from Angel One")
                    return False
                
                # Create SmartConnect object
                if self.api_key is None:
                    logger.error("API_KEY is None")
                    return False
                self.api = SmartConnect(api_key=self.api_key)
                
                # Login using client credentials
                if self.client_id is None or self.password is None:
                    logger.error("CLIENT_ID or PASSWORD is None")
                    return False
                    
                login_data = self.api.generateSession(self.client_id, self.password, totp)
                
                # Type-safe access to login_data
                if isinstance(login_data, dict) and login_data.get('status'):
                    data_section = login_data.get('data')
                    if isinstance(data_section, dict):
                        # Get tokens from response
                        # Note: SmartConnect adds "Bearer " prefix to jwtToken in the response
                        self.jwt_token = data_section.get('jwtToken')
                        self.refresh_token = data_section.get('refreshToken')
                        self.feed_token = self.api.getfeedToken()
                        
                        # CRITICAL: generateSession() already calls setAccessToken() internally
                        # But we need to ensure it's set correctly for historical data requests
                        # Remove "Bearer " prefix if present before calling setAccessToken
                        token_to_set = self.jwt_token.replace("Bearer ", "") if self.jwt_token and self.jwt_token.startswith("Bearer ") else self.jwt_token
                        self.api.setAccessToken(token_to_set)
                        logger.info("✅ Access token set in API object")
                        
                        # Save token for the day
                        self._save_daily_token()
                        
                        logger.info("Successfully connected to SmartAPI with new daily token")
                        self.connected = True
                        return True
                    else:
                        logger.error("Invalid login response structure: missing data section")
                        self.connected = False
                        return False
                else:
                    error_message = "Unknown error"
                    if isinstance(login_data, dict):
                        error_message = login_data.get('message', 'Unknown error')
                    logger.error(f"Failed to connect to SmartAPI: {error_message}")
                    self.connected = False
                    return False
                    
            except Exception as e:
                logger.error(f"FATAL error during SmartAPI connection: {e}", exc_info=True)
                self.connected = False
                return False
    
    def init_websocket(self, on_message_callback=None, on_open_callback=None, on_error_callback=None, on_close_callback=None, 
                      on_control_message_callback=None, max_retry_attempt=3, retry_strategy=1, retry_delay=10, retry_multiplier=2):
        """Initialize SmartAPI WebSocket V2 connection with enhanced features
        Args:
            on_message_callback: Callback for market data messages
            on_open_callback: Callback when connection opens
            on_error_callback: Callback for errors
            on_close_callback: Callback when connection closes
            on_control_message_callback: Callback for control messages (ping/pong)
            max_retry_attempt: Maximum retry attempts (default: 3)
            retry_strategy: 0=simple retry, 1=exponential backoff (default: 1)
            retry_delay: Initial retry delay in seconds (default: 10)
            retry_multiplier: Multiplier for exponential backoff (default: 2)
        """
        with self._websocket_lock:
            if not self.connected:
                logger.error("Cannot initialize WebSocket: Not connected to SmartAPI")
                return False
                
            if not self.feed_token:
                logger.error("Cannot initialize WebSocket: No feed token available")
                return False
                
            # Check if WebSocket is already initialized
            if self.websocket is not None:
                logger.info("WebSocket already initialized")
                return True
            
        try:
            logger.info(f"🔧 Initializing SmartWebSocketV2 with enhanced features...")
            logger.info(f"   📊 Feed Token: {self.feed_token[:10]}...")
            logger.info(f"   🔄 Max Retries: {max_retry_attempt}")
            logger.info(f"   ⚡ Retry Strategy: {'Exponential' if retry_strategy == 1 else 'Simple'}")
            
            # Create WebSocket V2 object with retry configuration
            # Ensure all required parameters are not None
            if not all([self.jwt_token, self.api_key, self.client_id, self.feed_token]):
                logger.error("Missing required parameters for WebSocket initialization")
                return False
                
            self.websocket = SmartWebSocketV2(
                auth_token=self.jwt_token,
                api_key=self.api_key,
                client_code=self.client_id,
                feed_token=self.feed_token,
                max_retry_attempt=max_retry_attempt,
                retry_strategy=retry_strategy,
                retry_delay=retry_delay,
                retry_multiplier=retry_multiplier,
                retry_duration=60  # 60 minutes timeout
            )
            
            # Enhanced callback wrappers with better error handling
            def safe_on_open(wsapp):
                try:
                    logger.info("🟢 SmartAPI WebSocket V2 opened successfully")
                    self.feed_opened = True
                    if on_open_callback:
                        on_open_callback(wsapp)
                    else:
                        self._default_on_open(wsapp)
                except Exception as e:
                    logger.error(f"❌ Error in on_open callback: {e}")
            
            def safe_on_close(wsapp, *args):
                try:
                    logger.info("🔴 SmartAPI WebSocket V2 closed")
                    self.feed_opened = False
                    # Don't set connected=False here as it might be a temporary disconnection
                    if on_close_callback:
                        on_close_callback(wsapp)
                    else:
                        self._default_on_close(wsapp, *args)
                except Exception as e:
                    logger.error(f"❌ Error in on_close callback: {e}")
            
            def safe_on_error(error_type, error_message):
                try:
                    logger.error(f"⚠️ SmartAPI WebSocket V2 error: {error_type} - {error_message}")
                    if on_error_callback:
                        # Convert to the format the user callback expects
                        on_error_callback(self.websocket.wsapp if self.websocket else None, f"{error_type}: {error_message}")
                    else:
                        self._default_on_error(error_type, error_message)
                except Exception as e:
                    logger.error(f"❌ Error in on_error callback: {e}")
            
            def safe_on_data(wsapp, data):
                try:
                    if on_message_callback:
                        on_message_callback(wsapp, data)
                    else:
                        self._default_on_message(wsapp, data)
                except Exception as e:
                    logger.error(f"❌ Error in on_data callback: {e}")
            
            def safe_on_control_message(wsapp, message):
                try:
                    logger.debug(f"🎛️ Control message: {message}")
                    if on_control_message_callback:
                        on_control_message_callback(wsapp, message)
                    else:
                        self._default_on_control_message(wsapp, message)
                except Exception as e:
                    logger.error(f"❌ Error in on_control_message callback: {e}")
            
            # Assign enhanced callback functions
            self.websocket.on_open = safe_on_open
            self.websocket.on_close = safe_on_close
            self.websocket.on_error = safe_on_error
            self.websocket.on_data = safe_on_data
            self.websocket.on_control_message = safe_on_control_message
            
            logger.info("✅ SmartAPI WebSocket V2 initialized with enhanced features")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing WebSocket V2: {e}", exc_info=True)
            return False
    
    def connect_websocket(self, max_retries=3, retry_delay=5):
        """Connect to WebSocket with retry logic - runs in background thread"""
        with self._websocket_lock:
            if not self.websocket:
                logger.error("WebSocket not initialized")
                return False
                
            # Check if already connected
            if self.feed_opened:
                logger.info("WebSocket already connected")
                return True
                
            logger.info("Attempting to connect SmartAPI WebSocket...")
            
            # Run WebSocket connection in a background thread since it's blocking
            def _connect_in_thread():
                retries = 0
                while retries < max_retries:
                    try:
                        logger.info(f"WebSocket connection attempt {retries + 1}/{max_retries}")
                        self.websocket.connect()  # This is a blocking call
                        logger.info("SmartAPI WebSocket connected successfully")
                        return True
                    except Exception as e:
                        retries += 1
                        logger.error(f"Error connecting to WebSocket (attempt {retries}/{max_retries}): {e}", exc_info=True)
                        if retries < max_retries:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                
                logger.error("Failed to connect to WebSocket after maximum retries")
                return False
            
            # Start connection in background thread
            ws_thread = threading.Thread(target=_connect_in_thread, daemon=True, name="SmartAPI-WebSocket")
            ws_thread.start()
            
            # Wait a moment for connection to establish
            time.sleep(2)
            
            # Check if connection was successful
            if self.feed_opened:
                logger.info("✅ WebSocket connection established in background")
                return True
            else:
                logger.warning("⚠️ WebSocket connection started but not yet opened (may take a few seconds)")
                return True  # Return True as connection is in progress
    
    def subscribe(self, token_list, mode=1, correlation_id=None):
        """Subscribe to market data feeds using SmartWebSocketV2 enhanced features
        Args:
            token_list: List of token groups in SmartAPI format [{"exchangeType": 1, "tokens": ["token1", "token2"]}]
            mode: Subscription mode
                  1 -> LTP_MODE (Last Traded Price)
                  2 -> QUOTE (Basic quote data)
                  3 -> SNAP_QUOTE (Detailed quote with best 5 buy/sell)
                  4 -> DEPTH (20-level market depth - NSE only)
            correlation_id: Optional 10-character alphanumeric ID for tracking requests
        """
        if not self.websocket:
            logger.error("WebSocket not initialized")
            return False
            
        try:
            # Generate correlation ID if not provided
            if not correlation_id:
                correlation_id = f"REQ{int(time.time())}"[-10:]  # Last 10 chars of timestamp
            
            # Count total tokens for logging
            total_tokens = 0
            for token_group in token_list:
                if 'tokens' in token_group:
                    total_tokens += len(token_group['tokens'])
            
            # Mode validation and logging
            mode_names = {1: "LTP_MODE", 2: "QUOTE", 3: "SNAP_QUOTE", 4: "DEPTH"}
            mode_name = mode_names.get(mode, f"UNKNOWN({mode})")
            logger.info(f"Subscribing to {total_tokens} tokens in {mode_name} mode (ID: {correlation_id})")
            
            # DEPTH mode validation - only NSE supported
            if mode == 4:  # DEPTH mode
                for token_group in token_list:
                    if token_group.get('exchangeType') != 1:  # NSE_CM = 1
                        logger.error(f"DEPTH mode only supports NSE (exchangeType=1), got {token_group.get('exchangeType')}")
                        return False
                
                # DEPTH mode has a 50 token limit
                if total_tokens > 50:
                    logger.error(f"DEPTH mode supports maximum 50 tokens, requested {total_tokens}")
                    return False
            
            # BATCH SUBSCRIPTION: SmartAPI V2 limits
            MAX_TOKENS_PER_BATCH = 50 if mode != 4 else 50  # DEPTH already validated above
            
            if total_tokens <= MAX_TOKENS_PER_BATCH:
                # Single subscription for small lists
                self.websocket.subscribe(correlation_id, mode, token_list)
                logger.info(f"✅ Subscribed to {total_tokens} instruments in single batch")
            else:
                # Batch subscription for large lists
                logger.info(f"📦 Large subscription detected, batching into smaller groups...")
                subscribed_count = 0
                batch_num = 1
                
                for token_group in token_list:
                    if 'tokens' in token_group:
                        tokens = token_group['tokens']
                        exchange_type = token_group['exchangeType']
                        
                        # Split tokens into batches
                        for i in range(0, len(tokens), MAX_TOKENS_PER_BATCH):
                            batch_tokens = tokens[i:i + MAX_TOKENS_PER_BATCH]
                            batch_group = [{
                                "exchangeType": exchange_type,
                                "tokens": batch_tokens
                            }]
                            
                            # Generate unique correlation ID for each batch
                            batch_correlation_id = f"{correlation_id}B{batch_num}"[-10:]
                            
                            self.websocket.subscribe(batch_correlation_id, mode, batch_group)
                            subscribed_count += len(batch_tokens)
                            logger.info(f"✅ Batch {batch_num}: {len(batch_tokens)} tokens (total: {subscribed_count}/{total_tokens})")
                            
                            batch_num += 1
                            # Small delay between batches to avoid overwhelming the server
                            time.sleep(0.1)
                
                logger.info(f"🎯 Completed batched subscription: {subscribed_count} instruments")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error subscribing to market data: {e}", exc_info=True)
            return False
    
    def _session_expired_hook(self):
        """Handle session expiry"""
        logger.warning("SmartAPI session expired, will need to reconnect")
        self.connected = False
    
    def disconnect(self):
        """Disconnect from SmartAPI"""
        with self._connection_lock:
            try:
                if self.websocket:
                    with self._websocket_lock:
                        self.websocket.close_connection()
                        self.websocket = None
                self.connected = False
                self.feed_opened = False
                logger.info("Disconnected from SmartAPI")
            except Exception as e:
                logger.error(f"Error disconnecting from SmartAPI: {e}", exc_info=True)
    
    def unsubscribe(self, token_list, mode=1, correlation_id=None):
        """Unsubscribe from market data feeds
        Args:
            token_list: List of token groups to unsubscribe from
            mode: Subscription mode to unsubscribe from
            correlation_id: Optional correlation ID for tracking
        """
        if not self.websocket:
            logger.error("WebSocket not initialized")
            return False
            
        try:
            if not correlation_id:
                correlation_id = f"UNSUB{int(time.time())}"[-10:]
            
            total_tokens = sum(len(group.get('tokens', [])) for group in token_list)
            logger.info(f"🚫 Unsubscribing from {total_tokens} tokens (ID: {correlation_id})")
            
            self.websocket.unsubscribe(correlation_id, mode, token_list)
            logger.info("✅ Unsubscription request sent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error unsubscribing: {e}", exc_info=True)
            return False
    
    def get_subscription_modes(self):
        """Get available subscription modes"""
        return {
            "LTP_MODE": 1,      # Last Traded Price only
            "QUOTE": 2,         # Basic quote data (LTP + volume + OHLC)
            "SNAP_QUOTE": 3,    # Detailed quote with best 5 buy/sell
            "DEPTH": 4          # 20-level market depth (NSE only, max 50 tokens)
        }
    
    def get_exchange_types(self):
        """Get available exchange types"""
        return {
            "NSE_CM": 1,    # NSE Cash Market
            "NSE_FO": 2,    # NSE Futures & Options
            "MCX_FO": 5,    # MCX Futures & Options
            "NCX_FO": 7,    # NCX Futures & Options
            "CDE_FO": 13    # CDE Futures & Options
        }
    
    # Enhanced default callback methods
    def _default_on_open(self, wsapp):
        """Enhanced default WebSocket open callback"""
        logger.info("🟢 SmartAPI WebSocket V2 connection established")
        self.feed_opened = True
        logger.info("📡 Ready to receive market data streams")
    
    def _default_on_close(self, wsapp, *args):
        """Enhanced default WebSocket close callback"""
        logger.info("🔴 SmartAPI WebSocket V2 connection closed")
        self.feed_opened = False
        if args:
            logger.debug(f"Close arguments: {args}")
    
    def _default_on_error(self, error_type, error_message):
        """Enhanced default WebSocket error callback"""
        logger.error(f"⚠️ SmartAPI WebSocket V2 error: {error_type} - {error_message}")
        # Check if it's a connection error that might trigger retry
        if "connection" in str(error_message).lower() or "reconnect" in str(error_type).lower():
            logger.info("🔄 Connection error detected - automatic retry will be attempted")
    
    def _default_on_message(self, wsapp, message):
        """Enhanced default WebSocket message callback with data parsing"""
        try:
            if isinstance(message, dict):
                # Parsed market data
                token = message.get('token', 'Unknown')
                mode = message.get('subscription_mode_val', 'Unknown')
                ltp = message.get('last_traded_price', 0)
                
                if ltp > 0:
                    # Convert from paise to rupees if needed (SmartAPI sends in paise)
                    ltp_formatted = ltp / 100.0 if ltp > 1000 else ltp
                    logger.debug(f"📊 {token} ({mode}): ₹{ltp_formatted:.2f}")
                else:
                    logger.debug(f"📊 {token} ({mode}): {message}")
            else:
                logger.debug(f"📨 Raw message: {str(message)[:100]}...")
                
            # Store last message for inspection
            self.last_message = message
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    def _default_on_control_message(self, wsapp, message):
        """Default control message callback for ping/pong handling"""
        try:
            msg_type = message.get('type', 'unknown')
            if msg_type == 'ping':
                logger.debug("🏓 Received ping - connection alive")
            elif msg_type == 'pong':
                logger.debug("🏓 Received pong - heartbeat confirmed")
            else:
                logger.debug(f"🎛️ Control message: {message}")
        except Exception as e:
            logger.error(f"❌ Error handling control message: {e}")
    
    def get_historical_candles(self, exchange: str, symbol_token: str, interval: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch historical candle data from SmartAPI with retry logic
        Args:
            exchange: Exchange string ('NSE', 'NFO', etc.)
            symbol_token: Instrument token
            interval: Time interval ('ONE_MINUTE', 'FIVE_MINUTE', etc.)
            from_date: Start date string 'YYYY-MM-DD HH:MM'
            to_date: End date string 'YYYY-MM-DD HH:MM'
        Returns:
            List of candle dictionaries or empty list on failure
        """
        if not self.connected or not self.api:
            logger.error("Cannot fetch history: SmartAPI not connected")
            return []

        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date
        }
        
        # Retry logic for rate limiting (AB1004 errors)
        max_retries = 3
        base_delay = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(f"Retrying historical data fetch for token {symbol_token} (attempt {attempt + 1}/{max_retries}) after {delay}s delay")
                    time.sleep(delay)
                else:
                    logger.info(f"Fetching historical data for token {symbol_token} from {from_date} to {to_date}")
                
                response = self.api.getCandleData(params)
                
                if response and response.get('status') and response.get('data'):
                    # Format: [timestamp, open, high, low, close, volume]
                    raw_data = response.get('data')
                    formatted_data = []
                    for item in raw_data:
                        if len(item) >= 6:
                            formatted_data.append({
                                "timestamp": item[0],
                                "open": float(item[1]),
                                "high": float(item[2]),
                                "low": float(item[3]),
                                "close": float(item[4]),
                                "volume": int(item[5])
                            })
                    logger.info(f"✅ Successfully fetched {len(formatted_data)} candles for token {symbol_token}")
                    return formatted_data
                else:
                    error = response.get('message', 'Unknown error') if response else 'No response'
                    error_code = response.get('errorcode', '') if response else ''
                    
                    # Check if it's a rate limiting error (AB1004)
                    if error_code == 'AB1004' and attempt < max_retries - 1:
                        logger.warning(f"Rate limit error (AB1004) for token {symbol_token}, will retry...")
                        continue
                    else:
                        logger.error(f"Failed to fetch historical data for token {symbol_token}: {error}")
                        return []
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Exception fetching historical data for token {symbol_token} (attempt {attempt + 1}): {e}")
                    continue
                else:
                    logger.error(f"Error fetching historical data for token {symbol_token}: {e}", exc_info=True)
                    return []
        
        return []

    def get_profile(self) -> Optional[Dict[str, Any]]:
        """Get user profile information"""
        if not self.connected or not self.api:
            return None
            
        try:
            profile = self.api.getProfile(self.refresh_token)
            return profile
        except Exception as e:
            logger.error(f"Error getting profile: {e}", exc_info=True)
            return None
    
    def get_available_margin(self):
        """Get available margin"""
        # Removed actual margin fetching since we're using a fixed default value
        # This prevents unnecessary API calls to fetch margin data
        logger.info("Margin fetching disabled - using default capital value")
        return None

# Global singleton instance
smartapi_client = SmartAPIClient()

def get_smartapi_client():
    """Get the global SmartAPI client singleton instance"""
    return smartapi_client

def ensure_single_connection():
    """Ensure only one WebSocket connection is active"""
    global smartapi_client
    if smartapi_client.websocket and smartapi_client.feed_opened:
        logger.warning("WebSocket connection already active - using existing connection")
        return True
    return False

# Add a utility function to demonstrate usage of the global instance
def get_system_margin():
    """Utility function to get system margin using the global SmartAPI client"""
    # Removed actual margin fetching since we're using a fixed default value
    # This prevents unnecessary API calls to fetch margin data
    logger.info("System margin fetching disabled - using default capital value")
    return None

if __name__ == "__main__":
    # Test the SmartAPI client and validate token
    print("Initializing SmartAPI connection...")
    client = SmartAPIClient()
    
    if client.connect():
        print("Successfully connected to SmartAPI")
        print(f"JWT Token obtained: {client.jwt_token[:20]}...")
        
        # Get profile to validate the connection
        profile = client.get_profile()
        if profile and isinstance(profile, dict) and profile.get('status'):
            data_section = profile.get('data')
            if isinstance(data_section, dict):
                user_name = data_section.get('name', 'Unknown')
                print(f"User Profile: {user_name}")
            else:
                print("User Profile: Unknown")
            print("SmartAPI authentication successful!")
        else:
            print("Profile fetch failed, but connection established")
        
        # Token is automatically saved by the connect() method
        
        print("SmartAPI client ready for trading system!")
        
    else:
        print("Failed to connect to SmartAPI")
        print("Please check your credentials in .env file:")
        print("- API_KEY")
        print("- CLIENT_ID") 
        print("- PASSWORD")
        print("- TOTP_SECRET")
        exit(1)