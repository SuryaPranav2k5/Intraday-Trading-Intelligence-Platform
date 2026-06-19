"""
ORDER FLOW HARVESTER 3.0 — Production Grade (1-Year Unattended Operation)

Captures live 5-level market depth, order flow imbalances, and microstructure
features from SmartAPI WebSocket for all 6 priority symbols.

Designed for:
  - Daily auto-authentication (tokens expire every 24h)
  - Weekend/holiday recovery (reconnects every Monday/post-holiday)
  - Thread-safe buffer access (no race conditions)
  - Cross-day price reset (no stale prev_last_price leakage)
  - Graceful disk monitoring

Usage:
  python orderflow_harvester.py

  Leave running permanently. It will sleep outside market hours,
  re-authenticate every morning at 09:10, and harvest 09:15-15:30.
"""

import os
import sys
import time
import logging
import threading
from logging.handlers import RotatingFileHandler
from datetime import datetime, time as datetime_time, timedelta
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pandas_market_calendars as mcal

# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            "orderflow_harvester.log",
            maxBytes=10 * 1024 * 1024,  # 10MB per file
            backupCount=5               # Keep last 5 = max 50MB total
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Harvester")

load_dotenv(os.path.join(os.path.dirname(__file__), 'SmartApi', '.env'))
API_KEY = os.getenv("API_KEY")
CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")

TARGET_SYMBOLS = {
    'TVSMOTOR': '8479',
    'RELIANCE': '2885',
    'LT': '11483',
    'TITAN': '3506',
    'SIEMENS': '3150',
    'TATAELXSI': '3411',
}

HARVEST_DIR = Path("market_data/orderflow")
HARVEST_DIR.mkdir(parents=True, exist_ok=True)

SAVE_INTERVAL_SECONDS = 300  # Flush buffer to disk every 5 minutes
DISK_WARN_GB = 5  # Warn if free disk space drops below this

# ============================================================
# THREAD-SAFE BUFFER
# ============================================================

buffer_lock = threading.Lock()
data_buffer = {sym: [] for sym in TARGET_SYMBOLS.keys()}


def save_buffer_to_disk():
    """Thread-safe flush of tick buffer to daily parquet files."""
    today = datetime.now().strftime("%Y-%m-%d")

    with buffer_lock:
        snapshot = {sym: list(ticks) for sym, ticks in data_buffer.items()}
        for sym in data_buffer:
            data_buffer[sym] = []

    saved_any = False
    for symbol, ticks in snapshot.items():
        if not ticks:
            continue

        df_new = pd.DataFrame(ticks)
        file_path = HARVEST_DIR / f"{symbol}_orderflow_{today}.parquet"

        if file_path.exists():
            try:
                df_old = pd.read_parquet(file_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
            except Exception as e:
                logger.error(f"Corrupted parquet {file_path.name}, starting fresh: {e}")
                df_combined = df_new
            df_combined.to_parquet(file_path, index=False)
        else:
            df_new.to_parquet(file_path, index=False)

        logger.info(f"Saved {len(ticks)} ticks to {file_path.name}")
        saved_any = True

    if not saved_any:
        logger.debug("Buffer empty, nothing to save.")


def check_disk_space():
    """Log a warning if free disk space is critically low."""
    import shutil
    total, used, free = shutil.disk_usage(HARVEST_DIR)
    free_gb = free / (1024 ** 3)
    if free_gb < DISK_WARN_GB:
        logger.warning(f"LOW DISK SPACE: Only {free_gb:.1f} GB remaining!")
    else:
        logger.debug(f"Disk space OK: {free_gb:.1f} GB free.")


# ============================================================
# AUTHENTICATION
# ============================================================

def generate_totp(secret):
    """Generate TOTP token for SmartAPI login."""
    try:
        import pyotp
        return pyotp.TOTP(secret).now()
    except ImportError:
        logger.error("pyotp not installed. Run: pip install pyotp")
        sys.exit(1)


def authenticate():
    """Authenticate with SmartAPI and return (api, auth_token, feed_token).
    Retries up to 3 times on failure."""
    for attempt in range(1, 4):
        try:
            logger.info(f"Authenticating with SmartAPI (attempt {attempt}/3)...")
            smart_api = SmartConnect(api_key=API_KEY)
            totp = generate_totp(TOTP_SECRET)
            data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp)

            if data['status']:
                logger.info("Login Successful!")
                auth_token = data['data']['jwtToken']
                feed_token = data['data']['feedToken']
                return smart_api, auth_token, feed_token
            else:
                logger.error(f"Login Failed: {data}")
        except Exception as e:
            logger.error(f"Auth error: {e}")

        if attempt < 3:
            logger.info("Retrying in 30 seconds...")
            time.sleep(30)

    logger.critical("All 3 authentication attempts failed. Will retry next cycle.")
    return None, None, None


# ============================================================
# HARVESTER (WebSocket Handler)
# ============================================================

class OrderFlowHarvester:
    def __init__(self, auth_token, feed_token):
        self.feed_token = feed_token
        self.sws = SmartWebSocketV2(
            auth_token=auth_token,
            api_key=API_KEY,
            client_code=CLIENT_ID,
            feed_token=feed_token
        )
        self.token_to_symbol = {token: sym for sym, token in TARGET_SYMBOLS.items()}
        self.prev_last_prices = {sym: 0.0 for sym in TARGET_SYMBOLS}
        self.last_save_time = time.time()
        self._stopped = threading.Event()

        self.sws.on_data = self.on_data
        self.sws.on_open = self.on_open
        self.sws.on_error = self.on_error
        self.sws.on_close = self.on_close

    def reset_daily_state(self):
        """Reset cross-day stale state. Called at market open each day."""
        self.prev_last_prices = {sym: 0.0 for sym in TARGET_SYMBOLS}
        logger.info("Daily state reset (prev_last_prices cleared).")

    def on_data(self, wsapp, message):
        """Process each incoming SnapQuote tick."""
        now = datetime.now().time()
        if not (datetime_time(9, 15) <= now <= datetime_time(15, 30)):
            return

        if 'token' not in message or 'best_5_buy_data' not in message:
            return

        token = message.get('token')
        symbol = self.token_to_symbol.get(token)
        if not symbol:
            return

        last_price = message.get('last_traded_price', 0)
        prev_last_price = self.prev_last_prices[symbol]

        total_buy_qty = message.get('tbq', 0)
        total_sell_qty = message.get('tsq', 0)

        buy_depth = message.get('best_5_buy_data', [])
        sell_depth = message.get('best_5_sell_data', [])

        top_bid = buy_depth[0].get('price', 0) if buy_depth else 0
        top_ask = sell_depth[0].get('price', 0) if sell_depth else 0

        bid_depth_total = sum(b.get('quantity', 0) for b in buy_depth)
        ask_depth_total = sum(a.get('quantity', 0) for a in sell_depth)

        imbalance_ratio = 0
        if total_buy_qty + total_sell_qty > 0:
            imbalance_ratio = (total_buy_qty - total_sell_qty) / (total_buy_qty + total_sell_qty)

        exchange_ts_ms = message.get('exchange_timestamp', int(time.time() * 1000))
        tick_data = {
            'timestamp': pd.Timestamp.fromtimestamp(exchange_ts_ms / 1000.0),
            'exchange_epoch_ms': exchange_ts_ms,
            'last_traded_price': last_price,
            'total_buy_qty': total_buy_qty,
            'total_sell_qty': total_sell_qty,
            'imbalance_ratio': imbalance_ratio,
            'top_bid_price': top_bid,
            'top_ask_price': top_ask,
            'volume': message.get('volume_trade_for_the_day', 0),
            'bid_ask_spread': top_ask - top_bid,
            'depth_imbalance': bid_depth_total / (ask_depth_total + 1e-9),
            'bid_depth_total': bid_depth_total,
            'ask_depth_total': ask_depth_total,
            'weighted_mid_price': (top_bid + top_ask) / 2 if (top_bid + top_ask) > 0 else last_price,
            'price_impact': abs(last_price - prev_last_price) if prev_last_price > 0 else 0,
            'tick_direction': 1 if last_price > prev_last_price else (-1 if last_price < prev_last_price else 0),
        }

        # Full 5-level depth expansion
        for i, (b, a) in enumerate(zip(buy_depth, sell_depth), 1):
            tick_data[f'bid_price_{i}'] = b.get('price', 0)
            tick_data[f'bid_qty_{i}'] = b.get('quantity', 0)
            tick_data[f'ask_price_{i}'] = a.get('price', 0)
            tick_data[f'ask_qty_{i}'] = a.get('quantity', 0)

        if last_price > 0:
            self.prev_last_prices[symbol] = last_price

        with buffer_lock:
            data_buffer[symbol].append(tick_data)

        # Periodic flush
        if time.time() - self.last_save_time > SAVE_INTERVAL_SECONDS:
            save_buffer_to_disk()
            self.last_save_time = time.time()

    def on_open(self, wsapp):
        logger.info("WebSocket Connected! Subscribing to SnapQuote (Mode 3)...")
        token_list = [{"exchangeType": 1, "tokens": list(TARGET_SYMBOLS.values())}]
        self.sws.subscribe("corrid", 3, token_list)

    def on_error(self, wsapp, error):
        logger.error(f"WebSocket Error: {error}")
        now = datetime.now().time()
        if datetime_time(9, 15) <= now <= datetime_time(15, 30):
            time.sleep(5)
            logger.info("Attempting WebSocket reconnect...")
            try:
                self.sws.connect()
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")
                self._stopped.set()
        else:
            logger.info("Outside market hours — skipping reconnect.")
            self._stopped.set()

    def on_close(self, wsapp):
        logger.warning("WebSocket Closed.")
        save_buffer_to_disk()
        self._stopped.set()

    def start(self):
        """Start the blocking WebSocket connection."""
        self._stopped.clear()
        logger.info("Starting WebSocket stream...")
        try:
            self.sws.connect()
        except Exception as e:
            logger.error(f"WebSocket connect failed: {e}")
            self._stopped.set()

    def wait_until_stopped(self):
        """Block until the WebSocket session ends."""
        self._stopped.wait()


# ============================================================
# DAILY LIFECYCLE LOOP
# ============================================================

def sleep_until(target_time: datetime_time):
    """Sleep until a specific time today (or tomorrow if already past)."""
    now = datetime.now()
    target = now.replace(
        hour=target_time.hour,
        minute=target_time.minute,
        second=0, microsecond=0
    )
    if target <= now:
        target += timedelta(days=1)

    delta = (target - now).total_seconds()
    logger.info(f"Sleeping until {target.strftime('%Y-%m-%d %H:%M')} ({delta / 60:.0f} minutes)...")
    time.sleep(delta)


def is_trading_day(date=None):
    """Check if given date is an NSE trading day (handles holidays + weekends)."""
    if date is None:
        date = datetime.now().date()
    nse = mcal.get_calendar('NSE')
    schedule = nse.schedule(start_date=str(date), end_date=str(date))
    return not schedule.empty


def run_daily_session():
    """
    Run a single day's harvesting session:
      1. Authenticate with fresh token
      2. Reset daily state
      3. Stream 09:15 - 15:35
      4. Save final buffer
      5. Check disk space
    """
    api, auth_token, feed_token = authenticate()
    if not feed_token:
        logger.error("Skipping today's session — authentication failed.")
        return

    harvester = OrderFlowHarvester(auth_token, feed_token)
    harvester.reset_daily_state()

    # Start WebSocket in a separate thread so we can monitor time
    ws_thread = threading.Thread(target=harvester.start, daemon=True)
    ws_thread.start()

    # Monitor loop: keep running until 15:35, then gracefully stop
    try:
        while True:
            now = datetime.now().time()

            # Past 15:35 — market is over, end this session
            if now >= datetime_time(15, 35):
                logger.info("Market session over (15:35). Ending today's harvest.")
                break

            # If WebSocket died mid-session, stop waiting
            if harvester._stopped.is_set():
                logger.warning("WebSocket stopped unexpectedly during market hours.")
                break

            time.sleep(10)  # Check every 10 seconds

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt during session.")

    # Final flush
    save_buffer_to_disk()
    check_disk_space()
    logger.info("Daily session complete.\n")


def main():
    """
    Outer infinite loop — runs one session per trading day, forever.
    Handles weekends, holidays (via auth failure), and token refresh.
    """
    print("=" * 60)
    print("  ORDER FLOW HARVESTER 3.0 — Production Grade")
    print("  Designed for 1-year unattended operation")
    print("=" * 60)
    print(f"  Symbols:  {', '.join(TARGET_SYMBOLS.keys())}")
    print(f"  Output:   {HARVEST_DIR.resolve()}")
    print(f"  Interval: Save every {SAVE_INTERVAL_SECONDS}s")
    print("=" * 60)

    while True:
        # Skip non-trading days (weekends + NSE holidays)
        if not is_trading_day():
            # Find next trading day
            next_day = datetime.now().date() + timedelta(days=1)
            while not is_trading_day(next_day):
                next_day += timedelta(days=1)

            wake_time = datetime.combine(next_day, datetime_time(9, 10))
            delta = (wake_time - datetime.now()).total_seconds()

            logger.info(
                f"Non-trading day ({datetime.now().strftime('%A')}). "
                f"Next trading day: {next_day}. Sleeping..."
            )
            if delta > 0:
                time.sleep(delta)
            continue

        now = datetime.now().time()

        # If before 09:10, sleep until 09:10 (authenticate 5 min before market open)
        if now < datetime_time(9, 10):
            sleep_until(datetime_time(9, 10))

        # If after 15:35, today is done — sleep until next trading day 09:10
        elif now >= datetime_time(15, 35):
            logger.info("Past market close. Sleeping until next trading day 09:10...")
            next_day = datetime.now().date() + timedelta(days=1)
            while not is_trading_day(next_day):
                next_day += timedelta(days=1)
            wake_time = datetime.combine(next_day, datetime_time(9, 10))
            delta = (wake_time - datetime.now()).total_seconds()
            if delta > 0:
                time.sleep(delta)
            continue

        # Run today's session
        logger.info(f"=== Starting session for {datetime.now().strftime('%Y-%m-%d (%A)')} ===")
        run_daily_session()

        # After session ends, sleep until next trading day's 09:10
        logger.info("Session finished. Sleeping until next trading day 09:10...")
        next_day = datetime.now().date() + timedelta(days=1)
        while not is_trading_day(next_day):
            next_day += timedelta(days=1)
        wake_time = datetime.combine(next_day, datetime_time(9, 10))
        delta = (wake_time - datetime.now()).total_seconds()
        if delta > 0:
            time.sleep(delta)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nManual shutdown. Saving remaining buffer...")
        save_buffer_to_disk()
        print("Harvester stopped cleanly.")
