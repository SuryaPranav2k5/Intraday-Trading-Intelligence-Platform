"""
Phase 2: Professional Exit Engine (Institutional-Grade)

This Phase-2 does NOT predict price.
It monitors the ORIGINAL HYPOTHESIS and exits when it degrades.

The 3 Core Layers:
1. Level State Awareness - Modulates exits near key levels
2. Failure Mode Awareness - Detects when hypothesis is invalid
3. Trade Quality Memory - Session-level adaptation

Exit Philosophy:
"Is the original reason for this trade still valid?"
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, Dict, List
from enum import Enum
import numpy as np


# ============================================================
# ENUMS & CONSTANTS
# ============================================================

class LevelState(Enum):
    FRESH = "FRESH"           # touch_count <= 1
    WEAKENING = "WEAKENING"   # touch_count 2-3
    EXHAUSTED = "EXHAUSTED"   # touch_count >= 4
    BROKEN = "BROKEN"         # strong close beyond level


class ExitReason(Enum):
    # Hard stops
    MAX_LOSS = "Max loss hit"
    
    # Failure modes
    TIME_FAILURE = "Time failure (hypothesis expired)"
    VOLATILITY_FAILURE = "Volatility failure (environment changed)"
    DIRECTIONAL_FAILURE = "Directional failure (wrong side)"
    ABSORPTION_FAILURE = "Absorption failure (stalled)"
    
    # Profit management
    PROFIT_FLOOR = "Profit floor hit"
    PARTIAL_EXIT = "Partial exit at level"
    TARGET_EXHAUSTION = "Target exhaustion"
    
    # Session management
    TIME_EXIT = "Time exit"
    EOD_EXIT = "End of day exit"
    SESSION_STOP = "Session stopped (bad day)"


# ============================================================
# CONFIGURATION (Tunable Parameters)
# ============================================================

@dataclass
class Phase2Config:
    # Hard stops
    max_loss_atr: float = 1.5
    
    # Profit floor
    profit_floor_trigger_atr: float = 1.0  # Trigger when profit >= this
    profit_floor_level_atr: float = 0.5    # Lock floor at this
    
    # Time limits
    max_duration_minutes: int = 45
    time_failure_multiplier: float = 1.5   # Exit if time > expected * this
    time_failure_progress_threshold: float = 0.3  # Min progress required
    
    # Volatility failure
    volatility_contraction_threshold: float = 0.5  # ATR ratio for compression
    
    # Directional failure
    directional_mae_threshold: float = 0.5  # MAE ratio before directional exit
    
    # Absorption failure
    absorption_candle_count: int = 5   # Candles to check for stall
    absorption_progress_min: float = 0.2  # Min ATR progress in window
    
    # Level interaction
    level_tighten_distance_atr: float = 0.5  # Tighten when within this of level
    level_partial_exit_pct: float = 0.3      # Partial exit at level reaction
    
    # Session management
    consecutive_failures_for_risk_decay: int = 2
    risk_decay_factor: float = 0.5
    min_trades_for_session_stop: int = 3
    rolling_expectancy_threshold: float = 0.0  # Stop if below
    
    # EOD
    eod_exit_time: time = time(15, 20)


# ============================================================
# TRADE CONTEXT (Input from Phase-1)
# ============================================================

@dataclass
class TradeContext:
    """Phase-1 output consumed by Phase-2"""
    entry_prob: float
    expected_mfe_atr: float
    expected_mae_atr: float
    expected_time_to_resolution: float
    risk_multiplier: float
    directional_bias: int  # -1, 0, +1
    session_quality: float
    volatility_regime: int  # 0=low, 1=normal, 2=high
    trend_regime: int  # 0=ranging, 1=weak, 2=strong
    market_state: int  # 0=compression, 1=normal, 2=expansion


# ============================================================
# LEVEL TRACKING
# ============================================================

@dataclass
class LevelInfo:
    """Tracks state of a price level"""
    price: float
    level_type: str  # "session_high", "session_low", "vwap", "htf_high", "htf_low"
    touch_count: int = 0
    last_reaction_strength: float = 0.0  # ATR units
    last_touch_time: Optional[datetime] = None
    state: LevelState = LevelState.FRESH
    
    def update_touch(self, reaction_atr: float, now: datetime):
        self.touch_count += 1
        self.last_reaction_strength = reaction_atr
        self.last_touch_time = now
        
        # Update state
        if self.touch_count <= 1:
            self.state = LevelState.FRESH
        elif self.touch_count <= 3:
            self.state = LevelState.WEAKENING
        else:
            self.state = LevelState.EXHAUSTED
    
    def mark_broken(self):
        self.state = LevelState.BROKEN


# ============================================================
# SESSION MEMORY
# ============================================================

@dataclass
class SessionMemory:
    """Tracks session-level performance for adaptive behavior"""
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_atr: float = 0.0
    consecutive_failures: int = 0
    is_stopped: bool = False
    
    # Recent trade outcomes
    recent_outcomes: List[float] = field(default_factory=list)  # Last N PnLs
    
    @property
    def rolling_expectancy(self) -> float:
        if len(self.recent_outcomes) == 0:
            return 0.0
        return sum(self.recent_outcomes) / len(self.recent_outcomes)
    
    @property
    def win_rate(self) -> float:
        if self.trades_taken == 0:
            return 0.0
        return self.wins / self.trades_taken
    
    def record_trade(self, pnl_atr: float, config: Phase2Config):
        self.trades_taken += 1
        self.total_pnl_atr += pnl_atr
        
        if pnl_atr > 0:
            self.wins += 1
            self.consecutive_failures = 0
        else:
            self.losses += 1
            self.consecutive_failures += 1
        
        # Update rolling window (last 5 trades)
        self.recent_outcomes.append(pnl_atr)
        if len(self.recent_outcomes) > 5:
            self.recent_outcomes.pop(0)
        
        # Check for session stop
        if (self.trades_taken >= config.min_trades_for_session_stop 
            and self.rolling_expectancy < config.rolling_expectancy_threshold):
            self.is_stopped = True
    
    def get_risk_multiplier_adjustment(self, config: Phase2Config) -> float:
        """Returns multiplier based on session performance"""
        if self.consecutive_failures >= config.consecutive_failures_for_risk_decay:
            return config.risk_decay_factor
        return 1.0


# ============================================================
# TRADE STATE
# ============================================================

@dataclass
class TradeState:
    """Complete state of an active trade"""
    symbol: str
    entry_price: float
    entry_time: datetime
    entry_atr: float
    
    # Phase-1 context (hypothesis)
    context: TradeContext
    
    # Trade direction (determined at entry)
    direction: int = 1  # 1=long, -1=short
    
    # Status
    trade_status: str = "IN_TRADE"
    exit_reason: Optional[str] = None
    
    # Running metrics
    minutes_in_trade: int = 0
    unrealized_pnl: float = 0.0
    unrealized_pnl_atr: float = 0.0
    mfe: float = 0.0
    mfe_atr: float = 0.0
    mae: float = 0.0
    mae_atr: float = 0.0
    
    # Profit management
    profit_floor: Optional[float] = None
    partial_exit_taken: bool = False
    
    # For absorption detection
    recent_closes: List[float] = field(default_factory=list)
    
    # Entry volatility state (for failure detection)
    entry_market_state: int = 1  # From Phase-1
    
    # Position size (for backtest/execution)
    quantity: int = 0


# ============================================================
# PHASE-2 ENGINE
# ============================================================

class Phase2Engine:
    """
    Professional Exit Engine
    
    Monitors hypothesis validity and manages exits adaptively.
    Does NOT predict - only validates original trade thesis.
    """
    
    def __init__(self, config: Optional[Phase2Config] = None):
        self.config = config or Phase2Config()
        self.session_memory = SessionMemory()
        self.levels: Dict[str, LevelInfo] = {}  # symbol -> List of levels
    
    def reset_session(self):
        """Call at start of each trading day"""
        self.session_memory = SessionMemory()
        self.levels = {}
    
    def update_levels(self, symbol: str, session_high: float, session_low: float,
                      vwap: float, htf_high: float, htf_low: float):
        """
        Update level tracking from live data.
        FIX #3: Only update prices, preserve touch_count/state memory.
        """
        level_data = [
            (f"{symbol}_session_high", session_high, "session_high"),
            (f"{symbol}_session_low", session_low, "session_low"),
            (f"{symbol}_vwap", vwap, "vwap"),
            (f"{symbol}_htf_high", htf_high, "htf_high"),
            (f"{symbol}_htf_low", htf_low, "htf_low"),
        ]
        
        for key, price, level_type in level_data:
            if key not in self.levels:
                # Create new level
                self.levels[key] = LevelInfo(price, level_type)
            else:
                # Only update price, preserve touch history
                self.levels[key].price = price
    
    def _get_nearby_levels(self, symbol: str, price: float, atr: float) -> List[LevelInfo]:
        """Get levels within interaction distance"""
        nearby = []
        threshold = self.config.level_tighten_distance_atr * atr
        
        for key, level in self.levels.items():
            if key.startswith(symbol):
                if abs(price - level.price) <= threshold:
                    nearby.append(level)
        
        return nearby
    
    def _exit(self, trade: TradeState, reason: ExitReason) -> TradeState:
        trade.trade_status = "EXIT_RECOMMENDED"
        trade.exit_reason = reason.value
        return trade
    
    def _check_hard_stop(self, trade: TradeState) -> Optional[TradeState]:
        """Check if hard stop is hit"""
        max_loss = -self.config.max_loss_atr * trade.entry_atr
        if trade.unrealized_pnl <= max_loss:
            return self._exit(trade, ExitReason.MAX_LOSS)
        return None
    
    def _check_time_failure(self, trade: TradeState) -> Optional[TradeState]:
        """Check if trade has failed on time"""
        expected_time = trade.context.expected_time_to_resolution
        if expected_time <= 0:
            expected_time = 30  # Default
        
        time_threshold = expected_time * self.config.time_failure_multiplier
        progress = trade.mfe_atr / max(trade.context.expected_mfe_atr, 0.1)
        
        if (trade.minutes_in_trade > time_threshold 
            and progress < self.config.time_failure_progress_threshold):
            return self._exit(trade, ExitReason.TIME_FAILURE)
        return None
    
    def _check_market_state_failure(self, trade: TradeState, current_market_state: int) -> Optional[TradeState]:
        """
        FIX #2: Renamed from _check_volatility_failure.
        Check if market state has changed unfavorably (expansion -> compression).
        """
        # Entry was in expansion/normal, now compression
        if trade.entry_market_state >= 1 and current_market_state == 0:
            # Give it some time to develop before exiting
            if trade.minutes_in_trade >= 10 and trade.mfe_atr < 0.5:
                return self._exit(trade, ExitReason.VOLATILITY_FAILURE)
        return None
    
    def _check_directional_failure(self, trade: TradeState) -> Optional[TradeState]:
        """
        FIX #1: Check if trade is against directional bias and losing.
        Uses abs(mae_atr) since mae_atr is already negative.
        """
        # Trade direction doesn't match bias
        if trade.context.directional_bias != 0:
            if trade.direction != trade.context.directional_bias:
                # Wrong direction AND adverse excursion exceeded threshold
                mae_threshold = self.config.directional_mae_threshold * trade.context.expected_mae_atr
                # FIX: mae_atr is negative, use abs() for correct comparison
                if abs(trade.mae_atr) >= mae_threshold:
                    return self._exit(trade, ExitReason.DIRECTIONAL_FAILURE)
        return None
    
    def _check_absorption_failure(self, trade: TradeState) -> Optional[TradeState]:
        """Check if trade is stalling (absorption)"""
        if len(trade.recent_closes) >= self.config.absorption_candle_count:
            recent = trade.recent_closes[-self.config.absorption_candle_count:]
            price_range = max(recent) - min(recent)
            progress_atr = price_range / trade.entry_atr
            
            # Stalling: minimal progress over N candles
            if progress_atr < self.config.absorption_progress_min:
                # Only exit if not already winning
                if trade.unrealized_pnl_atr < 0.5:
                    return self._exit(trade, ExitReason.ABSORPTION_FAILURE)
        return None
    
    def _check_profit_floor(self, trade: TradeState) -> Optional[TradeState]:
        """
        FIX #4: Professional trailing profit floor.
        Floor trails upward as profits increase.
        """
        trigger = self.config.profit_floor_trigger_atr * trade.entry_atr
        
        # Set/update floor when trigger reached - trails upward
        if trade.unrealized_pnl >= trigger:
            # Calculate new floor as percentage of current profit
            # Lock 50% of profits as the floor
            new_floor = trade.unrealized_pnl * 0.5
            
            if trade.profit_floor is None:
                trade.profit_floor = new_floor
            else:
                # FIX: Floor only moves UP, never down (true trailing)
                trade.profit_floor = max(trade.profit_floor, new_floor)
        
        # Check floor violation
        if trade.profit_floor is not None:
            if trade.unrealized_pnl < trade.profit_floor:
                return self._exit(trade, ExitReason.PROFIT_FLOOR)
        return None
    
    def _adjust_trailing_stop(self, trade: TradeState, nearby_levels: List[LevelInfo]) -> float:
        """Calculate dynamic trailing stop based on regime and levels"""
        base_stop_atr = self.config.max_loss_atr
        
        # Tighten near fresh opposing levels
        for level in nearby_levels:
            is_opposing = (
                (trade.direction == 1 and level.price > trade.entry_price) or
                (trade.direction == -1 and level.price < trade.entry_price)
            )
            if is_opposing and level.state == LevelState.FRESH:
                base_stop_atr *= 0.7  # Tighten 30%
        
        # Loosen in strong trends
        if trade.context.trend_regime == 2:  # Strong trend
            base_stop_atr *= 1.2  # Loosen 20%
        
        # Tighten in compression
        if trade.context.market_state == 0:  # Compression
            base_stop_atr *= 0.8
        
        return base_stop_atr
    
    def evaluate(
        self,
        trade: TradeState,
        candle: dict,
        vwap: float,
        now: datetime,
        current_market_state: int = 1
    ) -> TradeState:
        """
        Main evaluation loop - called on each closed 1-min candle.
        
        Decision Flow:
        1. Hard stop hit? -> EXIT
        2. Failure signature? -> EXIT
        3. Profit floor violated? -> EXIT
        4. Partial exit condition? -> PARTIAL
        5. Adjust trailing, continue
        """
        
        # Already exited
        if trade.trade_status != "IN_TRADE":
            return trade
        
        # Session stopped
        if self.session_memory.is_stopped:
            return self._exit(trade, ExitReason.SESSION_STOP)
        
        atr = trade.entry_atr if trade.entry_atr > 0 else 1.0
        close_price = candle["close"]
        
        # ────────────────────────────────────────────────────────────
        # UPDATE RUNNING METRICS
        # ────────────────────────────────────────────────────────────
        trade.minutes_in_trade = int((now - trade.entry_time).total_seconds() / 60)
        trade.unrealized_pnl = (close_price - trade.entry_price) * trade.direction
        trade.unrealized_pnl_atr = trade.unrealized_pnl / atr
        trade.mfe = max(trade.mfe, trade.unrealized_pnl)
        trade.mfe_atr = trade.mfe / atr
        trade.mae = min(trade.mae, trade.unrealized_pnl)
        trade.mae_atr = trade.mae / atr
        
        # Track recent closes for absorption detection
        trade.recent_closes.append(close_price)
        if len(trade.recent_closes) > 10:
            trade.recent_closes.pop(0)
        
        # ────────────────────────────────────────────────────────────
        # 1. HARD STOP
        # ────────────────────────────────────────────────────────────
        result = self._check_hard_stop(trade)
        if result:
            return result
        
        # ────────────────────────────────────────────────────────────
        # 2. FAILURE MODES
        # ────────────────────────────────────────────────────────────
        
        # Time failure
        result = self._check_time_failure(trade)
        if result:
            return result
        
        # Market state failure (FIX #2: renamed from volatility_failure)
        result = self._check_market_state_failure(trade, current_market_state)
        if result:
            return result
        
        # Directional failure
        result = self._check_directional_failure(trade)
        if result:
            return result
        
        # Absorption failure
        result = self._check_absorption_failure(trade)
        if result:
            return result
        
        # ────────────────────────────────────────────────────────────
        # 3. PROFIT FLOOR
        # ────────────────────────────────────────────────────────────
        result = self._check_profit_floor(trade)
        if result:
            return result
        
        # ────────────────────────────────────────────────────────────
        # 4. TIME EXIT
        # ────────────────────────────────────────────────────────────
        if trade.minutes_in_trade >= self.config.max_duration_minutes:
            return self._exit(trade, ExitReason.TIME_EXIT)
        
        # ────────────────────────────────────────────────────────────
        # 5. END OF DAY
        # ────────────────────────────────────────────────────────────
        if now.time() >= self.config.eod_exit_time:
            return self._exit(trade, ExitReason.EOD_EXIT)
        
        # ────────────────────────────────────────────────────────────
        # PARTIAL EXIT DISABLED - User trades 1 lot only
        # ────────────────────────────────────────────────────────────
        # NOTE: Partial exit is only useful with multiple lots.
        # Since we're trading 1 lot, skip this check.
        nearby_levels = self._get_nearby_levels(trade.symbol, close_price, atr)
        
        # ────────────────────────────────────────────────────────────
        # STILL IN TRADE - Adjust trailing stop (for UI display)
        # ────────────────────────────────────────────────────────────
        adjusted_stop_atr = self._adjust_trailing_stop(trade, nearby_levels)
        
        return trade
    
    def record_completed_trade(self, final_pnl_atr: float):
        """Call when trade is closed to update session memory"""
        self.session_memory.record_trade(final_pnl_atr, self.config)
    
    def should_take_trade(self, context: TradeContext) -> tuple[bool, float]:
        """
        Pre-trade check: Should we take this trade?
        Returns (should_trade, adjusted_risk_multiplier)
        """
        if self.session_memory.is_stopped:
            return False, 0.0
        
        if context.session_quality < 0.3:
            return False, 0.0
        
        # Adjust risk based on session performance
        risk_adj = self.session_memory.get_risk_multiplier_adjustment(self.config)
        final_risk = context.risk_multiplier * risk_adj
        
        return True, final_risk
    
    def create_trade(
        self,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
        entry_atr: float,
        context: TradeContext,
        direction: int = 1
    ) -> TradeState:
        """Create a new trade state"""
        return TradeState(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            entry_atr=entry_atr,
            context=context,
            direction=direction,
            entry_market_state=context.market_state
        )


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_engine: Optional[Phase2Engine] = None

def get_phase2_engine() -> Phase2Engine:
    global _engine
    if _engine is None:
        _engine = Phase2Engine()
    return _engine

def reset_phase2_engine():
    global _engine
    _engine = Phase2Engine()