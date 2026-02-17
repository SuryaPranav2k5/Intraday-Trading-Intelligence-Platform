"""
Trade Simulator - Phase 2 Training Data Generator

This module:
1. Loads historical OHLCV data
2. Uses frozen EntryInferenceEngine to find entries
3. Opens trades and tracks full lifecycle
4. Logs state features for every candle after entry
5. Closes trades using rule-based exits
6. Generates training data for exit models

Output:
- trades.csv: One row per trade (summary)
- trade_states.parquet: Millions of rows (per-candle state features)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime, time
from pathlib import Path
import json

from entry_inference import EntryInferenceEngine
from feature_engineering import compute_all_features


# ============================================================
# TRADE DATA STRUCTURE
# ============================================================

@dataclass
class TradeState:
    """State snapshot for one candle during trade lifecycle"""
    step: int  # Bars since entry (0 = entry bar)
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Trade-relative features (EXIT CONTEXT)
    price_from_entry_atr: float
    vwap_from_entry_atr: float
    mfe_atr: float  # Maximum Favorable Excursion
    mae_atr: float  # Maximum Adverse Excursion
    bars_since_entry: int
    pullback_depth: float
    volatility_expansion_ratio: float
    momentum_decay: float
    unrealized_pnl_atr: float
    
    # Current market context
    current_atr: float
    current_vwap: float
    
    # Future labels (will be computed after trade closes)
    bars_to_exit: Optional[int] = None
    final_pnl_atr: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class Trade:
    """Complete trade object with full lifecycle tracking"""
    trade_id: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    
    # Entry information
    entry_time: datetime
    entry_price: float
    entry_idx: int
    initial_atr: float
    entry_vwap: float
    
    # Entry features (frozen snapshot)
    entry_features: Dict[str, float] = field(default_factory=dict)
    
    # Trade lifecycle
    states: List[TradeState] = field(default_factory=list)
    
    # Exit information (filled when trade closes)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_idx: Optional[int] = None
    exit_reason: Optional[str] = None
    
    # Trade results
    pnl_points: Optional[float] = None
    pnl_atr: Optional[float] = None
    bars_held: Optional[int] = None
    mfe: Optional[float] = None  # Best price achieved
    mae: Optional[float] = None  # Worst price achieved
    
    def is_closed(self) -> bool:
        return self.exit_time is not None
    
    def add_state(self, state: TradeState):
        self.states.append(state)
    
    def close_trade(self, exit_time: datetime, exit_price: float, exit_idx: int, exit_reason: str):
        """Close the trade and compute final metrics"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_idx = exit_idx
        self.exit_reason = exit_reason
        self.bars_held = len(self.states)
        
        # Compute P&L
        if self.direction == 'LONG':
            self.pnl_points = exit_price - self.entry_price
        else:  # SHORT
            self.pnl_points = self.entry_price - exit_price
        
        self.pnl_atr = self.pnl_points / self.initial_atr if self.initial_atr > 0 else 0.0
        
        # Compute MFE/MAE from states
        if self.states:
            self.mfe = max(s.mfe_atr for s in self.states)
            self.mae = min(s.mae_atr for s in self.states)
        
        # Update future labels in all states
        total_bars = len(self.states)
        for state in self.states:
            state.bars_to_exit = total_bars - state.step
            state.final_pnl_atr = self.pnl_atr
            state.exit_reason = exit_reason


# ============================================================
# FEATURE COMPUTATION
# ============================================================

class FeatureComputer:
    """Compute trade-relative features for each state"""
    
    @staticmethod
    def compute_state_features(
        trade: Trade,
        step: int,
        current_bar: pd.Series,
        df_slice: pd.DataFrame,
        entry_mfe: float,
        entry_mae: float
    ) -> TradeState:
        """
        Compute all state features for current bar
        
        Args:
            trade: The trade object
            step: Bars since entry (0 = entry bar)
            current_bar: Current OHLCV bar
            df_slice: Historical data from entry to current (for computing context)
            entry_mfe: Running maximum favorable excursion
            entry_mae: Running maximum adverse excursion
        """
        
        current_price = current_bar['close']
        current_vwap = current_bar.get('VWAP', current_price)  # Use VWAP if available
        current_atr = current_bar.get('ATR_14', trade.initial_atr)
        
        # ============================================================
        # ATR CONSISTENCY: Always use trade.initial_atr for normalization
        # current_atr is stored but NOT used as denominator
        # ============================================================
        
        # Price movement from entry (in ATR units) - Use INITIAL_ATR
        price_diff = current_price - trade.entry_price
        if trade.direction == 'SHORT':
            price_diff = -price_diff
        
        price_from_entry_atr = price_diff / trade.initial_atr
        
        # VWAP distance from entry - Use INITIAL_ATR
        vwap_diff = current_vwap - trade.entry_vwap
        if trade.direction == 'SHORT':
            vwap_diff = -vwap_diff
        
        vwap_from_entry_atr = vwap_diff / trade.initial_atr
        
        # MFE/MAE (Maximum Favorable/Adverse Excursion) - Use INITIAL_ATR
        mfe_atr = entry_mfe / trade.initial_atr
        mae_atr = entry_mae / trade.initial_atr
        
        # Pullback depth - Use INITIAL_ATR
        pullback_depth = (entry_mfe - price_diff) / trade.initial_atr if entry_mfe > 0 else 0.0
        
        # Volatility expansion - current_atr / initial_atr (this is correct)
        volatility_expansion_ratio = current_atr / trade.initial_atr if trade.initial_atr > 0 else 1.0
        
        # ============================================================
        # MOMENTUM DECAY - STABILIZED VERSION
        # ============================================================
        if len(df_slice) >= 5:
            recent_momentum = (df_slice['close'].iloc[-1] - df_slice['close'].iloc[-5]) / df_slice['close'].iloc[-5]
            entry_momentum = (df_slice['close'].iloc[min(4, len(df_slice)-1)] - df_slice['close'].iloc[0]) / df_slice['close'].iloc[0] if len(df_slice) >= 5 else 0.01
            
            # Compute ratio with safety checks
            if abs(entry_momentum) > 1e-6:
                momentum_decay_raw = recent_momentum / entry_momentum
                # CRITICAL: Clip to prevent explosion (Transformer stability)
                momentum_decay = np.clip(momentum_decay_raw, -3.0, 3.0)
            else:
                momentum_decay = 0.0
        else:
            momentum_decay = 0.0
        
        # Unrealized P&L - Use INITIAL_ATR
        unrealized_pnl_atr = price_from_entry_atr
        
        return TradeState(
            step=step,
            timestamp=current_bar['timestamp'],
            open=current_bar['open'],
            high=current_bar['high'],
            low=current_bar['low'],
            close=current_bar['close'],
            volume=current_bar['volume'],
            price_from_entry_atr=price_from_entry_atr,
            vwap_from_entry_atr=vwap_from_entry_atr,
            mfe_atr=mfe_atr,
            mae_atr=mae_atr,
            bars_since_entry=step,
            pullback_depth=pullback_depth,
            volatility_expansion_ratio=volatility_expansion_ratio,
            momentum_decay=momentum_decay,
            unrealized_pnl_atr=unrealized_pnl_atr,
            current_atr=current_atr,
            current_vwap=current_vwap
        )


# ============================================================
# EXIT RULES
# ============================================================

class ExitRules:
    """Rule-based exit logic (no ML)"""
    
    def __init__(
        self,
        stop_loss_atr: float = -0.6,
        take_profit_atr: float = 1.0,
        max_bars: int = 60,
        market_close_time: time = time(15, 15)
    ):
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.max_bars = max_bars
        self.market_close_time = market_close_time
    
    def check_exit(
        self,
        trade: Trade,
        current_bar: pd.Series,
        step: int
    ) -> Optional[tuple]:
        """
        Check if trade should exit
        
        Returns:
            None if no exit
            (exit_price, exit_reason) if exit triggered
        """
        
        current_price = current_bar['close']
        current_time = current_bar['timestamp']
        
        # Compute current P&L in ATR units
        price_diff = current_price - trade.entry_price
        if trade.direction == 'SHORT':
            price_diff = -price_diff
        
        pnl_atr = price_diff / trade.initial_atr
        
        # Exit Rule 1: Stop Loss
        if pnl_atr <= self.stop_loss_atr:
            return (current_price, 'stop_loss')
        
        # Exit Rule 2: Take Profit
        if pnl_atr >= self.take_profit_atr:
            return (current_price, 'take_profit')
        
        # Exit Rule 3: Time Stop
        if step >= self.max_bars:
            return (current_price, 'time_exit')
        
        # Exit Rule 4: End of Day
        if current_time.time() >= self.market_close_time:
            return (current_price, 'end_of_day')
        
        return None


# ============================================================
# TRADE SIMULATOR
# ============================================================

class TradeSimulator:
    """Main simulator class"""
    
    def __init__(
        self,
        symbol: str,
        data_path: str,
        model_artifacts_path: str,
        exit_rules: Optional[ExitRules] = None
    ):
        self.symbol = symbol
        self.data_path = data_path
        self.model_artifacts_path = model_artifacts_path
        
        # Load entry engine
        self.entry_engine = EntryInferenceEngine(model_artifacts_path)
        
        # Exit rules
        self.exit_rules = exit_rules or ExitRules()
        
        # Storage
        self.trades: List[Trade] = []
        self.trade_counter = 0
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare historical data"""
        print(f"Loading data from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"Loaded {len(df):,} bars")
        return df
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute entry features for the entire dataset
        Uses the same feature engineering as training
        """
        print("Computing features...")
        return compute_all_features(df, symbol=self.symbol)
    
    def simulate(self, df: pd.DataFrame, max_trades: Optional[int] = None) -> None:
        """
        Main simulation loop
        
        Args:
            df: DataFrame with OHLCV and features
            max_trades: Maximum number of trades to simulate (None = all)
        """
        print(f"\n{'='*60}")
        print(f"SIMULATING TRADES: {self.symbol}")
        print(f"{'='*60}\n")
        
        active_trade: Optional[Trade] = None
        warmup_bars = 300  # Skip first 300 bars for feature warmup
        
        total_bars = len(df) - warmup_bars
        progress_step = total_bars // 10  # Report every 10%
        
        for idx in range(warmup_bars, len(df)):
            current_bar = df.iloc[idx]
            
            # Progress indicator
            if (idx - warmup_bars) % progress_step == 0 and (idx - warmup_bars) > 0:
                pct = ((idx - warmup_bars) / total_bars) * 100
                print(f"⏳ Progress: {pct:.0f}% | Trades: {len(self.trades)}")
            
            # ============================================================
            # ACTIVE TRADE UPDATE
            # ============================================================
            if active_trade and not active_trade.is_closed():
                # ============================================================
                # CRITICAL FIX #1: Entry Bar Leakage Prevention
                # Do NOT log state at entry candle - start from NEXT candle
                # ============================================================
                step = idx - active_trade.entry_idx
                
                # Skip entry candle itself (step 0)
                if step == 0:
                    continue
                
                # Adjust step to start from 0 at first candle AFTER entry
                step = step - 1
                
                # Update MFE/MAE
                current_price = current_bar['close']
                price_diff = current_price - active_trade.entry_price
                if active_trade.direction == 'SHORT':
                    price_diff = -price_diff
                
                # Track best and worst
                if len(active_trade.states) == 0:
                    running_mfe = max(0, price_diff)
                    running_mae = min(0, price_diff)
                else:
                    last_mfe = active_trade.states[-1].mfe_atr * active_trade.initial_atr
                    last_mae = active_trade.states[-1].mae_atr * active_trade.initial_atr
                    running_mfe = max(last_mfe, price_diff)
                    running_mae = min(last_mae, price_diff)
                
                # Compute state features
                df_slice = df.iloc[active_trade.entry_idx:idx+1]
                state = FeatureComputer.compute_state_features(
                    trade=active_trade,
                    step=step,
                    current_bar=current_bar,
                    df_slice=df_slice,
                    entry_mfe=running_mfe,
                    entry_mae=running_mae
                )
                
                active_trade.add_state(state)
                
                # Check exit rules
                exit_result = self.exit_rules.check_exit(active_trade, current_bar, step)
                
                if exit_result:
                    exit_price, exit_reason = exit_result
                    active_trade.close_trade(
                        exit_time=current_bar['timestamp'],
                        exit_price=exit_price,
                        exit_idx=idx,
                        exit_reason=exit_reason
                    )
                    
                    self.trades.append(active_trade)
                    active_trade = None
                    
                    # Check if we've hit max trades
                    if max_trades and len(self.trades) >= max_trades:
                        print(f"\nReached max trades limit: {max_trades}")
                        break
            
            # ============================================================
            # ENTRY SIGNAL DETECTION
            # ============================================================
            # CRITICAL FIX #2: Explicit one-trade-per-symbol guard
            if active_trade is None:
                # Build feature dict
                feature_dict = df.iloc[idx][self.entry_engine.feature_list].to_dict()
                
                # Get entry decision
                decision = self.entry_engine.predict(
                    feature_dict=feature_dict,
                    symbol=self.symbol,
                    timestamp=current_bar['timestamp']
                )
                
                if decision in ['LONG', 'SHORT']:
                    # Open new trade
                    self.trade_counter += 1
                    trade_id = f"{self.symbol}_{self.trade_counter:04d}"
                    
                    active_trade = Trade(
                        trade_id=trade_id,
                        symbol=self.symbol,
                        direction=decision,
                        entry_time=current_bar['timestamp'],
                        entry_price=current_bar['close'],
                        entry_idx=idx,
                        initial_atr=current_bar.get('ATR_14', 10.0),
                        entry_vwap=current_bar.get('VWAP', current_bar['close']),
                        entry_features=feature_dict
                    )
        
        # Close any remaining open trade at end of data
        if active_trade and not active_trade.is_closed():
            last_bar = df.iloc[-1]
            active_trade.close_trade(
                exit_time=last_bar['timestamp'],
                exit_price=last_bar['close'],
                exit_idx=len(df) - 1,
                exit_reason='end_of_data'
            )
            self.trades.append(active_trade)
        
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total trades: {len(self.trades)}")
    
    def export_results(self, output_dir: str = "simulation_results", append_mode: bool = False):
        """
        Export trades and states to files
        
        Args:
            output_dir: Directory to save results
            append_mode: If True, append to existing combined files instead of creating new per-symbol files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not append_mode:
            print(f"\n{'='*60}")
            print(f"EXPORTING RESULTS: {self.symbol}")
            print(f"{'='*60}\n")
        
        # 1. Export trade summaries
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'direction': trade.direction,
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'exit_reason': trade.exit_reason,
                'pnl_points': trade.pnl_points,
                'pnl_atr': trade.pnl_atr,
                'bars_held': trade.bars_held,
                'mfe': trade.mfe,
                'mae': trade.mae,
                'initial_atr': trade.initial_atr
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        if append_mode:
            # Append to combined file
            trades_file = output_path / "trades.csv"
            if trades_file.exists():
                trades_df.to_csv(trades_file, mode='a', header=False, index=False)
            else:
                trades_df.to_csv(trades_file, index=False)
        else:
            # Save per-symbol file
            trades_file = output_path / f"{self.symbol}_trades.csv"
            trades_df.to_csv(trades_file, index=False)
        
        if not append_mode:
            print(f"✅ Saved trade summaries: {trades_file}")
            print(f"   Total trades: {len(trades_df)}")
        
        # 2. Export trade states
        states_data = []
        for trade in self.trades:
            for state in trade.states:
                state_dict = asdict(state)
                state_dict['trade_id'] = trade.trade_id
                state_dict['symbol'] = trade.symbol
                state_dict['direction'] = trade.direction
                states_data.append(state_dict)
        
        states_df = pd.DataFrame(states_data)
        
        if append_mode:
            # Append to combined file
            states_file = output_path / "trade_states.parquet"
            if states_file.exists():
                existing_df = pd.read_parquet(states_file)
                combined_df = pd.concat([existing_df, states_df], ignore_index=True)
                combined_df.to_parquet(states_file, index=False)
            else:
                states_df.to_parquet(states_file, index=False)
        else:
            # Save per-symbol file
            states_file = output_path / f"{self.symbol}_trade_states.parquet"
            states_df.to_parquet(states_file, index=False)
        
        if not append_mode:
            print(f"✅ Saved trade states: {states_file}")
            print(f"   Total state records: {len(states_df):,}")
            
            # 3. Print summary statistics
            print(f"\n{'='*60}")
            print(f"SUMMARY STATISTICS")
            print(f"{'='*60}\n")
            
            if len(trades_df) > 0:
                winners = trades_df[trades_df['pnl_atr'] > 0]
                losers = trades_df[trades_df['pnl_atr'] <= 0]
                
                print(f"Win Rate: {len(winners)/len(trades_df)*100:.1f}%")
                print(f"Avg P&L (ATR): {trades_df['pnl_atr'].mean():.3f}")
                print(f"Best Trade (ATR): {trades_df['pnl_atr'].max():.3f}")
                print(f"Worst Trade (ATR): {trades_df['pnl_atr'].min():.3f}")
                print(f"Avg Bars Held: {trades_df['bars_held'].mean():.1f}")
                print(f"\nExit Reasons:")
                print(trades_df['exit_reason'].value_counts())


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run simulation for all symbols and create combined dataset"""
    
    symbols = ['LT', 'RELIANCE', 'SIEMENS', 'TATAELXSI', 'TITAN', 'TVSMOTOR']
    output_dir = "simulation_results"
    
    # Clean output directory for fresh start
    output_path = Path(output_dir)
    if output_path.exists():
        # Remove old combined files
        for f in ['trades.csv', 'trade_states.parquet']:
            file_path = output_path / f
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️  Removed old {f}")
    
    print(f"\n\n{'#'*60}")
    print(f"# MULTI-SYMBOL TRADE SIMULATION")
    print(f"# Creating COMBINED dataset for exit model training")
    print(f"{'#'*60}\n")
    
    total_trades = 0
    total_states = 0
    symbol_stats = []
    
    for idx, symbol in enumerate(symbols, 1):
        print(f"\n\n{'#'*60}")
        print(f"# [{idx}/{len(symbols)}] PROCESSING: {symbol}")
        print(f"{'#'*60}\n")
        
        try:
            simulator = TradeSimulator(
                symbol=symbol,
                data_path=f"Dataset/{symbol}_2years_1min_fixed.parquet",
                model_artifacts_path=f"{symbol}/model_artifacts",
                exit_rules=ExitRules(
                    stop_loss_atr=-0.6,
                    take_profit_atr=1.0,
                    max_bars=60,
                    market_close_time=time(15, 15)
                )
            )
            
            # Load data
            df = simulator.load_data()
            
            # Compute features
            df = simulator.compute_features(df)
            
            # Run simulation
            simulator.simulate(df, max_trades=None)  # All trades
            
            # Export to combined files
            simulator.export_results(output_dir=output_dir, append_mode=True)
            
            # Track statistics
            num_trades = len(simulator.trades)
            num_states = sum(len(t.states) for t in simulator.trades)
            total_trades += num_trades
            total_states += num_states
            
            symbol_stats.append({
                'symbol': symbol,
                'trades': num_trades,
                'states': num_states
            })
            
            print(f"\n✅ {symbol} complete: {num_trades} trades, {num_states:,} states")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n\n{'='*60}")
    print(f"COMBINED DATASET GENERATION COMPLETE")
    print(f"{'='*60}\n")
    
    print(f"📊 Per-Symbol Statistics:")
    print(f"{'Symbol':<12} {'Trades':>8} {'States':>12}")
    print(f"{'-'*32}")
    for stat in symbol_stats:
        print(f"{stat['symbol']:<12} {stat['trades']:>8,} {stat['states']:>12,}")
    print(f"{'-'*32}")
    print(f"{'TOTAL':<12} {total_trades:>8,} {total_states:>12,}")
    
    print(f"\n📁 Output Files:")
    print(f"   ✅ {output_dir}/trades.csv")
    print(f"   ✅ {output_dir}/trade_states.parquet")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Analyze combined dataset")
    print(f"   2. Train exit models on trade_states.parquet")
    print(f"   3. Symbol is now a feature (cross-symbol learning enabled)")
    
    # Load and show combined dataset info
    try:
        trades_df = pd.read_parquet(f"{output_dir}/trade_states.parquet")
        print(f"\n📈 Combined Dataset Info:")
        print(f"   Total records: {len(trades_df):,}")
        print(f"   Columns: {len(trades_df.columns)}")
        print(f"   Memory usage: {trades_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        print(f"\n   Sample columns:")
        for col in list(trades_df.columns)[:10]:
            print(f"      - {col}")
        if len(trades_df.columns) > 10:
            print(f"      ... and {len(trades_df.columns) - 10} more")
    except Exception as e:
        print(f"\n⚠️  Could not load combined dataset: {e}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
