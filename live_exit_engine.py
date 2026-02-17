"""
Live Exit Engine

PURPOSE:
Real-time exit decision loop using:
- Temporal Transformer (context understanding)
- Exit LightGBM (action decision)
- Hard risk rules (safety guardrails)

FINAL DECISION FLOW:
  New candle → Update trade state → Transformer inference 
  → LightGBM inference → Risk overrides → EXIT or HOLD

This is the PRODUCTION inference module.
"""

import torch
import numpy as np
import os
import joblib
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_MODEL = os.path.join(_SCRIPT_DIR, "exit_transformer_checkpoints", "best_model.pt")
EXIT_LGBM_MODEL = os.path.join(_SCRIPT_DIR, "exit_lgbm_policy.pkl")

DEVICE = "cpu"  # Use CPU for consistent low-latency inference
MAX_SEQ_LEN = 32

# Hard risk rules (MANDATORY OVERRIDES)
HARD_STOP_ATR = -1.2      # Absolute disaster stop
MAX_BARS = 120            # Time stop (2 hours in 1-min data)
END_OF_DAY_CUTOFF = 90    # Minutes before market close to force exit

# Symbol and direction encoding
SYMBOL_MAP = {"LT": 0, "RELIANCE": 1, "SIEMENS": 2, "TATAELXSI": 3, "TITAN": 4, "TVSMOTOR": 5}
DIRECTION_MAP = {"LONG": 1, "SHORT": -1}

# State feature order (must match training)
STATE_FEATURES = [
    "price_from_entry_atr",
    "vwap_from_entry_atr",
    "unrealized_pnl_atr",
    "mfe_atr",
    "mae_atr",
    "step_log",
    "volatility_expansion_ratio",
    "pullback_depth",
    "momentum_decay",
]

# ============================================================
# LIVE EXIT ENGINE CLASS
# ============================================================

class LiveExitEngine:
    """
    Real-time exit decision engine.
    
    Usage:
        engine = LiveExitEngine()
        engine.reset_trade()
        
        for candle in trade_candles:
            decision = engine.update_and_decide(state_features, entry_features)
            if decision == "EXIT":
                execute_exit()
                break
    """
    
    def __init__(self):
        print("=" * 60)
        print("INITIALIZING LIVE EXIT ENGINE")
        print("=" * 60)
        
        # Load Transformer
        self._load_transformer()
        
        # Load Exit LightGBM
        self._load_lgbm()
        
        # Initialize state buffer
        self.state_buffer = []
        self.trade_active = False
        
        print("\n[OK] Live Exit Engine ready")
        print("=" * 60)
    
    def _load_transformer(self):
        """Load trained Transformer model"""
        from exit_transformer import ExitTransformer, TransformerConfig
        
        print(f"\n[LOAD] Loading Transformer from: {TRANSFORMER_MODEL}")
        checkpoint = torch.load(TRANSFORMER_MODEL, map_location=DEVICE)
        
        self.tf_config = TransformerConfig()
        self.transformer = ExitTransformer(self.tf_config)
        self.transformer.load_state_dict(checkpoint["model_state_dict"])
        self.transformer.eval().to(DEVICE)
        
        print(f"   [OK] Loaded (best_val_loss: {checkpoint.get('best_val_loss', 'N/A'):.4f})")
    
    def _load_lgbm(self):
        """Load trained Exit LightGBM"""
        print(f"\n[LOAD] Loading Exit LightGBM from: {EXIT_LGBM_MODEL}")
        
        lgbm_pack = joblib.load(EXIT_LGBM_MODEL)
        self.lgbm = lgbm_pack["model"]
        self.lgbm_features = lgbm_pack["features"]
        self.exit_threshold = lgbm_pack["threshold"]
        
        print(f"   [OK] Loaded (AUC: {lgbm_pack.get('auc', 'N/A'):.4f}, threshold: {self.exit_threshold})")
    
    def reset_trade(self):
        """Call this when a new trade is opened"""
        self.state_buffer = []
        self.trade_active = True
    
    def update_and_decide(
        self,
        state_features: Dict[str, float],
        entry_features: Dict[str, float]
    ) -> str:
        """
        Main decision function. Call this on every new candle.
        
        Args:
            state_features: Current trade state
                - price_from_entry_atr
                - vwap_from_entry_atr
                - unrealized_pnl_atr
                - mfe_atr
                - mae_atr
                - step (or step_log)
                - volatility_expansion_ratio
                - pullback_depth
                - momentum_decay
            
            entry_features: Static entry context
                - direction: "LONG" or "SHORT"
                - initial_atr
                - symbol: e.g., "TITAN"
        
        Returns:
            "HOLD" or "EXIT"
        """
        
        if not self.trade_active:
            return "HOLD"
        
        # ---------- HARD RISK OVERRIDES ----------
        # These ALWAYS trigger, regardless of model predictions
        
        # 1. Disaster stop
        if state_features.get("unrealized_pnl_atr", 0) <= HARD_STOP_ATR:
            return "EXIT"
        
        # 2. Time stop
        step = state_features.get("step", 0)
        if step >= MAX_BARS:
            return "EXIT"
        
        # ---------- UPDATE STATE BUFFER ----------
        # Compute step_log if not provided
        step_log = state_features.get("step_log", np.log1p(step))
        
        state_vector = [
            state_features.get("price_from_entry_atr", 0),
            state_features.get("vwap_from_entry_atr", 0),
            state_features.get("unrealized_pnl_atr", 0),
            state_features.get("mfe_atr", 0),
            state_features.get("mae_atr", 0),
            step_log,
            state_features.get("volatility_expansion_ratio", 1.0),
            state_features.get("pullback_depth", 0),
            state_features.get("momentum_decay", 0),
        ]
        
        self.state_buffer.append(state_vector)
        
        # ---------- PREPARE TRANSFORMER INPUT ----------
        seq = np.array(self.state_buffer, dtype=np.float32)
        
        # Truncate if too long
        if len(seq) > MAX_SEQ_LEN - 1:
            seq = seq[-(MAX_SEQ_LEN - 1):]
        
        seq_len = len(seq)
        
        # Pad sequence
        padded = np.zeros((MAX_SEQ_LEN - 1, len(STATE_FEATURES)), dtype=np.float32)
        padded[:seq_len] = seq
        
        # Create attention mask
        attention_mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        attention_mask[:seq_len + 1] = 1.0  # +1 for entry token
        
        # Entry features - handle both numeric and string formats
        symbol = entry_features.get("symbol", None)
        symbol_id = entry_features.get("symbol_id", None)
        direction = entry_features.get("direction", 1)
        initial_atr = entry_features.get("initial_atr", 1.0)
        
        # Symbol: prefer numeric symbol_id, fallback to string lookup
        if symbol_id is not None:
            symbol_id = int(symbol_id)
        elif symbol is not None:
            symbol_id = SYMBOL_MAP.get(symbol, 0)
        else:
            symbol_id = 0
        
        # Direction: handle numeric (+1/-1) or string ("LONG"/"SHORT")
        if isinstance(direction, str):
            direction_encoded = DIRECTION_MAP.get(direction, 1)
        else:
            direction_encoded = int(direction)  # Already numeric
        
        # Convert to tensors
        features_t = torch.tensor(padded).unsqueeze(0).to(DEVICE)
        mask_t = torch.tensor(attention_mask).unsqueeze(0).to(DEVICE)
        sym_t = torch.tensor([symbol_id], dtype=torch.long).to(DEVICE)
        dir_t = torch.tensor([direction_encoded], dtype=torch.float32).to(DEVICE)
        atr_t = torch.tensor([initial_atr], dtype=torch.float32).to(DEVICE)
        seq_len_t = torch.tensor([seq_len + 1], dtype=torch.long).to(DEVICE)
        
        # ---------- TRANSFORMER INFERENCE ----------
        with torch.no_grad():
            tf_out = self.transformer(features_t, sym_t, dir_t, atr_t, mask_t, seq_len_t)
        
        # ---------- BUILD LightGBM INPUT (20 FEATURES) ----------
        lgbm_row = {
            # Transformer outputs (4)
            "tf_pred_future_max_mfe_atr": tf_out["future_max_mfe_atr"].item(),
            "tf_pred_future_max_mae_atr": tf_out["future_max_mae_atr"].item(),
            "tf_pred_future_return_atr": tf_out["future_return_atr"].item(),
            "tf_pred_continuation_score": tf_out["continuation_score"].item(),
            
            # Current state (5)
            "unrealized_pnl_atr": state_features.get("unrealized_pnl_atr", 0),
            "price_from_entry_atr": state_features.get("price_from_entry_atr", 0),
            "mfe_atr": state_features.get("mfe_atr", 0),
            "mae_atr": state_features.get("mae_atr", 0),
            "step_log": step_log,
            
            # Risk & regime (4)
            "volatility_expansion_ratio": state_features.get("volatility_expansion_ratio", 1.0),
            "momentum_decay": state_features.get("momentum_decay", 0),
            "pullback_depth": state_features.get("pullback_depth", 0),
            "vwap_from_entry_atr": state_features.get("vwap_from_entry_atr", 0),
            
            # Extended state (3)
            "step": step,
            "current_atr": state_features.get("current_atr", initial_atr),
            "initial_atr": initial_atr,
            
            # Entry context (4)
            "entry_confidence": entry_features.get("entry_confidence", 0.5),
            "direction": direction_encoded,
            "symbol_id": symbol_id,
            "direction_x_pnl": direction_encoded * state_features.get("unrealized_pnl_atr", 0),
        }
        
        # Create feature array in correct order WITH STRICT ASSERTION
        for f in self.lgbm_features:
            if f not in lgbm_row:
                raise ValueError(f"Missing LightGBM feature: {f}")
        
        X = np.array([[lgbm_row[f] for f in self.lgbm_features]])
        
        # ---------- LightGBM INFERENCE ----------
        exit_prob = self.lgbm.predict(X)[0]
        
        # ---------- FINAL DECISION ----------
        if exit_prob >= self.exit_threshold:
            self.trade_active = False
            return "EXIT"
        
        return "HOLD"
    
    def get_exit_probability(
        self,
        state_features: Dict[str, float],
        entry_features: Dict[str, float]
    ) -> float:
        """
        Get exit probability without making decision.
        Useful for monitoring and logging.
        """
        # This is a simplified version that doesn't update state
        # For full accuracy, use update_and_decide
        return 0.0  # Placeholder


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_usage():
    """
    Example showing how to use LiveExitEngine in a trading loop.
    """
    
    print("\n" + "=" * 60)
    print("EXAMPLE: LIVE EXIT ENGINE USAGE")
    print("=" * 60)
    
    # Initialize engine (do this once at startup)
    engine = LiveExitEngine()
    
    # Simulate a trade
    print("\n[TRADE] Starting example trade...")
    
    # Entry context (set once when trade opens)
    entry_features = {
        "direction": "LONG",
        "symbol": "TITAN",
        "initial_atr": 25.5,
    }
    
    # Reset for new trade
    engine.reset_trade()
    
    # Simulate candles
    simulated_states = [
        {"step": 1, "price_from_entry_atr": 0.1, "unrealized_pnl_atr": 0.08, "mfe_atr": 0.1, "mae_atr": 0.0,
         "vwap_from_entry_atr": 0.05, "volatility_expansion_ratio": 1.0, "pullback_depth": 0.0, "momentum_decay": 0.0},
        
        {"step": 2, "price_from_entry_atr": 0.3, "unrealized_pnl_atr": 0.25, "mfe_atr": 0.3, "mae_atr": 0.0,
         "vwap_from_entry_atr": 0.15, "volatility_expansion_ratio": 1.1, "pullback_depth": 0.0, "momentum_decay": 0.1},
        
        {"step": 3, "price_from_entry_atr": 0.2, "unrealized_pnl_atr": 0.15, "mfe_atr": 0.3, "mae_atr": 0.0,
         "vwap_from_entry_atr": 0.12, "volatility_expansion_ratio": 1.0, "pullback_depth": 0.1, "momentum_decay": 0.0},
        
        {"step": 4, "price_from_entry_atr": 0.5, "unrealized_pnl_atr": 0.45, "mfe_atr": 0.5, "mae_atr": 0.0,
         "vwap_from_entry_atr": 0.3, "volatility_expansion_ratio": 0.9, "pullback_depth": 0.0, "momentum_decay": 0.2},
    ]
    
    for state in simulated_states:
        decision = engine.update_and_decide(state, entry_features)
        print(f"   Step {state['step']}: PnL={state['unrealized_pnl_atr']:+.2f} ATR → {decision}")
        
        if decision == "EXIT":
            print("   [EXIT] EXIT triggered!")
            break
    
    print("\n[OK] Example complete")


if __name__ == "__main__":
    example_usage()
