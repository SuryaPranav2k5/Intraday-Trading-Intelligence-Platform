"""
Transformer Inference Pipeline

PURPOSE:
Run trained Temporal Transformer on all exit states to generate predictions.
These predictions become features for Exit LightGBM training.

INPUT:  simulation_results/trade_states_labeled.parquet
        exit_transformer_checkpoints/best_model.pt

OUTPUT: simulation_results/trade_states_with_tf.parquet

This is a ONE-TIME batch inference step for training data preparation.
"""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "simulation_results/trade_states_labeled.parquet"
MODEL_FILE = "exit_transformer_checkpoints/best_model.pt"
OUTPUT_FILE = "simulation_results/trade_states_with_tf.parquet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 32
BATCH_SIZE = 256  # Process multiple trades at once for speed

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Feature definitions (must match training)
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

SYMBOL_MAP = {"LT": 0, "RELIANCE": 1, "SIEMENS": 2, "TATAELXSI": 3, "TITAN": 4, "TVSMOTOR": 5}
DIRECTION_MAP = {"LONG": 1, "SHORT": -1}

# ============================================================
# MODEL CLASSES (EMBEDDED FOR KAGGLE COMPATIBILITY)
# ============================================================

from typing import Dict, List

class TransformerConfig:
    """Model and training configuration"""
    
    # Model architecture
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    ffn_dim: int = 128
    dropout: float = 0.1
    
    # Sequence handling
    max_seq_len: int = 32
    
    # Input features
    state_features: List[str] = [
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
    
    # Symbol embedding
    n_symbols: int = 6
    symbol_embed_dim: int = 8


class RelativePositionalEncoding(torch.nn.Module):
    """Relative positional bias for varying sequence lengths"""
    
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.rel_pos_embed = torch.nn.Embedding(2 * max_len - 1, d_model)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.rel_pos_embed.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions + self.max_len - 1
        relative_positions = relative_positions.clamp(0, 2 * self.max_len - 2)
        return self.rel_pos_embed(relative_positions)


class EntryToken(torch.nn.Module):
    """Creates the entry token that anchors the sequence"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.entry_proj = torch.nn.Linear(2 + config.symbol_embed_dim, config.d_model)
        
    def forward(self, direction, initial_atr, symbol_embed):
        entry_features = torch.cat([
            direction.unsqueeze(-1),
            initial_atr.unsqueeze(-1),
            symbol_embed
        ], dim=-1)
        return self.entry_proj(entry_features)


class ExitTransformerBlock(torch.nn.Module):
    """Single transformer block with causal attention"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.ffn_dim, config.d_model),
            torch.nn.Dropout(config.dropout),
        )
        self.norm1 = torch.nn.LayerNorm(config.d_model)
        self.norm2 = torch.nn.LayerNorm(config.d_model)
        self.dropout = torch.nn.Dropout(config.dropout)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ExitTransformer(torch.nn.Module):
    """Temporal Transformer for Exit Prediction"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Symbol embedding
        self.symbol_embed = torch.nn.Embedding(config.n_symbols, config.symbol_embed_dim)
        
        # Entry token generator
        self.entry_token = EntryToken(config)
        
        # State feature projection
        self.state_proj = torch.nn.Linear(len(config.state_features), config.d_model)
        
        # Positional encoding
        self.pos_encoding = RelativePositionalEncoding(config.d_model, config.max_seq_len)
        
        # Transformer blocks
        self.blocks = torch.nn.ModuleList([
            ExitTransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output heads
        self.output_heads = torch.nn.ModuleDict({
            "future_max_mfe_atr": torch.nn.Linear(config.d_model, 1),
            "future_max_mae_atr": torch.nn.Linear(config.d_model, 1),
            "future_return_atr": torch.nn.Linear(config.d_model, 1),
            "continuation_score": torch.nn.Linear(config.d_model, 1),
        })
    
    def _create_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, features, symbol_id, direction, initial_atr, attention_mask, seq_len):
        batch_size = features.size(0)
        device = features.device
        
        # Get symbol embeddings
        symbol_emb = self.symbol_embed(symbol_id)
        
        # Create entry token
        entry_tok = self.entry_token(direction, initial_atr, symbol_emb)
        entry_tok = entry_tok.unsqueeze(1)
        
        # Project state features
        state_emb = self.state_proj(features)
        
        # Concatenate entry token with states
        x = torch.cat([entry_tok, state_emb], dim=1)
        
        # Add positional encoding
        full_seq_len = x.size(1)
        pos_bias = self.pos_encoding(full_seq_len)
        pos_emb = pos_bias.diagonal(dim1=0, dim2=1).T
        x = x + pos_emb.unsqueeze(0)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(full_seq_len, device)
        
        # Key padding mask
        key_padding_mask = (attention_mask == 0).bool()
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        
        # Get last valid position
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = seq_len - 1
        last_hidden = x[batch_indices, last_indices]
        
        # Apply output heads
        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(last_hidden).squeeze(-1)
        
        return outputs

def load_model():
    """Load trained Transformer model"""
    print(f"📁 Loading model from: {MODEL_FILE}")
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    
    # CRITICAL: Load config from checkpoint to ensure exact match with training
    saved_config = checkpoint.get("config", None)
    if saved_config is not None and isinstance(saved_config, dict):
        # Config was saved as dict, create object with those values
        config = TransformerConfig()
        for key, value in saved_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        print(f"   ✅ Config loaded from checkpoint")
    else:
        # Fallback to default (should match training)
        config = TransformerConfig()
        print(f"   ⚠️ Using default config (checkpoint config not found)")
    
    model = ExitTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE)
    
    print(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"   Device: {DEVICE}")
    
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_mem:.1f} GB")
        print(f"   cuDNN benchmark: enabled")
        print(f"   TF32: enabled")
    
    return model, config

# ============================================================
# INFERENCE FUNCTION
# ============================================================

def run_inference_for_trade(model, trade_df, config):
    """
    Run Transformer inference for all states in a single trade.
    Returns predictions for each state.
    
    BATCHED VERSION: Process all states at once for GPU efficiency
    """
    states = trade_df[STATE_FEATURES].values.astype(np.float32)
    n_states = len(states)
    
    # Get entry context
    symbol = trade_df["symbol"].iloc[0]
    direction = trade_df["direction"].iloc[0]
    initial_atr = trade_df.get("initial_atr", pd.Series([1.0])).iloc[0] if "initial_atr" in trade_df.columns else 1.0
    
    symbol_id = SYMBOL_MAP.get(symbol, 0)
    direction_encoded = DIRECTION_MAP.get(direction, 0)
    
    # Build batch tensors for ALL states at once
    batch_features = []
    batch_masks = []
    batch_seq_lens = []
    
    for i in range(n_states):
        # Only use states up to current position (no leakage)
        seq = states[:i + 1]
        
        # Truncate if too long
        if len(seq) > MAX_SEQ_LEN - 1:
            seq = seq[-(MAX_SEQ_LEN - 1):]
        
        seq_len = len(seq)
        
        # Pad sequence
        padded = np.zeros((MAX_SEQ_LEN - 1, len(STATE_FEATURES)), dtype=np.float32)
        padded[:seq_len] = seq
        
        # Create attention mask
        attention_mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        attention_mask[:seq_len + 1] = 1.0
        
        batch_features.append(padded)
        batch_masks.append(attention_mask)
        batch_seq_lens.append(seq_len + 1)
    
    # Convert to batched tensors - ONE transfer to GPU
    features_t = torch.tensor(np.array(batch_features)).to(DEVICE)
    masks_t = torch.tensor(np.array(batch_masks)).to(DEVICE)
    seq_lens_t = torch.tensor(batch_seq_lens, dtype=torch.long).to(DEVICE)
    
    # Entry context expanded to batch size
    sym_t = torch.tensor([symbol_id] * n_states, dtype=torch.long).to(DEVICE)
    dir_t = torch.tensor([direction_encoded] * n_states, dtype=torch.float32).to(DEVICE)
    atr_t = torch.tensor([initial_atr] * n_states, dtype=torch.float32).to(DEVICE)
    
    # Run model ONCE for entire trade
    use_amp = DEVICE == "cuda"
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(features_t, sym_t, dir_t, atr_t, masks_t, seq_lens_t)
    
    # Extract predictions
    pred_mfe = outputs["future_max_mfe_atr"].cpu().numpy()
    pred_mae = outputs["future_max_mae_atr"].cpu().numpy()
    pred_ret = outputs["future_return_atr"].cpu().numpy()
    pred_cont = outputs["continuation_score"].cpu().numpy()
    
    return pred_mfe, pred_mae, pred_ret, pred_cont

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TRANSFORMER INFERENCE PIPELINE")
    print("=" * 60)
    
    # Load data
    print(f"\n📁 Loading data from: {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values(["trade_id", "step"]).reset_index(drop=True)
    
    print(f"   Rows: {len(df):,}")
    print(f"   Trades: {df['trade_id'].nunique():,}")
    
    # Add initial_atr if missing
    if 'initial_atr' not in df.columns:
        if 'current_atr' in df.columns:
            df['initial_atr'] = df.groupby('trade_id')['current_atr'].transform('first')
        else:
            df['initial_atr'] = 1.0
    
    # Load model
    model, config = load_model()
    
    # Allocate output columns
    print(f"\n🔄 Running inference on {df['trade_id'].nunique():,} trades...")
    
    pred_mfe = np.zeros(len(df))
    pred_mae = np.zeros(len(df))
    pred_ret = np.zeros(len(df))
    pred_cont = np.zeros(len(df))
    
    # Process each trade
    trade_groups = df.groupby("trade_id")
    
    for trade_id, group in tqdm(trade_groups, desc="Processing trades"):
        # Run inference for this trade
        mfe, mae, ret, cont = run_inference_for_trade(model, group, config)
        
        # Store in output arrays
        indices = group.index.values
        pred_mfe[indices] = mfe
        pred_mae[indices] = mae
        pred_ret[indices] = ret
        pred_cont[indices] = cont
    
    # Add predictions to dataframe
    df["tf_pred_future_max_mfe_atr"] = pred_mfe
    df["tf_pred_future_max_mae_atr"] = pred_mae
    df["tf_pred_future_return_atr"] = pred_ret
    df["tf_pred_continuation_score"] = pred_cont
    
    # Sanity check
    print(f"\n📊 Prediction Statistics:")
    print(f"   MFE pred:  [{pred_mfe.min():.2f}, {pred_mfe.max():.2f}], mean={pred_mfe.mean():.3f}")
    print(f"   MAE pred:  [{pred_mae.min():.2f}, {pred_mae.max():.2f}], mean={pred_mae.mean():.3f}")
    print(f"   Ret pred:  [{pred_ret.min():.2f}, {pred_ret.max():.2f}], mean={pred_ret.mean():.3f}")
    print(f"   Cont pred: [{pred_cont.min():.2f}, {pred_cont.max():.2f}], mean={pred_cont.mean():.3f}")
    
    # Check for NaN/Inf
    nan_count = df[["tf_pred_future_max_mfe_atr", "tf_pred_future_max_mae_atr", 
                    "tf_pred_future_return_atr", "tf_pred_continuation_score"]].isna().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️ Warning: {nan_count} NaN values in predictions!")
    else:
        print(f"   ✅ No NaN values in predictions")
    
    # Save
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    
    # Create output directory if it doesn't exist (for Kaggle)
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n" + "=" * 60)
    print("TRANSFORMER INFERENCE COMPLETE")
    print("=" * 60)
    print(f"   Rows: {len(df):,}")
    print(f"   New columns:")
    print(f"      - tf_pred_future_max_mfe_atr")
    print(f"      - tf_pred_future_max_mae_atr")
    print(f"      - tf_pred_future_return_atr")
    print(f"      - tf_pred_continuation_score")
    print(f"\n   Output: {OUTPUT_FILE}")
    print(f"\n🎯 NEXT: Run exit_lgbm_training.py")


if __name__ == "__main__":
    main()
