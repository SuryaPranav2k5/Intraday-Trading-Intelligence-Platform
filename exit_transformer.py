"""
Temporal Transformer for Exit Prediction

PURPOSE:
The Transformer learns: "Given the trade story so far, what is future opportunity vs risk?"
It does NOT decide exits - that's the policy's job.

ARCHITECTURE:
- d_model = 64, n_heads = 4, n_layers = 3
- Entry token + trade state sequence
- Multi-task output: MFE, MAE, Return, Continuation Score
- Causal attention with relative positional bias

INPUTS:
- Trade-relative features only (no raw OHLC)
- Entry token anchors the sequence

OUTPUTS (for policy consumption):
- ŷ_future_max_mfe_atr
- ŷ_future_max_mae_atr
- ŷ_future_return_atr
- ŷ_continuation_score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import warnings

# Suppress PyTorch attention mask warnings
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")

# ============================================================
# CONFIGURATION
# ============================================================

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
    
    # Entry token features
    entry_features: List[str] = [
        "direction_encoded",  # +1 for LONG, -1 for SHORT
        "initial_atr",
    ]
    
    # Symbol embedding
    n_symbols: int = 6
    symbol_embed_dim: int = 8
    
    # Total input dimension
    @property
    def input_dim(self) -> int:
        return len(self.state_features) + self.symbol_embed_dim
    
    # Output targets
    targets: List[str] = [
        "future_max_mfe_atr",
        "future_max_mae_atr",
        "future_return_atr",
        "continuation_score",
    ]
    
    # Training
    batch_size: int = 256  # Larger batch for GPU utilization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 50
    patience: int = 7
    
    # Loss weights
    loss_weights: Dict[str, float] = {
        "future_max_mfe_atr": 1.0,
        "future_max_mae_atr": 1.0,
        "future_return_atr": 0.5,
        "continuation_score": 1.5,
    }


# ============================================================
# DATASET
# ============================================================

class TradeSequenceDataset(Dataset):
    """
    Dataset that groups trade states into sequences.
    Each sample is one complete trade (or truncated to max_seq_len).
    """
    
    SYMBOL_MAP = {"LT": 0, "RELIANCE": 1, "SIEMENS": 2, "TATAELXSI": 3, "TITAN": 4, "TVSMOTOR": 5}
    DIRECTION_MAP = {"LONG": 1, "SHORT": -1}
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: TransformerConfig,
        trade_ids: Optional[List[str]] = None
    ):
        self.config = config
        self.max_seq_len = config.max_seq_len
        
        # Filter to specific trade_ids if provided
        if trade_ids is not None:
            df = df[df["trade_id"].isin(trade_ids)]
        
        # Group by trade_id
        self.trades = []
        for trade_id, group in df.groupby("trade_id"):
            group = group.sort_values("step")
            self.trades.append({
                "trade_id": trade_id,
                "data": group.reset_index(drop=True)
            })
        
        self.state_features = config.state_features
        self.targets = config.targets
    
    def __len__(self) -> int:
        return len(self.trades)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trade = self.trades[idx]
        df = trade["data"]
        
        # Get sequence length (cap at max_seq_len - 1 for entry token)
        seq_len = min(len(df), self.max_seq_len - 1)
        
        # Extract features for each state
        features = df[self.state_features].values[:seq_len].astype(np.float32)
        
        # Extract targets (from last state)
        targets = df[self.targets].values[-1].astype(np.float32)
        
        # Create entry token (direction + initial_atr)
        direction = self.DIRECTION_MAP.get(df["direction"].iloc[0], 0)
        initial_atr = df.get("initial_atr", pd.Series([1.0])).iloc[0] if "initial_atr" in df.columns else 1.0
        
        # Symbol encoding
        symbol = df["symbol"].iloc[0]
        symbol_id = self.SYMBOL_MAP.get(symbol, 0)
        
        # Pad sequence to max_seq_len - 1 (entry token takes 1 slot)
        padded_features = np.zeros((self.max_seq_len - 1, len(self.state_features)), dtype=np.float32)
        padded_features[:seq_len] = features
        
        # Create attention mask (1 = attend, 0 = ignore)
        # +1 for entry token
        attention_mask = np.zeros(self.max_seq_len, dtype=np.float32)
        attention_mask[:seq_len + 1] = 1.0  # Entry token + actual states
        
        return {
            "features": torch.tensor(padded_features),
            "targets": torch.tensor(targets),
            "attention_mask": torch.tensor(attention_mask),
            "symbol_id": torch.tensor(symbol_id, dtype=torch.long),
            "direction": torch.tensor(direction, dtype=torch.float32),
            "initial_atr": torch.tensor(initial_atr, dtype=torch.float32),
            "seq_len": torch.tensor(seq_len + 1, dtype=torch.long),  # +1 for entry token
        }


# ============================================================
# MODEL COMPONENTS
# ============================================================

class RelativePositionalEncoding(nn.Module):
    """Relative positional bias for varying sequence lengths"""
    
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable relative position embeddings
        self.rel_pos_embed = nn.Embedding(2 * max_len - 1, d_model)
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns relative positional encoding for sequence"""
        positions = torch.arange(seq_len, device=self.rel_pos_embed.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = relative_positions + self.max_len - 1  # Shift to positive
        relative_positions = relative_positions.clamp(0, 2 * self.max_len - 2)
        return self.rel_pos_embed(relative_positions)


class EntryToken(nn.Module):
    """Creates the entry token that anchors the sequence"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Entry token projection
        # Input: direction (1) + initial_atr (1) + symbol_embed (8) = 10
        self.entry_proj = nn.Linear(2 + config.symbol_embed_dim, config.d_model)
        
    def forward(
        self,
        direction: torch.Tensor,      # (batch,)
        initial_atr: torch.Tensor,    # (batch,)
        symbol_embed: torch.Tensor    # (batch, symbol_embed_dim)
    ) -> torch.Tensor:
        """Returns entry token embedding (batch, d_model)"""
        entry_features = torch.cat([
            direction.unsqueeze(-1),
            initial_atr.unsqueeze(-1),
            symbol_embed
        ], dim=-1)
        
        return self.entry_proj(entry_features)


class ExitTransformerBlock(nn.Module):
    """Single transformer block with causal attention"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


# ============================================================
# MAIN MODEL
# ============================================================

class ExitTransformer(nn.Module):
    """
    Temporal Transformer for Exit Prediction
    
    Input: Trade state sequence with entry token
    Output: Future path predictions (MFE, MAE, Return, Continuation)
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Symbol embedding
        self.symbol_embed = nn.Embedding(config.n_symbols, config.symbol_embed_dim)
        
        # Entry token generator
        self.entry_token = EntryToken(config)
        
        # State feature projection (no symbol embedding - it's in entry token only)
        self.state_proj = nn.Linear(
            len(config.state_features),  # Only state features, no symbol
            config.d_model
        )
        
        # Positional encoding
        self.pos_encoding = RelativePositionalEncoding(config.d_model, config.max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ExitTransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        
        # Output heads (multi-task)
        self.output_heads = nn.ModuleDict({
            "future_max_mfe_atr": nn.Linear(config.d_model, 1),
            "future_max_mae_atr": nn.Linear(config.d_model, 1),
            "future_return_atr": nn.Linear(config.d_model, 1),
            "continuation_score": nn.Linear(config.d_model, 1),
        })
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        features: torch.Tensor,        # (batch, seq_len-1, n_features)
        symbol_id: torch.Tensor,       # (batch,)
        direction: torch.Tensor,       # (batch,)
        initial_atr: torch.Tensor,     # (batch,)
        attention_mask: torch.Tensor,  # (batch, seq_len)
        seq_len: torch.Tensor,         # (batch,)
    ) -> Dict[str, torch.Tensor]:
        
        batch_size = features.size(0)
        device = features.device
        
        # Get symbol embeddings (only for entry token)
        symbol_emb = self.symbol_embed(symbol_id)  # (batch, symbol_embed_dim)
        
        # Create entry token (contains symbol context)
        entry_tok = self.entry_token(direction, initial_atr, symbol_emb)  # (batch, d_model)
        entry_tok = entry_tok.unsqueeze(1)  # (batch, 1, d_model)
        
        # Project state features (NO symbol embedding - focus on trade dynamics)
        state_emb = self.state_proj(features)  # (batch, seq_len-1, d_model)
        
        # Concatenate entry token with states
        x = torch.cat([entry_tok, state_emb], dim=1)  # (batch, seq_len, d_model)
        
        # Add positional encoding (CRITICAL for order awareness)
        full_seq_len = x.size(1)
        pos_bias = self.pos_encoding(full_seq_len)  # (seq_len, seq_len, d_model)
        # Average over relative positions for additive bias
        pos_emb = pos_bias.diagonal(dim1=0, dim2=1).T  # (seq_len, d_model)
        x = x + pos_emb.unsqueeze(0)  # Add positional info
        
        # Create causal mask
        full_seq_len = x.size(1)
        causal_mask = self._create_causal_mask(full_seq_len, device)
        
        # Create key padding mask from attention mask (invert: 1->False, 0->True)
        # Must be bool type to avoid warning
        key_padding_mask = (attention_mask == 0).bool()
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        
        # Get last valid position for each sequence
        # seq_len includes entry token, so last valid index is seq_len - 1
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = seq_len - 1
        last_hidden = x[batch_indices, last_indices]  # (batch, d_model)
        
        # Apply output heads
        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(last_hidden).squeeze(-1)
        
        return outputs


# ============================================================
# LOSS FUNCTION
# ============================================================

class ExitTransformerLoss(nn.Module):
    """Multi-task loss with Huber losses"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.huber = nn.HuberLoss(reduction='mean', delta=1.0)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor  # (batch, 4) in order of config.targets
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        losses = {}
        
        for i, target_name in enumerate(self.config.targets):
            pred = predictions[target_name]
            target = targets[:, i]
            
            if target_name == "continuation_score":
                loss = self.smooth_l1(pred, target)
            else:
                loss = self.huber(pred, target)
            
            losses[target_name] = loss
        
        # Weighted sum
        total_loss = sum(
            self.config.loss_weights[name] * loss
            for name, loss in losses.items()
        )
        
        # Convert to float for logging
        loss_dict = {name: loss.item() for name, loss in losses.items()}
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================================
# TRAINER
# ============================================================

class ExitTransformerTrainer:
    """Training loop with early stopping"""
    
    def __init__(
        self,
        model: ExitTransformer,
        config: TransformerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str = "exit_transformer_checkpoints"
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = ExitTransformerLoss(config)
        
        # Mixed precision training (AMP) for GPU
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {name: 0.0 for name in self.config.targets}
        total_losses["total"] = 0.0
        n_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # Move to device
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            symbol_id = batch["symbol_id"].to(self.device)
            direction = batch["direction"].to(self.device)
            initial_atr = batch["initial_atr"].to(self.device)
            seq_len = batch["seq_len"].to(self.device)
            
            # Forward pass with AMP autocast
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                predictions = self.model(
                    features, symbol_id, direction, initial_atr, attention_mask, seq_len
                )
                loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Accumulate losses
            for name, val in loss_dict.items():
                total_losses[name] += val
            n_batches += 1
        
        # Average losses
        return {name: val / n_batches for name, val in total_losses.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        total_losses = {name: 0.0 for name in self.config.targets}
        total_losses["total"] = 0.0
        n_batches = 0
        
        for batch in self.val_loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            symbol_id = batch["symbol_id"].to(self.device)
            direction = batch["direction"].to(self.device)
            initial_atr = batch["initial_atr"].to(self.device)
            seq_len = batch["seq_len"].to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                predictions = self.model(
                    features, symbol_id, direction, initial_atr, attention_mask, seq_len
                )
                _, loss_dict = self.criterion(predictions, targets)
            
            for name, val in loss_dict.items():
                total_losses[name] += val
            n_batches += 1
        
        return {name: val / n_batches for name, val in total_losses.items()}
    
    def train(self) -> Dict:
        """Full training loop with early stopping"""
        history = {"train": [], "val": []}
        
        for epoch in range(self.config.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            print("-" * 40)
            
            # Train
            train_losses = self.train_epoch()
            history["train"].append(train_losses)
            
            # Validate
            val_losses = self.validate()
            history["val"].append(val_losses)
            
            # Print losses
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss:   {val_losses['total']:.4f}")
            print(f"  MFE: {val_losses['future_max_mfe_atr']:.4f}")
            print(f"  MAE: {val_losses['future_max_mae_atr']:.4f}")
            print(f"  Ret: {val_losses['future_return_atr']:.4f}")
            print(f"  Cont: {val_losses['continuation_score']:.4f}")
            
            # Early stopping on continuation_score loss
            val_cont_loss = val_losses["continuation_score"]
            
            if val_cont_loss < self.best_val_loss:
                self.best_val_loss = val_cont_loss
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print(f"✅ New best model saved!")
            else:
                self.patience_counter += 1
                print(f"⏳ Patience: {self.patience_counter}/{self.config.patience}")
                
                if self.patience_counter >= self.config.patience:
                    print(f"\n🛑 Early stopping triggered!")
                    break
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.config),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.save_dir / filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

def main():
    print("=" * 60)
    print("TEMPORAL TRANSFORMER FOR EXIT PREDICTION")
    print("=" * 60)
    
    # Device setup with GPU optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
    if device.type == "cuda":
        # Enable cuDNN auto-tuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 for Ampere GPUs (faster matmul)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_mem:.1f} GB")
        print(f"   cuDNN benchmark: enabled")
        print(f"   TF32: enabled")
    
    # Configuration
    config = TransformerConfig()
    
    # Load labeled data
    print(f"\n📁 Loading labeled data...")
    df = pd.read_parquet("simulation_results/trade_states_labeled.parquet")
    print(f"   Rows: {len(df):,}")
    print(f"   Trades: {df['trade_id'].nunique():,}")
    
    # Add initial_atr if missing (from first state of each trade)
    if 'initial_atr' not in df.columns:
        # Estimate from current_atr of first state
        df['initial_atr'] = df.groupby('trade_id')['current_atr'].transform('first')
    
    # Encode direction
    df['direction_encoded'] = df['direction'].map({'LONG': 1, 'SHORT': -1})
    
    # Split by trade_id (70/15/15)
    trade_ids = df['trade_id'].unique()
    np.random.seed(42)
    np.random.shuffle(trade_ids)
    
    n_trades = len(trade_ids)
    train_end = int(0.7 * n_trades)
    val_end = int(0.85 * n_trades)
    
    train_ids = trade_ids[:train_end]
    val_ids = trade_ids[train_end:val_end]
    test_ids = trade_ids[val_end:]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(train_ids):,} trades")
    print(f"   Val:   {len(val_ids):,} trades")
    print(f"   Test:  {len(test_ids):,} trades")
    
    # Create datasets
    train_dataset = TradeSequenceDataset(df, config, train_ids)
    val_dataset = TradeSequenceDataset(df, config, val_ids)
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train: {len(train_dataset):,} sequences")
    print(f"   Val:   {len(val_dataset):,} sequences")
    
    # Create data loaders with parallel loading
    num_workers = 4 if device.type == "cuda" else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create model
    print(f"\n🧠 Creating model...")
    model = ExitTransformer(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Create trainer
    trainer = ExitTransformerTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    history = trainer.train()
    
    # Save final model
    trainer.save_checkpoint("final_model.pt")
    
    # Save training history
    with open("exit_transformer_checkpoints/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n📁 Saved to: exit_transformer_checkpoints/")
    print(f"   - best_model.pt")
    print(f"   - final_model.pt")
    print(f"   - training_history.json")


if __name__ == "__main__":
    main()
