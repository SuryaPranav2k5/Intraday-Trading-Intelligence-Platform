"""
Create feature_list.json and entry_config.yaml for all symbols
This freezes the entry model behavior for consistent reproduction
"""
import json
import yaml
from pathlib import Path

# List of all symbols
SYMBOLS = ['LT', 'RELIANCE', 'SIEMENS', 'TATAELXSI', 'TITAN', 'TVSMOTOR']

def create_feature_list(symbol_dir):
    """Extract feature list from metrics.json and save as feature_list.json"""
    metrics_path = symbol_dir / "model_artifacts" / "metrics.json"
    output_path = symbol_dir / "model_artifacts" / "feature_list.json"
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    feature_list = metrics["feature_names"]
    
    with open(output_path, "w") as f:
        json.dump(feature_list, f, indent=2)
    
    print(f"✅ {symbol_dir.name}: Saved feature_list.json with {len(feature_list)} features")
    return len(feature_list)

def create_entry_config(symbol_dir):
    """Create entry_config.yaml with frozen behavior parameters"""
    output_path = symbol_dir / "model_artifacts" / "entry_config.yaml"
    
    config = {
        'market': {
            'open': '09:15',
            'close': '15:30',
            'tradable_start_offset_min': 60,
            'tradable_end_offset_min': 15
        },
        'entry': {
            'max_trades_per_symbol': 1,
            'allow_long': True,
            'allow_short': True
        },
        'ensemble': {
            'lgb_weight': 0.5,
            'xgb_weight': 0.5,
            'conflict_policy': 'skip'  # or "stronger_prob"
        },
        'symbols': {
            symbol_dir.name: 1
        }
    }
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ {symbol_dir.name}: Saved entry_config.yaml")

def main():
    print("=" * 60)
    print("CREATING ENTRY ARTIFACTS FOR ALL SYMBOLS")
    print("=" * 60)
    print()
    
    base_dir = Path(__file__).parent
    
    for symbol in SYMBOLS:
        symbol_dir = base_dir / symbol
        
        if not symbol_dir.exists():
            print(f"⚠️  {symbol}: Directory not found, skipping")
            continue
        
        try:
            # Create feature_list.json
            num_features = create_feature_list(symbol_dir)
            
            # Create entry_config.yaml
            create_entry_config(symbol_dir)
            
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    print()
    print("=" * 60)
    print("🔒 ENTRY IS OFFICIALLY FROZEN")
    print("=" * 60)
    print()
    print("All model_artifacts folders now contain:")
    print("  ├── lgb_long.txt")
    print("  ├── lgb_short.txt")
    print("  ├── xgb_long.json")
    print("  ├── xgb_short.json")
    print("  ├── thresholds.json")
    print("  ├── metrics.json")
    print("  ├── feature_list.json     ✅ NEW")
    print("  └── entry_config.yaml     ✅ NEW")
    print()
    print("No more training changes allowed.")
    print("Exit simulation + live trading will use these exact configs.")

if __name__ == "__main__":
    main()
