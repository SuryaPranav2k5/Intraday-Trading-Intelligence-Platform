"""
Master Training Script
Trains setup quality models for all symbols sequentially
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# List of all symbols to train
SYMBOLS = [
    "LT",
    "RELIANCE",
    "SIEMENS",
    "TATAELXSI",
    "TITAN",
    "TVSMOTOR"
]

def train_symbol(symbol: str) -> bool:
    """
    Train model for a specific symbol
    Returns True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Starting training for {symbol}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    script_path = Path(__file__).parent / symbol / "train_setup_quality.py"
    
    if not script_path.exists():
        print(f"❌ Training script not found: {script_path}")
        return False
    
    try:
        # Run the training script (Stream output to console)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            timeout=3600  # 60 minutes timeout
        )
        
        # Output is streamed directly, so no need to print result.stdout
        
        if result.returncode == 0:
            print(f"\n✅ {symbol} training completed successfully!")
            return True
        else:
            print(f"\n❌ {symbol} training failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ {symbol} training timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"\n❌ {symbol} training failed with exception: {e}")
        return False


def main():
    """
    Train all symbols and report results
    """
    print("="*60)
    print("MULTI-SYMBOL MODEL TRAINING")
    print(f"Total symbols: {len(SYMBOLS)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    for symbol in SYMBOLS:
        success = train_symbol(symbol)
        results[symbol] = success
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    successful = [s for s, success in results.items() if success]
    failed = [s for s, success in results.items() if not success]
    
    print(f"\n✅ Successful: {len(successful)}/{len(SYMBOLS)}")
    for symbol in successful:
        print(f"   - {symbol}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(SYMBOLS)}")
        for symbol in failed:
            print(f"   - {symbol}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Exit with non-zero code if any training failed
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
