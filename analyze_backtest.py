import pandas as pd
import numpy as np

def analyze_results(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print("No trades found in results.")
            return

        total_trades = len(df)
        wins = df[df['PnL'] > 0]
        losses = df[df['PnL'] <= 0]
        
        win_rate = len(wins) / total_trades * 100
        total_pnl = df['PnL'].sum()
        avg_pnl = df['PnL'].mean()
        
        gross_profit = wins['PnL'].sum()
        gross_loss = abs(losses['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        print(f"Analysis of {file_path}")
        print("=" * 40)
        print(f"Total Trades:       {total_trades}")
        print(f"Win Rate:           {win_rate:.2f}%")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print(f"Total PnL:          ₹{total_pnl:.2f}")
        print(f"Avg PnL per Trade:  ₹{avg_pnl:.2f}")
        print("-" * 40)
        print("Exit Reasons:")
        print(df['Reason'].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    analyze_results("backtest_results_TVSMOTOR.csv")
