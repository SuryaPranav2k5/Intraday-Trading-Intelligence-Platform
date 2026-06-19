import os
import re

def generate_universal_script():
    src_file = "d:/Model/Human_ML/TATAELXSI/train_setup_quality.py"
    dst_file = "d:/Model/Human_ML/train_universal.py"
    
    with open(src_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 1. Update Header
    content = content.replace(
        "# Symbol: TATAELXSI", 
        "# Symbols: UNIVERSAL (All 6 Symbols Combined)"
    )
    
    # 2. Update DATA Loading Block (Lines ~880 to ~960 depending on where __main__ starts)
    # We will replace the entire __main__ setup section using regex.
    main_block_pattern = r'def main\(\):.*?#( \S+)+ TRAINING(.*?)\n\n'
    
    # Let's find def main(): to the start of the training loop
    main_start = "def main():"
    train_start = "    # ========== TRAINING =========="
    
    if main_start in content and train_start in content:
        head, rest = content.split("def main():", 1)
        main_content, tail = rest.split(train_start, 1)
        
        universal_main = """
    print("=" * 60)
    print("UNIVERSAL ENTRY MODEL TRAINING")
    print("=" * 60)
    
    symbol_folders = ["LT", "RELIANCE", "SIEMENS", "TATAELXSI", "TITAN", "TVSMOTOR"]
    symbol_map = {s: i for i, s in enumerate(symbol_folders)}
    
    all_X = []
    all_labels = []
    all_symbols = []
    
    for symbol in symbol_folders:
        data_file = Path(f"market_data/{symbol}_1min.parquet")
        if not data_file.exists():
            print(f"Skipping {symbol} - No data file found.")
            continue
            
        print(f"\\nProcessing {symbol}...")
        df = load_raw_data(data_file)
        df, labels_df = process_symbol_data(df)
        
        # Add symbol ID
        df['symbol_id'] = symbol_map[symbol]
        
        all_X.append(df)
        all_labels.append(labels_df)
        all_symbols.append(pd.Series([symbol]*len(df), index=df.index, name='symbol_name'))
        
    print("\\nConcatenating Universal Dataset...")
    X_full = pd.concat(all_X).sort_index()
    labels_full = pd.concat(all_labels).sort_index()
    symbols_full = pd.concat(all_symbols).sort_index()
    
    print(f"Universal Dataset Size: {len(X_full):,} rows")
    
    # ========== CORRELATION PRUNING (Fixed: Split-Aware) ==========
    print("\\nPruning highly correlated features based ONLY on First Fold Train Data (>0.95)...")
    n_splits_sim = 5
    split_idx_1st = int(len(X_full) / (n_splits_sim + 1))
    X_safe_train = X_full.iloc[:split_idx_1st].drop(columns=['symbol_id']) # Don't prune symbol_id
    
    corr_matrix = X_safe_train.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    if to_drop:
        print(f"  Dropping {len(to_drop)} redundant features: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")
        X_full = X_full.drop(columns=to_drop)
    else:
        print("  No features dropped.")
        
    X = X_full
    y_long = labels_full['label_long']
    y_short = labels_full['label_short']
    
"""
        # We need to wrap the single-file data prep into a function
        data_prep_func = """
def process_symbol_data(df: pd.DataFrame):
""" + "\n".join([line for line in main_content.split("\n") 
                 if "print(" not in line 
                 and "DATA_FILE" not in line 
                 and "def main()" not in line
                 and "data_file =" not in line
                 and "df = load_raw_" not in line
                 and "return" not in line]) + "\n    return X, labels_df\n\n"
                 
        content = head + data_prep_func + "def main():\n" + universal_main + "    # ========== TRAINING ==========" + tail
        
        # We have to fix one specific thing in data_prep: "X, labels_df = clean_training_data(df, labels_df)"
        # The X that comes out does not have the 'symbol_id'.
        # Wait, the universal_main explicitly defines symbol_id: df['symbol_id'] = symbol_map[symbol]
        # BUT 'df' in universal_main expects the return from process_symbol_data to be 'X'.
        # Let's cleanly define process_symbol_data return:
        content = content.replace("df, labels_df = process_symbol_data(df)", "X_df, labels_df = process_symbol_data(df)\n        X_df['symbol_id'] = symbol_map[symbol]")
        content = content.replace("all_X.append(df)", "all_X.append(X_df)")


    # 3. Update the Optimize Ensemble logic to be PER-SYMBOL
    old_opt_header = """def optimize_ensemble(y_val, preds_lgb, preds_xgb, name):"""
    new_opt_header = """def optimize_ensemble(y_val, preds_lgb, preds_xgb, name, sym_series_val):"""
    content = content.replace(old_opt_header, new_opt_header)
    
    old_targ_logic = """    # 2. Optimize Threshold for this Weight (on set T)
    final_preds_t = best_w * preds_lgb_t + (1 - best_w) * preds_xgb_t
    best_threshold = 0.5
    best_score = 0
    
    for threshold in np.linspace(0.3, 0.7, 41):
        preds_binary = (final_preds_t >= threshold).astype(int)
        if preds_binary.sum() < 5:  # lowered slightly for the 30% split size
            continue
        prec = precision_score(y_thresh, preds_binary, zero_division=0)
        trade_frac = preds_binary.mean()
        
        if prec >= 0.55 and trade_frac >= 0.05:
            score = prec * trade_frac
            if score > best_score:
                best_score = score
                best_threshold = threshold"""
                
    new_targ_logic = """    # 2. Optimize Threshold for this Weight PER SYMBOL (on set T)
    final_preds_t = best_w * preds_lgb_t + (1 - best_w) * preds_xgb_t
    
    symbol_thresholds = {}
    
    # We slice sym_series_val just like we slice y_val
    sym_thresh = sym_series_val.iloc[split:]
    unique_symbols = sym_thresh.unique()
    
    for sym in unique_symbols:
        sym_mask = (sym_thresh == sym).values
        if sum(sym_mask) < 50:  # If barely any trades in OOS
            symbol_thresholds[sym] = 0.5
            continue
            
        y_sym = y_thresh.iloc[sym_mask]
        preds_sym = final_preds_t[sym_mask]
        
        sym_best_thresh = 0.5
        sym_best_score = 0
        
        for threshold in np.linspace(0.4, 0.7, 31):
            preds_binary = (preds_sym >= threshold).astype(int)
            if preds_binary.sum() < 3: 
                continue
            prec = precision_score(y_sym, preds_binary, zero_division=0)
            trade_frac = preds_binary.mean()
            
            if prec >= 0.53 and trade_frac >= 0.02:
                score = prec * trade_frac
                if score > sym_best_score:
                    sym_best_score = score
                    sym_best_thresh = threshold
                    
        symbol_thresholds[sym] = sym_best_thresh
        print(f"    -> {sym} best threshold: {sym_best_thresh:.3f}")
        
    best_threshold = symbol_thresholds  # Return the dict instead of scalar
"""
    content = content.replace(old_targ_logic, new_targ_logic)

    # 4. Update the calls to optimize_ensemble
    content = content.replace(
        'w_long, thresh_long, auc_long = optimize_ensemble(y_long_val, lgb_long_preds, xgb_long_preds, "LONG")',
        'sym_val = symbols_full.iloc[lgb_val_idx]\n    w_long, thresh_long, auc_long = optimize_ensemble(y_long_val, lgb_long_preds, xgb_long_preds, "LONG", sym_val)'
    )
    content = content.replace(
        'w_short, thresh_short, auc_short = optimize_ensemble(y_short_val, lgb_short_preds, xgb_short_preds, "SHORT")',
        'w_short, thresh_short, auc_short = optimize_ensemble(y_short_val, lgb_short_preds, xgb_short_preds, "SHORT", sym_val)'
    )

    # 5. Fix the dictionary JSON saving so it accepts the dict
    content = content.replace(
        '''        "ensemble_long_thresh": float(thresh_long),
        "ensemble_short_weight_lgb": float(w_short),
        "ensemble_short_thresh": float(thresh_short)''',
        '''        "ensemble_long_thresh": thresh_long,  # Now a dict
        "ensemble_short_weight_lgb": float(w_short),
        "ensemble_short_thresh": thresh_short  # Now a dict'''
    )

    with open(dst_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Universal script successfully generated at: {dst_file}")

if __name__ == "__main__":
    generate_universal_script()
