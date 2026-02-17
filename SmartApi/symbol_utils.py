"""
Symbol Utilities for SmartAPI Compatibility
This module provides utilities for normalizing and mapping symbols for SmartAPI
"""

# Symbol mapping for SmartAPI compatibility - PRIORITY SYMBOLS
SYMBOL_MAPPING = {
    # Priority symbol mappings (direct mapping - no aliases needed)
    'TVSMOTOR': 'TVSMOTOR',    
    'RELIANCE': 'RELIANCE',   
    'LT': 'LT',               
    'TITAN': 'TITAN',        
    'SIEMENS': 'SIEMENS',      
    'TATAELXSI': 'TATAELXSI', 
}

# Complete SmartAPI symbol mappings with tokens
SMARTAPI_SYMBOL_MAPPINGS = {
    # 🥇 Verified Equity Symbols
    'TVSMOTOR': {'symbol': 'TVSMOTOR-EQ', 'token': '8479'}, 
    'RELIANCE': {'symbol': 'RELIANCE-EQ', 'token': '2885'}, 
    'LT': {'symbol': 'LT-EQ', 'token': '11483'},          
    'TITAN': {'symbol': 'TITAN-EQ', 'token': '3506'},    
    'SIEMENS': {'symbol': 'SIEMENS-EQ', 'token': '3150'},   
    'TATAELXSI': {'symbol': 'TATAELXSI-EQ', 'token': '3411'}, 
}

# Reverse mapping for display purposes
REVERSE_SYMBOL_MAPPING = {v: k for k, v in SYMBOL_MAPPING.items()}

def normalize_symbol_for_smartapi(symbol):
    """
    Normalize symbol names for SmartAPI compatibility
    
    Args:
        symbol (str): Original symbol name
        
    Returns:
        str: SmartAPI compatible symbol name
    """
    normalized = symbol
    
    # Apply symbol mapping
    normalized = SYMBOL_MAPPING.get(normalized, normalized)
    
    return normalized

def normalize_symbol_for_display(symbol):
    """
    Normalize symbol for display purposes (remove exchange suffixes)
    
    Args:
        symbol (str): Symbol name with potential suffixes
        
    Returns:
        str: Clean symbol name for display
    """
    return symbol

def get_original_symbol(smartapi_symbol):
    """
    Get the original symbol name from SmartAPI symbol
    
    Args:
        smartapi_symbol (str): SmartAPI compatible symbol
        
    Returns:
        str: Original symbol name if mapped, otherwise returns input
    """
    return REVERSE_SYMBOL_MAPPING.get(smartapi_symbol, smartapi_symbol)

def validate_symbol_format(symbol):
    """
    Validate if symbol is in correct format for SmartAPI
    
    Args:
        symbol (str): Symbol to validate
        
    Returns:
        bool: True if valid format, False otherwise
    """
    # Basic validation rules
    if not symbol or not isinstance(symbol, str):
        return False
    
    
    # Should be uppercase
    if symbol != symbol.upper():
        return False
    
    # Should not be empty after stripping
    if not symbol.strip():
        return False
    
    return True

def get_smartapi_watchlist():
    """
    Get the verified SmartAPI watchlist
    
    Returns:
        list: List of verified equity symbols
    """
    # Priority symbols that are verified to work with SmartAPI
    return list(SYMBOL_MAPPING.keys())

def get_verified_smartapi_watchlist():
    """
    Get the verified SmartAPI watchlist with priority symbols confirmed working
    
    Returns:
        list: List of verified SmartAPI symbols that have confirmed market feeds
    """
    # Priority symbols that are verified to work with SmartAPI
    return list(SMARTAPI_SYMBOL_MAPPINGS.keys())

def get_smartapi_token(symbol):
    """
    Get SmartAPI token for a symbol
    
    Args:
        symbol (str): Original symbol name
        
    Returns:
        str: SmartAPI token if found, None otherwise
    """
    normalized = normalize_symbol_for_smartapi(symbol)
    mapping = SMARTAPI_SYMBOL_MAPPINGS.get(normalized)
    return mapping['token'] if mapping else None

def get_smartapi_trading_symbol(symbol):
    """
    Get SmartAPI trading symbol for a symbol
    
    Args:
        symbol (str): Original symbol name
        
    Returns:
        str: SmartAPI trading symbol if found, None otherwise
    """
    normalized = normalize_symbol_for_smartapi(symbol)
    mapping = SMARTAPI_SYMBOL_MAPPINGS.get(normalized)
    return mapping['symbol'] if mapping else None

def get_all_smartapi_mappings():
    """
    Get all SmartAPI symbol mappings
    
    Returns:
        dict: Complete mapping of all symbols with tokens
    """
    return SMARTAPI_SYMBOL_MAPPINGS.copy()


if __name__ == "__main__":
    # Test the symbol utilities
    print("🔧 Testing Symbol Utilities - Priority Symbols")
    print("=" * 50)
    
    test_symbols = ['TVSMOTOR', 'RELIANCE', 'LT', 'TITAN']
    
    for symbol in test_symbols:
        normalized = normalize_symbol_for_smartapi(symbol)
        display = normalize_symbol_for_display(symbol)
        valid = validate_symbol_format(normalized)
        
        print(f"Original: {symbol}")
        print(f"SmartAPI: {normalized}")
        print(f"Display:  {display}")
        print(f"Valid:    {valid}")
        print("-" * 30)
    
    print(f"\nVerified SmartAPI Watchlist ({len(get_smartapi_watchlist())} symbols):")
    for i, symbol in enumerate(get_smartapi_watchlist(), 1):
        token = get_smartapi_token(symbol)
        print(f"{i}. {symbol} (Token: {token})")