import { 
  Symbol, 
  MarketData, 
  ActiveTrade, 
  ClosedTrade, 
  PerformanceMetrics,
  Candle 
} from '@/types/trading';

export const symbols: Symbol[] = [
  { id: 'RELIANCE', name: 'Reliance Industries', exchange: 'NSE' },
  { id: 'TCS', name: 'Tata Consultancy Services', exchange: 'NSE' },
  { id: 'INFY', name: 'Infosys Ltd', exchange: 'NSE' },
  { id: 'HDFCBANK', name: 'HDFC Bank', exchange: 'NSE' },
  { id: 'ICICIBANK', name: 'ICICI Bank', exchange: 'NSE' },
  { id: 'SBIN', name: 'State Bank of India', exchange: 'NSE' },
  { id: 'TATAMOTORS', name: 'Tata Motors', exchange: 'NSE' },
  { id: 'WIPRO', name: 'Wipro Ltd', exchange: 'NSE' },
];

const generateCandle = (basePrice: number, timestamp: Date): Candle => {
  const volatility = 0.002;
  const open = basePrice * (1 + (Math.random() - 0.5) * volatility);
  const close = basePrice * (1 + (Math.random() - 0.5) * volatility);
  const high = Math.max(open, close) * (1 + Math.random() * volatility);
  const low = Math.min(open, close) * (1 - Math.random() * volatility);
  
  return {
    timestamp,
    open: parseFloat(open.toFixed(2)),
    high: parseFloat(high.toFixed(2)),
    low: parseFloat(low.toFixed(2)),
    close: parseFloat(close.toFixed(2)),
    volume: Math.floor(Math.random() * 100000) + 50000,
  };
};

export const generateMarketData = (symbolId: string): MarketData => {
  const basePrices: Record<string, number> = {
    'RELIANCE': 2456.75,
    'TCS': 3892.40,
    'INFY': 1523.65,
    'HDFCBANK': 1678.90,
    'ICICIBANK': 1045.25,
    'SBIN': 628.40,
    'TATAMOTORS': 745.80,
    'WIPRO': 456.15,
  };

  const basePrice = basePrices[symbolId] || 1000;
  const currentPrice = basePrice * (1 + (Math.random() - 0.5) * 0.02);
  const change = currentPrice - basePrice;
  const setupQuality = Math.random();
  const threshold = 0.65;

  let tradeStatus: MarketData['tradeStatus'] = 'NO_TRADE';
  if (setupQuality >= threshold) {
    const rand = Math.random();
    if (rand > 0.7) tradeStatus = 'IN_TRADE_LONG';
    else if (rand > 0.4) tradeStatus = 'IN_TRADE_SHORT';
    else tradeStatus = 'WAITING';
  }

  return {
    symbol: symbolId,
    currentPrice: parseFloat(currentPrice.toFixed(2)),
    change: parseFloat(change.toFixed(2)),
    changePercent: parseFloat(((change / basePrice) * 100).toFixed(2)),
    currentCandle: generateCandle(currentPrice, new Date()),
    setupQuality: parseFloat(setupQuality.toFixed(3)),
    threshold,
    tradeStatus,
    vwap: parseFloat((currentPrice * (1 + (Math.random() - 0.5) * 0.005)).toFixed(2)),
    ema20: parseFloat((currentPrice * (1 + (Math.random() - 0.5) * 0.008)).toFixed(2)),
  };
};

export const generateActiveTrade = (symbolId: string, currentPrice: number): ActiveTrade | null => {
  const isLong = Math.random() > 0.5;
  const entryOffset = (Math.random() * 0.01 + 0.005) * (isLong ? -1 : 1);
  const entryPrice = currentPrice * (1 + entryOffset);
  const atr = currentPrice * 0.012;
  
  const direction = isLong ? 'LONG' : 'SHORT';
  const targetPrice = isLong 
    ? entryPrice + (atr * 2) 
    : entryPrice - (atr * 2);
  const stopLoss = isLong 
    ? entryPrice - atr 
    : entryPrice + atr;

  const unrealizedPnl = isLong 
    ? currentPrice - entryPrice 
    : entryPrice - currentPrice;
  const unrealizedPnlPercent = (unrealizedPnl / entryPrice) * 100;

  return {
    symbol: symbolId,
    entryTime: new Date(Date.now() - Math.floor(Math.random() * 30) * 60000),
    entryPrice: parseFloat(entryPrice.toFixed(2)),
    direction,
    targetPrice: parseFloat(targetPrice.toFixed(2)),
    stopLoss: parseFloat(stopLoss.toFixed(2)),
    currentPrice,
    unrealizedPnl: parseFloat(unrealizedPnl.toFixed(2)),
    unrealizedPnlPercent: parseFloat(unrealizedPnlPercent.toFixed(2)),
    timeInTrade: Math.floor(Math.random() * 30) + 1,
  };
};

export const generateClosedTrades = (count: number = 20): ClosedTrade[] => {
  const trades: ClosedTrade[] = [];
  const exitReasons: ClosedTrade['exitReason'][] = ['TARGET_HIT', 'STOP_HIT', 'TIME_DECAY_EXIT', 'TIMEOUT'];

  for (let i = 0; i < count; i++) {
    const symbolIndex = Math.floor(Math.random() * symbols.length);
    const symbol = symbols[symbolIndex].id;
    const direction = Math.random() > 0.5 ? 'LONG' : 'SHORT';
    const exitReason = exitReasons[Math.floor(Math.random() * exitReasons.length)];
    
    const basePrices: Record<string, number> = {
      'RELIANCE': 2456.75,
      'TCS': 3892.40,
      'INFY': 1523.65,
      'HDFCBANK': 1678.90,
      'ICICIBANK': 1045.25,
      'SBIN': 628.40,
      'TATAMOTORS': 745.80,
      'WIPRO': 456.15,
    };
    
    const entryPrice = basePrices[symbol] * (1 + (Math.random() - 0.5) * 0.02);
    const pnlPercent = exitReason === 'TARGET_HIT' 
      ? Math.random() * 2 + 0.5 
      : exitReason === 'STOP_HIT' 
        ? -(Math.random() * 1.5 + 0.3)
        : (Math.random() - 0.5) * 1;
    
    const exitPrice = entryPrice * (1 + pnlPercent / 100 * (direction === 'LONG' ? 1 : -1));
    const pnl = (exitPrice - entryPrice) * (direction === 'LONG' ? 1 : -1);

    const entryTime = new Date(Date.now() - (count - i) * 3600000 - Math.random() * 1800000);
    const exitTime = new Date(entryTime.getTime() + Math.random() * 1800000 + 300000);

    trades.push({
      id: `trade-${i + 1}`,
      symbol,
      entryTime,
      exitTime,
      direction,
      entryPrice: parseFloat(entryPrice.toFixed(2)),
      exitPrice: parseFloat(exitPrice.toFixed(2)),
      pnl: parseFloat(pnl.toFixed(2)),
      pnlPercent: parseFloat(pnlPercent.toFixed(2)),
      exitReason,
    });
  }

  return trades.sort((a, b) => b.exitTime.getTime() - a.exitTime.getTime());
};

export const calculatePerformanceMetrics = (trades: ClosedTrade[]): PerformanceMetrics => {
  if (trades.length === 0) {
    return {
      totalTrades: 0,
      winRate: 0,
      avgPnlPerTrade: 0,
      cumulativeReturn: 0,
      maxDrawdown: 0,
      tradesToday: 0,
      profitFactor: 0,
      sharpeRatio: 0,
    };
  }

  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl <= 0);
  
  const totalPnl = trades.reduce((sum, t) => sum + t.pnlPercent, 0);
  const avgPnl = totalPnl / trades.length;
  
  const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
  const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));
  
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const tradesToday = trades.filter(t => t.exitTime >= today).length;

  // Calculate max drawdown
  let peak = 0;
  let maxDrawdown = 0;
  let cumulative = 0;
  
  for (const trade of [...trades].reverse()) {
    cumulative += trade.pnlPercent;
    if (cumulative > peak) peak = cumulative;
    const drawdown = peak - cumulative;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  }

  // Sharpe ratio approximation
  const returns = trades.map(t => t.pnlPercent);
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  const sharpeRatio = stdDev > 0 ? (mean / stdDev) * Math.sqrt(252) : 0;

  return {
    totalTrades: trades.length,
    winRate: parseFloat(((winningTrades.length / trades.length) * 100).toFixed(1)),
    avgPnlPerTrade: parseFloat(avgPnl.toFixed(2)),
    cumulativeReturn: parseFloat(totalPnl.toFixed(2)),
    maxDrawdown: parseFloat(maxDrawdown.toFixed(2)),
    tradesToday,
    profitFactor: grossLoss > 0 ? parseFloat((grossProfit / grossLoss).toFixed(2)) : grossProfit > 0 ? Infinity : 0,
    sharpeRatio: parseFloat(sharpeRatio.toFixed(2)),
  };
};

export const generateHistoricalCandles = (basePrice: number, count: number = 100): Candle[] => {
  const candles: Candle[] = [];
  let currentPrice = basePrice;
  
  for (let i = count; i > 0; i--) {
    const timestamp = new Date(Date.now() - i * 60000);
    const trend = Math.sin(i / 10) * 0.001;
    currentPrice = currentPrice * (1 + trend + (Math.random() - 0.5) * 0.003);
    candles.push(generateCandle(currentPrice, timestamp));
  }
  
  return candles;
};
