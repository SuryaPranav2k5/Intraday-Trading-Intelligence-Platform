export interface Candle {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export type TradeDirection = 'LONG' | 'SHORT';

export type TradeStatus = 'NO_TRADE' | 'IN_TRADE_LONG' | 'IN_TRADE_SHORT' | 'WAITING';

export type ExitReason = 'TARGET_HIT' | 'STOP_HIT' | 'TIME_DECAY_EXIT' | 'TIMEOUT' | 'MANUAL';

export interface ActiveTrade {
  symbol: string;
  entryTime: Date;
  entryPrice: number;
  direction: TradeDirection;
  targetPrice: number;
  stopLoss: number;
  currentPrice: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  timeInTrade: number; // minutes
}

export interface ClosedTrade {
  id: string;
  symbol: string;
  entryTime: Date;
  exitTime: Date;
  direction: TradeDirection;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  pnlPercent: number;
  exitReason: ExitReason;
}

export interface MarketData {
  symbol: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  dailyCandle: Candle;      // Daily OHLC (main context)
  currentCandle: Candle;    // Current 1-min candle (execution level)
  setupQuality: number;
  threshold: number;
  tradeStatus: TradeStatus;
  vwap: number;
  ema20: number;
  // Phase-1 model outputs
  expectedMfeAtr?: number;
  expectedMaeAtr?: number;
  riskMultiplier?: number;
  directionalBias?: number;  // -1 = bearish, 0 = neutral, 1 = bullish
  sessionQuality?: number;
  shouldNotTrade?: boolean;
  regime?: {
    volatility: number;  // 0=low, 1=normal, 2=high
    trend: number;       // 0=ranging, 1=weak, 2=strong
    marketState: number; // 0=compression, 1=normal, 2=expansion
  };
}

export interface PerformanceMetrics {
  totalTrades: number;
  winRate: number;
  avgPnlPerTrade: number;
  cumulativeReturn: number;
  maxDrawdown: number;
  tradesToday: number;
  profitFactor: number;
  sharpeRatio: number;
}

export interface Symbol {
  id: string;
  name: string;
  exchange: string;
}
