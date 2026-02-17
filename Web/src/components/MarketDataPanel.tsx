import { TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import { MarketData, TradeStatus } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { SetupQualityGauge } from './SetupQualityGauge';
import { cn } from '@/lib/utils';

interface MarketDataPanelProps {
  data: MarketData | null;
}

const tradeStatusConfig: Record<TradeStatus, { label: string; variant: 'long' | 'short' | 'warning' | 'neutral' }> = {
  'NO_TRADE': { label: 'NO TRADE', variant: 'neutral' },
  'IN_TRADE_LONG': { label: 'IN TRADE (LONG)', variant: 'long' },
  'IN_TRADE_SHORT': { label: 'IN TRADE (SHORT)', variant: 'short' },
  'WAITING': { label: 'WAITING', variant: 'warning' },
};

export function MarketDataPanel({ data }: MarketDataPanelProps) {
  if (!data) {
    return (
      <div className="flex flex-col gap-4">
        <Card variant="trading" className="animate-pulse">
          <CardContent className="h-40 flex items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Waiting for market data...</span>
            </div>
          </CardContent>
        </Card>
        <Card variant="glass" className="h-32 animate-pulse" />
        <Card variant="glass" className="h-40 animate-pulse" />
      </div>
    );
  }

  const isPositive = data.change >= 0;
  const statusConfig = tradeStatusConfig[data.tradeStatus];

  return (
    <div className="grid gap-4 lg:gap-6">
      {/* Price Card */}
      <Card variant="trading" className="animate-fade-in">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Current Price
            </CardTitle>
            <Badge variant={statusConfig.variant}>
              {statusConfig.label}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-4xl font-bold tracking-tight">
              ₹{data.currentPrice.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </span>
            <div className={cn(
              'flex items-center gap-1 text-sm font-medium',
              isPositive ? 'text-profit' : 'text-loss'
            )}>
              {isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
              <span>{isPositive ? '+' : ''}{data.change.toFixed(2)}</span>
              <span>({isPositive ? '+' : ''}{data.changePercent}%)</span>
            </div>
          </div>

          {/* Indicators */}
          <div className="grid grid-cols-2 gap-4 mt-6">
            <div className="metric-card">
              <span className="data-label">VWAP</span>
              <span className="block font-mono text-lg font-semibold mt-1">
                ₹{data.vwap.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
              </span>
            </div>
            <div className="metric-card">
              <span className="data-label">EMA 20</span>
              <span className="block font-mono text-lg font-semibold mt-1">
                ₹{data.ema20.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Daily OHLC - Main Context Panel */}
      <Card variant="glass">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            📊 Daily OHLC
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="metric-card text-center">
              <span className="data-label">Open</span>
              <span className="block font-mono text-base font-semibold mt-1">
                {data.dailyCandle?.open?.toFixed(2) ?? '—'}
              </span>
            </div>
            <div className="metric-card text-center">
              <span className="data-label">High</span>
              <span className="block font-mono text-base font-semibold mt-1 text-profit">
                {data.dailyCandle?.high?.toFixed(2) ?? '—'}
              </span>
            </div>
            <div className="metric-card text-center">
              <span className="data-label">Low</span>
              <span className="block font-mono text-base font-semibold mt-1 text-loss">
                {data.dailyCandle?.low?.toFixed(2) ?? '—'}
              </span>
            </div>
            <div className="metric-card text-center">
              <span className="data-label">Prev Close</span>
              <span className="block font-mono text-base font-semibold mt-1">
                {data.dailyCandle?.close?.toFixed(2) ?? '—'}
              </span>
            </div>
          </div>
          <div className="mt-3 text-center">
            <span className="data-label">Volume</span>
            <span className="block font-mono text-sm text-muted-foreground mt-1">
              {data.dailyCandle?.volume?.toLocaleString('en-IN') ?? '—'}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Current 1-Min Candle - Execution Level Panel */}
      <Card variant="glass">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            ⏱️ Last 1-Min Candle
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="metric-card text-center">
              <span className="data-label">Open</span>
              <span className="block font-mono text-base font-semibold mt-1">
                {data.currentCandle?.open?.toFixed(2) ?? '—'}
              </span>
            </div>
            <div className="metric-card text-center">
              <span className="data-label">High</span>
              <span className="block font-mono text-base font-semibold mt-1 text-profit">
                {data.currentCandle?.high?.toFixed(2) ?? '—'}
              </span>
            </div>
            <div className="metric-card text-center">
              <span className="data-label">Low</span>
              <span className="block font-mono text-base font-semibold mt-1 text-loss">
                {data.currentCandle?.low?.toFixed(2) ?? '—'}
              </span>
            </div>
            <div className="metric-card text-center">
              <span className="data-label">Close</span>
              <span className="block font-mono text-base font-semibold mt-1">
                {data.currentCandle?.close?.toFixed(2) ?? '—'}
              </span>
            </div>
          </div>
          <div className="mt-3 text-center">
            <span className="data-label">Volume</span>
            <span className="block font-mono text-sm text-muted-foreground mt-1">
              {data.currentCandle?.volume?.toLocaleString('en-IN') ?? '—'}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Setup Quality */}
      <Card variant="glass">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            🤖 AI Signal
          </CardTitle>
        </CardHeader>
        <CardContent>
          <SetupQualityGauge
            quality={data.setupQuality}
            threshold={data.threshold}
            expectedMfeAtr={data.expectedMfeAtr}
            expectedMaeAtr={data.expectedMaeAtr}
            directionalBias={data.directionalBias}
            sessionQuality={data.sessionQuality}
            shouldNotTrade={data.shouldNotTrade}
            regime={data.regime}
          />
        </CardContent>
      </Card>
    </div>
  );
}

