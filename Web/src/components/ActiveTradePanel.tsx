import { Clock, Target, ShieldAlert, TrendingUp, TrendingDown } from 'lucide-react';
import { ActiveTrade } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';

interface ActiveTradePanelProps {
  trade: ActiveTrade | null;
}

export function ActiveTradePanel({ trade }: ActiveTradePanelProps) {
  if (!trade) {
    return (
      <Card variant="glass" className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-muted-foreground" />
            Active Trade
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="h-16 w-16 rounded-full bg-muted/30 flex items-center justify-center mb-4">
              <TrendingUp className="h-8 w-8 text-muted-foreground/50" />
            </div>
            <p className="text-muted-foreground text-sm">No active trade</p>
            <p className="text-muted-foreground/60 text-xs mt-1">
              Waiting for AI signal...
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const isLong = trade.direction === 'LONG';
  const isProfitable = trade.unrealizedPnl >= 0;
  
  // Calculate progress to target/stop
  const range = Math.abs(trade.targetPrice - trade.stopLoss);
  const currentPosition = Math.abs(trade.currentPrice - trade.stopLoss);
  const progressToTarget = Math.min(100, Math.max(0, (currentPosition / range) * 100));

  return (
    <Card variant="trading" className={cn(
      'h-full border-2 transition-all duration-300',
      isLong ? 'border-profit/30 shadow-profit/5' : 'border-loss/30 shadow-loss/5',
      'shadow-lg'
    )}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <div className={cn(
              'h-2 w-2 rounded-full animate-pulse',
              isLong ? 'bg-profit' : 'bg-loss'
            )} />
            Active Trade
          </CardTitle>
          <Badge variant={isLong ? 'long' : 'short'} className="text-sm px-3">
            {isLong ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
            {trade.direction}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Symbol & Time */}
        <div className="flex items-center justify-between">
          <span className="font-mono font-semibold text-lg">{trade.symbol}</span>
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Clock className="h-4 w-4" />
            <span className="font-mono text-sm">{trade.timeInTrade} min</span>
          </div>
        </div>

        {/* Entry Info */}
        <div className="grid grid-cols-2 gap-3">
          <div className="metric-card">
            <span className="data-label">Entry Price</span>
            <span className="block font-mono text-base font-semibold mt-1">
              ₹{trade.entryPrice.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </span>
          </div>
          <div className="metric-card">
            <span className="data-label">Entry Time</span>
            <span className="block font-mono text-base font-semibold mt-1">
              {trade.entryTime.toLocaleTimeString('en-IN', { 
                hour: '2-digit', 
                minute: '2-digit',
                hour12: false 
              })}
            </span>
          </div>
        </div>

        {/* Target & Stop Loss */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1.5 text-profit">
              <Target className="h-4 w-4" />
              <span>Target</span>
            </div>
            <span className="font-mono font-medium">
              ₹{trade.targetPrice.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </span>
          </div>
          
          <Progress 
            value={progressToTarget} 
            className="h-2"
          />
          
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1.5 text-loss">
              <ShieldAlert className="h-4 w-4" />
              <span>Stop Loss</span>
            </div>
            <span className="font-mono font-medium">
              ₹{trade.stopLoss.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </span>
          </div>
        </div>

        {/* Current Price & PnL */}
        <div className="pt-3 border-t border-border/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">Current Price</span>
            <span className="font-mono font-semibold text-lg">
              ₹{trade.currentPrice.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </span>
          </div>
          
          <div className={cn(
            'p-3 rounded-lg text-center',
            isProfitable 
              ? 'bg-profit/10 border border-profit/20' 
              : 'bg-loss/10 border border-loss/20'
          )}>
            <span className="data-label">Unrealized P&L</span>
            <div className={cn(
              'font-mono text-2xl font-bold mt-1',
              isProfitable ? 'text-profit' : 'text-loss'
            )}>
              {isProfitable ? '+' : ''}₹{trade.unrealizedPnl.toFixed(2)}
              <span className="text-sm ml-2">
                ({isProfitable ? '+' : ''}{trade.unrealizedPnlPercent.toFixed(2)}%)
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
