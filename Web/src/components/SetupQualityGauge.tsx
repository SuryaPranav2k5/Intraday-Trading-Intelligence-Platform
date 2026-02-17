import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react';

interface SetupQualityGaugeProps {
  quality: number;
  threshold: number;
  expectedMfeAtr?: number;
  expectedMaeAtr?: number;
  directionalBias?: number;
  sessionQuality?: number;
  shouldNotTrade?: boolean;
  regime?: {
    volatility: number;
    trend: number;
    marketState: number;
  };
}

const getRegimeLabel = (value: number, type: 'volatility' | 'trend' | 'marketState'): string => {
  if (type === 'volatility') {
    return ['Low', 'Normal', 'High'][value] || 'Unknown';
  }
  if (type === 'trend') {
    return ['Ranging', 'Weak', 'Strong'][value] || 'Unknown';
  }
  if (type === 'marketState') {
    return ['Compression', 'Normal', 'Expansion'][value] || 'Unknown';
  }
  return 'Unknown';
};

export function SetupQualityGauge({
  quality,
  threshold,
  expectedMfeAtr = 0,
  expectedMaeAtr = 0,
  directionalBias = 0,
  sessionQuality = 0,
  shouldNotTrade = false,
  regime
}: SetupQualityGaugeProps) {
  const percentage = quality * 100;
  const thresholdPercentage = threshold * 100;

  const getQualityColor = () => {
    if (shouldNotTrade) return 'text-muted-foreground';
    if (quality >= threshold) {
      // High Quality: Color depends on direction
      return directionalBias < 0 ? 'text-loss' : 'text-profit';
    }
    if (quality >= threshold * 0.7) return 'text-warning';
    return 'text-muted-foreground'; // Muted if very low quality
  };

  const getGaugeColor = () => {
    if (shouldNotTrade) return 'bg-muted';
    if (quality >= threshold) {
      // Valid Signal: Color depends on direction
      return directionalBias < 0 ? 'bg-loss' : 'bg-profit';
    }
    if (quality >= threshold * 0.7) return 'bg-warning';
    return 'bg-muted'; // Muted gray if below threshold
  };

  const getGlowClass = () => {
    if (quality >= threshold && !shouldNotTrade) {
      return directionalBias < 0 ? 'glow-loss' : 'glow-profit';
    }
    return '';
  };

  const getDirectionalIcon = () => {
    if (directionalBias > 0) return <TrendingUp className="h-4 w-4 text-profit" />;
    if (directionalBias < 0) return <TrendingDown className="h-4 w-4 text-loss" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  const getDirectionalLabel = () => {
    if (directionalBias > 0) return 'Bullish';
    if (directionalBias < 0) return 'Bearish';
    return 'Neutral';
  };

  return (
    <div className="space-y-4">
      {/* Session Warning */}
      {shouldNotTrade && (
        <div className="flex items-center gap-2 p-2 rounded-md bg-warning/10 border border-warning/20">
          <AlertTriangle className="h-4 w-4 text-warning" />
          <span className="text-xs text-warning font-medium">Session Quality Too Low</span>
        </div>
      )}

      {/* Main Quality Display */}
      <div className="flex items-center justify-between">
        <span className="data-label">Setup Quality</span>
        <span className={cn('font-mono text-2xl font-bold', getQualityColor())}>
          {percentage.toFixed(1)}%
        </span>
      </div>

      {/* Gauge bar */}
      <div className="relative h-3 rounded-full bg-muted/50 overflow-hidden">
        <div className="absolute inset-0 quality-gradient opacity-20" />
        <div
          className={cn(
            'absolute left-0 top-0 h-full rounded-full transition-all duration-500',
            getGaugeColor(),
            getGlowClass()
          )}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-foreground/80"
          style={{ left: `${thresholdPercentage}%` }}
        />
      </div>

      {/* Labels */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>0%</span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 bg-foreground/80 rounded-full" />
          Threshold: {thresholdPercentage.toFixed(0)}%
        </span>
        <span>100%</span>
      </div>

      {/* Model Predictions Grid */}
      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-border/30">
        {/* MFE/MAE */}
        <div className="metric-card p-2 text-center">
          <span className="data-label text-[10px]">Exp. MFE</span>
          <span className="block font-mono text-sm font-semibold text-profit mt-0.5">
            {expectedMfeAtr.toFixed(2)} ATR
          </span>
        </div>
        <div className="metric-card p-2 text-center">
          <span className="data-label text-[10px]">Exp. MAE</span>
          <span className="block font-mono text-sm font-semibold text-loss mt-0.5">
            {expectedMaeAtr.toFixed(2)} ATR
          </span>
        </div>
        {/* Directional Bias */}
        <div className="metric-card p-2 text-center">
          <span className="data-label text-[10px]">Bias</span>
          <div className="flex items-center justify-center gap-1 mt-0.5">
            {getDirectionalIcon()}
            <span className="font-mono text-sm font-semibold">
              {getDirectionalLabel()}
            </span>
          </div>
        </div>
      </div>

      {/* Regime Info */}
      {regime && (
        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-border/30">
          <span>Vol: {getRegimeLabel(regime.volatility, 'volatility')}</span>
          <span>Trend: {getRegimeLabel(regime.trend, 'trend')}</span>
          <span>State: {getRegimeLabel(regime.marketState, 'marketState')}</span>
        </div>
      )}

      {/* Status indicator */}
      <div className={cn(
        'text-center py-2 rounded-md text-sm font-medium uppercase tracking-wider',
        shouldNotTrade
          ? 'bg-muted/30 text-muted-foreground border border-border/30'
          : quality >= threshold
            ? 'bg-profit/10 text-profit border border-profit/20'
            : 'bg-muted/30 text-muted-foreground border border-border/30'
      )}>
        {shouldNotTrade
          ? '⚠ Session Stopped'
          : quality >= threshold
            ? '✓ Trade Signal Active'
            : 'Below Threshold'
        }
      </div>
    </div>
  );
}
