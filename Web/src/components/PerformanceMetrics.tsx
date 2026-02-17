import { 
  TrendingUp, 
  Trophy, 
  BarChart3, 
  Target, 
  AlertTriangle,
  Calendar,
  Percent,
  Activity
} from 'lucide-react';
import { PerformanceMetrics as Metrics } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface PerformanceMetricsProps {
  metrics: Metrics;
}

export function PerformanceMetrics({ metrics }: PerformanceMetricsProps) {
  const metricItems = [
    {
      label: 'Total Trades',
      value: metrics.totalTrades.toString(),
      icon: BarChart3,
      color: 'text-primary',
      bgColor: 'bg-primary/10',
    },
    {
      label: 'Win Rate',
      value: `${metrics.winRate}%`,
      icon: Trophy,
      color: metrics.winRate >= 50 ? 'text-profit' : 'text-loss',
      bgColor: metrics.winRate >= 50 ? 'bg-profit/10' : 'bg-loss/10',
    },
    {
      label: 'Avg P&L/Trade',
      value: `${metrics.avgPnlPerTrade >= 0 ? '+' : ''}${metrics.avgPnlPerTrade}%`,
      icon: Percent,
      color: metrics.avgPnlPerTrade >= 0 ? 'text-profit' : 'text-loss',
      bgColor: metrics.avgPnlPerTrade >= 0 ? 'bg-profit/10' : 'bg-loss/10',
    },
    {
      label: 'Cumulative Return',
      value: `${metrics.cumulativeReturn >= 0 ? '+' : ''}${metrics.cumulativeReturn}%`,
      icon: TrendingUp,
      color: metrics.cumulativeReturn >= 0 ? 'text-profit' : 'text-loss',
      bgColor: metrics.cumulativeReturn >= 0 ? 'bg-profit/10' : 'bg-loss/10',
    },
    {
      label: 'Max Drawdown',
      value: `-${metrics.maxDrawdown}%`,
      icon: AlertTriangle,
      color: 'text-warning',
      bgColor: 'bg-warning/10',
    },
    {
      label: 'Trades Today',
      value: metrics.tradesToday.toString(),
      icon: Calendar,
      color: 'text-primary',
      bgColor: 'bg-primary/10',
    },
    {
      label: 'Profit Factor',
      value: metrics.profitFactor === Infinity ? '∞' : metrics.profitFactor.toFixed(2),
      icon: Target,
      color: metrics.profitFactor >= 1 ? 'text-profit' : 'text-loss',
      bgColor: metrics.profitFactor >= 1 ? 'bg-profit/10' : 'bg-loss/10',
    },
    {
      label: 'Sharpe Ratio',
      value: metrics.sharpeRatio.toFixed(2),
      icon: Activity,
      color: metrics.sharpeRatio >= 1 ? 'text-profit' : metrics.sharpeRatio >= 0 ? 'text-warning' : 'text-loss',
      bgColor: metrics.sharpeRatio >= 1 ? 'bg-profit/10' : metrics.sharpeRatio >= 0 ? 'bg-warning/10' : 'bg-loss/10',
    },
  ];

  return (
    <Card variant="glass">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Performance Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {metricItems.map((item) => (
            <div 
              key={item.label}
              className="metric-card group hover:bg-muted/40 transition-colors"
            >
              <div className="flex items-center gap-2 mb-2">
                <div className={cn('p-1.5 rounded-md', item.bgColor)}>
                  <item.icon className={cn('h-3.5 w-3.5', item.color)} />
                </div>
              </div>
              <span className="data-label">{item.label}</span>
              <span className={cn(
                'block font-mono text-lg font-bold mt-1',
                item.color
              )}>
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
