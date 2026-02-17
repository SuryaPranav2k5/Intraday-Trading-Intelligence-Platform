import { useState, useEffect, useCallback, useRef } from 'react';
import {
    Play,
    TrendingUp,
    Clock,
    AlertCircle,
    CheckCircle2,
    XCircle,
    Activity,
    BarChart3,
    TrendingDown,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import {
    SupervisionState,
    SupervisionEntryType,
    SupervisionTradeStatus,
    SupervisedTradeEntry,
    SupervisedTradeMetrics,
    ExitRecommendationReason,
    INITIAL_SUPERVISION_STATE,
} from '@/types/supervision';
import { Socket } from 'socket.io-client';

interface Phase2SupervisionPanelProps {
    symbol: string;
    currentPrice: number;
    exchangeTime?: Date;
    socket?: Socket | null;
}

// Status banner configurations
const STATUS_CONFIG: Record<SupervisionTradeStatus, {
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    bgClass: string;
    textClass: string;
    borderClass: string;
}> = {
    'NO_ACTIVE_TRADE': {
        label: 'No Active Trade',
        icon: Activity,
        bgClass: 'bg-muted/40',
        textClass: 'text-muted-foreground',
        borderClass: 'border-border/50',
    },
    'TRADE_ACTIVE_MONITORED': {
        label: 'Trade Active — Logic Valid',
        icon: CheckCircle2,
        bgClass: 'bg-profit/10',
        textClass: 'text-profit',
        borderClass: 'border-profit/30',
    },
    'EXIT_RECOMMENDED': {
        label: 'Exit Recommended',
        icon: AlertCircle,
        bgClass: 'bg-warning/10',
        textClass: 'text-warning',
        borderClass: 'border-warning/30',
    },
};

export function Phase2SupervisionPanel({
    symbol,
    currentPrice,
    exchangeTime,
    socket,
}: Phase2SupervisionPanelProps) {
    // === State Management ===
    const [supervisionState, setSupervisionState] = useState<SupervisionState>(
        INITIAL_SUPERVISION_STATE
    );

    // Use ref to access currentPrice in callbacks without triggering re-renders
    const currentPriceRef = useRef(currentPrice);
    useEffect(() => {
        currentPriceRef.current = currentPrice;
    }, [currentPrice]);

    // === Socket.IO Event Listeners ===
    useEffect(() => {
        if (!socket) return;

        // Listen for supervision updates from backend
        const handleSupervisionUpdate = (payload: any) => {
            const data = payload.data;
            if (data.symbol !== symbol) return;

            // Map backend status to frontend status
            const statusMap: Record<string, SupervisionTradeStatus> = {
                'IN_TRADE': 'TRADE_ACTIVE_MONITORED',
                'EXIT_RECOMMENDED': 'EXIT_RECOMMENDED',
                'NO_ACTIVE_TRADE': 'NO_ACTIVE_TRADE',
            };

            const newStatus = statusMap[data.status] || 'NO_ACTIVE_TRADE';

            if (data.entry && data.metrics) {
                setSupervisionState({
                    status: newStatus,
                    entry: {
                        symbol: data.entry.symbol,
                        entryType: data.entry.entry_type as SupervisionEntryType,
                        entryPrice: data.entry.entry_price,
                        entryTime: new Date(data.entry.entry_time),
                        exchangeTime: new Date(data.entry.entry_time),
                    },
                    metrics: {
                        minutesInTrade: data.metrics.minutes_in_trade,
                        unrealizedPnL: data.metrics.unrealized_pnl,
                        unrealizedPnLPercent: data.metrics.unrealized_pnl_percent,
                        maxFavorableExcursion: data.metrics.max_favorable_excursion,
                        maxAdverseExcursion: data.metrics.max_adverse_excursion,
                        entryAtr: data.metrics.entry_atr || 0,
                        profitFloor: data.metrics.profit_floor,
                        profitFloorActive: data.metrics.profit_floor_active || false,
                        lastUpdated: new Date(),
                    },
                    exitReason: data.exit_reason as ExitRecommendationReason | null,
                    currentPrice: currentPriceRef.current,
                });
            } else {
                // No active trade
                setSupervisionState(INITIAL_SUPERVISION_STATE);
            }
        };

        socket.on('phase2_supervision_update', handleSupervisionUpdate);

        // Request current state on mount
        socket.emit('phase2_get_state', { symbol });

        return () => {
            socket.off('phase2_supervision_update', handleSupervisionUpdate);
        };
    }, [socket, symbol]);

    // === Request state when symbol changes ===
    useEffect(() => {
        if (!socket || !symbol) return;
        socket.emit('phase2_get_state', { symbol });
    }, [socket, symbol]);

    // === Entry Registration ===
    const handleEntryClick = useCallback((entryType: SupervisionEntryType, direction: number = 1) => {
        const now = exchangeTime || new Date();

        // Optimistic UI update
        const entry: SupervisedTradeEntry = {
            symbol,
            entryType,
            entryPrice: currentPrice,
            entryTime: now,
            exchangeTime: now,
        };

        const initialMetrics: SupervisedTradeMetrics = {
            minutesInTrade: 0,
            unrealizedPnL: 0,
            unrealizedPnLPercent: 0,
            maxFavorableExcursion: 0,
            maxAdverseExcursion: 0,
            entryAtr: 0,
            profitFloor: null,
            profitFloorActive: false,
            lastUpdated: now,
        };

        setSupervisionState({
            status: 'TRADE_ACTIVE_MONITORED',
            entry,
            metrics: initialMetrics,
            exitReason: null,
            currentPrice,
        });

        // Send to backend
        if (socket) {
            socket.emit('phase2_register_entry', {
                symbol,
                entry_type: entryType,
                entry_price: currentPrice,
                entry_time: now.toISOString(),
                direction: direction,
            });
        }
    }, [symbol, currentPrice, exchangeTime, socket]);

    // === Clear Trade (Manual Exit) ===
    const handleClearTrade = useCallback(() => {
        setSupervisionState(INITIAL_SUPERVISION_STATE);

        // Notify backend
        if (socket) {
            socket.emit('phase2_clear_trade', { symbol });
        }
    }, [socket, symbol]);

    // === Reset when symbol changes during active trade ===
    useEffect(() => {
        if (supervisionState.entry && supervisionState.entry.symbol !== symbol) {
            handleClearTrade();
        }
    }, [symbol, supervisionState.entry, handleClearTrade]);

    // === Computed Values ===
    const statusConfig = STATUS_CONFIG[supervisionState.status];
    const StatusIcon = statusConfig.icon;
    const isActive = supervisionState.status !== 'NO_ACTIVE_TRADE';
    const isProfitable = (supervisionState.metrics?.unrealizedPnL ?? 0) >= 0;

    return (
        <Card variant="glass" className="h-full">
            <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-primary" />
                        Phase 2 Supervision
                    </CardTitle>
                    <Badge variant="outline" className="text-xs font-normal">
                        {symbol}
                    </Badge>
                </div>
            </CardHeader>

            <CardContent className="space-y-4">
                {/* === Status Banner === */}
                <div className={cn(
                    'p-3 rounded-lg border flex items-center gap-3',
                    statusConfig.bgClass,
                    statusConfig.borderClass
                )}>
                    <StatusIcon className={cn('h-5 w-5 shrink-0', statusConfig.textClass)} />
                    <span className={cn('font-medium text-sm', statusConfig.textClass)}>
                        {statusConfig.label}
                    </span>
                </div>

                {/* === Exit Reason (only when exit recommended) === */}
                {supervisionState.status === 'EXIT_RECOMMENDED' && supervisionState.exitReason && (
                    <div className="p-3 rounded-lg bg-warning/5 border border-warning/20">
                        <p className="text-sm text-warning font-medium">
                            {supervisionState.exitReason}
                        </p>
                    </div>
                )}

                {/* === Entry Button (only when no active trade) === */}
                {!isActive && (
                    <div className="space-y-3">
                        <p className="text-xs text-muted-foreground text-center">
                            Register trade entry manually
                        </p>
                        <div className="grid grid-cols-2 gap-3">
                            <Button
                                variant="long"
                                size="lg"
                                className="flex items-center justify-center h-auto py-3 gap-2 bg-profit text-white hover:bg-profit/90"
                                onClick={() => handleEntryClick('VWAP', 1)}
                            >
                                <TrendingUp className="h-4 w-4" />
                                <span className="text-sm font-semibold">Long</span>
                            </Button>
                            <Button
                                variant="destructive"
                                size="lg"
                                className="flex items-center justify-center h-auto py-3 gap-2 bg-loss text-white hover:bg-loss/90"
                                onClick={() => handleEntryClick('VWAP', -1)}
                            >
                                <TrendingDown className="h-4 w-4" />
                                <span className="text-sm font-semibold">Short</span>
                            </Button>
                        </div>
                        <p className="text-[10px] text-muted-foreground/70 text-center">
                            No orders will be placed. You remain in full control.
                        </p>
                    </div>
                )}

                {/* === Entry Summary (locked after entry) === */}
                {isActive && supervisionState.entry && (
                    <div className="space-y-3">
                        <div className="flex items-center gap-2">
                            <div className="h-1.5 w-1.5 rounded-full bg-profit" />
                            <span className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                                Entry Summary
                            </span>
                            <span className="text-[10px] text-muted-foreground/50 ml-auto">
                                Locked
                            </span>
                        </div>

                        <div className="grid grid-cols-3 gap-2">
                            <div className="metric-card p-3">
                                <span className="data-label">Symbol</span>
                                <span className="block font-mono text-sm font-semibold mt-0.5">
                                    {supervisionState.entry.symbol}
                                </span>
                            </div>
                            <div className="metric-card p-3">
                                <span className="data-label">Entry Price</span>
                                <span className="block font-mono text-sm font-semibold mt-0.5">
                                    ₹{supervisionState.entry.entryPrice.toLocaleString('en-IN', {
                                        minimumFractionDigits: 2,
                                        maximumFractionDigits: 2
                                    })}
                                </span>
                            </div>
                            <div className="metric-card p-3">
                                <span className="data-label">Entry Time</span>
                                <span className="block font-mono text-sm font-semibold mt-0.5">
                                    {supervisionState.entry.entryTime.toLocaleTimeString('en-IN', {
                                        hour: '2-digit',
                                        minute: '2-digit',
                                        hour12: false,
                                    })}
                                </span>
                            </div>
                        </div>
                    </div>
                )}

                {/* === Live Trade Metrics === */}
                {isActive && supervisionState.metrics && (
                    <div className="space-y-3">
                        <div className="flex items-center gap-2">
                            <Clock className="h-3 w-3 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                                Live Metrics
                            </span>
                            <span className="text-[10px] text-muted-foreground/50 ml-auto">
                                Updates on 1-min close
                            </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                            <div className="metric-card p-3">
                                <span className="data-label">Time in Trade</span>
                                <span className="block font-mono text-sm font-semibold mt-0.5">
                                    {supervisionState.metrics.minutesInTrade} min
                                </span>
                            </div>
                            <div className="metric-card p-3">
                                <span className="data-label">Unrealized P&L</span>
                                <span className={cn(
                                    'block font-mono text-sm font-semibold mt-0.5',
                                    isProfitable ? 'text-profit' : 'text-loss'
                                )}>
                                    {isProfitable ? '+' : ''}
                                    ₹{supervisionState.metrics.unrealizedPnL.toFixed(2)}
                                    <span className="text-xs ml-1 opacity-80">
                                        ({isProfitable ? '+' : ''}{supervisionState.metrics.unrealizedPnLPercent.toFixed(2)}%)
                                    </span>
                                </span>
                            </div>
                            <div className="metric-card p-3">
                                <span className="data-label">Max Favorable (MFE)</span>
                                <span className={cn(
                                    'block font-mono text-sm font-semibold mt-0.5',
                                    supervisionState.metrics.maxFavorableExcursion >= 0 ? 'text-profit' : 'text-muted-foreground'
                                )}>
                                    {supervisionState.metrics.maxFavorableExcursion >= 0 ? '+' : ''}
                                    ₹{supervisionState.metrics.maxFavorableExcursion.toFixed(2)}
                                </span>
                            </div>
                            <div className="metric-card p-3">
                                <span className="data-label">Max Adverse (MAE)</span>
                                <span className={cn(
                                    'block font-mono text-sm font-semibold mt-0.5',
                                    supervisionState.metrics.maxAdverseExcursion < 0 ? 'text-loss' : 'text-muted-foreground'
                                )}>
                                    ₹{supervisionState.metrics.maxAdverseExcursion.toFixed(2)}
                                </span>
                            </div>
                        </div>

                        {/* === Profit Floor Status === */}
                        {supervisionState.metrics.entryAtr > 0 && (
                            <div className={cn(
                                'p-3 rounded-lg border mt-3',
                                supervisionState.metrics.profitFloorActive
                                    ? 'bg-profit/10 border-profit/30'
                                    : 'bg-muted/30 border-border/50'
                            )}>
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-muted-foreground">Profit Floor</span>
                                    <Badge variant={supervisionState.metrics.profitFloorActive ? 'default' : 'secondary'} className="text-xs">
                                        {supervisionState.metrics.profitFloorActive ? '🔒 LOCKED' : 'Not Active'}
                                    </Badge>
                                </div>
                                {supervisionState.metrics.profitFloorActive && supervisionState.metrics.profitFloor !== null && (
                                    <div className="mt-2 text-sm font-mono text-profit font-semibold">
                                        Floor: ₹{supervisionState.metrics.profitFloor.toFixed(2)}
                                    </div>
                                )}
                                <div className="text-[10px] text-muted-foreground/70 mt-1">
                                    Stop: -{(supervisionState.metrics.entryAtr * 1.5).toFixed(2)} | Trigger: +{supervisionState.metrics.entryAtr.toFixed(2)}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* === Manual Exit Button === */}
                {isActive && (
                    <div className="pt-2 border-t border-border/30">
                        <Button
                            variant="ghost"
                            size="sm"
                            className="w-full text-muted-foreground hover:text-foreground"
                            onClick={handleClearTrade}
                        >
                            <XCircle className="h-4 w-4 mr-2" />
                            Clear Supervision (I exited manually)
                        </Button>
                        <p className="text-[10px] text-muted-foreground/50 text-center mt-2">
                            This only clears the supervision — no orders are affected.
                        </p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
