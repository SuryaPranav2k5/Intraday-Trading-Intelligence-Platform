import { useState, useCallback, useEffect, useRef } from 'react';
import {
    SupervisionState,
    SupervisionEntryType,
    SupervisedTradeEntry,
    SupervisedTradeMetrics,
    SupervisionTradeStatus,
    ExitRecommendationReason,
    INITIAL_SUPERVISION_STATE,
} from '@/types/supervision';

interface UseTradeSupervisionOptions {
    symbol: string;
    currentPrice: number;
    onEntryRegistered?: (entry: SupervisedTradeEntry) => void;
}

interface TradeSupervisionHook {
    state: SupervisionState;
    registerEntry: (entryType: SupervisionEntryType) => void;
    clearTrade: () => void;
    isActive: boolean;
    canEnterTrade: boolean;
}

/**
 * Hook for managing Phase 2 Trade Supervision state.
 * 
 * Key principles:
 * - Never auto-executes trades
 * - Only updates metrics on completed 1-minute candles
 * - User remains the final decision-maker
 * - Provides clear, unambiguous status information
 */
export function useTradeSupervision({
    symbol,
    currentPrice,
    onEntryRegistered,
}: UseTradeSupervisionOptions): TradeSupervisionHook {
    const [state, setState] = useState<SupervisionState>(INITIAL_SUPERVISION_STATE);

    // Track the last minute we updated metrics
    const lastMetricUpdateMinuteRef = useRef<number>(-1);

    // Track MFE/MAE across the trade
    const mfeRef = useRef<number>(0);
    const maeRef = useRef<number>(0);

    // Register a new trade entry
    const registerEntry = useCallback((entryType: SupervisionEntryType) => {
        const now = new Date();

        const entry: SupervisedTradeEntry = {
            symbol,
            entryType,
            entryPrice: currentPrice,
            entryTime: now,
            exchangeTime: now, // In production, this would come from exchange
        };

        const initialMetrics: SupervisedTradeMetrics = {
            minutesInTrade: 0,
            unrealizedPnL: 0,
            unrealizedPnLPercent: 0,
            maxFavorableExcursion: 0,
            maxAdverseExcursion: 0,
            lastUpdated: now,
        };

        // Reset MFE/MAE tracking
        mfeRef.current = 0;
        maeRef.current = 0;
        lastMetricUpdateMinuteRef.current = now.getMinutes();

        setState({
            status: 'TRADE_ACTIVE_MONITORED',
            entry,
            metrics: initialMetrics,
            exitReason: null,
            currentPrice,
        });

        onEntryRegistered?.(entry);
    }, [symbol, currentPrice, onEntryRegistered]);

    // Clear the current trade (manual action)
    const clearTrade = useCallback(() => {
        setState(INITIAL_SUPERVISION_STATE);
        mfeRef.current = 0;
        maeRef.current = 0;
        lastMetricUpdateMinuteRef.current = -1;
    }, []);

    // Update metrics when price changes (only on new minute boundaries)
    useEffect(() => {
        if (state.status === 'NO_ACTIVE_TRADE' || !state.entry || !state.metrics) {
            return;
        }

        const now = new Date();
        const currentMinute = now.getMinutes();

        // Only update on new completed 1-minute candles
        if (currentMinute === lastMetricUpdateMinuteRef.current) {
            // Still same minute - just update current price for display
            setState(prev => ({
                ...prev,
                currentPrice,
            }));
            return;
        }

        // New minute - update all metrics
        lastMetricUpdateMinuteRef.current = currentMinute;

        const entryPrice = state.entry.entryPrice;
        const pnl = currentPrice - entryPrice;
        const pnlPercent = (pnl / entryPrice) * 100;

        // Update MFE/MAE
        if (pnl > mfeRef.current) {
            mfeRef.current = pnl;
        }
        if (pnl < maeRef.current) {
            maeRef.current = pnl;
        }

        // Calculate minutes in trade
        const minutesInTrade = Math.floor(
            (now.getTime() - state.entry.entryTime.getTime()) / 60000
        );

        const updatedMetrics: SupervisedTradeMetrics = {
            minutesInTrade,
            unrealizedPnL: pnl,
            unrealizedPnLPercent: pnlPercent,
            maxFavorableExcursion: mfeRef.current,
            maxAdverseExcursion: maeRef.current,
            lastUpdated: now,
        };

        setState(prev => ({
            ...prev,
            metrics: updatedMetrics,
            currentPrice,
        }));
    }, [currentPrice, state.status, state.entry, state.metrics]);

    // Reset supervision when symbol changes
    useEffect(() => {
        if (state.entry && state.entry.symbol !== symbol) {
            // Symbol changed - keep the trade data but note it's for a different symbol
            // Don't auto-clear in case user switches back
        }
    }, [symbol, state.entry]);

    return {
        state: {
            ...state,
            currentPrice, // Always reflect latest price
        },
        registerEntry,
        clearTrade,
        isActive: state.status !== 'NO_ACTIVE_TRADE',
        canEnterTrade: state.status === 'NO_ACTIVE_TRADE',
    };
}

/**
 * Storage for supervised trades across all symbols.
 * This allows maintaining trade supervision state when switching between symbols.
 */
export function useMultiSymbolSupervision() {
    const [trades, setTrades] = useState<Record<string, SupervisionState>>({});

    const getTradeForSymbol = useCallback((symbol: string): SupervisionState => {
        return trades[symbol] || INITIAL_SUPERVISION_STATE;
    }, [trades]);

    const setTradeForSymbol = useCallback((symbol: string, state: SupervisionState) => {
        setTrades(prev => ({
            ...prev,
            [symbol]: state,
        }));
    }, []);

    const clearTradeForSymbol = useCallback((symbol: string) => {
        setTrades(prev => ({
            ...prev,
            [symbol]: INITIAL_SUPERVISION_STATE,
        }));
    }, []);

    return {
        trades,
        getTradeForSymbol,
        setTradeForSymbol,
        clearTradeForSymbol,
    };
}
