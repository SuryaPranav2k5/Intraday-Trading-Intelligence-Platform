// Phase 2 Trade Supervision Types
// These types define the UI state for the trade supervision panel.
// No auto-execution - purely informational and for manual decision-making.

export type SupervisionEntryType = 'VWAP' | 'Trend';

export type SupervisionTradeStatus =
    | 'NO_ACTIVE_TRADE'
    | 'TRADE_ACTIVE_MONITORED'
    | 'EXIT_RECOMMENDED';

// Reasons for exit recommendation (only ONE is shown at a time)
// Updated to match Phase-2 engine exit reasons
export type ExitRecommendationReason =
    | 'Max loss hit'
    | 'Profit floor hit'
    | 'Time exit'
    | 'End of day exit'
    | 'Session stopped (bad day)'
    | 'Absorption failure (stalled)'
    | 'Time failure (hypothesis expired)'
    | 'Volatility failure (environment changed)';

export interface SupervisedTradeEntry {
    symbol: string;
    entryType: SupervisionEntryType;
    entryPrice: number;
    entryTime: Date;
    exchangeTime: Date;  // The exchange time at entry
}

export interface SupervisedTradeMetrics {
    minutesInTrade: number;
    unrealizedPnL: number;           // In currency (₹)
    unrealizedPnLPercent: number;    // Percentage
    maxFavorableExcursion: number;   // MFE - best unrealized P&L seen
    maxAdverseExcursion: number;     // MAE - worst unrealized P&L seen
    entryAtr: number;                // ATR at entry (for stop/floor calculation)
    profitFloor: number | null;      // Profit floor level (set when profit > 1 ATR)
    profitFloorActive: boolean;      // Is profit floor currently active?
    lastUpdated: Date;
}

export interface SupervisionState {
    status: SupervisionTradeStatus;
    entry: SupervisedTradeEntry | null;
    metrics: SupervisedTradeMetrics | null;
    exitReason: ExitRecommendationReason | null;  // Only shown when status is EXIT_RECOMMENDED
    currentPrice: number;
}

// Initial empty state
export const INITIAL_SUPERVISION_STATE: SupervisionState = {
    status: 'NO_ACTIVE_TRADE',
    entry: null,
    metrics: null,
    exitReason: null,
    currentPrice: 0,
};
