import { useState, useEffect, useCallback, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import {
  MarketData,
  ActiveTrade,
  ClosedTrade,
  PerformanceMetrics,
  Symbol,
  Candle
} from '@/types/trading';
// NO MOCK DATA - PURE REAL-TIME IMPORTS

const API_BASE_URL = 'http://localhost:5000';

// Utility function to get previous trading day (skip weekends)
function getPreviousTradingDay(date: Date): Date {
  const previousDay = new Date(date);
  previousDay.setDate(date.getDate() - 1);

  // If current day is Monday (1), go back to Friday
  if (date.getDay() === 1) {
    previousDay.setDate(date.getDate() - 3);
  }
  // If current day is Sunday (0), go back to Friday  
  else if (date.getDay() === 0) {
    previousDay.setDate(date.getDate() - 2);
  }

  return previousDay;
}

// Type for storing data per symbol
interface SymbolDataStore {
  historicalCandles: Candle[];
  currentMinuteCandle: Candle | null;      // In-progress candle (updates with every tick)
  lastClosedMinuteCandle: Candle | null;   // Last completed 1-min candle (for display)
  dailyCandle: Candle | null;
  marketData: MarketData | null;
}

export function useTradingData() {
  const [availableSymbols, setAvailableSymbols] = useState<Symbol[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<Symbol | null>(null);
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [activeTrade, setActiveTrade] = useState<ActiveTrade | null>(null);
  const [closedTrades, setClosedTrades] = useState<ClosedTrade[]>([]);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [historicalCandles, setHistoricalCandles] = useState<Candle[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);

  const socketRef = useRef<Socket | null>(null);
  const selectedSymbolRef = useRef<Symbol | null>(null);
  const initialLoadCompleteRef = useRef(false);

  // Store data for ALL symbols (background processing)
  const allSymbolsDataRef = useRef<Record<string, SymbolDataStore>>({});

  // Sync ref with state
  useEffect(() => {
    selectedSymbolRef.current = selectedSymbol;
  }, [selectedSymbol]);

  // Initialize symbols and fetch history for ALL of them
  useEffect(() => {
    const initializeData = async () => {
      try {
        // Fetch watchlist first
        const watchlistRes = await fetch(`${API_BASE_URL}/api/watchlist`);
        const watchlistData = await watchlistRes.json();

        const symbols: Symbol[] = watchlistData.symbols.map((s: string) => ({
          id: s,
          name: s,
          exchange: 'NSE'
        }));
        setAvailableSymbols(symbols);

        // Initialize data store for all symbols
        symbols.forEach(sym => {
          if (!allSymbolsDataRef.current[sym.id]) {
            allSymbolsDataRef.current[sym.id] = {
              historicalCandles: [],
              currentMinuteCandle: null,
              lastClosedMinuteCandle: null,
              dailyCandle: null,
              marketData: null
            };
          }
        });

        // Fetch history for ALL symbols at startup (don't await - let it load in background)
        symbols.forEach(sym => {
          fetchHistoryForSymbol(sym.id);
        });

        // Set the first symbol and IMMEDIATELY fetch its prediction
        if (symbols.length > 0) {
          const firstSymbol = symbols[0];
          setSelectedSymbol(firstSymbol);
          selectedSymbolRef.current = firstSymbol;

          // Fetch prediction immediately for the first symbol
          try {
            const predRes = await fetch(`${API_BASE_URL}/api/prediction/${firstSymbol.id}`);
            const predData = await predRes.json();

            if (predData.prediction) {
              const p = predData.prediction;
              const initialMarketData: MarketData = {
                symbol: firstSymbol.id,
                currentPrice: p.last_price || 0,
                change: p.change || 0,
                changePercent: p.change_percent || 0,
                dailyCandle: {
                  timestamp: new Date(),
                  open: p.last_price || 0,
                  high: p.last_price || 0,
                  low: p.last_price || 0,
                  close: p.last_price || 0,
                  volume: 0
                },
                currentCandle: {
                  timestamp: new Date(),
                  open: p.last_price || 0,
                  high: p.last_price || 0,
                  low: p.last_price || 0,
                  close: p.last_price || 0,
                  volume: 0
                },
                vwap: p.vwap || p.last_price || 0,
                ema20: p.ema20 || p.last_price || 0,
                threshold: p.threshold || 0.65,
                setupQuality: p.setup_quality || 0,
                tradeStatus: (p.direction === 'BUY' && p.setup_quality > (p.threshold || 0.7))
                  ? 'IN_TRADE_LONG'
                  : (p.direction === 'SELL' && p.setup_quality > (p.threshold || 0.7))
                    ? 'IN_TRADE_SHORT'
                    : 'NO_TRADE',
                // Phase-1 model outputs
                expectedMfeAtr: p.expected_mfe_atr || 0,
                expectedMaeAtr: p.expected_mae_atr || 0,
                riskMultiplier: p.risk_multiplier || 0,
                directionalBias: p.directional_bias || 0,
                sessionQuality: p.session_quality || 0,
                shouldNotTrade: p.should_not_trade || false,
                regime: p.regime ? {
                  volatility: p.regime.volatility ?? 1,
                  trend: p.regime.trend ?? 0,
                  marketState: p.regime.market_state ?? p.regime.marketState ?? 1,
                } : { volatility: 1, trend: 0, marketState: 1 },
              };

              // Set marketData BEFORE marking loading as complete
              setMarketData(initialMarketData);
              allSymbolsDataRef.current[firstSymbol.id].marketData = initialMarketData;
            }
          } catch (predErr) {
            console.error('Failed to fetch initial prediction:', predErr);
          }

          initialLoadCompleteRef.current = true;
        }
      } catch (err) {
        console.error('Failed to initialize:', err);
      } finally {
        // Always mark loading as complete, even on error
        setIsInitialLoading(false);
      }
    };

    initializeData();

    // Initialize with empty real data - NO MOCK DATA
    setClosedTrades([]);
    setMetrics({
      totalTrades: 0,
      winRate: 0,
      avgPnlPerTrade: 0,
      cumulativeReturn: 0,
      maxDrawdown: 0,
      tradesToday: 0,
      profitFactor: 0,
      sharpeRatio: 0,
    });
  }, []);

  // Fetch history for a specific symbol
  const fetchHistoryForSymbol = useCallback((symbolId: string) => {
    fetch(`${API_BASE_URL}/api/history/${symbolId}`)
      .then(res => res.json())
      .then(data => {
        if (!data || !Array.isArray(data) || data.length === 0) {
          console.log(`No historical data for ${symbolId}`);
          return;
        }

        const candles: Candle[] = data.map((item: any) => ({
          timestamp: new Date(item.timestamp),
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume
        }));

        // Store in the all-symbols data store
        if (allSymbolsDataRef.current[symbolId]) {
          allSymbolsDataRef.current[symbolId].historicalCandles = candles;
        }

        console.log(`Loaded ${candles.length} historical candles for ${symbolId}`);

        // If this is the currently selected symbol, update the display
        if (selectedSymbolRef.current?.id === symbolId) {
          setHistoricalCandles(candles);
        }
      })
      .catch(err => console.error(`Failed to fetch history for ${symbolId}:`, err));
  }, []);

  // Handle Socket.IO connection - process ALL symbols
  useEffect(() => {
    const socket = io(API_BASE_URL, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
    socketRef.current = socket;

    socket.on('connect', () => {
      setIsConnected(true);
      console.log('Connected to SmartAPI Hub');
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      console.log('Disconnected from SmartAPI Hub');
    });

    socket.on('market_data_update', (payload) => {
      const outerData = typeof payload === 'string' ? JSON.parse(payload) : payload;
      const { symbol, data } = outerData.data;

      // Process tick for this symbol (background update for ALL symbols)
      processTickForSymbol(symbol, data);

      // If this is the selected symbol, also update the display
      const currentSelected = selectedSymbolRef.current;
      if (currentSelected && symbol === currentSelected.id) {
        updateDisplayForSelectedSymbol(symbol, data);
      }
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  // Process tick for any symbol (background processing)
  const processTickForSymbol = useCallback((symbol: string, data: any) => {
    const store = allSymbolsDataRef.current[symbol];
    if (!store) {
      // Initialize if not exists
      allSymbolsDataRef.current[symbol] = {
        historicalCandles: [],
        currentMinuteCandle: null,
        lastClosedMinuteCandle: null,
        dailyCandle: null,
        marketData: null
      };
    }

    const tickTime = new Date(data.timestamp);
    const minuteStart = new Date(tickTime);
    minuteStart.setSeconds(0, 0);
    const minuteTimestamp = minuteStart.getTime();

    const today = new Date();
    const previousDay = getPreviousTradingDay(today);
    const previousDayStart = new Date(previousDay.getFullYear(), previousDay.getMonth(), previousDay.getDate());
    const todayEnd = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 1);

    // Skip ticks not from current or previous trading day
    if (minuteStart < previousDayStart || minuteStart >= todayEnd) {
      return;
    }

    const symbolStore = allSymbolsDataRef.current[symbol];

    // Update current minute candle for this symbol
    let currentCandle = symbolStore.currentMinuteCandle;
    if (!currentCandle || currentCandle.timestamp.getTime() !== minuteTimestamp) {
      // New minute started - save previous candle as "lastClosedMinuteCandle"
      if (currentCandle) {
        symbolStore.lastClosedMinuteCandle = { ...currentCandle };
      }
      // Start a new in-progress candle
      currentCandle = {
        timestamp: minuteStart,
        open: data.ltp,
        high: data.ltp,
        low: data.ltp,
        close: data.ltp,
        volume: data.volume
      };
    } else {
      // Same minute - update in-progress candle
      currentCandle = {
        ...currentCandle,
        high: Math.max(currentCandle.high, data.ltp),
        low: Math.min(currentCandle.low, data.ltp),
        close: data.ltp,
        volume: data.volume
      };
    }
    symbolStore.currentMinuteCandle = currentCandle;

    // Update daily candle
    // IMPORTANT: Daily close should be the PREVIOUS day's close (prevclose), not LTP
    // The backend sends data.close as the previous day's closing price
    symbolStore.dailyCandle = {
      timestamp: new Date(),
      open: data.open ?? data.ltp,
      high: data.high ?? data.ltp,
      low: data.low ?? data.ltp,
      close: data.close ?? data.ltp,  // Use prevclose from backend, not LTP
      volume: data.volume
    };

    // Update historical candles for this symbol
    const candleMap = new Map<number, Candle>();
    symbolStore.historicalCandles.forEach(candle => {
      candleMap.set(candle.timestamp.getTime(), candle);
    });

    const timeKey = minuteStart.getTime();
    const existingCandle = candleMap.get(timeKey);

    if (existingCandle) {
      candleMap.set(timeKey, {
        ...existingCandle,
        close: data.ltp,
        high: Math.max(existingCandle.high, data.ltp),
        low: Math.min(existingCandle.low, data.ltp),
        volume: data.volume
      });
    } else {
      const lastCandle = symbolStore.historicalCandles.length > 0
        ? symbolStore.historicalCandles[symbolStore.historicalCandles.length - 1]
        : null;
      const openPrice = lastCandle ? lastCandle.close : data.ltp;

      candleMap.set(timeKey, {
        timestamp: minuteStart,
        open: openPrice,
        high: Math.max(openPrice, data.ltp),
        low: Math.min(openPrice, data.ltp),
        close: data.ltp,
        volume: data.volume
      });
    }

    symbolStore.historicalCandles = Array.from(candleMap.values())
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    // Update market data snapshot
    // For the 1-min candle display, use lastClosedMinuteCandle if available (shows closed candle)
    // Fall back to currentCandle if no closed candle exists yet (e.g., first minute of trading)
    const displayCandle = symbolStore.lastClosedMinuteCandle || currentCandle;

    symbolStore.marketData = {
      symbol: symbol,
      currentPrice: data.ltp,
      change: data.change,
      changePercent: data.change_percent,
      setupQuality: data.setup_quality ?? 0,
      threshold: data.threshold ?? 0.65,
      tradeStatus: data.trade_status ?? 'NO_TRADE',
      vwap: data.vwap ?? data.ltp,
      ema20: data.ema20 ?? data.ltp,
      dailyCandle: symbolStore.dailyCandle!,
      currentCandle: displayCandle,  // Show last closed candle's data, not in-progress
      // Phase-1 model outputs
      expectedMfeAtr: data.expected_mfe_atr ?? 0,
      expectedMaeAtr: data.expected_mae_atr ?? 0,
      riskMultiplier: data.risk_multiplier ?? 0,
      directionalBias: data.directional_bias ?? 0,
      sessionQuality: data.session_quality ?? 0,
      shouldNotTrade: data.should_not_trade ?? false,
      regime: data.regime ? {
        volatility: data.regime.volatility ?? 1,
        trend: data.regime.trend ?? 0,
        marketState: data.regime.market_state ?? data.regime.marketState ?? 1,
      } : { volatility: 1, trend: 0, marketState: 1 },
    };
  }, []);

  // Update display for the currently selected symbol
  const updateDisplayForSelectedSymbol = useCallback((symbol: string, data: any) => {
    const store = allSymbolsDataRef.current[symbol];
    if (!store) return;

    // Update market data display
    setMarketData(store.marketData);

    // Update historical candles display
    setHistoricalCandles([...store.historicalCandles]);
  }, []);

  // When selected symbol changes, immediately load from cache
  useEffect(() => {
    if (!selectedSymbol) return;

    // Skip prediction fetch on initial load - it's handled in initializeData
    if (!initialLoadCompleteRef.current) return;

    const store = allSymbolsDataRef.current[selectedSymbol.id];
    if (store) {
      // Immediately set cached data
      if (store.historicalCandles.length > 0) {
        setHistoricalCandles([...store.historicalCandles]);
      }
      if (store.marketData) {
        setMarketData(store.marketData);
      }
    }

    // Fetch prediction for this symbol - this is critical for initial load!
    // We MUST set marketData directly, not just update it conditionally
    fetch(`${API_BASE_URL}/api/prediction/${selectedSymbol.id}`)
      .then(res => res.json())
      .then(data => {
        if (data.prediction) {
          const p = data.prediction;

          // Create the market data object from prediction
          const newMarketData: MarketData = {
            symbol: selectedSymbol.id,
            currentPrice: p.last_price || 0,
            change: p.change || 0,
            changePercent: p.change_percent || 0,
            dailyCandle: {
              timestamp: new Date(),
              open: p.last_price || 0,
              high: p.last_price || 0,
              low: p.last_price || 0,
              close: p.last_price || 0,
              volume: 0
            },
            currentCandle: {
              timestamp: new Date(),
              open: p.last_price || 0,
              high: p.last_price || 0,
              low: p.last_price || 0,
              close: p.last_price || 0,
              volume: 0
            },
            vwap: p.vwap || p.last_price || 0,
            ema20: p.ema20 || p.last_price || 0,
            threshold: p.threshold || 0.65,
            setupQuality: p.setup_quality || 0,
            tradeStatus: (p.direction === 'BUY' && p.setup_quality > (p.threshold || 0.7))
              ? 'IN_TRADE_LONG'
              : (p.direction === 'SELL' && p.setup_quality > (p.threshold || 0.7))
                ? 'IN_TRADE_SHORT'
                : 'NO_TRADE',
            // Phase-1 model outputs
            expectedMfeAtr: p.expected_mfe_atr || 0,
            expectedMaeAtr: p.expected_mae_atr || 0,
            riskMultiplier: p.risk_multiplier || 0,
            directionalBias: p.directional_bias || 0,
            sessionQuality: p.session_quality || 0,
            shouldNotTrade: p.should_not_trade || false,
            regime: p.regime ? {
              volatility: p.regime.volatility ?? 1,
              trend: p.regime.trend ?? 0,
              marketState: p.regime.market_state ?? p.regime.marketState ?? 1,
            } : { volatility: 1, trend: 0, marketState: 1 },
          };

          // ALWAYS set marketData from prediction - this fixes the initial load issue
          // The prediction API returns real data, so we should use it directly
          setMarketData(prev => {
            if (!prev) {
              // No existing data - use the new data directly
              return newMarketData;
            }
            // Merge with existing data, preferring real-time values when available
            return {
              ...newMarketData,
              // Keep existing currentPrice/change if they're non-zero (from live ticks)
              currentPrice: prev.currentPrice > 0 ? prev.currentPrice : newMarketData.currentPrice,
              change: prev.change !== 0 ? prev.change : newMarketData.change,
              changePercent: prev.changePercent !== 0 ? prev.changePercent : newMarketData.changePercent,
              // Keep existing candles if they have real data
              dailyCandle: prev.dailyCandle.volume > 0 ? prev.dailyCandle : newMarketData.dailyCandle,
              currentCandle: prev.currentCandle.volume > 0 ? prev.currentCandle : newMarketData.currentCandle,
            };
          });

          // Also update the store so symbol switching is instant
          if (allSymbolsDataRef.current[selectedSymbol.id]) {
            allSymbolsDataRef.current[selectedSymbol.id].marketData = newMarketData;
          }
        }
      })
      .catch(err => console.error('Failed to fetch prediction:', err));

  }, [selectedSymbol?.id]);


  const handleSymbolChange = useCallback((symbol: Symbol) => {
    setSelectedSymbol(symbol);
    setActiveTrade(null);

    // Immediately load cached data for the new symbol
    const store = allSymbolsDataRef.current[symbol.id];
    if (store) {
      setHistoricalCandles([...store.historicalCandles]);
      if (store.marketData) {
        setMarketData(store.marketData);
      } else {
        setMarketData(null);
      }
    } else {
      setHistoricalCandles([]);
      setMarketData(null);
    }
  }, []);

  return {
    symbols: availableSymbols,
    selectedSymbol: selectedSymbol || (availableSymbols.length > 0 ? availableSymbols[0] : null),
    setSelectedSymbol: handleSymbolChange,
    marketData,
    activeTrade,
    closedTrades,
    metrics,
    historicalCandles,
    isConnected,
    isInitialLoading,
    socket: socketRef.current,  // Expose socket for Phase2 integration
  };
}
