import { Header } from '@/components/Header';
import { SymbolSelector } from '@/components/SymbolSelector';
import { MarketDataPanel } from '@/components/MarketDataPanel';
import { ActiveTradePanel } from '@/components/ActiveTradePanel';
import { Phase2SupervisionPanel } from '@/components/Phase2SupervisionPanel';
import { PerformanceMetrics } from '@/components/PerformanceMetrics';
import { TradeLogTable } from '@/components/TradeLogTable';
import { useTradingData } from '@/hooks/useTradingData';
import { Skeleton } from '@/components/ui/skeleton';

const Index = () => {
  const {
    symbols,
    selectedSymbol,
    setSelectedSymbol,
    marketData,
    activeTrade,
    closedTrades,
    metrics,
    isConnected,
    isInitialLoading,
    socket,
  } = useTradingData();

  return (
    <div className="min-h-screen bg-background">
      <Header isConnected={isConnected} />

      <main className="container mx-auto px-4 py-6 space-y-6">
        {/* Symbol Selector */}
        <div className="max-w-xs">
          {isInitialLoading ? (
            <Skeleton className="h-11 w-full" />
          ) : (
            <SymbolSelector
              symbols={symbols}
              selectedSymbol={selectedSymbol}
              onSymbolChange={setSelectedSymbol}
            />
          )}
        </div>

        {/* Main Grid - 3 Columns */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Market Data & AI Signal */}
          <div className="lg:col-span-1">
            {isInitialLoading ? (
              <div className="space-y-4">
                <Skeleton className="h-48 w-full" />
                <Skeleton className="h-32 w-full" />
                <Skeleton className="h-40 w-full" />
              </div>
            ) : (
              <MarketDataPanel data={marketData} />
            )}
          </div>

          {/* Center Column - Phase 2 Supervision */}
          <div className="lg:col-span-1">
            {isInitialLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <Phase2SupervisionPanel
                symbol={selectedSymbol?.id || ''}
                currentPrice={marketData?.currentPrice || 0}
                socket={socket}
              />
            )}
          </div>

          {/* Right Column - Active Trade */}
          <div className="lg:col-span-1">
            {isInitialLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : (
              <ActiveTradePanel trade={activeTrade} />
            )}
          </div>
        </div>

        {/* Performance Metrics */}
        {metrics && <PerformanceMetrics metrics={metrics} />}

        {/* Trade Log */}
        <TradeLogTable
          trades={closedTrades}
          symbols={symbols.map(s => s.id)}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-4 mt-8">
        <div className="container mx-auto px-4">
          <p className="text-center text-xs text-muted-foreground">
            AI Trading Terminal • Paper Trading Only • No Real Money Involved
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
