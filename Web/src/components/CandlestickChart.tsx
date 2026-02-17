import { useEffect, useRef, useMemo } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickSeries, LineSeries, createSeriesMarkers } from 'lightweight-charts';
import { Candle, ActiveTrade } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CandlestickChartProps {
  candles: Candle[];
  vwap?: number;
  ema20?: number;
  activeTrade?: ActiveTrade | null;
}

export function CandlestickChart({ candles, vwap, ema20, activeTrade }: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const vwapSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const ema20SeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const markersRef = useRef<any>(null);


  // Calculate date range for display
  const dateRange = useMemo(() => {
    if (!candles || candles.length === 0) return '';

    const firstCandle = candles[0];
    const lastCandle = candles[candles.length - 1];

    const firstDate = firstCandle.timestamp.toLocaleDateString('en-IN', {
      month: 'short',
      day: 'numeric'
    });
    const lastDate = lastCandle.timestamp.toLocaleDateString('en-IN', {
      month: 'short',
      day: 'numeric'
    });

    return firstDate === lastDate ? firstDate : `${firstDate} - ${lastDate}`;
  }, [candles]);

  // Map candles to Lightweight Charts format
  const chartData = useMemo(() => {
    if (!candles || candles.length === 0) return [];

    // Convert to lightweight charts format with proper timestamp handling
    const chartPoints = candles.map(candle => {
      // Convert to Unix timestamp in seconds (LWC requirement)
      const timestamp = Math.floor(candle.timestamp.getTime() / 1000);
      return {
        time: timestamp as any,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      };
    });

    // Remove duplicates and ensure ascending order (LWC requirement)
    const uniquePoints = new Map();
    chartPoints.forEach(point => {
      // Keep the latest data for each timestamp
      uniquePoints.set(point.time, point);
    });

    return Array.from(uniquePoints.values())
      .sort((a, b) => a.time - b.time);
  }, [candles]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    console.log("Initializing chart...");
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.05)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.05)' },
      },
      width: chartContainerRef.current.clientWidth || 600,
      height: 400,
      crosshair: {
        mode: 0, // CrosshairMode.Normal
        vertLine: {
          width: 1,
          color: 'rgba(148, 163, 184, 0.4)',
          style: 3, // Dotted
          labelBackgroundColor: '#1e293b',
        },
        horzLine: {
          width: 1,
          color: 'rgba(148, 163, 184, 0.4)',
          style: 3, // Dotted
          labelBackgroundColor: '#1e293b',
        },
      },
      localization: {
        timeFormatter: (time: number) => {
          // Convert UTC timestamp to local time string
          const date = new Date(time * 1000);
          return date.toLocaleTimeString('en-IN', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
          });
        },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: 'rgba(148, 163, 184, 0.1)',
        rightOffset: 12, // Space for real-time updates like brokers
        barSpacing: 6, // Optimal spacing for 1-minute candles
        minBarSpacing: 0.5,
        tickMarkFormatter: (time: number) => {
          // Show local time on the x-axis tick marks
          const date = new Date(time * 1000);
          return date.toLocaleTimeString('en-IN', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
          });
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(148, 163, 184, 0.1)',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
        autoScale: true, // Auto-scale like broker charts
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });

    const vwapSeries = chart.addSeries(LineSeries, {
      color: '#3b82f6',
      lineWidth: 2,
      lineStyle: 2, // Dashed
      title: 'VWAP',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });

    const ema20Series = chart.addSeries(LineSeries, {
      color: '#f59e0b',
      lineWidth: 2,
      lineStyle: 2, // Dashed
      title: 'EMA20',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });



    candleSeriesRef.current = candleSeries;
    vwapSeriesRef.current = vwapSeries;
    ema20SeriesRef.current = ema20Series;
    chartRef.current = chart;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    // Initialize markers primitive
    markersRef.current = createSeriesMarkers(candleSeries, []);

    // Initial fit
    if (chartData.length > 0) {
      candleSeries.setData(chartData);
      chart.timeScale().fitContent();
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []); // Only once on mount

  // Track data state
  const isFirstDataLoad = useRef(true);
  const lastTimestampRef = useRef<number | null>(null);

  // Reset chart for new symbol
  useEffect(() => {
    isFirstDataLoad.current = true;
    lastTimestampRef.current = null;
    if (candleSeriesRef.current) {
      candleSeriesRef.current.setData([]);
      if (markersRef.current) {
        markersRef.current.setMarkers([]);
      }
    }
  }, [candles.length > 0 ? candles[0].timestamp.getTime() : 0]);

  // Update data whenever it changes - EXACTLY LIKE REAL BROKERS
  useEffect(() => {
    if (!candleSeriesRef.current || chartData.length === 0) return;

    const latestPoint = chartData[chartData.length - 1];

    if (isFirstDataLoad.current) {
      // First load of historical data for current + previous day
      console.log('Setting initial chart data:', chartData.length, 'candles');
      candleSeriesRef.current.setData(chartData);

      // Add "Market Open" marker for the first candle of today
      if (chartData.length > 1) {
        const today = new Date();
        const todayStart = new Date(today.getFullYear(), today.getMonth(), today.getDate());
        const todayStartTimestamp = Math.floor(todayStart.getTime() / 1000);

        // Find the first candle of today
        const todayFirstCandle = chartData.find(candle => candle.time >= todayStartTimestamp);
        if (todayFirstCandle) {
          // Add a marker at market open (9:15 AM IST usually, but we'll use the first candle found today)
          const marketOpenTime = todayStartTimestamp + (9 * 60 + 15) * 60; // 9:15 AM IST
          const nearestCandle = chartData.find(c => c.time >= marketOpenTime) || todayFirstCandle;

          if (markersRef.current) {
            markersRef.current.setMarkers([
              {
                time: nearestCandle.time,
                position: 'belowBar',
                color: 'rgba(148, 163, 184, 0.8)',
                shape: 'arrowUp',
                text: 'Market Open',
              }
            ]);
          }
        }
      }

      chartRef.current?.timeScale().fitContent();
      isFirstDataLoad.current = false;
      lastTimestampRef.current = latestPoint.time;
    } else {
      // Real-time update logic - EXACTLY how broker charts work
      if (latestPoint.time !== lastTimestampRef.current) {
        // Only update if timestamp is different to avoid duplicate time errors
        candleSeriesRef.current.update(latestPoint);
        lastTimestampRef.current = latestPoint.time;

        // Auto-scroll to show latest data (like real brokers)
        setTimeout(() => {
          chartRef.current?.timeScale().scrollToRealTime();
        }, 100);
      } else {
        // Same timestamp, update the current candle (live OHLC movement within the same minute)
        candleSeriesRef.current.update(latestPoint);
      }
    }
  }, [chartData]);

  // Update indicators - REAL-TIME like brokers
  useEffect(() => {
    if (vwapSeriesRef.current && vwap && chartData.length > 0) {
      const latestPoint = chartData[chartData.length - 1];
      vwapSeriesRef.current.update({ time: latestPoint.time, value: vwap });
    }
  }, [vwap, chartData]);

  useEffect(() => {
    if (ema20SeriesRef.current && ema20 && chartData.length > 0) {
      const latestPoint = chartData[chartData.length - 1];
      ema20SeriesRef.current.update({ time: latestPoint.time, value: ema20 });
    }
  }, [ema20, chartData]);

  return (
    <Card variant="glass" className="h-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-primary" />
            Intraday Chart
            <Badge variant="outline" className="text-xs font-mono">1M</Badge>
            {dateRange && (
              <Badge variant="secondary" className="text-xs">
                {dateRange}
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-3 text-xs">
            {vwap && (
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-primary" />
                <span className="text-muted-foreground">VWAP</span>
                <span className="font-mono">{vwap.toFixed(2)}</span>
              </div>
            )}
            {ema20 && (
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-warning" />
                <span className="text-muted-foreground">EMA20</span>
                <span className="font-mono">{ema20.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="h-[420px] p-0 overflow-hidden relative">
        <div
          ref={chartContainerRef}
          className={cn("w-full h-full cursor-crosshair", chartData.length === 0 && "invisible")}
        />
        {chartData.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground bg-card/20 backdrop-blur-[2px] z-10">
            <div className="flex flex-col items-center gap-3">
              <div className="h-4 w-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              <span>Loading intraday market data...</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
