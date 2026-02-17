import { useState } from 'react';
import { 
  ArrowUpDown, 
  Download, 
  Filter, 
  TrendingUp, 
  TrendingDown,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { ClosedTrade, ExitReason } from '@/types/trading';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { cn } from '@/lib/utils';

interface TradeLogTableProps {
  trades: ClosedTrade[];
  symbols: string[];
}

type SortField = 'exitTime' | 'pnlPercent';
type SortOrder = 'asc' | 'desc';

const exitReasonLabels: Record<ExitReason, { label: string; color: string }> = {
  'TARGET_HIT': { label: 'Target', color: 'text-profit' },
  'STOP_HIT': { label: 'Stop', color: 'text-loss' },
  'TIME_DECAY_EXIT': { label: 'Time Decay', color: 'text-warning' },
  'TIMEOUT': { label: 'Timeout', color: 'text-muted-foreground' },
  'MANUAL': { label: 'Manual', color: 'text-primary' },
};

const ITEMS_PER_PAGE = 10;

export function TradeLogTable({ trades, symbols }: TradeLogTableProps) {
  const [symbolFilter, setSymbolFilter] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField>('exitTime');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [currentPage, setCurrentPage] = useState(1);

  // Filter and sort trades
  const filteredTrades = trades
    .filter(trade => symbolFilter === 'all' || trade.symbol === symbolFilter)
    .sort((a, b) => {
      let comparison = 0;
      if (sortField === 'exitTime') {
        comparison = a.exitTime.getTime() - b.exitTime.getTime();
      } else if (sortField === 'pnlPercent') {
        comparison = a.pnlPercent - b.pnlPercent;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

  const totalPages = Math.ceil(filteredTrades.length / ITEMS_PER_PAGE);
  const paginatedTrades = filteredTrades.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  const exportToCsv = () => {
    const headers = ['Symbol', 'Entry Time', 'Exit Time', 'Direction', 'Entry Price', 'Exit Price', 'P&L', 'P&L %', 'Exit Reason'];
    const rows = filteredTrades.map(trade => [
      trade.symbol,
      trade.entryTime.toISOString(),
      trade.exitTime.toISOString(),
      trade.direction,
      trade.entryPrice,
      trade.exitPrice,
      trade.pnl,
      trade.pnlPercent,
      trade.exitReason,
    ]);

    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `trade_log_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card variant="glass">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-primary" />
            Trade Log
          </CardTitle>
          <div className="flex items-center gap-2">
            <Select value={symbolFilter} onValueChange={setSymbolFilter}>
              <SelectTrigger className="w-36 h-8">
                <SelectValue placeholder="All Symbols" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Symbols</SelectItem>
                {symbols.map(symbol => (
                  <SelectItem key={symbol} value={symbol}>
                    {symbol}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={exportToCsv}
              className="h-8"
            >
              <Download className="h-3.5 w-3.5 mr-1.5" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-lg border border-border/50 overflow-hidden">
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="w-24">Symbol</TableHead>
                  <TableHead className="w-20">Dir</TableHead>
                  <TableHead 
                    className="cursor-pointer hover:text-foreground transition-colors"
                    onClick={() => handleSort('exitTime')}
                  >
                    <div className="flex items-center gap-1">
                      Exit Time
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Exit</TableHead>
                  <TableHead 
                    className="text-right cursor-pointer hover:text-foreground transition-colors"
                    onClick={() => handleSort('pnlPercent')}
                  >
                    <div className="flex items-center justify-end gap-1">
                      P&L
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </TableHead>
                  <TableHead className="w-24">Reason</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {paginatedTrades.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-24 text-center text-muted-foreground">
                      No trades found
                    </TableCell>
                  </TableRow>
                ) : (
                  paginatedTrades.map((trade) => (
                    <TableRow key={trade.id} className="trade-row">
                      <TableCell className="font-mono font-medium">
                        {trade.symbol}
                      </TableCell>
                      <TableCell>
                        <Badge variant={trade.direction === 'LONG' ? 'long' : 'short'} className="text-xs">
                          {trade.direction === 'LONG' ? (
                            <TrendingUp className="h-3 w-3 mr-1" />
                          ) : (
                            <TrendingDown className="h-3 w-3 mr-1" />
                          )}
                          {trade.direction}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-mono text-sm text-muted-foreground">
                        {trade.exitTime.toLocaleString('en-IN', {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                          hour12: false,
                        })}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        ₹{trade.entryPrice.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        ₹{trade.exitPrice.toFixed(2)}
                      </TableCell>
                      <TableCell className={cn(
                        'text-right font-mono font-medium',
                        trade.pnl >= 0 ? 'text-profit' : 'text-loss'
                      )}>
                        <div>
                          {trade.pnl >= 0 ? '+' : ''}₹{trade.pnl.toFixed(2)}
                        </div>
                        <div className="text-xs opacity-80">
                          {trade.pnlPercent >= 0 ? '+' : ''}{trade.pnlPercent.toFixed(2)}%
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className={cn(
                          'text-xs font-medium',
                          exitReasonLabels[trade.exitReason].color
                        )}>
                          {exitReasonLabels[trade.exitReason].label}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-4">
            <span className="text-sm text-muted-foreground">
              Showing {(currentPage - 1) * ITEMS_PER_PAGE + 1}-{Math.min(currentPage * ITEMS_PER_PAGE, filteredTrades.length)} of {filteredTrades.length}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-sm font-medium px-2">
                {currentPage} / {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
