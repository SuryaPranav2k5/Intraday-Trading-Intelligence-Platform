import { ChevronDown, Search } from 'lucide-react';
import { useState } from 'react';
import { Symbol } from '@/types/trading';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';

interface SymbolSelectorProps {
  symbols: Symbol[];
  selectedSymbol: Symbol | null;
  onSymbolChange: (symbol: Symbol) => void;
}

export function SymbolSelector({ symbols, selectedSymbol, onSymbolChange }: SymbolSelectorProps) {
  const [search, setSearch] = useState('');

  const filteredSymbols = symbols.filter(
    s => s.id.toLowerCase().includes(search.toLowerCase()) ||
      s.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          className="w-full justify-between bg-card hover:bg-muted/50 h-11"
        >
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
            <span className="font-mono font-semibold">{selectedSymbol?.id || 'Loading...'}</span>
            {selectedSymbol && (
              <span className="text-muted-foreground text-xs hidden sm:inline">
                {selectedSymbol.exchange}
              </span>
            )}
          </div>
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-64" align="start">
        <div className="p-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search symbols..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-8 h-9 bg-muted/50"
            />
          </div>
        </div>
        <div className="max-h-64 overflow-y-auto custom-scrollbar">
          {filteredSymbols.map((symbol) => (
            <DropdownMenuItem
              key={symbol.id}
              onClick={() => {
                onSymbolChange(symbol);
                setSearch('');
              }}
              className={`flex items-center justify-between cursor-pointer ${symbol.id === selectedSymbol?.id ? 'bg-primary/10' : ''
                }`}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono font-medium">{symbol.id}</span>
              </div>
              <span className="text-xs text-muted-foreground">{symbol.exchange}</span>
            </DropdownMenuItem>
          ))}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

