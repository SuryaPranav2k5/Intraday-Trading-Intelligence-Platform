import { Activity, Bell, Settings, Wifi } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

interface HeaderProps {
  isConnected: boolean;
}

export function Header({ isConnected }: HeaderProps) {
  const currentTime = new Date().toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-card/95 backdrop-blur-sm">
      <div className="flex h-14 items-center justify-between px-4 lg:px-6">
        {/* Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 border border-primary/20">
            <Activity className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">AI Trading Terminal</h1>
            <p className="text-xs text-muted-foreground hidden sm:block">Paper Trading Dashboard</p>
          </div>
        </div>

        {/* Center - Status & Time */}
        <div className="hidden md:flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Wifi className={`h-4 w-4 ${isConnected ? 'text-success' : 'text-loss'}`} />
            <Badge variant={isConnected ? 'live' : 'destructive'}>
              {isConnected ? 'LIVE' : 'DISCONNECTED'}
            </Badge>
          </div>
          <div className="h-4 w-px bg-border" />
          <div className="font-mono text-sm text-muted-foreground">
            {currentTime}
          </div>
          <Badge variant="outline" className="font-mono text-xs">
            NSE
          </Badge>
        </div>

        {/* Right - Actions */}
        <div className="flex items-center gap-2">
          <Badge variant="warning" className="hidden sm:flex">
            PAPER TRADING
          </Badge>
          <Button variant="ghost" size="icon" className="relative">
            <Bell className="h-4 w-4" />
            <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-primary animate-pulse" />
          </Button>
          <Button variant="ghost" size="icon">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
}
