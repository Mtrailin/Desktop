import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

class PerformanceTracker:
    def __init__(self, initial_balance: float = 0.0):
        self.reset(initial_balance)

    def reset(self, initial_balance: float = 0.0):
        """Reset all tracking data"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades: List[Dict] = []
        self.positions: Dict[str, Dict] = {}
        self.pnl_history: List[Tuple[datetime, float]] = []
        self.hourly_returns: List[float] = []
        self.daily_returns: List[float] = []

    def update_balance(self, new_balance: float):
        """Update current balance and calculate returns"""
        self.pnl_history.append((datetime.now(), new_balance - self.current_balance))
        self.current_balance = new_balance
        self._calculate_returns()

    def add_trade(self, trade: Dict):
        """Add a new trade to history"""
        trade['timestamp'] = datetime.now()
        self.trades.append(trade)
        self._update_position(trade)

    def _update_position(self, trade: Dict):
        """Update position tracking"""
        symbol = trade['symbol']
        if trade['side'] == 'buy':
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'size': trade['size'],
                    'entry_price': trade['price'],
                    'current_price': trade['price']
                }
            else:
                # Average entry price for additional buys
                current_pos = self.positions[symbol]
                total_size = current_pos['size'] + trade['size']
                avg_price = (current_pos['size'] * current_pos['entry_price'] +
                           trade['size'] * trade['price']) / total_size
                self.positions[symbol]['size'] = total_size
                self.positions[symbol]['entry_price'] = avg_price

        elif trade['side'] == 'sell':
            if symbol in self.positions:
                current_pos = self.positions[symbol]
                current_pos['size'] -= trade['size']
                if current_pos['size'] <= 0:
                    del self.positions[symbol]

    def update_position_prices(self, price_updates: Dict[str, float]):
        """Update current prices for open positions"""
        for symbol, price in price_updates.items():
            if symbol in self.positions:
                self.positions[symbol]['current_price'] = price

    def get_open_pnl(self) -> float:
        """Calculate unrealized P&L from open positions"""
        total_pnl = 0.0
        for pos in self.positions.values():
            pnl = (pos['current_price'] - pos['entry_price']) * pos['size']
            total_pnl += pnl
        return total_pnl

    def get_closed_pnl(self) -> float:
        """Calculate realized P&L from closed trades"""
        return sum(pnl for _, pnl in self.pnl_history)

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.get_closed_pnl() + self.get_open_pnl()

    def get_roi(self) -> float:
        """Calculate Return on Investment"""
        if self.initial_balance == 0:
            return 0.0
        return (self.get_total_pnl() / self.initial_balance) * 100

    def _calculate_returns(self):
        """Calculate hourly and daily returns"""
        if len(self.pnl_history) < 2:
            return

        latest_pnl = self.pnl_history[-1][1]

        # Hourly returns
        hour_ago = datetime.now() - timedelta(hours=1)
        hour_pnls = [pnl for dt, pnl in self.pnl_history if dt >= hour_ago]
        if hour_pnls:
            self.hourly_returns.append(sum(hour_pnls))

        # Daily returns
        day_ago = datetime.now() - timedelta(days=1)
        day_pnls = [pnl for dt, pnl in self.pnl_history if dt >= day_ago]
        if day_pnls:
            self.daily_returns.append(sum(day_pnls))

    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])

        # Calculate average trade metrics
        if total_trades > 0:
            avg_win = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0) / max(winning_trades, 1)
            avg_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0)) / max(total_trades - winning_trades, 1)
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        metrics = {
            'total_pnl': self.get_total_pnl(),
            'open_pnl': self.get_open_pnl(),
            'closed_pnl': self.get_closed_pnl(),
            'roi': self.get_roi(),
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_trades': total_trades,
            'open_positions': len(self.positions),
            'hourly_sharpe': self._calculate_sharpe(self.hourly_returns),
            'daily_sharpe': self._calculate_sharpe(self.daily_returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

        return metrics

    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 365)  # Daily risk-free rate
        if len(excess_returns) < 2:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(365)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.pnl_history:
            return 0.0

        cumulative = np.cumsum([pnl for _, pnl in self.pnl_history])
        max_so_far = np.maximum.accumulate(cumulative)
        drawdowns = (max_so_far - cumulative) / max_so_far
        return np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0

    def save_history(self, filepath: str = 'performance_history.json'):
        """Save performance history to file"""
        data = {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'trades': self.trades,
            'pnl_history': [(dt.isoformat(), pnl) for dt, pnl in self.pnl_history],
            'hourly_returns': self.hourly_returns,
            'daily_returns': self.daily_returns
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def load_history(self, filepath: str = 'performance_history.json'):
        """Load performance history from file"""
        if not Path(filepath).exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.initial_balance = data['initial_balance']
        self.current_balance = data['current_balance']
        self.trades = data['trades']
        self.pnl_history = [(datetime.fromisoformat(dt), pnl)
                           for dt, pnl in data['pnl_history']]
        self.hourly_returns = data['hourly_returns']
        self.daily_returns = data['daily_returns']

    def plot_pnl_chart(self, figure: Figure, time_range: str = 'all'):
        """Create P&L chart for GUI"""
        figure.clear()
        if not self.pnl_history:
            return

        # Convert history to DataFrame for easier analysis
        df = pd.DataFrame(self.pnl_history, columns=['timestamp', 'pnl'])
        df['cumulative_pnl'] = df['pnl'].cumsum()

        # Filter data based on time range
        now = datetime.now()
        if time_range == '1d':
            df = df[df['timestamp'] >= now - timedelta(days=1)]
        elif time_range == '1w':
            df = df[df['timestamp'] >= now - timedelta(weeks=1)]
        elif time_range == '1m':
            df = df[df['timestamp'] >= now - timedelta(days=30)]
        elif time_range == '3m':
            df = df[df['timestamp'] >= now - timedelta(days=90)]

        # Create subplot grid
        fig_width = figure.get_size_inches()[0]
        is_wide = fig_width > 8

        if is_wide:
            grid = figure.add_gridspec(2, 2)
            ax1 = figure.add_subplot(grid[0, :])  # Main P&L chart spans both columns
            ax2 = figure.add_subplot(grid[1, 0])  # Returns distribution
            ax3 = figure.add_subplot(grid[1, 1])  # Drawdown chart
        else:
            grid = figure.add_gridspec(3, 1)
            ax1 = figure.add_subplot(grid[0, :])
            ax2 = figure.add_subplot(grid[1, :])
            ax3 = figure.add_subplot(grid[2, :])

        # Plot main P&L chart
        ax1.plot(df['timestamp'], df['cumulative_pnl'], label='Cumulative P&L', color='blue')
        ax1.set_title('Trading Performance')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Profit/Loss ($)')
        ax1.grid(True)
        ax1.legend()

        # Plot returns distribution
        returns = df['pnl'].dropna()
        if len(returns) > 1:
            ax2.hist(returns, bins=30, density=True, alpha=0.75, color='green')
            ax2.set_title('Returns Distribution')
            ax2.set_xlabel('Return ($)')
            ax2.set_ylabel('Frequency')

        # Plot drawdown chart
        rolling_max = df['cumulative_pnl'].expanding().max()
        drawdown = (df['cumulative_pnl'] - rolling_max) / rolling_max * 100
        ax3.fill_between(df['timestamp'], drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Drawdown (%)')

        # Rotate x-axis labels for better readability
        for ax in [ax1, ax2, ax3]:
            plt.setp(ax.get_xticklabels(), rotation=45)

        figure.tight_layout()

    def get_historical_analysis(self, time_range: str = 'all') -> Dict:
        """Get historical performance analysis"""
        if not self.pnl_history:
            return {}

        # Convert history to DataFrame
        df = pd.DataFrame(self.pnl_history, columns=['timestamp', 'pnl'])
        df['cumulative_pnl'] = df['pnl'].cumsum()

        # Filter data based on time range
        now = datetime.now()
        if time_range == '1d':
            df = df[df['timestamp'] >= now - timedelta(days=1)]
        elif time_range == '1w':
            df = df[df['timestamp'] >= now - timedelta(weeks=1)]
        elif time_range == '1m':
            df = df[df['timestamp'] >= now - timedelta(days=30)]
        elif time_range == '3m':
            df = df[df['timestamp'] >= now - timedelta(days=90)]

        # Calculate metrics
        returns = df['pnl'].dropna()
        if len(returns) < 2:
            return {}

        rolling_max = df['cumulative_pnl'].expanding().max()
        drawdown = (df['cumulative_pnl'] - rolling_max) / rolling_max * 100

        analysis = {
            'period_pnl': df['pnl'].sum(),
            'avg_daily_return': df.groupby(df['timestamp'].dt.date)['pnl'].sum().mean(),
            'volatility': returns.std() * np.sqrt(252),  # Annualized
            'max_drawdown': drawdown.min(),
            'max_drawdown_duration': self._get_max_drawdown_duration(drawdown),
            'best_trade': returns.max(),
            'worst_trade': returns.min(),
            'profitable_days': (df.groupby(df['timestamp'].dt.date)['pnl'].sum() > 0).mean() * 100,
            'avg_trade_duration': self._calculate_avg_trade_duration()
        }

        return analysis

    def _get_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate the longest drawdown duration in days"""
        in_drawdown = False
        current_duration = 0
        max_duration = 0
        prev_value = 0

        for value in drawdown_series:
            if value < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    current_duration = 0
                    in_drawdown = False
            prev_value = value

        if in_drawdown:
            max_duration = max(max_duration, current_duration)

        return max_duration

    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in minutes"""
        if not self.trades or len(self.trades) < 2:
            return 0.0

        durations = []
        open_trades = {}

        for trade in self.trades:
            if trade['side'] == 'buy':
                open_trades[trade['symbol']] = trade['timestamp']
            elif trade['side'] == 'sell' and trade['symbol'] in open_trades:
                start_time = open_trades[trade['symbol']]
                duration = (trade['timestamp'] - start_time).total_seconds() / 60
                durations.append(duration)
                del open_trades[trade['symbol']]

        return np.mean(durations) if durations else 0.0
