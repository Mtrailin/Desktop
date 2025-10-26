import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from crypto_trader import CryptoTrader
from market_data_collector import MarketDataCollector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
class CryptoTraderGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Crypto Trader Control Panel")
        self.geometry("1200x800")

        # Initialize configuration
        self.config = self.load_config()

        # Setup logger
        self.setup_logging()

        # Build UI
        self.create_notebook()
        self.create_status_bar()

        # Initialize trader instance
        self.trader = None

        # Initialize performance tracker with type hint
        self.performance_tracker: Performancetracker = Performancetracker()
        self.last_update_time = datetime.now()

        # Load previous performance history if available
        self.performance_tracker.load_history()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        config_path = Path("config/gui_config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)

        # Default configuration
        return {
            "exchange": {
                "id": "binance",
                "api_key": "",
                "secret_key": "",
                "symbols": [
                    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
                    "ADA/USDT", "DOGE/USDT", "SOL/USDT", "DOT/USDT",
                    "AVAX/USDT", "MATIC/USDT", "LINK/USDT", "UNI/USDT",
                    "ATOM/USDT", "LTC/USDT", "ETC/USDT"
                ],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            "training": {
                "mode": True,
                "days_history": 30,
                "epochs": 20,
                "batch_size": 32,
                "learning_rate": 0.001,
                "sequence_length": 60
            },
            "trading": {
                "risk_per_trade": 0.02,
                "stop_loss": 0.02,
                "take_profit": 0.03,
                "leverage": 1
            },
            "model": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2
            }
        }

    def save_config(self):
        """Save current configuration to file"""
        config_path = Path("config/gui_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def setup_logging(self):
        """Setup logging configuration"""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)

        self.logger = logging.getLogger("crypto_trader_gui")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(
            log_path / f"gui_{datetime.now().strftime('%Y%m%d')}.log"
        )
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)

    def create_notebook(self):
        """Create main notebook with tabs"""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.create_exchange_tab()
        self.create_training_tab()
        self.create_trading_tab()
        self.create_model_tab()
        self.create_monitor_tab()
        self.create_performance_tab()  # Add performance tracking tab

    def create_exchange_tab(self):
        """Create exchange settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Exchange")

        # Exchange selection
        ttk.Label(tab, text="Exchange:").grid(row=0, column=0, padx=5, pady=5)
        self.exchange_var = tk.StringVar(value=self.config["exchange"]["id"])
        ttk.Combobox(
            tab,
            textvariable=self.exchange_var,
            values=["binance", "kucoin", "bybit"]
        ).grid(row=0, column=1, padx=5, pady=5)

        # API Credentials
        ttk.Label(tab, text="API Key:").grid(row=1, column=0, padx=5, pady=5)
        self.api_key_var = tk.StringVar(value=self.config["exchange"]["api_key"])
        ttk.Entry(
            tab,
            textvariable=self.api_key_var,
            width=50,
            show="*"
        ).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(tab, text="Secret Key:").grid(row=2, column=0, padx=5, pady=5)
        self.secret_key_var = tk.StringVar(value=self.config["exchange"]["secret_key"])
        ttk.Entry(
            tab,
            textvariable=self.secret_key_var,
            width=50,
            show="*"
        ).grid(row=2, column=1, padx=5, pady=5)

        # Trading pairs
        ttk.Label(tab, text="Trading Pairs:").grid(row=3, column=0, padx=5, pady=5)
        self.pairs_listbox = tk.Listbox(tab, selectmode=tk.MULTIPLE, height=6)
        self.pairs_listbox.grid(row=3, column=1, padx=5, pady=5)
        for symbol in self.config["exchange"]["symbols"]:
            self.pairs_listbox.insert(tk.END, symbol)

        # Timeframes
        ttk.Label(tab, text="Timeframes:").grid(row=4, column=0, padx=5, pady=5)
        self.timeframes_listbox = tk.Listbox(tab, selectmode=tk.MULTIPLE, height=6)
        self.timeframes_listbox.grid(row=4, column=1, padx=5, pady=5)
        for tf in self.config["exchange"]["timeframes"]:
            self.timeframes_listbox.insert(tk.END, tf)

        # Save button
        ttk.Button(
            tab,
            text="Save Exchange Settings",
            command=self.save_exchange_settings
        ).grid(row=5, column=0, columnspan=2, pady=20)

    def create_training_tab(self):
        """Create training settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Training")

        # Training mode
        self.train_mode_var = tk.BooleanVar(value=self.config["training"]["mode"])
        ttk.Checkbutton(
            tab,
            text="Training Mode",
            variable=self.train_mode_var
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        # Training parameters
        params = [
            ("Historical Days:", "days_history", 30),
            ("Epochs:", "epochs", 20),
            ("Batch Size:", "batch_size", 32),
            ("Learning Rate:", "learning_rate", 0.001),
            ("Sequence Length:", "sequence_length", 60)
        ]

        for i, (label, key, default) in enumerate(params, start=1):
            ttk.Label(tab, text=label).grid(row=i, column=0, padx=5, pady=5)
            var = tk.StringVar(value=str(self.config["training"].get(key, default)))
            ttk.Entry(
                tab,
                textvariable=var
            ).grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"train_{key}_var", var)

        # Control buttons
        ttk.Button(
            tab,
            text="Start Data Collection",
            command=self.start_data_collection
        ).grid(row=len(params)+1, column=0, pady=20)

        ttk.Button(
            tab,
            text="Start Training",
            command=self.start_training
        ).grid(row=len(params)+1, column=1, pady=20)

    def create_trading_tab(self):
        """Create trading settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Trading")

        # Trading parameters
        params = [
            ("Risk Per Trade (%):", "risk_per_trade", 2.0),
            ("Stop Loss (%):", "stop_loss", 2.0),
            ("Take Profit (%):", "take_profit", 3.0),
            ("Leverage:", "leverage", 1)
        ]

        for i, (label, key, default) in enumerate(params):
            ttk.Label(tab, text=label).grid(row=i, column=0, padx=5, pady=5)
            var = tk.StringVar(value=str(self.config["trading"].get(key, default)))
            ttk.Entry(
                tab,
                textvariable=var
            ).grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"trade_{key}_var", var)

        # Control buttons
        ttk.Button(
            tab,
            text="Start Live Trading",
            command=self.start_live_trading
        ).grid(row=len(params)+1, column=0, pady=20)

        ttk.Button(
            tab,
            text="Stop Trading",
            command=self.stop_trading
        ).grid(row=len(params)+1, column=1, pady=20)

    def create_model_tab(self):
        """Create model settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Model")

        # Model parameters
        params = [
            ("Hidden Size:", "hidden_size", 64),
            ("Number of Layers:", "num_layers", 2),
            ("Dropout Rate:", "dropout", 0.2)
        ]

        for i, (label, key, default) in enumerate(params):
            ttk.Label(tab, text=label).grid(row=i, column=0, padx=5, pady=5)
            var = tk.StringVar(value=str(self.config["model"].get(key, default)))
            ttk.Entry(
                tab,
                textvariable=var
            ).grid(row=i, column=1, padx=5, pady=5)
            setattr(self, f"model_{key}_var", var)

        # Model controls
        ttk.Button(
            tab,
            text="Save Model",
            command=self.save_model
        ).grid(row=len(params)+1, column=0, pady=20)

        ttk.Button(
            tab,
            text="Load Model",
            command=self.load_model
        ).grid(row=len(params)+1, column=1, pady=20)

    def create_monitor_tab(self):
        """Create monitoring tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Monitor")

        # Log viewer
        self.log_text = scrolledtext.ScrolledText(tab, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add handler to display logs in GUI
        gui_handler = logging.StreamHandler(self.LogTextHandler(self.log_text))
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(gui_handler)

    def create_performance_tab(self):
        """Create performance tracking tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Performance")

        # Top frame for time range selection
        time_frame = ttk.Frame(tab)
        time_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(time_frame, text="Time Range:").pack(side=tk.LEFT, padx=5)
        self.time_range_var = tk.StringVar(value="all")
        time_range_combo = ttk.Combobox(
            time_frame,
            textvariable=self.time_range_var,
            values=["all", "1d", "1w", "1m", "3m"],
            width=10,
            state="readonly"
        )
        time_range_combo.pack(side=tk.LEFT, padx=5)
        time_range_combo.bind('<<ComboboxSelected>>', lambda e: self.force_performance_update())

        # Left side - Performance metrics
        metrics_frame = ttk.LabelFrame(tab, text="Performance Metrics")
        metrics_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Performance labels
        metrics = [
            ("Total P&L:", "total_pnl_var"),
            ("Open P&L:", "open_pnl_var"),
            ("ROI (%):", "roi_var"),
            ("Win Rate (%):", "win_rate_var"),
            ("Total Trades:", "total_trades_var"),
            ("Open Positions:", "open_positions_var"),
            ("Avg Win:", "avg_win_var"),
            ("Avg Loss:", "avg_loss_var"),
            ("Profit Factor:", "profit_factor_var"),
            ("Sharpe Ratio:", "sharpe_var"),
            ("Max Drawdown (%):", "drawdown_var")
        ]

        for i, (label, var_name) in enumerate(metrics):
            ttk.Label(metrics_frame, text=label).grid(row=i, column=0, padx=5, pady=2, sticky="w")
            var = tk.StringVar(value="0.00")
            ttk.Label(metrics_frame, textvariable=var).grid(row=i, column=1, padx=5, pady=2, sticky="e")
            setattr(self, var_name, var)

        # Right side - Charts
        charts_frame = ttk.LabelFrame(tab, text="Performance Charts")
        charts_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(self.fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom frame - Historical Statistics
        stats_frame = ttk.LabelFrame(tab, text="Historical Statistics")
        stats_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Create labels for historical stats
        stats_labels = [
            ("Period P&L:", "period_pnl_var"),
            ("Avg Daily Return:", "avg_daily_return_var"),
            ("Volatility:", "volatility_var"),
            ("Max Drawdown Duration:", "max_dd_duration_var"),
            ("Best Trade:", "best_trade_var"),
            ("Worst Trade:", "worst_trade_var"),
            ("Profitable Days %:", "profitable_days_var"),
            ("Avg Trade Duration:", "avg_trade_duration_var")
        ]

        for i, (label, var_name) in enumerate(stats_labels):
            col = i % 4
            row = i // 4
            ttk.Label(stats_frame, text=label).grid(row=row, column=col*2, padx=5, pady=2, sticky="w")
            var = tk.StringVar(value="0.00")
            ttk.Label(stats_frame, textvariable=var).grid(row=row, column=col*2+1, padx=5, pady=2, sticky="e")
            setattr(self, var_name, var)

        # Configure grid weights
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=2)
        tab.grid_rowconfigure(1, weight=1)

    def update_performance(self):
        """Update performance metrics and chart"""
        try:
            if self.trader and not self.trader.train_mode:
                # Get current account balance
                current_balance = self.trader.get_account_balance()
                self.performance_tracker.update_balance(current_balance)

                # Get open positions and their current prices
                positions = self.trader.get_open_positions()
                position_prices = {pos['symbol']: pos['current_price'] for pos in positions}
                self.performance_tracker.update_position_prices(position_prices)

                # Update metrics
                metrics = self.performance_tracker.get_performance_metrics()

                # Update display variables
                self.total_pnl_var.set(f"${metrics['total_pnl']:.2f}")
                self.open_pnl_var.set(f"${metrics['open_pnl']:.2f}")
                self.roi_var.set(f"{metrics['roi']:.2f}")
                self.win_rate_var.set(f"{metrics['win_rate']:.1f}")
                self.total_trades_var.set(str(metrics['total_trades']))
                self.open_positions_var.set(str(metrics['open_positions']))
                self.avg_win_var.set(f"${metrics['avg_win']:.2f}")
                self.avg_loss_var.set(f"${metrics['avg_loss']:.2f}")
                self.profit_factor_var.set(f"{metrics['profit_factor']:.2f}")
                self.sharpe_var.set(f"{metrics['daily_sharpe']:.2f}")
                self.drawdown_var.set(f"{metrics['max_drawdown']:.2f}")

                # Update chart every minute
                if (datetime.now() - self.last_update_time).seconds >= 60:
                    # Update main charts with current time range
                    self.performance_tracker.plot_pnl_chart(self.fig, self.time_range_var.get())
                    self.fig.canvas.draw()
                    self.last_update_time = datetime.now()

                    # Update historical statistics
                    historical_stats = self.performance_tracker.get_historical_analysis(self.time_range_var.get())
                    if historical_stats:
                        self.period_pnl_var.set(f"${historical_stats['period_pnl']:.2f}")
                        self.avg_daily_return_var.set(f"${historical_stats['avg_daily_return']:.2f}")
                        self.volatility_var.set(f"{historical_stats['volatility']:.2f}%")
                        self.max_dd_duration_var.set(f"{historical_stats['max_drawdown_duration']} days")
                        self.best_trade_var.set(f"${historical_stats['best_trade']:.2f}")
                        self.worst_trade_var.set(f"${historical_stats['worst_trade']:.2f}")
                        self.profitable_days_var.set(f"{historical_stats['profitable_days']:.1f}%")
                        self.avg_trade_duration_var.set(f"{historical_stats['avg_trade_duration']:.1f} min")

                # Save performance history periodically
                self.performance_tracker.save_history()

            # Schedule next update
            self.after(1000, self.update_performance)

        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
            self.status_var.set(f"Performance update error: {e}")
            # Still try to schedule next update
            self.after(1000, self.update_performance)

    def create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5)
        self.status_var.set("Ready")

    # Event handlers
    def save_exchange_settings(self):
        """Save exchange settings"""
        try:
            self.config["exchange"]["id"] = self.exchange_var.get()
            self.config["exchange"]["api_key"] = self.api_key_var.get()
            self.config["exchange"]["secret_key"] = self.secret_key_var.get()
            self.config["exchange"]["symbols"] = [
                self.pairs_listbox.get(i) for i in self.pairs_listbox.curselection()
            ]
            self.config["exchange"]["timeframes"] = [
                self.timeframes_listbox.get(i) for i in self.timeframes_listbox.curselection()
            ]

            self.save_config()
            self.logger.info("Exchange settings saved")
            messagebox.showinfo("Success", "Exchange settings saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving exchange settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def start_data_collection(self):
        """Start historical data collection"""
        try:
            days = int(self.train_days_history_var.get())
            symbols = [
                self.pairs_listbox.get(i) for i in self.pairs_listbox.curselection()
            ]
            timeframes = [
                self.timeframes_listbox.get(i) for i in self.timeframes_listbox.curselection()
            ]

            collector = MarketDataCollector(
                exchange_id=self.exchange_var.get(),
                symbols=symbols,
                timeframes=timeframes
            )

            self.status_var.set("Collecting historical data...")
            collector.collect_all_markets(days_history=days)

            self.logger.info("Historical data collection completed")
            messagebox.showinfo("Success", "Data collection completed successfully")
            self.status_var.set("Ready")

        except Exception as e:
            self.logger.error(f"Error collecting data: {e}")
            messagebox.showerror("Error", f"Data collection failed: {e}")
            self.status_var.set("Ready")

    def start_training(self):
        """Start model training"""
        try:
            if self.trader is None:
                self.trader = CryptoTrader(
                    exchange_id=self.exchange_var.get(),
                    symbol=self.pairs_listbox.get(self.pairs_listbox.curselection()[0]),
                    timeframe=self.timeframes_listbox.get(self.timeframes_listbox.curselection()[0]),
                    train_mode=True
                )

            self.status_var.set("Training model...")
            self.trader.train_on_historical(epochs=int(self.train_epochs_var.get()))

            self.logger.info("Model training completed")
            messagebox.showinfo("Success", "Model training completed successfully")
            self.status_var.set("Ready")

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            messagebox.showerror("Error", f"Training failed: {e}")
            self.status_var.set("Ready")

    def start_live_trading(self):
        """Start live trading"""
        try:
            if not messagebox.askyesno("Confirm", "Start live trading with real money?"):
                return

            if self.trader is None or self.trader.train_mode:
                self.trader = CryptoTrader(
                    exchange_id=self.exchange_var.get(),
                    symbol=self.pairs_listbox.get(self.pairs_listbox.curselection()[0]),
                    timeframe=self.timeframes_listbox.get(self.timeframes_listbox.curselection()[0]),
                    train_mode=False
                )

            self.status_var.set("Live trading started")
            # Set initial balance from exchange
            balance = self.trader.get_account_balance()
            self.performance_tracker.initial_balance = balance
            self.performance_tracker.current_balance = balance

            # Start balance tracking
            self.after(1000, self.update_performance)

            self.trader.run()

        except Exception as e:
            self.logger.error(f"Error starting live trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")
            self.status_var.set("Ready")

    def stop_trading(self):
        """Stop live trading"""
        try:
            if self.trader:
                self.trader.stop()
                self.logger.info("Trading stopped")

                # Save final performance snapshot
                self.performance_tracker.save_history()

                # Reset performance tracker
                self.performance_tracker.reset()
        finally:
                # Force update of all displays
            self.force_performance_update()

    def force_performance_update(self):
        """Force update of all performance displays"""
        # Update main chart with current time range
        self.performance_tracker.plot_pnl_chart(self.fig, self.time_range_var.get())
        self.fig.canvas.draw()

        # Update metrics display
        metrics = self.performance_tracker.get_performance_metrics()
        self.total_pnl_var.set(f"${metrics['total_pnl']:.2f}")
        self.open_pnl_var.set(f"${metrics['open_pnl']:.2f}")
        self.roi_var.set(f"{metrics['roi']:.2f}")
        self.win_rate_var.set(f"{metrics['win_rate']:.1f}")
        self.total_trades_var.set(str(metrics['total_trades']))
        self.open_positions_var.set(str(metrics['open_positions']))
        self.avg_win_var.set(f"${metrics['avg_win']:.2f}")
        self.avg_loss_var.set(f"${metrics['avg_loss']:.2f}")
        self.profit_factor_var.set(f"{metrics['profit_factor']:.2f}")
        self.sharpe_var.set(f"{metrics['daily_sharpe']:.2f}")
        self.drawdown_var.set(f"{metrics['max_drawdown']:.2f}")

        # Update historical statistics
        historical_stats = self.performance_tracker.get_historical_analysis(self.time_range_var.get())
        if historical_stats:
            self.period_pnl_var.set(f"${historical_stats['period_pnl']:.2f}")
            self.avg_daily_return_var.set(f"${historical_stats['avg_daily_return']:.2f}")
            self.volatility_var.set(f"{historical_stats['volatility']:.2f}%")
            self.max_dd_duration_var.set(f"{historical_stats['max_drawdown_duration']} days")
            self.best_trade_var.set(f"${historical_stats['best_trade']:.2f}")
            self.worst_trade_var.set(f"${historical_stats['worst_trade']:.2f}")
            self.profitable_days_var.set(f"{historical_stats['profitable_days']:.1f}%")
            self.avg_trade_duration_var.set(f"{historical_stats['avg_trade_duration']:.1f} min")
            self.status_var.set("Ready")

    def save_model(self):
        """Save current model"""
        try:
            if self.trader and self.trader.model:
                messagebox.showwarning("Warning", "No model to save")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            messagebox.showerror("Error", f"Failed to save model: {e}")

    def load_model(self):
        """Load saved model"""
        try:
            if self.trader:
                self.trader.load_model()
                self.logger.info("Model loaded successfully")
                messagebox.showinfo("Success", "Model loaded successfully")
            else:
                messagebox.showwarning("Warning", "Initialize trader first")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")

    class LogTextHandler:
        """Handler for redirecting logs to ScrolledText widget"""
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, message):
            self.text_widget.configure(state='normal')
            self.text_widget.insert('end', message)
            self.text_widget.see('end')
            self.text_widget.configure(state='disabled')

        def flush(self):
            pass

if __name__ == "__main__":
    app = CryptoTraderGUI()
    app.mainloop()
