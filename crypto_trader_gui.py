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
from performance_tracker import PerformanceTracker
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
        self.performance_tracker: PerformanceTracker = PerformanceTracker()
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
                "id": "coinbase",
                "api_key": "",
                "secret_key": "",
                "symbols": [
                    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
                    "ADA/USDT", "DOGE/USDT", "SOL/USDT", "DOT/USDT",
                    "AVAX/USDT", "MATIC/USDT", "LINK/USDT", "UNI/USDT",
                    "ATOM/USDT", "LTC/USDT", "ETC/USDT",
                    # Canadian CAD pairs for Coinbase
                    "BTC/CAD", "ETH/CAD", "XRP/CAD",
                    "ADA/CAD", "DOGE/CAD", "SOL/CAD",
                    "MATIC/CAD", "LINK/CAD", "LTC/CAD"
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
        self.create_settings_tab()  # Add comprehensive settings tab

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
            values=["binance", "coinbase", "kucoin", "bybit", "mexc", "gate", "huobi"]
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

    def create_settings_tab(self):
        """Create comprehensive multi-level settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Settings")
        
        # Create a canvas with scrollbar for settings
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create nested notebooks for multi-level settings
        settings_notebook = ttk.Notebook(scrollable_frame)
        settings_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Level 1: General Settings
        self.create_general_settings(settings_notebook)
        
        # Level 2: Advanced Settings
        self.create_advanced_settings(settings_notebook)
        
        # Level 3: Risk Management Settings
        self.create_risk_management_settings(settings_notebook)
        
        # Level 4: Notification Settings
        self.create_notification_settings(settings_notebook)
        
        # Level 5: API & Integration Settings
        self.create_api_integration_settings(settings_notebook)
        
        # Save all settings button at bottom
        save_frame = ttk.Frame(scrollable_frame)
        save_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(
            save_frame,
            text="Save All Settings",
            command=self.save_all_settings
        ).pack(pady=10)
        
        ttk.Button(
            save_frame,
            text="Reset to Defaults",
            command=self.reset_to_defaults
        ).pack(pady=5)
    
    def create_general_settings(self, parent_notebook):
        """Create general settings sub-tab"""
        tab = ttk.Frame(parent_notebook)
        parent_notebook.add(tab, text="General")
        
        # Application Settings
        app_frame = ttk.LabelFrame(tab, text="Application Settings", padding=10)
        app_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(app_frame, text="Theme:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.theme_var = tk.StringVar(value=self.config.get("theme", "default"))
        ttk.Combobox(
            app_frame,
            textvariable=self.theme_var,
            values=["default", "dark", "light"],
            state="readonly"
        ).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(app_frame, text="Language:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.language_var = tk.StringVar(value=self.config.get("language", "English"))
        ttk.Combobox(
            app_frame,
            textvariable=self.language_var,
            values=["English", "French", "Spanish", "German"],
            state="readonly"
        ).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(app_frame, text="Auto-save interval (min):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.autosave_var = tk.StringVar(value=str(self.config.get("autosave_interval", 5)))
        ttk.Entry(app_frame, textvariable=self.autosave_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # Display Settings
        display_frame = ttk.LabelFrame(tab, text="Display Settings", padding=10)
        display_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.show_tooltips_var = tk.BooleanVar(value=self.config.get("show_tooltips", True))
        ttk.Checkbutton(
            display_frame,
            text="Show Tooltips",
            variable=self.show_tooltips_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.show_grid_var = tk.BooleanVar(value=self.config.get("show_grid", True))
        ttk.Checkbutton(
            display_frame,
            text="Show Grid in Charts",
            variable=self.show_grid_var
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Label(display_frame, text="Chart Refresh Rate (sec):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.chart_refresh_var = tk.StringVar(value=str(self.config.get("chart_refresh", 5)))
        ttk.Entry(display_frame, textvariable=self.chart_refresh_var, width=10).grid(row=2, column=1, padx=5, pady=5)
    
    def create_advanced_settings(self, parent_notebook):
        """Create advanced settings sub-tab"""
        tab = ttk.Frame(parent_notebook)
        parent_notebook.add(tab, text="Advanced")
        
        # Performance Settings
        perf_frame = ttk.LabelFrame(tab, text="Performance Optimization", padding=10)
        perf_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(perf_frame, text="Max Concurrent Requests:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_requests_var = tk.StringVar(value=str(self.config.get("max_concurrent_requests", 5)))
        ttk.Entry(perf_frame, textvariable=self.max_requests_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(perf_frame, text="Request Timeout (sec):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.request_timeout_var = tk.StringVar(value=str(self.config.get("request_timeout", 30)))
        ttk.Entry(perf_frame, textvariable=self.request_timeout_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.enable_caching_var = tk.BooleanVar(value=self.config.get("enable_caching", True))
        ttk.Checkbutton(
            perf_frame,
            text="Enable Data Caching",
            variable=self.enable_caching_var
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Data Storage Settings
        storage_frame = ttk.LabelFrame(tab, text="Data Storage", padding=10)
        storage_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(storage_frame, text="Max History Days:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_history_var = tk.StringVar(value=str(self.config.get("max_history_days", 365)))
        ttk.Entry(storage_frame, textvariable=self.max_history_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(storage_frame, text="Log Retention Days:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.log_retention_var = tk.StringVar(value=str(self.config.get("log_retention_days", 30)))
        ttk.Entry(storage_frame, textvariable=self.log_retention_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.compress_old_data_var = tk.BooleanVar(value=self.config.get("compress_old_data", True))
        ttk.Checkbutton(
            storage_frame,
            text="Compress Old Data",
            variable=self.compress_old_data_var
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    def create_risk_management_settings(self, parent_notebook):
        """Create risk management settings sub-tab"""
        tab = ttk.Frame(parent_notebook)
        parent_notebook.add(tab, text="Risk Management")
        
        # Position Sizing
        position_frame = ttk.LabelFrame(tab, text="Position Sizing", padding=10)
        position_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(position_frame, text="Max Position Size (%):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_position_var = tk.StringVar(value=str(self.config.get("max_position_size", 10)))
        ttk.Entry(position_frame, textvariable=self.max_position_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(position_frame, text="Max Open Positions:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.max_open_pos_var = tk.StringVar(value=str(self.config.get("max_open_positions", 5)))
        ttk.Entry(position_frame, textvariable=self.max_open_pos_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(position_frame, text="Position Sizing Method:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.position_method_var = tk.StringVar(value=self.config.get("position_sizing_method", "Fixed"))
        ttk.Combobox(
            position_frame,
            textvariable=self.position_method_var,
            values=["Fixed", "Percentage", "Kelly Criterion", "ATR-based"],
            state="readonly"
        ).grid(row=2, column=1, padx=5, pady=5)
        
        # Loss Limits
        loss_frame = ttk.LabelFrame(tab, text="Loss Limits", padding=10)
        loss_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(loss_frame, text="Max Daily Loss (%):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_daily_loss_var = tk.StringVar(value=str(self.config.get("max_daily_loss", 5)))
        ttk.Entry(loss_frame, textvariable=self.max_daily_loss_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(loss_frame, text="Max Weekly Loss (%):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.max_weekly_loss_var = tk.StringVar(value=str(self.config.get("max_weekly_loss", 10)))
        ttk.Entry(loss_frame, textvariable=self.max_weekly_loss_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.auto_stop_on_limit_var = tk.BooleanVar(value=self.config.get("auto_stop_on_limit", True))
        ttk.Checkbutton(
            loss_frame,
            text="Auto-stop Trading on Limit Hit",
            variable=self.auto_stop_on_limit_var
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    def create_notification_settings(self, parent_notebook):
        """Create notification settings sub-tab"""
        tab = ttk.Frame(parent_notebook)
        parent_notebook.add(tab, text="Notifications")
        
        # Alert Settings
        alert_frame = ttk.LabelFrame(tab, text="Alert Preferences", padding=10)
        alert_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        self.notify_trades_var = tk.BooleanVar(value=self.config.get("notify_trades", True))
        ttk.Checkbutton(
            alert_frame,
            text="Notify on Trades",
            variable=self.notify_trades_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.notify_errors_var = tk.BooleanVar(value=self.config.get("notify_errors", True))
        ttk.Checkbutton(
            alert_frame,
            text="Notify on Errors",
            variable=self.notify_errors_var
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.notify_limits_var = tk.BooleanVar(value=self.config.get("notify_limits", True))
        ttk.Checkbutton(
            alert_frame,
            text="Notify on Loss Limits",
            variable=self.notify_limits_var
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        # Notification Methods
        method_frame = ttk.LabelFrame(tab, text="Notification Methods", padding=10)
        method_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.notify_email_var = tk.BooleanVar(value=self.config.get("notify_email", False))
        ttk.Checkbutton(
            method_frame,
            text="Email Notifications",
            variable=self.notify_email_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Label(method_frame, text="Email Address:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.email_address_var = tk.StringVar(value=self.config.get("email_address", ""))
        ttk.Entry(method_frame, textvariable=self.email_address_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        
        self.notify_sound_var = tk.BooleanVar(value=self.config.get("notify_sound", True))
        ttk.Checkbutton(
            method_frame,
            text="Sound Alerts",
            variable=self.notify_sound_var
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")
    
    def create_api_integration_settings(self, parent_notebook):
        """Create API & integration settings sub-tab"""
        tab = ttk.Frame(parent_notebook)
        parent_notebook.add(tab, text="API & Integration")
        
        # Exchange API Settings
        api_frame = ttk.LabelFrame(tab, text="Exchange API Configuration", padding=10)
        api_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(api_frame, text="Rate Limit Buffer (%):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.rate_limit_buffer_var = tk.StringVar(value=str(self.config.get("rate_limit_buffer", 20)))
        ttk.Entry(api_frame, textvariable=self.rate_limit_buffer_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(api_frame, text="Retry Attempts:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.retry_attempts_var = tk.StringVar(value=str(self.config.get("retry_attempts", 3)))
        ttk.Entry(api_frame, textvariable=self.retry_attempts_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.use_testnet_var = tk.BooleanVar(value=self.config.get("use_testnet", False))
        ttk.Checkbutton(
            api_frame,
            text="Use Testnet (Paper Trading)",
            variable=self.use_testnet_var
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # External Integrations
        integration_frame = ttk.LabelFrame(tab, text="External Integrations", padding=10)
        integration_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.enable_telegram_var = tk.BooleanVar(value=self.config.get("enable_telegram", False))
        ttk.Checkbutton(
            integration_frame,
            text="Enable Telegram Bot",
            variable=self.enable_telegram_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Label(integration_frame, text="Telegram Bot Token:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.telegram_token_var = tk.StringVar(value=self.config.get("telegram_token", ""))
        ttk.Entry(integration_frame, textvariable=self.telegram_token_var, width=30, show="*").grid(row=1, column=1, padx=5, pady=5)
        
        self.enable_webhook_var = tk.BooleanVar(value=self.config.get("enable_webhook", False))
        ttk.Checkbutton(
            integration_frame,
            text="Enable Webhook Notifications",
            variable=self.enable_webhook_var
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Label(integration_frame, text="Webhook URL:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.webhook_url_var = tk.StringVar(value=self.config.get("webhook_url", ""))
        ttk.Entry(integration_frame, textvariable=self.webhook_url_var, width=30).grid(row=3, column=1, padx=5, pady=5)
    
    def save_all_settings(self):
        """Save all settings from the settings tab"""
        try:
            # General settings
            self.config["theme"] = self.theme_var.get()
            self.config["language"] = self.language_var.get()
            self.config["autosave_interval"] = int(self.autosave_var.get())
            self.config["show_tooltips"] = self.show_tooltips_var.get()
            self.config["show_grid"] = self.show_grid_var.get()
            self.config["chart_refresh"] = int(self.chart_refresh_var.get())
            
            # Advanced settings
            self.config["max_concurrent_requests"] = int(self.max_requests_var.get())
            self.config["request_timeout"] = int(self.request_timeout_var.get())
            self.config["enable_caching"] = self.enable_caching_var.get()
            self.config["max_history_days"] = int(self.max_history_var.get())
            self.config["log_retention_days"] = int(self.log_retention_var.get())
            self.config["compress_old_data"] = self.compress_old_data_var.get()
            
            # Risk management settings
            self.config["max_position_size"] = float(self.max_position_var.get())
            self.config["max_open_positions"] = int(self.max_open_pos_var.get())
            self.config["position_sizing_method"] = self.position_method_var.get()
            self.config["max_daily_loss"] = float(self.max_daily_loss_var.get())
            self.config["max_weekly_loss"] = float(self.max_weekly_loss_var.get())
            self.config["auto_stop_on_limit"] = self.auto_stop_on_limit_var.get()
            
            # Notification settings
            self.config["notify_trades"] = self.notify_trades_var.get()
            self.config["notify_errors"] = self.notify_errors_var.get()
            self.config["notify_limits"] = self.notify_limits_var.get()
            self.config["notify_email"] = self.notify_email_var.get()
            self.config["email_address"] = self.email_address_var.get()
            self.config["notify_sound"] = self.notify_sound_var.get()
            
            # API & integration settings
            self.config["rate_limit_buffer"] = int(self.rate_limit_buffer_var.get())
            self.config["retry_attempts"] = int(self.retry_attempts_var.get())
            self.config["use_testnet"] = self.use_testnet_var.get()
            self.config["enable_telegram"] = self.enable_telegram_var.get()
            self.config["telegram_token"] = self.telegram_token_var.get()
            self.config["enable_webhook"] = self.enable_webhook_var.get()
            self.config["webhook_url"] = self.webhook_url_var.get()
            
            self.save_config()
            self.logger.info("All settings saved successfully")
            messagebox.showinfo("Success", "All settings saved successfully!")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def reset_to_defaults(self):
        """Reset all settings to default values"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?"):
            try:
                # Reset to default configuration
                self.config = self.load_config()
                
                # Reload all UI elements with default values
                # This would need to be implemented for each variable
                self.logger.info("Settings reset to defaults")
                messagebox.showinfo("Success", "Settings reset to defaults. Please restart the application.")
                
            except Exception as e:
                self.logger.error(f"Error resetting settings: {e}")
                messagebox.showerror("Error", f"Failed to reset settings: {e}")

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
    app.mainloop()
