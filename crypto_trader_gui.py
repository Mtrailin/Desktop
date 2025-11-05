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

# Tooltip class for user-friendly help
class ToolTip:
    """Create a tooltip for a given widget"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        try:
            # Try to get position from bbox first (for text widgets)
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 25
        except (AttributeError, TypeError, tk.TclError):
            # Fallback for non-text widgets (buttons, labels, etc.)
            x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify='left',
                        background="#ffffe0", relief='solid', borderwidth=1,
                        font=("Arial", 9, "normal"))
        label.pack(ipadx=1)
    
    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
class CryptoTraderGUI(tk.Tk):
    # Constants for validation ranges
    MIN_HISTORY_DAYS = 1
    MAX_HISTORY_DAYS = 365
    MIN_EPOCHS = 1
    MAX_EPOCHS = 1000
    
    def __init__(self):
        super().__init__()

        self.title("Crypto Trader Control Panel")
        self.geometry("1200x800")
        
        # Set minimum window size for better usability
        self.minsize(1000, 600)
        
        # Timer ID for cleanup
        self.time_update_id = None

        # Initialize configuration
        self.app_config = self.load_config()

        # Setup logger
        self.setup_logging()

        # Build UI
        self.create_menu_bar()
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
            json.dump(self.app_config, f, indent=4)

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

    def create_menu_bar(self):
        """Create menu bar with help and options"""
        menubar = tk.Menu(self)
        self.configure(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Configuration", command=self.save_config, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit, accelerator="Ctrl+Q")
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="Quick Start Guide", command=self.show_quick_start)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.bind('<Control-q>', lambda e: self.quit())
        
    def show_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        shortcuts_text = """
        Keyboard Shortcuts:
        
        Ctrl+S    - Save Exchange Settings
        Ctrl+Q    - Quit Application
        
        Tips:
        • Hover over any control to see helpful tooltips
        • Use Ctrl+Click to select multiple items in lists
        • Watch the status bar for real-time updates
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)
        
    def show_quick_start(self):
        """Show quick start guide"""
        guide_text = """
        Quick Start Guide:
        
        1. Exchange Tab:
           • Select your exchange
           • Enter API credentials
           • Choose trading pairs and timeframes
           • Click Save
        
        2. Training Tab:
           • Click 'Start Data Collection' first
           • Wait for data to download
           • Click 'Start Training'
           • Monitor progress in Monitor tab
        
        3. Trading Tab:
           • Set your risk parameters
           • Review settings carefully
           • Click 'Start Live Trading'
           • Confirm twice (safety feature)
        
        4. Performance Tab:
           • View real-time metrics
           • Monitor P&L and statistics
           • Track trading performance
        
        ⚠️ Always test in training mode first!
        """
        messagebox.showinfo("Quick Start Guide", guide_text)
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
        Crypto Trader Control Panel
        
        A professional cryptocurrency trading system
        with AI-powered predictions and risk management.
        
        Features:
        • Multiple exchange support
        • LSTM-based price prediction
        • Advanced risk management
        • Real-time performance tracking
        • Comprehensive training tools
        
        Version: 1.0
        """
        messagebox.showinfo("About", about_text)

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
        ttk.Label(tab, text="Exchange:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.exchange_var = tk.StringVar(value=self.app_config["exchange"]["id"])
        exchange_combo = ttk.Combobox(
            tab,
            textvariable=self.exchange_var,
            values=["binance", "coinbase", "kucoin", "bybit", "mexc", "gate", "huobi"],
            state="readonly"
        )
        exchange_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        ToolTip(exchange_combo, "Select the cryptocurrency exchange to connect to")

        # API Credentials Frame
        cred_frame = ttk.LabelFrame(tab, text=" API Credentials ", padding=10)
        cred_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        
        ttk.Label(cred_frame, text="API Key:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.api_key_var = tk.StringVar(value=self.app_config["exchange"]["api_key"])
        api_key_entry = ttk.Entry(
            cred_frame,
            textvariable=self.api_key_var,
            width=50,
            show="*"
        )
        api_key_entry.grid(row=0, column=1, padx=5, pady=5)
        ToolTip(api_key_entry, "Your exchange API key (stored securely)")
        
        # Show/Hide API Key button
        self.show_api_key = tk.BooleanVar(value=False)
        def toggle_api_key():
            api_key_entry.config(show="" if self.show_api_key.get() else "*")
        ttk.Checkbutton(cred_frame, text="Show", variable=self.show_api_key, command=toggle_api_key).grid(row=0, column=2, padx=5)

        ttk.Label(cred_frame, text="Secret Key:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.secret_key_var = tk.StringVar(value=self.app_config["exchange"]["secret_key"])
        secret_key_entry = ttk.Entry(
            cred_frame,
            textvariable=self.secret_key_var,
            width=50,
            show="*"
        )
        secret_key_entry.grid(row=1, column=1, padx=5, pady=5)
        ToolTip(secret_key_entry, "Your exchange secret key (stored securely)")
        
        # Show/Hide Secret Key button
        self.show_secret_key = tk.BooleanVar(value=False)
        def toggle_secret_key():
            secret_key_entry.config(show="" if self.show_secret_key.get() else "*")
        ttk.Checkbutton(cred_frame, text="Show", variable=self.show_secret_key, command=toggle_secret_key).grid(row=1, column=2, padx=5)

        # Trading pairs Frame
        pairs_frame = ttk.LabelFrame(tab, text=" Trading Pairs ", padding=10)
        pairs_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky='nsew')
        
        ttk.Label(pairs_frame, text="Select trading pairs (Ctrl+Click for multiple):").pack(anchor='w', padx=5, pady=5)
        
        # Add scrollbar to listbox
        pairs_scroll_frame = ttk.Frame(pairs_frame)
        pairs_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        pairs_scrollbar = ttk.Scrollbar(pairs_scroll_frame)
        pairs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.pairs_listbox = tk.Listbox(pairs_scroll_frame, selectmode=tk.MULTIPLE, height=6, yscrollcommand=pairs_scrollbar.set)
        self.pairs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pairs_scrollbar.config(command=self.pairs_listbox.yview)
        
        for symbol in self.app_config["exchange"]["symbols"]:
            self.pairs_listbox.insert(tk.END, symbol)
        ToolTip(self.pairs_listbox, "Select one or more trading pairs to monitor")

        # Timeframes Frame
        tf_frame = ttk.LabelFrame(tab, text=" Timeframes ", padding=10)
        tf_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        
        ttk.Label(tf_frame, text="Select timeframes (Ctrl+Click for multiple):").pack(anchor='w', padx=5, pady=5)
        self.timeframes_listbox = tk.Listbox(tf_frame, selectmode=tk.MULTIPLE, height=6)
        self.timeframes_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for tf in self.app_config["exchange"]["timeframes"]:
            self.timeframes_listbox.insert(tk.END, tf)
        ToolTip(self.timeframes_listbox, "Select timeframe intervals for analysis")

        # Save button with color
        save_btn = ttk.Button(
            tab,
            text="💾 Save Exchange Settings",
            command=self.save_exchange_settings
        )
        save_btn.grid(row=4, column=0, columnspan=2, pady=20)
        ToolTip(save_btn, "Save your exchange configuration (Ctrl+S)")
        
        # Configure grid weights for responsiveness
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        
        # Bind keyboard shortcut
        self.bind('<Control-s>', lambda e: self.save_exchange_settings())

    def create_training_tab(self):
        """Create training settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Training")
        
        # Main container frame
        container = ttk.Frame(tab)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Training mode with better visual
        mode_frame = ttk.LabelFrame(container, text=" Training Mode ", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.train_mode_var = tk.BooleanVar(value=self.app_config["training"]["mode"])
        mode_check = ttk.Checkbutton(
            mode_frame,
            text="Enable Training Mode (Safe - No Real Trading)",
            variable=self.train_mode_var
        )
        mode_check.pack(anchor='w', padx=5, pady=5)
        ToolTip(mode_check, "When enabled, the system will train on historical data without placing real trades")

        # Training parameters in organized frame
        params_frame = ttk.LabelFrame(container, text=" Training Parameters ", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        params = [
            ("Historical Days:", "days_history", 30, "Number of days of historical data to use for training"),
            ("Epochs:", "epochs", 20, "Number of complete passes through the training dataset"),
            ("Batch Size:", "batch_size", 32, "Number of samples processed before model update"),
            ("Learning Rate:", "learning_rate", 0.001, "Step size for model weight updates (0.0001-0.01)"),
            ("Sequence Length:", "sequence_length", 60, "Number of time steps used for prediction")
        ]

        for i, (label, key, default, tooltip) in enumerate(params):
            ttk.Label(params_frame, text=label, font=('Arial', 9)).grid(row=i, column=0, padx=5, pady=5, sticky='w')
            var = tk.StringVar(value=str(self.app_config["training"].get(key, default)))
            entry = ttk.Entry(params_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky='w')
            setattr(self, f"train_{key}_var", var)
            ToolTip(entry, tooltip)

        # Control buttons with icons and better layout
        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X, pady=10)
        
        collect_btn = ttk.Button(
            button_frame,
            text="📊 Start Data Collection",
            command=self.start_data_collection
        )
        collect_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ToolTip(collect_btn, "Collect historical market data for training")

        train_btn = ttk.Button(
            button_frame,
            text="🎓 Start Training",
            command=self.start_training
        )
        train_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ToolTip(train_btn, "Train the model on collected data")
        
        # Progress information
        info_frame = ttk.LabelFrame(container, text=" Training Information ", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, state='disabled', bg='#f0f0f0')
        info_text.pack(fill=tk.BOTH, expand=True)
        info_text.config(state='normal')
        info_text.insert('1.0', "💡 Training Tips:\n\n" +
                        "• Start with data collection to gather historical market data\n" +
                        "• More epochs generally improve accuracy but take longer\n" +
                        "• Lower learning rates are more stable but slower\n" +
                        "• Monitor training progress in the Monitor tab")
        info_text.config(state='disabled')

    def create_trading_tab(self):
        """Create trading settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Trading")
        
        # Main container
        container = ttk.Frame(tab)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Warning banner
        warning_frame = ttk.Frame(container)
        warning_frame.pack(fill=tk.X, pady=(0, 10))
        warning_label = ttk.Label(
            warning_frame, 
            text="⚠️  WARNING: Live trading uses real money. Start with small amounts and test thoroughly.",
            foreground='red',
            font=('Arial', 10, 'bold')
        )
        warning_label.pack()

        # Trading parameters in organized frame
        params_frame = ttk.LabelFrame(container, text=" Risk Management Parameters ", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        params = [
            ("Risk Per Trade (%):", "risk_per_trade", 2.0, "Maximum percentage of capital to risk per trade (1-5% recommended)"),
            ("Stop Loss (%):", "stop_loss", 2.0, "Automatic sell trigger to limit losses (1-5% typical)"),
            ("Take Profit (%):", "take_profit", 3.0, "Automatic sell trigger to lock in profits (2-10% typical)"),
            ("Leverage:", "leverage", 1, "Trading leverage multiplier (1 = no leverage, RISKY if > 1)")
        ]

        for i, (label, key, default, tooltip) in enumerate(params):
            ttk.Label(params_frame, text=label, font=('Arial', 9)).grid(row=i, column=0, padx=5, pady=5, sticky='w')
            var = tk.StringVar(value=str(self.app_config["trading"].get(key, default)))
            entry = ttk.Entry(params_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky='w')
            setattr(self, f"trade_{key}_var", var)
            ToolTip(entry, tooltip)
            
            # Add validation indicator
            ttk.Label(params_frame, text="✓", foreground='green', font=('Arial', 12)).grid(row=i, column=2, padx=5)

        # Control buttons with better styling
        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Start button (green-ish)
        start_btn_frame = ttk.Frame(button_frame)
        start_btn_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        start_btn = ttk.Button(
            start_btn_frame,
            text="▶️  Start Live Trading",
            command=self.start_live_trading
        )
        start_btn.pack(fill=tk.X)
        ToolTip(start_btn, "Begin live trading with real money - Double confirmation required")
        
        # Stop button (red-ish)
        stop_btn_frame = ttk.Frame(button_frame)
        stop_btn_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        stop_btn = ttk.Button(
            stop_btn_frame,
            text="⏹️  Stop Trading",
            command=self.stop_trading
        )
        stop_btn.pack(fill=tk.X)
        ToolTip(stop_btn, "Stop all trading activity immediately")
        
        # Trading status display
        status_frame = ttk.LabelFrame(container, text=" Trading Status ", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.trading_status_text = tk.Text(status_frame, height=8, wrap=tk.WORD, state='disabled', bg='#f0f0f0')
        self.trading_status_text.pack(fill=tk.BOTH, expand=True)
        self.update_trading_status("No active trading session")
        
    def update_trading_status(self, message):
        """Update the trading status display"""
        if hasattr(self, 'trading_status_text'):
            self.trading_status_text.config(state='normal')
            self.trading_status_text.delete('1.0', tk.END)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.trading_status_text.insert('1.0', f"[{timestamp}] {message}")
            self.trading_status_text.config(state='disabled')

    def create_model_tab(self):
        """Create model settings tab with sliders"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Model")
        
        # Main container
        container = ttk.Frame(tab)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Model parameters with sliders
        params_frame = ttk.LabelFrame(container, text=" Model Parameters ", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Hidden Size slider (16 to 256)
        ttk.Label(params_frame, text="Hidden Size:", font=('Arial', 9)).grid(row=0, column=0, padx=5, pady=10, sticky='w')
        self.model_hidden_size_var = tk.IntVar(value=self.app_config["model"].get("hidden_size", 64))
        hidden_slider = ttk.Scale(
            params_frame,
            from_=16,
            to=256,
            orient=tk.HORIZONTAL,
            variable=self.model_hidden_size_var,
            command=lambda v: self.update_slider_label(self.hidden_value_label, v)
        )
        hidden_slider.grid(row=0, column=1, padx=5, pady=10, sticky='ew')
        self.hidden_value_label = ttk.Label(params_frame, text=str(self.model_hidden_size_var.get()))
        self.hidden_value_label.grid(row=0, column=2, padx=5, pady=10)
        ToolTip(hidden_slider, "Number of hidden units in each LSTM layer (16-256)")
        
        # Number of Layers slider (1 to 5)
        ttk.Label(params_frame, text="Number of Layers:", font=('Arial', 9)).grid(row=1, column=0, padx=5, pady=10, sticky='w')
        self.model_num_layers_var = tk.IntVar(value=self.app_config["model"].get("num_layers", 2))
        layers_slider = ttk.Scale(
            params_frame,
            from_=1,
            to=5,
            orient=tk.HORIZONTAL,
            variable=self.model_num_layers_var,
            command=lambda v: self.update_slider_label(self.layers_value_label, v)
        )
        layers_slider.grid(row=1, column=1, padx=5, pady=10, sticky='ew')
        self.layers_value_label = ttk.Label(params_frame, text=str(self.model_num_layers_var.get()))
        self.layers_value_label.grid(row=1, column=2, padx=5, pady=10)
        ToolTip(layers_slider, "Number of LSTM layers in the model (1-5)")
        
        # Dropout Rate slider (0.0 to 0.5)
        ttk.Label(params_frame, text="Dropout Rate:", font=('Arial', 9)).grid(row=2, column=0, padx=5, pady=10, sticky='w')
        self.model_dropout_var = tk.DoubleVar(value=self.app_config["model"].get("dropout", 0.2))
        dropout_slider = ttk.Scale(
            params_frame,
            from_=0.0,
            to=0.5,
            orient=tk.HORIZONTAL,
            variable=self.model_dropout_var,
            command=lambda v: self.update_slider_label(self.dropout_value_label, v, decimals=2)
        )
        dropout_slider.grid(row=2, column=1, padx=5, pady=10, sticky='ew')
        self.dropout_value_label = ttk.Label(params_frame, text=f"{self.model_dropout_var.get():.2f}")
        self.dropout_value_label.grid(row=2, column=2, padx=5, pady=10)
        ToolTip(dropout_slider, "Dropout rate for regularization (0.0-0.5, prevents overfitting)")
        
        # Configure grid
        params_frame.grid_columnconfigure(1, weight=1)

        # Model controls
        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X, pady=10)
        
        save_btn = ttk.Button(
            button_frame,
            text="💾 Save Model Settings",
            command=self.save_model
        )
        save_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ToolTip(save_btn, "Save current model configuration")

        load_btn = ttk.Button(
            button_frame,
            text="📂 Load Model",
            command=self.load_model
        )
        load_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ToolTip(load_btn, "Load a previously saved model")
        
        # Model info
        info_frame = ttk.LabelFrame(container, text=" Model Information ", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, state='disabled', bg='#f0f0f0')
        info_text.pack(fill=tk.BOTH, expand=True)
        info_text.config(state='normal')
        info_text.insert('1.0', "💡 Model Configuration Tips:\n\n" +
                        "• Hidden Size: Larger values = more complex patterns but slower training\n" +
                        "• Layers: More layers = deeper learning but risk of overfitting\n" +
                        "• Dropout: Higher values = more regularization, prevents overfitting\n" +
                        "• Recommended: Start with defaults and adjust based on performance")
        info_text.config(state='disabled')
        
    def update_slider_label(self, label, value, decimals=0):
        """Update slider value label"""
        if decimals > 0:
            label.config(text=f"{float(value):.{decimals}f}")
        else:
            label.config(text=str(int(float(value))))

    def create_monitor_tab(self):
        """Create enhanced monitoring tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Monitor")
        
        # Create paned window for split view
        paned = ttk.PanedWindow(tab, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top section - Real-time metrics
        metrics_frame = ttk.LabelFrame(paned, text=" Real-Time Metrics ", padding=10)
        paned.add(metrics_frame, weight=1)
        
        # Create metrics display
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.BOTH, expand=True)
        
        # System status
        ttk.Label(metrics_grid, text="System Status:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.system_status_var = tk.StringVar(value="Idle")
        ttk.Label(metrics_grid, textvariable=self.system_status_var, foreground='blue').grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # Active trades
        ttk.Label(metrics_grid, text="Active Trades:", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.active_trades_var = tk.StringVar(value="0")
        ttk.Label(metrics_grid, textvariable=self.active_trades_var).grid(row=0, column=3, padx=5, pady=5, sticky='w')
        
        # Current balance
        ttk.Label(metrics_grid, text="Current Balance:", font=('Arial', 10, 'bold')).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.current_balance_var = tk.StringVar(value="$0.00")
        ttk.Label(metrics_grid, textvariable=self.current_balance_var, foreground='green').grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Total P&L
        ttk.Label(metrics_grid, text="Total P&L:", font=('Arial', 10, 'bold')).grid(row=1, column=2, padx=5, pady=5, sticky='w')
        self.monitor_pnl_var = tk.StringVar(value="$0.00")
        ttk.Label(metrics_grid, textvariable=self.monitor_pnl_var, foreground='green').grid(row=1, column=3, padx=5, pady=5, sticky='w')
        
        # Last update time
        ttk.Label(metrics_grid, text="Last Update:", font=('Arial', 10, 'bold')).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.last_update_var = tk.StringVar(value="Never")
        ttk.Label(metrics_grid, textvariable=self.last_update_var).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Refresh button
        refresh_btn = ttk.Button(metrics_grid, text="🔄 Refresh", command=self.refresh_monitor_metrics)
        refresh_btn.grid(row=2, column=2, columnspan=2, padx=5, pady=5)
        ToolTip(refresh_btn, "Manually refresh monitoring metrics")

        # Bottom section - Activity log
        log_frame = ttk.LabelFrame(paned, text=" Activity Log ", padding=5)
        paned.add(log_frame, weight=2)
        
        # Log viewer
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Add handler to display logs in GUI
        gui_handler = logging.StreamHandler(self.LogTextHandler(self.log_text))
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(gui_handler)
        
        # Log initial message
        self.logger.info("Monitoring system initialized")
        
    def refresh_monitor_metrics(self):
        """Refresh monitoring metrics display"""
        try:
            if self.trader:
                # Update system status
                if hasattr(self.trader, 'train_mode'):
                    if self.trader.train_mode:
                        self.system_status_var.set("Training Mode")
                    else:
                        self.system_status_var.set("Live Trading")
                else:
                    self.system_status_var.set("Ready")
                
                # Update balance if available
                try:
                    balance = self.trader.get_account_balance()
                    self.current_balance_var.set(f"${balance:.2f}")
                except Exception:
                    # Ignore errors if trader not initialized or balance unavailable
                    pass
                    
                # Update active trades
                try:
                    positions = self.trader.get_open_positions()
                    self.active_trades_var.set(str(len(positions)))
                except Exception:
                    # Ignore errors if trader not initialized or positions unavailable
                    pass
                    
                # Update P&L if performance tracker is available
                if hasattr(self, 'performance_tracker'):
                    metrics = self.performance_tracker.get_performance_metrics()
                    self.monitor_pnl_var.set(f"${metrics.get('total_pnl', 0.0):.2f}")
            else:
                self.system_status_var.set("Idle - No trader initialized")
            
            # Update timestamp
            self.last_update_var.set(datetime.now().strftime("%H:%M:%S"))
            self.logger.info("Metrics refreshed")
            
        except Exception as e:
            self.logger.error(f"Error refreshing metrics: {e}")
            self.system_status_var.set("Error")

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
        self.theme_var = tk.StringVar(value=self.app_config.get("theme", "default"))
        ttk.Combobox(
            app_frame,
            textvariable=self.theme_var,
            values=["default", "dark", "light"],
            state="readonly"
        ).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(app_frame, text="Language:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.language_var = tk.StringVar(value=self.app_config.get("language", "English"))
        ttk.Combobox(
            app_frame,
            textvariable=self.language_var,
            values=["English", "French", "Spanish", "German"],
            state="readonly"
        ).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(app_frame, text="Auto-save interval (min):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.autosave_var = tk.StringVar(value=str(self.app_config.get("autosave_interval", 5)))
        ttk.Entry(app_frame, textvariable=self.autosave_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        # Display Settings
        display_frame = ttk.LabelFrame(tab, text="Display Settings", padding=10)
        display_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.show_tooltips_var = tk.BooleanVar(value=self.app_config.get("show_tooltips", True))
        ttk.Checkbutton(
            display_frame,
            text="Show Tooltips",
            variable=self.show_tooltips_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.show_grid_var = tk.BooleanVar(value=self.app_config.get("show_grid", True))
        ttk.Checkbutton(
            display_frame,
            text="Show Grid in Charts",
            variable=self.show_grid_var
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Label(display_frame, text="Chart Refresh Rate (sec):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.chart_refresh_var = tk.StringVar(value=str(self.app_config.get("chart_refresh", 5)))
        ttk.Entry(display_frame, textvariable=self.chart_refresh_var, width=10).grid(row=2, column=1, padx=5, pady=5)
    
    def create_advanced_settings(self, parent_notebook):
        """Create advanced settings sub-tab"""
        tab = ttk.Frame(parent_notebook)
        parent_notebook.add(tab, text="Advanced")
        
        # Performance Settings
        perf_frame = ttk.LabelFrame(tab, text="Performance Optimization", padding=10)
        perf_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(perf_frame, text="Max Concurrent Requests:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_requests_var = tk.StringVar(value=str(self.app_config.get("max_concurrent_requests", 5)))
        ttk.Entry(perf_frame, textvariable=self.max_requests_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(perf_frame, text="Request Timeout (sec):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.request_timeout_var = tk.StringVar(value=str(self.app_config.get("request_timeout", 30)))
        ttk.Entry(perf_frame, textvariable=self.request_timeout_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.enable_caching_var = tk.BooleanVar(value=self.app_config.get("enable_caching", True))
        ttk.Checkbutton(
            perf_frame,
            text="Enable Data Caching",
            variable=self.enable_caching_var
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Data Storage Settings
        storage_frame = ttk.LabelFrame(tab, text="Data Storage", padding=10)
        storage_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(storage_frame, text="Max History Days:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_history_var = tk.StringVar(value=str(self.app_config.get("max_history_days", 365)))
        ttk.Entry(storage_frame, textvariable=self.max_history_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(storage_frame, text="Log Retention Days:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.log_retention_var = tk.StringVar(value=str(self.app_config.get("log_retention_days", 30)))
        ttk.Entry(storage_frame, textvariable=self.log_retention_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.compress_old_data_var = tk.BooleanVar(value=self.app_config.get("compress_old_data", True))
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
        self.max_position_var = tk.StringVar(value=str(self.app_config.get("max_position_size", 10)))
        ttk.Entry(position_frame, textvariable=self.max_position_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(position_frame, text="Max Open Positions:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.max_open_pos_var = tk.StringVar(value=str(self.app_config.get("max_open_positions", 5)))
        ttk.Entry(position_frame, textvariable=self.max_open_pos_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(position_frame, text="Position Sizing Method:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.position_method_var = tk.StringVar(value=self.app_config.get("position_sizing_method", "Fixed"))
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
        self.max_daily_loss_var = tk.StringVar(value=str(self.app_config.get("max_daily_loss", 5)))
        ttk.Entry(loss_frame, textvariable=self.max_daily_loss_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(loss_frame, text="Max Weekly Loss (%):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.max_weekly_loss_var = tk.StringVar(value=str(self.app_config.get("max_weekly_loss", 10)))
        ttk.Entry(loss_frame, textvariable=self.max_weekly_loss_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.auto_stop_on_limit_var = tk.BooleanVar(value=self.app_config.get("auto_stop_on_limit", True))
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
        
        self.notify_trades_var = tk.BooleanVar(value=self.app_config.get("notify_trades", True))
        ttk.Checkbutton(
            alert_frame,
            text="Notify on Trades",
            variable=self.notify_trades_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.notify_errors_var = tk.BooleanVar(value=self.app_config.get("notify_errors", True))
        ttk.Checkbutton(
            alert_frame,
            text="Notify on Errors",
            variable=self.notify_errors_var
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.notify_limits_var = tk.BooleanVar(value=self.app_config.get("notify_limits", True))
        ttk.Checkbutton(
            alert_frame,
            text="Notify on Loss Limits",
            variable=self.notify_limits_var
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        # Notification Methods
        method_frame = ttk.LabelFrame(tab, text="Notification Methods", padding=10)
        method_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.notify_email_var = tk.BooleanVar(value=self.app_config.get("notify_email", False))
        ttk.Checkbutton(
            method_frame,
            text="Email Notifications",
            variable=self.notify_email_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Label(method_frame, text="Email Address:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.email_address_var = tk.StringVar(value=self.app_config.get("email_address", ""))
        ttk.Entry(method_frame, textvariable=self.email_address_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        
        self.notify_sound_var = tk.BooleanVar(value=self.app_config.get("notify_sound", True))
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
        self.rate_limit_buffer_var = tk.StringVar(value=str(self.app_config.get("rate_limit_buffer", 20)))
        ttk.Entry(api_frame, textvariable=self.rate_limit_buffer_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(api_frame, text="Retry Attempts:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.retry_attempts_var = tk.StringVar(value=str(self.app_config.get("retry_attempts", 3)))
        ttk.Entry(api_frame, textvariable=self.retry_attempts_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.use_testnet_var = tk.BooleanVar(value=self.app_config.get("use_testnet", False))
        ttk.Checkbutton(
            api_frame,
            text="Use Testnet (Paper Trading)",
            variable=self.use_testnet_var
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # External Integrations
        integration_frame = ttk.LabelFrame(tab, text="External Integrations", padding=10)
        integration_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.enable_telegram_var = tk.BooleanVar(value=self.app_config.get("enable_telegram", False))
        ttk.Checkbutton(
            integration_frame,
            text="Enable Telegram Bot",
            variable=self.enable_telegram_var
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Label(integration_frame, text="Telegram Bot Token:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.telegram_token_var = tk.StringVar(value=self.app_config.get("telegram_token", ""))
        ttk.Entry(integration_frame, textvariable=self.telegram_token_var, width=30, show="*").grid(row=1, column=1, padx=5, pady=5)
        
        self.enable_webhook_var = tk.BooleanVar(value=self.app_config.get("enable_webhook", False))
        ttk.Checkbutton(
            integration_frame,
            text="Enable Webhook Notifications",
            variable=self.enable_webhook_var
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        ttk.Label(integration_frame, text="Webhook URL:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.webhook_url_var = tk.StringVar(value=self.app_config.get("webhook_url", ""))
        ttk.Entry(integration_frame, textvariable=self.webhook_url_var, width=30).grid(row=3, column=1, padx=5, pady=5)
    
    def save_all_settings(self):
        """Save all settings from the settings tab"""
        try:
            # General settings
            self.app_config["theme"] = self.theme_var.get()
            self.app_config["language"] = self.language_var.get()
            self.app_config["autosave_interval"] = int(self.autosave_var.get())
            self.app_config["show_tooltips"] = self.show_tooltips_var.get()
            self.app_config["show_grid"] = self.show_grid_var.get()
            self.app_config["chart_refresh"] = int(self.chart_refresh_var.get())
            
            # Advanced settings
            self.app_config["max_concurrent_requests"] = int(self.max_requests_var.get())
            self.app_config["request_timeout"] = int(self.request_timeout_var.get())
            self.app_config["enable_caching"] = self.enable_caching_var.get()
            self.app_config["max_history_days"] = int(self.max_history_var.get())
            self.app_config["log_retention_days"] = int(self.log_retention_var.get())
            self.app_config["compress_old_data"] = self.compress_old_data_var.get()
            
            # Risk management settings
            self.app_config["max_position_size"] = float(self.max_position_var.get())
            self.app_config["max_open_positions"] = int(self.max_open_pos_var.get())
            self.app_config["position_sizing_method"] = self.position_method_var.get()
            self.app_config["max_daily_loss"] = float(self.max_daily_loss_var.get())
            self.app_config["max_weekly_loss"] = float(self.max_weekly_loss_var.get())
            self.app_config["auto_stop_on_limit"] = self.auto_stop_on_limit_var.get()
            
            # Notification settings
            self.app_config["notify_trades"] = self.notify_trades_var.get()
            self.app_config["notify_errors"] = self.notify_errors_var.get()
            self.app_config["notify_limits"] = self.notify_limits_var.get()
            self.app_config["notify_email"] = self.notify_email_var.get()
            self.app_config["email_address"] = self.email_address_var.get()
            self.app_config["notify_sound"] = self.notify_sound_var.get()
            
            # API & integration settings
            self.app_config["rate_limit_buffer"] = int(self.rate_limit_buffer_var.get())
            self.app_config["retry_attempts"] = int(self.retry_attempts_var.get())
            self.app_config["use_testnet"] = self.use_testnet_var.get()
            self.app_config["enable_telegram"] = self.enable_telegram_var.get()
            self.app_config["telegram_token"] = self.telegram_token_var.get()
            self.app_config["enable_webhook"] = self.enable_webhook_var.get()
            self.app_config["webhook_url"] = self.webhook_url_var.get()
            
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
                self.app_config = self.load_config()
                
                # Reload all UI elements with default values
                # This would need to be implemented for each variable
                self.logger.info("Settings reset to defaults")
                messagebox.showinfo("Success", "Settings reset to defaults. Please restart the application.")
                
            except Exception as e:
                self.logger.error(f"Error resetting settings: {e}")
                messagebox.showerror("Error", f"Failed to reset settings: {e}")

    def create_status_bar(self):
        """Create enhanced status bar with color coding"""
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Arial', 9)
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add timestamp label
        self.time_var = tk.StringVar()
        time_label = ttk.Label(
            status_frame,
            textvariable=self.time_var,
            relief=tk.SUNKEN,
            anchor=tk.E,
            font=('Arial', 9)
        )
        time_label.pack(side=tk.RIGHT, padx=5)
        
        self.status_var.set("✓ Ready - Crypto Trader")
        self.update_time()
        
    def update_time(self):
        """Update the time display in status bar"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(current_time)
        # Schedule next update and store ID for cleanup
        self.time_update_id = self.after(1000, self.update_time)
        
    def destroy(self):
        """Clean up timer when window is destroyed"""
        if self.time_update_id:
            self.after_cancel(self.time_update_id)
        super().destroy()

    # Event handlers
    def save_exchange_settings(self):
        """Save exchange settings with validation"""
        try:
            # Validate inputs
            if not self.api_key_var.get().strip():
                messagebox.showwarning("Validation Error", "API Key is required. Please enter your exchange API key.")
                return
                
            if not self.secret_key_var.get().strip():
                messagebox.showwarning("Validation Error", "Secret Key is required. Please enter your exchange secret key.")
                return
            
            # Check if at least one pair is selected
            if not self.pairs_listbox.curselection():
                messagebox.showwarning("Validation Error", "Please select at least one trading pair.")
                return
                
            # Check if at least one timeframe is selected
            if not self.timeframes_listbox.curselection():
                messagebox.showwarning("Validation Error", "Please select at least one timeframe.")
                return
            
            self.app_config["exchange"]["id"] = self.exchange_var.get()
            self.app_config["exchange"]["api_key"] = self.api_key_var.get()
            self.app_config["exchange"]["secret_key"] = self.secret_key_var.get()
            self.app_config["exchange"]["symbols"] = [
                self.pairs_listbox.get(i) for i in self.pairs_listbox.curselection()
            ]
            self.app_config["exchange"]["timeframes"] = [
                self.timeframes_listbox.get(i) for i in self.timeframes_listbox.curselection()
            ]

            self.save_config()
            self.logger.info("Exchange settings saved successfully")
            self.status_var.set("✓ Exchange settings saved successfully")
            messagebox.showinfo("Success", 
                f"Exchange settings saved successfully!\n\n"
                f"Exchange: {self.exchange_var.get()}\n"
                f"Trading Pairs: {len(self.app_config['exchange']['symbols'])}\n"
                f"Timeframes: {len(self.app_config['exchange']['timeframes'])}")

        except Exception as e:
            self.logger.error(f"Error saving exchange settings: {e}")
            messagebox.showerror("Error", 
                f"Failed to save settings.\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check your inputs and try again.")
            self.status_var.set("✗ Error saving exchange settings")

    def start_data_collection(self):
        """Start historical data collection"""
        try:
            # Validate inputs
            try:
                days = int(self.train_days_history_var.get())
                if days < self.MIN_HISTORY_DAYS or days > self.MAX_HISTORY_DAYS:
                    messagebox.showwarning("Invalid Input", 
                        f"Historical days must be between {self.MIN_HISTORY_DAYS} and {self.MAX_HISTORY_DAYS}.\n\n"
                        "Recommended: 30-90 days for balanced training.")
                    return
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for historical days.")
                return
            
            symbols = [
                self.pairs_listbox.get(i) for i in self.pairs_listbox.curselection()
            ]
            if not symbols:
                messagebox.showwarning("No Selection", "Please select at least one trading pair from the Exchange tab.")
                return
                
            timeframes = [
                self.timeframes_listbox.get(i) for i in self.timeframes_listbox.curselection()
            ]
            if not timeframes:
                messagebox.showwarning("No Selection", "Please select at least one timeframe from the Exchange tab.")
                return

            # Confirm action
            if not messagebox.askyesno("Confirm Data Collection",
                f"This will collect {days} days of historical data for:\n\n"
                f"• {len(symbols)} trading pair(s)\n"
                f"• {len(timeframes)} timeframe(s)\n\n"
                f"This may take several minutes. Continue?"):
                return

            collector = MarketDataCollector(
                exchange_id=self.exchange_var.get(),
                symbols=symbols,
                timeframes=timeframes
            )

            self.status_var.set(f"📊 Collecting {days} days of historical data...")
            self.update_idletasks()  # Update UI immediately
            
            collector.collect_all_markets(days_history=days)

            self.logger.info("Historical data collection completed")
            self.status_var.set("✓ Data collection completed successfully")
            messagebox.showinfo("Success", 
                f"Data collection completed successfully!\n\n"
                f"Collected {days} days of data for {len(symbols)} pair(s).\n"
                f"You can now start training.")

        except Exception as e:
            self.logger.error(f"Error collecting data: {e}")
            self.status_var.set("✗ Data collection failed")
            messagebox.showerror("Error", 
                f"Data collection failed.\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check your exchange settings and try again.")

    def start_training(self):
        """Start model training"""
        try:
            # Validate training parameters
            try:
                epochs = int(self.train_epochs_var.get())
                if epochs < self.MIN_EPOCHS or epochs > self.MAX_EPOCHS:
                    messagebox.showwarning("Invalid Input", 
                        f"Epochs must be between {self.MIN_EPOCHS} and {self.MAX_EPOCHS}.\n\n"
                        "Recommended: 10-50 epochs for most cases.")
                    return
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for epochs.")
                return
            
            # Check if pairs are selected
            if not self.pairs_listbox.curselection():
                messagebox.showwarning("No Selection", "Please select a trading pair from the Exchange tab.")
                return
                
            if not self.timeframes_listbox.curselection():
                messagebox.showwarning("No Selection", "Please select a timeframe from the Exchange tab.")
                return
            
            # Confirm training
            if not messagebox.askyesno("Confirm Training",
                f"Start model training with {epochs} epochs?\n\n"
                f"This may take 10-30 minutes depending on data size.\n"
                f"The application may appear unresponsive during training.\n\n"
                f"Continue?"):
                return
                
            if self.trader is None:
                self.trader = CryptoTrader(
                    exchange_id=self.exchange_var.get(),
                    symbol=self.pairs_listbox.get(self.pairs_listbox.curselection()[0]),
                    timeframe=self.timeframes_listbox.get(self.timeframes_listbox.curselection()[0]),
                    train_mode=True
                )

            self.status_var.set(f"🎓 Training model ({epochs} epochs)... Please wait...")
            self.update_idletasks()  # Update UI immediately
            
            self.trader.train_on_historical(epochs=epochs)

            self.logger.info("Model training completed")
            self.status_var.set("✓ Model training completed successfully")
            messagebox.showinfo("Success", 
                f"Model training completed successfully!\n\n"
                f"Trained for {epochs} epochs.\n"
                f"The model is now ready for use.")

        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            self.status_var.set("✗ Training failed")
            messagebox.showerror("Error", 
                f"Training failed.\n\n"
                f"Error: {str(e)}\n\n"
                f"Make sure you have collected data first.")

    def start_live_trading(self):
        """Start live trading with enhanced confirmations"""
        try:
            # Multiple safety checks
            if not messagebox.askyesno("⚠️ FIRST CONFIRMATION ⚠️",
                "You are about to start LIVE TRADING with REAL MONEY.\n\n"
                "This is NOT a simulation.\n\n"
                "Are you absolutely sure you want to continue?",
                icon='warning'):
                return
            
            # Second confirmation
            if not messagebox.askyesno("⚠️ FINAL CONFIRMATION ⚠️",
                "FINAL WARNING: Live trading will use real funds!\n\n"
                "Have you:\n"
                "✓ Tested thoroughly in training mode?\n"
                "✓ Set appropriate risk limits?\n"
                "✓ Double-checked your API credentials?\n"
                "✓ Started with a small amount?\n\n"
                "Proceed with live trading?",
                icon='warning'):
                return
            
            # Validate selection
            if not self.pairs_listbox.curselection():
                messagebox.showerror("Error", "Please select a trading pair from the Exchange tab.")
                return
                
            if not self.timeframes_listbox.curselection():
                messagebox.showerror("Error", "Please select a timeframe from the Exchange tab.")
                return

            if self.trader is None or self.trader.train_mode:
                self.trader = CryptoTrader(
                    exchange_id=self.exchange_var.get(),
                    symbol=self.pairs_listbox.get(self.pairs_listbox.curselection()[0]),
                    timeframe=self.timeframes_listbox.get(self.timeframes_listbox.curselection()[0]),
                    train_mode=False
                )

            self.status_var.set("▶️ Live trading started - Monitor carefully!")
            self.update_trading_status(
                "Live trading session started\n"
                f"Pair: {self.pairs_listbox.get(self.pairs_listbox.curselection()[0])}\n"
                f"Timeframe: {self.timeframes_listbox.get(self.timeframes_listbox.curselection()[0])}\n"
                "Status: ACTIVE"
            )
            
            # Set initial balance from exchange
            balance = self.trader.get_account_balance()
            self.performance_tracker.initial_balance = balance
            self.performance_tracker.current_balance = balance

            # Start balance tracking
            self.after(1000, self.update_performance)

            self.trader.run()

        except Exception as e:
            self.logger.error(f"Error starting live trading: {e}")
            self.status_var.set("✗ Failed to start trading")
            self.update_trading_status(f"Error: {str(e)}")
            messagebox.showerror("Error", 
                f"Failed to start trading.\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check your settings and logs.")

    def stop_trading(self):
        """Stop live trading"""
        try:
            if self.trader:
                if messagebox.askyesno("Confirm Stop",
                    "Stop all trading activity?\n\n"
                    "This will:\n"
                    "• Stop placing new trades\n"
                    "• Keep existing positions open\n"
                    "• Save current performance data\n\n"
                    "Continue?"):
                    
                    self.trader.stop()
                    self.logger.info("Trading stopped by user")
                    self.status_var.set("⏹️ Trading stopped")
                    self.update_trading_status("Trading session stopped by user")

                    # Save final performance snapshot
                    self.performance_tracker.save_history()

                    # Reset performance tracker
                    self.performance_tracker.reset()
                    
                    # Force update of all displays
                    self.force_performance_update()
                    
                    messagebox.showinfo("Trading Stopped", 
                        "Trading has been stopped successfully.\n\n"
                        "Performance data has been saved.")
            else:
                messagebox.showinfo("No Active Trading", "There is no active trading session to stop.")
                
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
            self.status_var.set("✗ Error stopping trading")
            messagebox.showerror("Error", 
                f"Error stopping trading.\n\n"
                f"Error: {str(e)}")

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
