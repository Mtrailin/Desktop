"""
Build Your Own Trading System
A comprehensive GUI application for creating customized trading strategies with multi-level settings.
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from crypto_trader import CryptoTrader
from market_data_collector import MarketDataCollector
from performance_tracker import PerfomanceTracker
from trading_strategy import SystematicTradingStrategy, TradingParameters
from data_validator import DataValidator
from method_validator import MethodValidator


class MultiLevelSettingsDialog(tk.Toplevel):
    """Dialog for multi-level settings configuration"""
    
    def __init__(self, parent, settings_dict: Dict[str, Any], title: str = "Settings"):
        super().__init__(parent)
        self.title(title)
        self.geometry("800x600")
        self.settings_dict = settings_dict
        self.result = None
        
        # Create main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Category tree
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        ttk.Label(left_frame, text="Categories", font=('Arial', 12, 'bold')).pack()
        
        self.category_tree = ttk.Treeview(left_frame, selectmode='browse')
        self.category_tree.pack(fill=tk.BOTH, expand=True)
        self.category_tree.bind('<<TreeviewSelect>>', self.on_category_select)
        
        # Right side - Settings panel
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="Settings", font=('Arial', 12, 'bold')).pack()
        
        # Scrollable settings area
        self.canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.canvas.yview)
        self.settings_frame = ttk.Frame(self.canvas)
        
        self.settings_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom buttons
        button_frame = ttk.Frame(self, padding="10")
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Save", command=self.save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT)
        
        # Populate tree
        self.populate_tree()
        
        # Variables to store edited values
        self.setting_vars = {}
        
    def populate_tree(self):
        """Populate the category tree from settings dict"""
        for category in self.settings_dict.keys():
            self.category_tree.insert('', 'end', category, text=category.replace('_', ' ').title())
    
    def on_category_select(self, event):
        """Handle category selection"""
        selection = self.category_tree.selection()
        if not selection:
            return
            
        category = selection[0]
        self.display_settings(category)
    
    def display_settings(self, category: str):
        """Display settings for the selected category"""
        # Clear existing settings
        for widget in self.settings_frame.winfo_children():
            widget.destroy()
        
        if category not in self.settings_dict:
            return
        
        settings = self.settings_dict[category]
        
        row = 0
        for key, value in settings.items():
            # Label
            label = ttk.Label(
                self.settings_frame,
                text=key.replace('_', ' ').title() + ":",
                font=('Arial', 10)
            )
            label.grid(row=row, column=0, sticky='w', padx=5, pady=5)
            
            # Input widget based on type
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                widget = ttk.Checkbutton(self.settings_frame, variable=var)
            elif isinstance(value, (int, float)):
                var = tk.StringVar(value=str(value))
                widget = ttk.Entry(self.settings_frame, textvariable=var, width=30)
            elif isinstance(value, list):
                var = tk.StringVar(value=', '.join(map(str, value)))
                widget = ttk.Entry(self.settings_frame, textvariable=var, width=30)
                ttk.Label(
                    self.settings_frame,
                    text="(comma-separated)",
                    font=('Arial', 8)
                ).grid(row=row, column=2, sticky='w', padx=5)
            else:
                var = tk.StringVar(value=str(value))
                widget = ttk.Entry(self.settings_frame, textvariable=var, width=30)
            
            widget.grid(row=row, column=1, sticky='ew', padx=5, pady=5)
            
            # Store variable reference
            self.setting_vars[f"{category}.{key}"] = (var, type(value))
            
            row += 1
    
    def save_settings(self):
        """Save edited settings"""
        try:
            # Update settings dict from variables
            for key, (var, value_type) in self.setting_vars.items():
                category, setting = key.split('.', 1)
                
                if value_type == bool:
                    self.settings_dict[category][setting] = var.get()
                elif value_type == int:
                    self.settings_dict[category][setting] = int(var.get())
                elif value_type == float:
                    self.settings_dict[category][setting] = float(var.get())
                elif value_type == list:
                    # Parse comma-separated values
                    value_str = var.get()
                    self.settings_dict[category][setting] = [
                        x.strip() for x in value_str.split(',') if x.strip()
                    ]
                else:
                    self.settings_dict[category][setting] = var.get()
            
            self.result = self.settings_dict
            messagebox.showinfo("Success", "Settings saved successfully!")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def reset_defaults(self):
        """Reset to default settings"""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            # Re-populate with defaults would go here
            messagebox.showinfo("Info", "Please restart to load default settings")


class BuildYourOwnTrading(tk.Tk):
    """Main application for building custom trading systems"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Build Your Own Trading System")
        self.geometry("1400x900")
        
        # Initialize configuration
        self.config = self.load_config()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.trader = None
        self.performance_tracker = PerfomanceTracker()
        self.data_validator = DataValidator()
        self.method_validator = MethodValidator()
        
        # Build UI
        self.create_menu_bar()
        self.create_main_interface()
        self.create_status_bar()
        
        # Load saved strategies
        self.load_strategies()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        config_path = Path("config/build_your_own_trading.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        
        # Default comprehensive configuration
        return {
            "exchange": {
                "id": "binance",
                "api_key": "",
                "secret_key": "",
                "testnet": True,
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
            },
            "strategy": {
                "type": "custom",
                "name": "My Strategy",
                "entry_conditions": [],
                "exit_conditions": [],
                "indicators": ["RSI", "MACD", "BB"],
                "timeframe": "1h"
            },
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss_percent": 2.0,
                "take_profit_percent": 3.0,
                "trailing_stop": True,
                "max_drawdown": 10.0,
                "risk_per_trade": 1.0
            },
            "order_execution": {
                "order_type": "limit",
                "slippage_tolerance": 0.1,
                "timeout": 30,
                "partial_fills": True,
                "post_only": False
            },
            "backtesting": {
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "initial_capital": 10000.0,
                "commission": 0.001
            },
            "advanced": {
                "use_machine_learning": False,
                "model_type": "LSTM",
                "retrain_interval": 24,
                "feature_engineering": True,
                "portfolio_optimization": False
            }
        }
    
    def save_config(self):
        """Save current configuration to file"""
        config_path = Path("config/build_your_own_trading.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)
        
        self.logger.info("Configuration saved")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("build_your_own_trading")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(
            log_path / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        )
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Strategy", command=self.new_strategy)
        file_menu.add_command(label="Load Strategy", command=self.load_strategy)
        file_menu.add_command(label="Save Strategy", command=self.save_strategy)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Exchange Settings", command=lambda: self.open_settings("exchange"))
        settings_menu.add_command(label="Risk Management", command=lambda: self.open_settings("risk_management"))
        settings_menu.add_command(label="Order Execution", command=lambda: self.open_settings("order_execution"))
        settings_menu.add_command(label="Advanced Settings", command=lambda: self.open_settings("advanced"))
        settings_menu.add_separator()
        settings_menu.add_command(label="All Settings", command=self.open_all_settings)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Backtest Strategy", command=self.backtest_strategy)
        tools_menu.add_command(label="Optimize Parameters", command=self.optimize_parameters)
        tools_menu.add_command(label="Validate Data", command=self.validate_data)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_interface(self):
        """Create the main application interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_strategy_builder_tab()
        self.create_backtesting_tab()
        self.create_live_trading_tab()
        self.create_performance_tab()
        self.create_settings_tab()
    
    def create_strategy_builder_tab(self):
        """Create strategy builder tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Strategy Builder")
        
        # Strategy name
        name_frame = ttk.LabelFrame(tab, text="Strategy Information")
        name_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(name_frame, text="Strategy Name:").grid(row=0, column=0, padx=5, pady=5)
        self.strategy_name_var = tk.StringVar(value=self.config["strategy"]["name"])
        ttk.Entry(name_frame, textvariable=self.strategy_name_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(name_frame, text="Type:").grid(row=1, column=0, padx=5, pady=5)
        self.strategy_type_var = tk.StringVar(value=self.config["strategy"]["type"])
        ttk.Combobox(
            name_frame,
            textvariable=self.strategy_type_var,
            values=["custom", "momentum", "mean_reversion", "breakout", "scalping", "swing"],
            width=37
        ).grid(row=1, column=1, padx=5, pady=5)
        
        # Indicators
        indicators_frame = ttk.LabelFrame(tab, text="Technical Indicators")
        indicators_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side - available indicators
        left_ind = ttk.Frame(indicators_frame)
        left_ind.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(left_ind, text="Available Indicators").pack()
        
        available_indicators = [
            "SMA", "EMA", "RSI", "MACD", "Bollinger Bands",
            "Stochastic", "ATR", "ADX", "CCI", "Williams %R",
            "Volume", "OBV", "VWAP", "Ichimoku", "Parabolic SAR"
        ]
        
        self.available_ind_list = tk.Listbox(left_ind, selectmode=tk.MULTIPLE, height=15)
        self.available_ind_list.pack(fill=tk.BOTH, expand=True)
        
        for ind in available_indicators:
            self.available_ind_list.insert(tk.END, ind)
        
        # Middle - buttons
        middle_ind = ttk.Frame(indicators_frame)
        middle_ind.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(middle_ind, text="Add >>", command=self.add_indicator).pack(pady=5)
        ttk.Button(middle_ind, text="<< Remove", command=self.remove_indicator).pack(pady=5)
        
        # Right side - selected indicators
        right_ind = ttk.Frame(indicators_frame)
        right_ind.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(right_ind, text="Selected Indicators").pack()
        
        self.selected_ind_list = tk.Listbox(right_ind, selectmode=tk.MULTIPLE, height=15)
        self.selected_ind_list.pack(fill=tk.BOTH, expand=True)
        
        # Entry/Exit conditions
        conditions_frame = ttk.LabelFrame(tab, text="Trading Conditions")
        conditions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conditions_frame, text="Entry Conditions:").grid(row=0, column=0, padx=5, pady=5, sticky='nw')
        self.entry_conditions_text = scrolledtext.ScrolledText(conditions_frame, height=4, width=50)
        self.entry_conditions_text.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(conditions_frame, text="Exit Conditions:").grid(row=1, column=0, padx=5, pady=5, sticky='nw')
        self.exit_conditions_text = scrolledtext.ScrolledText(conditions_frame, height=4, width=50)
        self.exit_conditions_text.grid(row=1, column=1, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Save Strategy", command=self.save_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Test Strategy", command=self.test_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Deploy Strategy", command=self.deploy_strategy).pack(side=tk.LEFT, padx=5)
    
    def create_backtesting_tab(self):
        """Create backtesting tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Backtesting")
        
        # Parameters
        params_frame = ttk.LabelFrame(tab, text="Backtest Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Start Date:").grid(row=0, column=0, padx=5, pady=5)
        self.backtest_start_var = tk.StringVar(value=self.config["backtesting"]["start_date"])
        ttk.Entry(params_frame, textvariable=self.backtest_start_var).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="End Date:").grid(row=0, column=2, padx=5, pady=5)
        self.backtest_end_var = tk.StringVar(value=self.config["backtesting"]["end_date"])
        ttk.Entry(params_frame, textvariable=self.backtest_end_var).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Initial Capital:").grid(row=1, column=0, padx=5, pady=5)
        self.backtest_capital_var = tk.StringVar(value=str(self.config["backtesting"]["initial_capital"]))
        ttk.Entry(params_frame, textvariable=self.backtest_capital_var).grid(row=1, column=1, padx=5, pady=5)
        
        # Results
        results_frame = ttk.LabelFrame(tab, text="Backtest Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.backtest_results = scrolledtext.ScrolledText(results_frame, height=20)
        self.backtest_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Run Backtest", command=self.run_backtest).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Results", command=self.export_backtest_results).pack(side=tk.LEFT, padx=5)
    
    def create_live_trading_tab(self):
        """Create live trading tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Live Trading")
        
        # Controls
        control_frame = ttk.LabelFrame(tab, text="Trading Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Start Live Trading", command=self.start_live_trading).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Stop Trading", command=self.stop_trading).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Emergency Stop", command=self.emergency_stop).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(tab, text="Trading Status")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.trading_log = scrolledtext.ScrolledText(status_frame, height=20)
        self.trading_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_performance_tab(self):
        """Create performance monitoring tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Performance")
        
        # Metrics
        metrics_frame = ttk.LabelFrame(tab, text="Performance Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        metrics = [
            ("Total P&L:", "total_pnl"),
            ("Win Rate:", "win_rate"),
            ("Sharpe Ratio:", "sharpe"),
            ("Max Drawdown:", "max_dd")
        ]
        
        for i, (label, key) in enumerate(metrics):
            ttk.Label(metrics_frame, text=label).grid(row=i//2, column=(i%2)*2, padx=5, pady=5, sticky='w')
            var = tk.StringVar(value="0.00")
            ttk.Label(metrics_frame, textvariable=var).grid(row=i//2, column=(i%2)*2+1, padx=5, pady=5, sticky='e')
            setattr(self, f"{key}_var", var)
        
        # Charts placeholder
        charts_frame = ttk.LabelFrame(tab, text="Performance Charts")
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(charts_frame, text="Performance charts will appear here").pack(expand=True)
    
    def create_settings_tab(self):
        """Create comprehensive multi-level settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Settings")
        
        # Create hierarchical settings view
        settings_main = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        settings_main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left pane - Settings categories
        left_pane = ttk.Frame(settings_main)
        settings_main.add(left_pane, weight=1)
        
        ttk.Label(left_pane, text="Settings Categories", font=('Arial', 12, 'bold')).pack()
        
        # Category tree
        self.settings_tree = ttk.Treeview(left_pane, selectmode='browse')
        self.settings_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Populate categories
        categories = [
            ("exchange", "Exchange Configuration"),
            ("strategy", "Strategy Settings"),
            ("risk_management", "Risk Management"),
            ("order_execution", "Order Execution"),
            ("backtesting", "Backtesting"),
            ("advanced", "Advanced Settings")
        ]
        
        for cat_id, cat_name in categories:
            self.settings_tree.insert('', 'end', cat_id, text=cat_name)
        
        self.settings_tree.bind('<<TreeviewSelect>>', self.on_settings_category_select)
        
        # Right pane - Settings details
        right_pane = ttk.Frame(settings_main)
        settings_main.add(right_pane, weight=3)
        
        ttk.Label(right_pane, text="Settings", font=('Arial', 12, 'bold')).pack()
        
        # Scrollable settings panel
        settings_canvas = tk.Canvas(right_pane)
        settings_scrollbar = ttk.Scrollbar(right_pane, orient="vertical", command=settings_canvas.yview)
        self.settings_detail_frame = ttk.Frame(settings_canvas)
        
        self.settings_detail_frame.bind(
            "<Configure>",
            lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        )
        
        settings_canvas.create_window((0, 0), window=self.settings_detail_frame, anchor="nw")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        
        settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom buttons
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Open Advanced Settings Editor", command=self.open_all_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save All Settings", command=self.save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_to_defaults).pack(side=tk.RIGHT, padx=5)
    
    def on_settings_category_select(self, event):
        """Handle settings category selection"""
        selection = self.settings_tree.selection()
        if not selection:
            return
        
        category = selection[0]
        self.display_category_settings(category)
    
    def display_category_settings(self, category: str):
        """Display settings for selected category"""
        # Clear existing widgets
        for widget in self.settings_detail_frame.winfo_children():
            widget.destroy()
        
        if category not in self.config:
            return
        
        settings = self.config[category]
        
        row = 0
        for key, value in settings.items():
            # Create label
            label = ttk.Label(
                self.settings_detail_frame,
                text=key.replace('_', ' ').title() + ":",
                font=('Arial', 10)
            )
            label.grid(row=row, column=0, sticky='w', padx=10, pady=5)
            
            # Create appropriate input widget
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                widget = ttk.Checkbutton(self.settings_detail_frame, variable=var)
            elif isinstance(value, (int, float)):
                var = tk.StringVar(value=str(value))
                widget = ttk.Entry(self.settings_detail_frame, textvariable=var, width=40)
            elif isinstance(value, list):
                var = tk.StringVar(value=', '.join(map(str, value)))
                widget = ttk.Entry(self.settings_detail_frame, textvariable=var, width=40)
            else:
                var = tk.StringVar(value=str(value))
                widget = ttk.Entry(self.settings_detail_frame, textvariable=var, width=40)
            
            widget.grid(row=row, column=1, sticky='ew', padx=10, pady=5)
            
            # Store reference for saving
            setattr(self, f"setting_{category}_{key}_var", var)
            
            row += 1
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    # Event handlers
    def open_settings(self, category: str):
        """Open settings dialog for specific category"""
        settings = {category: self.config[category]}
        dialog = MultiLevelSettingsDialog(self, settings, f"{category.replace('_', ' ').title()} Settings")
        self.wait_window(dialog)
        
        if dialog.result:
            self.config.update(dialog.result)
            self.save_config()
    
    def open_all_settings(self):
        """Open comprehensive settings dialog"""
        dialog = MultiLevelSettingsDialog(self, self.config.copy(), "All Settings")
        self.wait_window(dialog)
        
        if dialog.result:
            self.config = dialog.result
            self.save_config()
    
    def add_indicator(self):
        """Add selected indicator to strategy"""
        selection = self.available_ind_list.curselection()
        for idx in selection:
            indicator = self.available_ind_list.get(idx)
            if indicator not in self.selected_ind_list.get(0, tk.END):
                self.selected_ind_list.insert(tk.END, indicator)
    
    def remove_indicator(self):
        """Remove selected indicator from strategy"""
        selection = self.selected_ind_list.curselection()
        for idx in reversed(selection):
            self.selected_ind_list.delete(idx)
    
    def new_strategy(self):
        """Create new strategy"""
        if messagebox.askyesno("New Strategy", "Clear current strategy and create new one?"):
            self.strategy_name_var.set("New Strategy")
            self.entry_conditions_text.delete('1.0', tk.END)
            self.exit_conditions_text.delete('1.0', tk.END)
            self.selected_ind_list.delete(0, tk.END)
            self.status_var.set("New strategy created")
    
    def load_strategy(self):
        """Load strategy from file"""
        filename = filedialog.askopenfilename(
            title="Load Strategy",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    strategy = json.load(f)
                
                self.strategy_name_var.set(strategy.get("name", "Loaded Strategy"))
                # Load other strategy details...
                
                self.status_var.set(f"Strategy loaded from {filename}")
                self.logger.info(f"Strategy loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load strategy: {str(e)}")
    
    def save_strategy(self):
        """Save current strategy"""
        strategy = {
            "name": self.strategy_name_var.get(),
            "type": self.strategy_type_var.get(),
            "indicators": list(self.selected_ind_list.get(0, tk.END)),
            "entry_conditions": self.entry_conditions_text.get('1.0', tk.END).strip(),
            "exit_conditions": self.exit_conditions_text.get('1.0', tk.END).strip()
        }
        
        filename = filedialog.asksaveasfilename(
            title="Save Strategy",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(strategy, f, indent=4)
                
                self.status_var.set(f"Strategy saved to {filename}")
                self.logger.info(f"Strategy saved to {filename}")
                messagebox.showinfo("Success", "Strategy saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save strategy: {str(e)}")
    
    def test_strategy(self):
        """Test current strategy"""
        self.status_var.set("Testing strategy...")
        messagebox.showinfo("Test", "Strategy testing not yet implemented")
    
    def deploy_strategy(self):
        """Deploy strategy to live trading"""
        if messagebox.askyesno("Deploy", "Deploy strategy to live trading?"):
            self.status_var.set("Strategy deployed")
            messagebox.showinfo("Deploy", "Strategy deployment not yet implemented")
    
    def backtest_strategy(self):
        """Run backtest on current strategy"""
        self.notebook.select(1)  # Switch to backtesting tab
        self.run_backtest()
    
    def run_backtest(self):
        """Execute backtest"""
        self.backtest_results.delete('1.0', tk.END)
        self.backtest_results.insert(tk.END, "Running backtest...\n")
        self.status_var.set("Backtesting in progress...")
        
        # Placeholder for actual backtesting logic
        self.backtest_results.insert(tk.END, "\nBacktest completed!\n")
        self.backtest_results.insert(tk.END, "Total Trades: 0\n")
        self.backtest_results.insert(tk.END, "Win Rate: 0%\n")
        self.backtest_results.insert(tk.END, "Net P&L: $0.00\n")
        
        self.status_var.set("Backtest completed")
    
    def export_backtest_results(self):
        """Export backtest results to file"""
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                results = self.backtest_results.get('1.0', tk.END)
                with open(filename, 'w') as f:
                    f.write(results)
                
                messagebox.showinfo("Success", "Results exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def optimize_parameters(self):
        """Optimize strategy parameters"""
        messagebox.showinfo("Optimize", "Parameter optimization not yet implemented")
    
    def validate_data(self):
        """Validate market data"""
        self.status_var.set("Validating data...")
        try:
            # Use data validator
            result = self.data_validator.validate_all()
            if result:
                messagebox.showinfo("Validation", "Data validation passed!")
            else:
                messagebox.showwarning("Validation", "Data validation found issues")
        except Exception as e:
            messagebox.showerror("Error", f"Validation error: {str(e)}")
        finally:
            self.status_var.set("Ready")
    
    def start_live_trading(self):
        """Start live trading"""
        if not messagebox.askyesno("Confirm", "Start live trading with real money?"):
            return
        
        self.status_var.set("Live trading started")
        self.trading_log.insert(tk.END, f"[{datetime.now()}] Trading started\n")
        self.logger.info("Live trading started")
    
    def stop_trading(self):
        """Stop live trading"""
        self.status_var.set("Trading stopped")
        self.trading_log.insert(tk.END, f"[{datetime.now()}] Trading stopped\n")
        self.logger.info("Trading stopped")
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        if messagebox.askyesno("Emergency Stop", "Close all positions immediately?"):
            self.status_var.set("Emergency stop executed")
            self.trading_log.insert(tk.END, f"[{datetime.now()}] EMERGENCY STOP\n")
            self.logger.warning("Emergency stop executed")
    
    def load_strategies(self):
        """Load saved strategies"""
        # Placeholder for loading saved strategies
        pass
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset", "Reset all settings to defaults?"):
            self.config = self.load_config()
            self.save_config()
            messagebox.showinfo("Reset", "Settings reset to defaults")
    
    def show_documentation(self):
        """Show documentation"""
        doc_text = """
Build Your Own Trading System - Documentation

1. Strategy Builder:
   - Create custom trading strategies
   - Select technical indicators
   - Define entry/exit conditions

2. Backtesting:
   - Test strategies on historical data
   - View performance metrics
   - Export results

3. Live Trading:
   - Deploy strategies to live markets
   - Monitor performance in real-time
   - Emergency stop functionality

4. Settings:
   - Multi-level configuration
   - Exchange settings
   - Risk management
   - Advanced options
        """
        
        doc_window = tk.Toplevel(self)
        doc_window.title("Documentation")
        doc_window.geometry("600x400")
        
        text = scrolledtext.ScrolledText(doc_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert('1.0', doc_text)
        text.config(state='disabled')
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Build Your Own Trading System\n\n"
            "Version 1.0.0\n\n"
            "A comprehensive platform for creating and deploying\n"
            "custom cryptocurrency trading strategies."
        )


if __name__ == "__main__":
    app = BuildYourOwnTrading()
    app.mainloop()
