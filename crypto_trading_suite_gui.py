"""
Main GUI application for the Crypto Trading Suite.
Integrates all components into a single unified interface.
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import queue
import json

from method_validator import MethodValidator
from data_validator import DataValidator
from trading_types import MarketData, ValidationError
from crypto_trader import CryptoTrader
from market_data_collector import MarketDataCollector
from performance_tracker import PerfomanceTracker
from market_data_aggregator import MarketDataAggregator, AGGREGATOR_CONFIG
from trading_strategy import SystematicTradingStrategy, TradingParameters
from endpoint_validator import EndpointValidator
from config import load_config, setup_logging

class CryptoTradingSuite(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Crypto Trading Suite")
        self.geometry("1200x800")

        # Initialize components
        self.config = load_config()
        self.logger = setup_logging()
        self.setup_gui_elements()
        self.setup_trading_components()

    def setup_gui_elements(self):
        """Setup all GUI elements"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # Create various tabs
        self.create_trading_tab()
        self.create_monitoring_tab()
        self.create_validation_tab()
        self.create_settings_tab()

        # Create status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_trading_components(self):
        """Initialize trading components"""
        try:
            self.trader = CryptoTrader()
            self.data_collector = MarketDataCollector()
            self.performance_tracker = PerfomanceTracker()
            self.endpoint_validator = EndpointValidator()
            self.method_validator = MethodValidator()
            self.data_validator = DataValidator()

            self.update_status("Components initialized successfully")
        except Exception as e:
            self.show_error(f"Error initializing components: {str(e)}")

    def create_trading_tab(self):
        """Create the main trading interface tab"""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="Trading")

        # Add trading controls
        controls_frame = ttk.LabelFrame(trading_frame, text="Trading Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Start Trading", command=self.start_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Stop Trading", command=self.stop_trading).pack(side=tk.LEFT, padx=5)

        # Add trading view
        self.trading_log = scrolledtext.ScrolledText(trading_frame, height=10)
        self.trading_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_monitoring_tab(self):
        """Create the monitoring and performance tab"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="Monitoring")

        # Add performance metrics
        metrics_frame = ttk.LabelFrame(monitor_frame, text="Performance Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add charts and graphs here
        self.setup_performance_charts(monitor_frame)

    def create_validation_tab(self):
        """Create the validation tools tab"""
        validation_frame = ttk.Frame(self.notebook)
        self.notebook.add(validation_frame, text="Validation")

        # Method validation section
        method_frame = ttk.LabelFrame(validation_frame, text="Method Validation")
        method_frame.pack(fill=tk.X, padx=5, pady=5)

        # Data validation section
        data_frame = ttk.LabelFrame(validation_frame, text="Data Validation")
        data_frame.pack(fill=tk.X, padx=5, pady=5)

    def create_settings_tab(self):
        """Create the settings and configuration tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")

        # Configuration section
        config_frame = ttk.LabelFrame(settings_frame, text="Configuration")
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add configuration options here
        self.setup_config_options(config_frame)

    def setup_performance_charts(self, parent):
        """Setup performance monitoring charts"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Create figure for charts
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_config_options(self, parent):
        """Setup configuration options"""
        # Trading parameters
        for key, value in self.config['trading'].items():
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=key).pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=tk.StringVar(value=str(value))).pack(side=tk.RIGHT)

    def start_trading(self):
        """Start the trading system"""
        try:
            self.trader.run()
            self.update_status("Trading started")
        except Exception as e:
            self.show_error(f"Error starting trading: {str(e)}")

    def stop_trading(self):
        """Stop the trading system"""
        try:
            self.trader.stop()
            self.update_status("Trading stopped")
        except Exception as e:
            self.show_error(f"Error stopping trading: {str(e)}")

    def update_status(self, message: str):
        """Update the status bar message"""
        self.status_bar.config(text=message)
        self.logger.info(message)

    def show_error(self, message: str):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.logger.error(message)

if __name__ == "__main__":
    app = CryptoTradingSuite()
    app.mainloop()
