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
        """Create the settings and configuration tab with multi-level menus"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Create hierarchical settings view
        settings_paned = ttk.PanedWindow(settings_frame, orient=tk.HORIZONTAL)
        settings_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left pane - Settings categories
        left_pane = ttk.Frame(settings_paned)
        settings_paned.add(left_pane, weight=1)
        
        ttk.Label(left_pane, text="Settings Categories", font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Category tree
        self.settings_tree = ttk.Treeview(left_pane, selectmode='browse')
        self.settings_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Populate categories
        categories = {
            'exchange': 'Exchange Configuration',
            'trading': 'Trading Parameters',
            'risk': 'Risk Management',
            'model': 'Model Settings',
            'performance': 'Performance Tracking',
            'validation': 'Data Validation',
            'advanced': 'Advanced Options'
        }
        
        for cat_id, cat_name in categories.items():
            self.settings_tree.insert('', 'end', cat_id, text=cat_name)
        
        self.settings_tree.bind('<<TreeviewSelect>>', self.on_settings_category_select)
        
        # Right pane - Settings details
        right_pane = ttk.Frame(settings_paned)
        settings_paned.add(right_pane, weight=3)
        
        ttk.Label(right_pane, text="Settings", font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Scrollable settings area
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
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Save All Settings", command=self.save_all_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Import Config", command=self.import_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Config", command=self.export_config).pack(side=tk.LEFT, padx=5)
    
    def on_settings_category_select(self, event):
        """Handle settings category selection"""
        selection = self.settings_tree.selection()
        if not selection:
            return
        
        category = selection[0]
        self.display_category_settings(category)
    
    def display_category_settings(self, category: str):
        """Display settings for the selected category"""
        # Clear existing widgets
        for widget in self.settings_detail_frame.winfo_children():
            widget.destroy()
        
        # Get settings for category
        settings_map = {
            'exchange': self.config.get('exchange', {}),
            'trading': self.config.get('trading', {}),
            'risk': self.config.get('risk_management', {}),
            'model': self.config.get('model', {}),
            'performance': self.config.get('performance', {}),
            'validation': self.config.get('validation', {}),
            'advanced': self.config.get('advanced', {})
        }
        
        if category not in settings_map:
            return
        
        settings = settings_map[category]
        
        # Initialize settings vars dictionary if not exists
        if not hasattr(self, 'settings_vars'):
            self.settings_vars = {}
        
        row = 0
        for key, value in settings.items():
            # Label
            label = ttk.Label(
                self.settings_detail_frame,
                text=key.replace('_', ' ').title() + ":",
                font=('Arial', 10)
            )
            label.grid(row=row, column=0, sticky='w', padx=10, pady=5)
            
            # Input widget based on type
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
            
            # Store variable reference with category and key for saving
            self.settings_vars[f"{category}.{key}"] = (var, type(value))
            
            row += 1
    
    def save_all_settings(self):
        """Save all settings to config"""
        try:
            # Update config from stored variables
            if hasattr(self, 'settings_vars'):
                for key, (var, value_type) in self.settings_vars.items():
                    parts = key.split('.', 1)
                    if len(parts) != 2:
                        continue
                    
                    category, setting = parts
                    
                    # Map category names to config keys
                    category_map = {
                        'exchange': 'exchange',
                        'trading': 'trading',
                        'risk': 'risk_management',
                        'model': 'model',
                        'performance': 'performance',
                        'validation': 'validation',
                        'advanced': 'advanced'
                    }
                    
                    config_key = category_map.get(category)
                    if not config_key or config_key not in self.config:
                        continue
                    
                    # Convert value based on type
                    try:
                        if value_type == bool:
                            self.config[config_key][setting] = var.get()
                        elif value_type == int:
                            self.config[config_key][setting] = int(var.get())
                        elif value_type == float:
                            self.config[config_key][setting] = float(var.get())
                        elif value_type == list:
                            # Parse comma-separated values
                            value_str = var.get()
                            self.config[config_key][setting] = [
                                x.strip() for x in value_str.split(',') if x.strip()
                            ]
                        else:
                            self.config[config_key][setting] = var.get()
                    except (ValueError, AttributeError) as e:
                        self.logger.warning(f"Error converting setting {key}: {e}")
                        continue
            
            # Save to file (would need to implement config persistence)
            # For now, just update in memory
            messagebox.showinfo("Success", "Settings saved successfully!")
            self.logger.info("Settings saved")
        except Exception as e:
            self.show_error(f"Error saving settings: {str(e)}")
    
    def reset_settings(self):
        """Reset settings to defaults"""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            self.config = load_config()
            self.update_status("Settings reset to defaults")
    
    def import_config(self):
        """Import configuration from file"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Import Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                import json
                with open(filename, 'r') as f:
                    self.config = json.load(f)
                self.update_status(f"Configuration imported from {filename}")
            except Exception as e:
                self.show_error(f"Error importing config: {str(e)}")
    
    def export_config(self):
        """Export configuration to file"""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="Export Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                import json
                with open(filename, 'w') as f:
                    json.dump(self.config, f, indent=4)
                self.update_status(f"Configuration exported to {filename}")
            except Exception as e:
                self.show_error(f"Error exporting config: {str(e)}")

    def setup_performance_charts(self, parent):
        """Setup performance monitoring charts"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Create figure for charts
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
