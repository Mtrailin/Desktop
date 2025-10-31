# Configuration file for PyInstaller
import PyInstaller.__main__
import os
import sys # pyright: ignore[reportUnusedImport]
from pathlib import Path

def build_exe(script_name='crypto_trader_gui.py', exe_name='CryptoTraderGUI'):
    """
    Build executable from Python script using PyInstaller
    
    Args:
        script_name: Name of the Python script to build (default: 'crypto_trader_gui.py')
        exe_name: Name for the output executable (default: 'CryptoTraderGUI')
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create necessary directories
    for dir_name in ['build', 'dist', 'logs', 'data', 'models']:
        Path(os.path.join(current_dir, dir_name)).mkdir(parents=True, exist_ok=True)

    # Define the paths for all required files
    main_script = os.path.join(current_dir, script_name)
    readme_path = os.path.join(current_dir, 'README.md')
    env_path = os.path.join(current_dir, '.env')
    config_path = os.path.join(current_dir, 'config.py')

    # Define all required hidden imports
    hidden_imports = [
        'numpy',
        'pandas',
        'torch',
        'sklearn',
        'matplotlib',
        'ccxt',
        'websockets',
        'aiohttp',
        'asyncio',
        'logging',
        'json',
        'os',
        'sys',
        'typing',
        'datetime',
        'tkinter',
        'method_validator',
        'data_validator',
        'trading_types',
        'trading_strategy',
        'market_data_aggregator',
        'endpoint_validator',
        'performance_tracker'
    ]

    # PyInstaller command line arguments
    args = [
        main_script,
        f'--name={exe_name}',
        '--onefile',
        '--windowed',  # For GUI applications
        '--clean',
        '--collect-all',
        *[f'--hidden-import={imp}' for imp in hidden_imports],
        '--add-data={};.'.format(readme_path),
        '--add-data={};.'.format(env_path),
        '--add-data={};.'.format(config_path),
        '--add-binary={};.'.format(os.path.join(current_dir, 'models')),
        '--add-binary={};.'.format(os.path.join(current_dir, 'data')),
        '--icon=app.ico',  # Add an icon if you have one
        '--specpath=build',
        '--workpath=build',
        '--distpath=dist',
        # Add hooks for GUI libraries
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=matplotlib.backends.backend_tkagg'
    ]

    # Run PyInstaller
    PyInstaller.__main__.run(args)

    # Copy configuration files to dist directory
    for file in ['.env', 'config.py', 'README.md']:
        src = os.path.join(current_dir, file)
        dst = os.path.join(current_dir, 'dist', file)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)

    print("Build completed successfully!")
    print(f"Executable created at: {os.path.join(current_dir, 'dist', f'{exe_name}.exe')}")

if __name__ == '__main__':
    build_exe()
