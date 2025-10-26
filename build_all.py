"""
Main build script for creating executables for all components
"""
import PyInstaller.__main__
import os
import shutil
from pathlib import Path

def clean_build_dirs():
    """Clean build and dist directories"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

def create_directories():
    """Create necessary directories"""
    dirs_to_create = ['build', 'dist', 'logs', 'data', 'models']
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)

def build_trader():
    """Build the main crypto trader executable"""
    PyInstaller.__main__.run([
        'crypto_trader.py',
        '--name=crypto_trader',
        '--onefile',
        '--add-data=config.py;.',
        '--add-data=.env;.',
        '--hidden-import=torch',
        '--hidden-import=sklearn',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=matplotlib',
        '--specpath=build',
        '--workpath=build/trader',
        '--distpath=dist'
    ])

def build_method_validator():
    """Build the method validator tool"""
    PyInstaller.__main__.run([
        'method_validator.py',
        '--name=method_validator',
        '--onefile',
        '--add-data=README.md;.',
        '--specpath=build',
        '--workpath=build/validator',
        '--distpath=dist'
    ])

def build_gui():
    """Build the GUI application"""
    PyInstaller.__main__.run([
        'crypto_trader_gui.py',
        '--name=crypto_trader_gui',
        '--onefile',
        '--windowed',
        '--add-data=config.py;.',
        '--add-data=.env;.',
        '--hidden-import=torch',
        '--hidden-import=sklearn',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=matplotlib',
        '--add-data=icons;icons',
        '--specpath=build',
        '--workpath=build/gui',
        '--distpath=dist'
    ])

def copy_resources():
    """Copy necessary resource files to dist directory"""
    resource_files = [
        'requirements.txt',
        'README.md',
        'config.py',
        '.env.example'
    ]

    for file in resource_files:
        if os.path.exists(file):
            shutil.copy2(file, 'dist')

def create_launcher():
    """Create a launcher script for easy startup"""
    launcher_content = """@echo off
echo Starting Crypto Trading Suite...
start dist\\crypto_trader_gui.exe
"""
    with open('dist/start_trading.bat', 'w') as f:
        f.write(launcher_content)

def main():
    print("Starting build process...")

    # Clean and create directories
    clean_build_dirs()
    create_directories()

    # Build all components
    build_method_validator()
    build_trader()
    build_gui()

    # Copy resources and create launcher
    copy_resources()
    create_launcher()

    print("Build completed successfully!")
    print("Executables and resources are available in the 'dist' directory")

if __name__ == '__main__':
    main()
