@echo off
echo Setting up build environment...

:: Create Python virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install required packages
echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

:: Create necessary directories
if not exist build mkdir build
if not exist dist mkdir dist
if not exist logs mkdir logs
if not exist data mkdir data
if not exist models mkdir models

:: Build the executables
echo Building Method Validator...
pyinstaller --name=method_validator --onefile --clean ^
    --specpath=build --workpath=build\validator --distpath=dist ^
    method_validator.py

echo Building Crypto Trader...
pyinstaller --name=crypto_trader --onefile --clean ^
    --hidden-import=pandas --hidden-import=numpy --hidden-import=torch ^
    --hidden-import=sklearn --hidden-import=matplotlib ^
    --add-data="config.py;." --add-data=".env;." ^
    --specpath=build --workpath=build\trader --distpath=dist ^
    crypto_trader.py

echo Building GUI...
pyinstaller --name=crypto_trader_gui --onefile --windowed --clean ^
    --hidden-import=pandas --hidden-import=numpy --hidden-import=torch ^
    --hidden-import=sklearn --hidden-import=matplotlib ^
    --add-data="config.py;." --add-data=".env;." ^
    --specpath=build --workpath=build\gui --distpath=dist ^
    crypto_trader_gui.py

:: Copy necessary files to dist
echo Copying support files...
copy requirements.txt dist\
copy README.md dist\
copy config.py dist\
copy .env.example dist\.env

:: Create launcher
echo Creating launcher...
echo @echo off > dist\start_trading.bat
echo start crypto_trader_gui.exe >> dist\start_trading.bat

echo Build completed! Check the dist directory for executables.
pause
