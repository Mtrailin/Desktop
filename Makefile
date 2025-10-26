# Makefile for building and managing the crypto trading suite

.PHONY: all clean install build test lint format

# Default target
all: clean install build test

# Clean build artifacts
clean:
	python -c "import shutil; import os; [shutil.rmtree(p) for p in ['build', 'dist', '__pycache__'] if os.path.exists(p)]"

# Install dependencies
install:
	pip install -e ".[dev]"

# Build executables
build:
	python build_all.py

# Run tests
test:
	pytest tests/ --cov=. --cov-report=html

# Run linting
lint:
	black .
	isort .
	mypy .

# Format code
format:
	black .
	isort .

# Create development environment
dev-setup:
	python -m venv venv
	venv\Scripts\activate
	pip install -e ".[dev]"

# Run the GUI application
run-gui:
	python crypto_trader_gui.py

# Run the trading bot
run-bot:
	python crypto_trader.py

# Generate documentation
docs:
	sphinx-build -b html docs/source docs/build/html
