#!/bin/bash

# Storage Forecasting - Complete Run Script
# This script runs the entire forecasting pipeline

echo "=================================================="
echo "   STORAGE FORECASTING - AUTOMATED RUN SCRIPT    "
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

# Check if Python is installed
if ! command -v python &> /dev/null; then
    print_error "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

print_status "Python is installed"

# Step 1: Install dependencies
print_info "Installing required packages..."
if pip install -r requirements.txt > /dev/null 2>&1; then
    print_status "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    echo "Try running: pip install -r requirements.txt"
    exit 1
fi

# Step 2: Check if data exists or generate it
if [ -f "data/disk_usage_1million.csv" ]; then
    print_info "Data file already exists, skipping generation"
else
    print_info "Generating 1 million record dataset..."
    if python scripts/generate_disk_usage_data.py; then
        print_status "Data generated successfully"
    else
        print_error "Failed to generate data"
        exit 1
    fi
fi

# Step 3: Run the forecasting
print_info "Running ARIMA forecasting model..."
echo ""
echo "This may take 30-60 seconds..."
echo ""

if python test_large_dataset.py; then
    echo ""
    print_status "Forecasting completed successfully!"
else
    print_error "Forecasting failed"
    exit 1
fi

# Step 4: Check outputs
echo ""
echo "=================================================="
echo "                   RESULTS                       "
echo "=================================================="

if [ -f "output/large_dataset_forecast.png" ]; then
    print_status "Forecast plot saved to: output/large_dataset_forecast.png"
    
    # Try to open the image (works on most systems)
    if command -v xdg-open &> /dev/null; then
        print_info "Opening forecast plot..."
        xdg-open output/large_dataset_forecast.png 2>/dev/null &
    elif command -v open &> /dev/null; then
        print_info "Opening forecast plot..."
        open output/large_dataset_forecast.png 2>/dev/null &
    else
        print_info "View the forecast at: output/large_dataset_forecast.png"
    fi
else
    print_error "Forecast plot not found"
fi

echo ""
echo "=================================================="
echo "              RUN COMPLETE!                      "
echo "=================================================="
echo ""
print_info "Next steps:"
echo "  1. View forecast: output/large_dataset_forecast.png"
echo "  2. Check metrics in the output above"
echo "  3. Run with your own data by modifying test_large_dataset.py"
echo "  4. See RUN_GUIDE.md for detailed documentation"
echo ""