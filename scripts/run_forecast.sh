#!/bin/bash

# Storage Forecasting Execution Script
# This script runs the storage forecasting pipeline with appropriate settings

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Default values
DATA_SOURCE="csv"
FORECAST_STEPS=30
OUTPUT_DIR="$PROJECT_ROOT/output"
LOG_FILE="$PROJECT_ROOT/logs/forecast_$(date +%Y%m%d_%H%M%S).log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --steps)
            FORECAST_STEPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --source SOURCE    Data source (csv, database, monitor)"
            echo "  --steps STEPS      Number of forecast steps"
            echo "  --output DIR       Output directory for results"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "========================================="
echo "Storage Forecasting Pipeline"
echo "========================================="
echo "Data Source: $DATA_SOURCE"
echo "Forecast Steps: $FORECAST_STEPS"
echo "Output Directory: $OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "========================================="

# Run the forecasting pipeline
cd "$PROJECT_ROOT"
python -m src.core.main \
    --data-source "$DATA_SOURCE" \
    --forecast-steps "$FORECAST_STEPS" \
    --save-plots \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "========================================="
    echo "Forecasting completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo "========================================="
else
    echo "========================================="
    echo "Forecasting failed! Check log file for details."
    echo "Log file: $LOG_FILE"
    echo "========================================="
    exit 1
fi