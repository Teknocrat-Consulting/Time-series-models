# Storage Forecasting - Complete Run Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Data Generation](#data-generation)
4. [Running Forecasts](#running-forecasts)
5. [Scripts Overview](#scripts-overview)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

Run these commands to get started immediately:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data (1 million records)
python scripts/generate_disk_usage_data.py

# 3. Run forecasting on large dataset
python test_large_dataset.py

# 4. View results
ls output/*.png
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 100MB free disk space for data

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install manually
pip install pandas numpy matplotlib statsmodels scikit-learn pyyaml
```

### Verify Installation

```bash
# Test imports
python -c "import pandas, numpy, statsmodels, matplotlib; print('All packages installed!')"
```

---

## Data Generation

### Generate Large Dataset (1 Million Records)

```bash
# Run the data generation script
python scripts/generate_disk_usage_data.py
```

**Output:**
- File: `data/disk_usage_1million.csv`
- Size: ~27MB
- Records: 1,000,000 hourly disk usage entries
- Format: Date (YYYY-MM-DD HH:MM:SS), Usage (GB)

### Generate Custom Dataset

Edit `scripts/generate_disk_usage_data.py` to customize:

```python
# Change number of records
df = generate_disk_usage_data(500000)  # 500K records

# Change starting parameters
start_date = datetime(2022, 1, 1)  # Different start date
initial_usage = 1000.0  # Start at 1TB
```

### Use Existing Sample Data

Small sample dataset available:
```bash
# View sample data
head data/sample_disk_usage.csv
```

---

## Running Forecasts

### 1. Run on Large Dataset (Recommended)

```bash
python test_large_dataset.py
```

**What it does:**
- Loads 1M records from `data/disk_usage_1million.csv`
- Aggregates hourly data to daily averages
- Uses last 365 days for analysis
- Trains ARIMA model with 80/20 train/test split
- Generates 30-day forecast
- Saves visualization to `output/large_dataset_forecast.png`

**Expected Output:**
```
============================================================
STORAGE FORECASTING - LARGE DATASET (1M RECORDS)
============================================================
1. Loading large dataset...
2. Aggregating hourly data to daily averages...
3. Selecting recent data for analysis...
4. Preprocessing data...
5. Splitting data for training and testing...
6. Creating ARIMA forecasting model...
7. Making predictions...
8. Evaluating model performance...
   - MAE:  ~1000 GB
   - RMSE: ~1300 GB
   - MAPE: ~1.5%
9. Performing anomaly detection...
10. Generating future forecast...
11. Creating visualizations...
✅ Test completed successfully!
```

### 2. Run on Sample Dataset

```bash
python test_run.py
```

**What it does:**
- Uses small sample dataset (52 records)
- Quick test of all components
- Good for debugging

### 3. Simple Test Run

```bash
python simple_test.py
```

**What it does:**
- Basic functionality test
- Minimal output
- Fastest execution

---

## Scripts Overview

### Main Scripts

| Script | Purpose | Input | Output | Runtime |
|--------|---------|-------|--------|---------|
| `test_large_dataset.py` | Full pipeline test | 1M records CSV | Forecast plot + metrics | ~30 seconds |
| `test_run.py` | Basic test | Sample data | Simple forecast | ~5 seconds |
| `simple_test.py` | Quick validation | Sample data | Console output | ~2 seconds |
| `scripts/generate_disk_usage_data.py` | Data generation | Parameters | CSV file | ~10 seconds |

### Core Modules

```
src/
├── forecasting/
│   ├── arima_model.py      # ARIMA forecasting implementation
│   └── lstm_model.py        # LSTM model (optional)
├── utils/
│   └── data_processing.py  # Data preprocessing utilities
├── visualization/
│   └── plots.py            # Plotting functions
└── monitoring/
    └── disk_monitor.py     # Real-time monitoring (optional)
```

---

## Running Custom Forecasts

### Create Your Own Script

```python
# custom_forecast.py
import pandas as pd
from src.forecasting.arima_model import ARIMAForecaster
from src.visualization.plots import ForecastVisualizer

# Load your data
df = pd.read_csv('your_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Create and fit model
forecaster = ARIMAForecaster()
forecaster.auto_fit(df['Usage'])

# Make predictions
forecast = forecaster.forecast(steps=30)

# Visualize
viz = ForecastVisualizer()
fig = viz.plot_forecast(
    historical_data=df['Usage'],
    forecast=forecast,
    title="My Custom Forecast"
)
```

### Command Line Usage

```bash
# Run with custom data file
python -c "
import pandas as pd
import sys
sys.path.insert(0, 'src')
from src.forecasting.arima_model import ARIMAForecaster

df = pd.read_csv('data/disk_usage_1million.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Quick forecast
forecaster = ARIMAForecaster()
forecaster.auto_fit(df['Usage'][-100:])  # Last 100 points
forecast = forecaster.forecast(steps=7)
print('7-day forecast:', forecast['forecast'].values)
"
```

---

## Output Files

After running the scripts, check these locations:

```bash
# Visualizations
output/
├── large_dataset_forecast.png  # Main forecast plot
├── large_dataset_anomalies.png # Anomaly detection plot
└── forecast.png                # Basic forecast plot

# Generated Data
data/
├── disk_usage_1million.csv     # Generated 1M records
└── sample_disk_usage.csv       # Small sample data

# Logs
logs/
└── forecast_*.log              # Execution logs
```

---

## Configuration

### Modify ARIMA Parameters

Edit in `test_large_dataset.py`:

```python
# Line ~79: Adjust ARIMA parameters
forecaster = ARIMAForecaster()
forecaster.auto_fit(
    train_data,
    max_p=5,  # Maximum AR order (default: 3)
    max_q=5,  # Maximum MA order (default: 3)
    max_d=2   # Maximum differencing (default: 2)
)
```

### Adjust Forecast Horizon

```python
# Line ~109: Change forecast period
future_forecast_df = forecaster.forecast(steps=60)  # 60 days instead of 30
```

### Change Data Aggregation

```python
# Line ~38: Change from daily to weekly
weekly_df = df.groupby(pd.Grouper(key='Date', freq='W'))['Usage'].mean()
```

---

## Performance Tips

### For Large Datasets

1. **Use data aggregation:**
   ```python
   # Aggregate to reduce data points
   daily_data = hourly_data.resample('D').mean()
   ```

2. **Limit analysis window:**
   ```python
   # Use recent data only
   recent_data = data.iloc[-1000:]  # Last 1000 points
   ```

3. **Reduce ARIMA search space:**
   ```python
   forecaster.auto_fit(data, max_p=2, max_q=2)  # Faster search
   ```

### Memory Management

```python
# Process in chunks for very large files
chunk_size = 100000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    chunks.append(chunk.groupby('Date')['Usage'].mean())
df = pd.concat(chunks)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'statsmodels'
# Solution:
pip install --upgrade statsmodels
```

#### 2. Memory Issues
```bash
# Error: MemoryError
# Solution: Aggregate data or use smaller subset
python -c "
import pandas as pd
df = pd.read_csv('data/disk_usage_1million.csv')
# Sample every 10th row
df_sample = df.iloc[::10]
df_sample.to_csv('data/disk_usage_sample.csv', index=False)
"
```

#### 3. ARIMA Convergence Issues
```python
# Error: ConvergenceWarning
# Solution: Try different parameters
forecaster.auto_fit(data, method='css-mle')  # Different optimization
```

#### 4. Plotting Issues
```bash
# Error: No display found
# Solution: Use non-interactive backend
export MPLBACKEND=Agg
python test_large_dataset.py
```

### Debug Mode

Enable verbose output:

```python
# Add to your script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or run with debug flag
python -u test_large_dataset.py 2>&1 | tee debug.log
```

---

## Advanced Usage

### Batch Processing Multiple Files

```bash
# Process all CSV files in data directory
for file in data/*.csv; do
    echo "Processing $file..."
    python test_large_dataset.py --input "$file" --output "output/$(basename $file .csv)_forecast.png"
done
```

### Scheduled Forecasting

Create a cron job for daily forecasts:

```bash
# Add to crontab
0 2 * * * cd /path/to/project && python test_large_dataset.py >> logs/daily_forecast.log 2>&1
```

### Real-time Monitoring Integration

```python
# monitor_and_forecast.py
from src.monitoring.disk_monitor import DiskMonitor
from src.forecasting.arima_model import ARIMAForecaster

monitor = DiskMonitor()
current_usage = monitor.get_current_usage()

# Add to historical data and reforecast
# ... implementation
```

---

## Expected Metrics

When running on the 1M record dataset, expect:

| Metric | Expected Range | Interpretation |
|--------|---------------|----------------|
| MAE | 800-1200 GB | Average absolute error |
| RMSE | 1000-1500 GB | Root mean squared error |
| MAPE | 1-3% | Mean absolute percentage error |
| Directional Accuracy | 65-75% | Trend prediction accuracy |

---

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Try with sample data first

---

## Next Steps

After successfully running the basic forecasts:

1. **Experiment with different models:**
   - Try LSTM model in `src/forecasting/lstm_model.py`
   - Adjust ARIMA parameters for your data

2. **Integrate with real data:**
   - Connect to your actual storage systems
   - Set up automated data collection

3. **Deploy to production:**
   - Set up scheduled runs
   - Create alerting based on forecasts
   - Build dashboard for visualization

---

Last Updated: August 2024