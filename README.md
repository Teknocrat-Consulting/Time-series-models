# Storage Forecasting Production System

A comprehensive system for monitoring disk storage usage and forecasting future storage needs using ARIMA models and InfluxDB time series database.

## Features

- **Real-time Monitoring**: Monitor disk usage across multiple mount points
- **Time Series Forecasting**: ARIMA-based forecasting with automatic parameter selection
- **Anomaly Detection**: Multiple methods for detecting unusual storage patterns
- **Database Integration**: InfluxDB support for time series data storage
- **Visualization**: Interactive plots and dashboards for monitoring and forecasting
- **Alerting**: Capacity breach predictions and threshold monitoring

## Project Structure

```
storage_forecasting_production/
├── src/
│   ├── core/               # Main application logic
│   │   └── main.py         # Pipeline orchestration
│   ├── database/           # Database operations
│   │   └── influx_client.py # InfluxDB client wrapper
│   ├── monitoring/         # System monitoring
│   │   └── disk_monitor.py # Disk usage monitoring
│   ├── forecasting/        # Forecasting models
│   │   └── arima_model.py  # ARIMA implementation
│   ├── visualization/      # Data visualization
│   │   └── plots.py        # Plotting utilities
│   └── utils/             # Helper utilities
│       └── data_processing.py # Data preprocessing
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── data/                   # Data directory
├── scripts/                # Utility scripts
├── docs/                   # Documentation
└── requirements.txt        # Dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- InfluxDB (optional, for database integration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd storage_forecasting_production
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the forecasting pipeline with default settings:

```bash
python -m src.core.main --data-source csv --csv-path data/disk_usage.csv
```

### With InfluxDB

```bash
python -m src.core.main \
    --data-source database \
    --db-host localhost \
    --db-port 8086 \
    --forecast-steps 30
```

### Real-time Monitoring

```python
from src.monitoring.disk_monitor import DiskMonitor

# Create monitor
monitor = DiskMonitor(mount_point='/', interval=60)

# Start monitoring
monitor.start_monitoring()

# ... let it run ...

# Stop and save data
monitor.stop_monitoring()
monitor.save_to_csv('disk_usage.csv')
```

### Forecasting

```python
from src.forecasting.arima_model import ARIMAForecaster
import pandas as pd

# Load data
data = pd.read_csv('disk_usage.csv', index_col=0, parse_dates=True)
series = data['usage']

# Create and fit model
forecaster = ARIMAForecaster()
forecaster.auto_fit(series)

# Generate forecast
forecast = forecaster.forecast(steps=30, confidence_level=0.95)
print(forecast)
```

### Anomaly Detection

```python
from src.forecasting.arima_model import AnomalyDetector

# Create detector
detector = AnomalyDetector(method='statistical', threshold=3.0)

# Detect anomalies
anomalies = detector.detect(series)
print(f"Found {anomalies.sum()} anomalies")
```

## Configuration

### Command Line Arguments

- `--data-source`: Data source (database/monitor/csv)
- `--forecast-steps`: Number of periods to forecast
- `--seasonal`: Enable seasonal ARIMA
- `--confidence-level`: Confidence level for intervals
- `--capacity-threshold`: Storage capacity threshold in GB
- `--save-plots`: Save visualization plots
- `--output-dir`: Output directory for plots

### Configuration File

Create a `config/config.yaml` file:

```yaml
database:
  host: localhost
  port: 8086
  username: root
  password: root
  database: telegraf

monitoring:
  mount_point: /
  interval: 60

forecasting:
  steps: 30
  seasonal: false
  confidence_level: 0.95

visualization:
  save_plots: true
  output_dir: output
```

## API Documentation

### InfluxDBManager

Manages InfluxDB connections and operations.

```python
db_manager = InfluxDBManager(host='localhost', port=8086)
df = db_manager.get_disk_usage_data(measurement='disk', limit=1000)
```

### DiskMonitor

Real-time disk usage monitoring.

```python
monitor = DiskMonitor(mount_point='/', interval=1)
monitor.start_monitoring()
data = monitor.get_data_as_dataframe()
```

### ARIMAForecaster

Time series forecasting with ARIMA models.

```python
forecaster = ARIMAForecaster()
forecaster.auto_fit(data, seasonal=True)
forecast = forecaster.forecast(steps=30)
```

## Testing

Run the test suite:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Performance Considerations

- **Data Volume**: The system handles up to 1M data points efficiently
- **Forecast Horizon**: Optimal performance for 30-90 day forecasts
- **Memory Usage**: ~100MB for typical workloads
- **Processing Time**: <5 seconds for standard forecasting tasks

## Troubleshooting

### Common Issues

1. **InfluxDB Connection Error**
   - Verify InfluxDB is running: `systemctl status influxdb`
   - Check connection parameters in configuration

2. **Insufficient Data for Forecasting**
   - Minimum 30 data points required for ARIMA
   - Check data quality and completeness

3. **Memory Issues with Large Datasets**
   - Use data sampling or aggregation
   - Increase system memory allocation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- ARIMA implementation based on statsmodels
- Auto-ARIMA using pmdarima library
- InfluxDB for time series storage

## Contact

For questions or support, please open an issue on GitHub.