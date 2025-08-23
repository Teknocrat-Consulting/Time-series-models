"""
Main Application Module

Orchestrates the storage forecasting pipeline including data collection,
processing, forecasting, and visualization.
"""

import argparse
import logging
from typing import Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.database.influx_client import InfluxDBManager
from src.monitoring.disk_monitor import DiskMonitor
from src.forecasting.arima_model import ARIMAForecaster, AnomalyDetector
from src.visualization.plots import ForecastVisualizer, create_dashboard
from src.utils.data_processing import (
    handle_missing_values,
    detect_outliers,
    calculate_metrics,
    estimate_time_to_threshold
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StorageForecastingPipeline:
    """
    Main pipeline for storage forecasting.
    
    Attributes:
        config (Dict): Configuration parameters
        db_manager (InfluxDBManager): Database manager
        monitor (DiskMonitor): Disk monitor
        forecaster (ARIMAForecaster): ARIMA forecaster
        visualizer (ForecastVisualizer): Visualization tool
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.db_manager = None
        self.monitor = None
        self.forecaster = ARIMAForecaster()
        self.visualizer = ForecastVisualizer()
        self.anomaly_detector = AnomalyDetector()
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        # Initialize database connection
        if self.config.get('use_database', False):
            self.db_manager = InfluxDBManager(
                host=self.config.get('db_host', 'localhost'),
                port=self.config.get('db_port', 8086),
                username=self.config.get('db_username', 'root'),
                password=self.config.get('db_password', 'root'),
                database=self.config.get('db_name', 'telegraf')
            )
            logger.info("Database connection initialized")
        
        # Initialize disk monitor
        if self.config.get('enable_monitoring', False):
            self.monitor = DiskMonitor(
                mount_point=self.config.get('mount_point', '/'),
                interval=self.config.get('monitor_interval', 60)
            )
            logger.info("Disk monitor initialized")
    
    def collect_data(self, source: str = 'database') -> pd.Series:
        """
        Collect data from specified source.
        
        Args:
            source: Data source ('database', 'monitor', 'csv')
            
        Returns:
            Time series data
        """
        logger.info(f"Collecting data from {source}")
        
        if source == 'database' and self.db_manager:
            df = self.db_manager.get_disk_usage_data(
                measurement=self.config.get('measurement', 'disk'),
                partition=self.config.get('partition', 'sda2'),
                limit=self.config.get('data_limit', 1000)
            )
            if 'used' in df.columns:
                return df['used'] / (1024**3)  # Convert to GB
            
        elif source == 'monitor' and self.monitor:
            self.monitor.start_monitoring()
            # Wait for data collection
            import time
            time.sleep(self.config.get('collection_duration', 60))
            self.monitor.stop_monitoring()
            
            df = self.monitor.get_data_as_dataframe()
            if 'used_gb' in df.columns:
                return df['used_gb']
            
        elif source == 'csv':
            filepath = self.config.get('csv_path', 'data/disk_usage.csv')
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df.iloc[:, 0]
        
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def preprocess_data(self, data: pd.Series) -> pd.Series:
        """
        Preprocess the data for forecasting.
        
        Args:
            data: Raw time series data
            
        Returns:
            Preprocessed data
        """
        logger.info("Preprocessing data")
        
        # Handle missing values
        data = handle_missing_values(data, method='interpolate')
        
        # Detect and handle outliers if configured
        if self.config.get('remove_outliers', False):
            outliers, data = detect_outliers(
                data, 
                method=self.config.get('outlier_method', 'iqr'),
                threshold=self.config.get('outlier_threshold', 1.5)
            )
            logger.info(f"Removed {outliers.sum()} outliers")
        
        return data
    
    def train_model(self, data: pd.Series) -> None:
        """
        Train the forecasting model.
        
        Args:
            data: Training data
        """
        logger.info("Training ARIMA model")
        
        self.forecaster.auto_fit(
            data,
            seasonal=self.config.get('seasonal', False),
            seasonal_period=self.config.get('seasonal_period', 12)
        )
        
        # Get model diagnostics
        diagnostics = self.forecaster.get_model_diagnostics()
        logger.info(f"Model AIC: {diagnostics['aic']:.2f}")
        logger.info(f"Model order: {diagnostics['order']}")
    
    def generate_forecast(self, steps: int) -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Args:
            steps: Number of periods to forecast
            
        Returns:
            Forecast DataFrame
        """
        logger.info(f"Generating forecast for {steps} periods")
        
        forecast = self.forecaster.forecast(
            steps=steps,
            confidence_level=self.config.get('confidence_level', 0.95)
        )
        
        return forecast
    
    def detect_anomalies(self, data: pd.Series) -> pd.Series:
        """
        Detect anomalies in the data.
        
        Args:
            data: Time series data
            
        Returns:
            Boolean series indicating anomalies
        """
        logger.info("Detecting anomalies")
        
        anomalies = self.anomaly_detector.detect(
            data,
            window=self.config.get('anomaly_window', 10)
        )
        
        if anomalies.any():
            logger.warning(f"Detected {anomalies.sum()} anomalies")
        
        return anomalies
    
    def visualize_results(self, 
                         data: pd.Series,
                         forecast: pd.DataFrame,
                         anomalies: Optional[pd.Series] = None) -> None:
        """
        Visualize the results.
        
        Args:
            data: Historical data
            forecast: Forecast data
            anomalies: Optional anomaly indicators
        """
        logger.info("Creating visualizations")
        
        # Create forecast plot
        fig1 = self.visualizer.plot_forecast(
            data, forecast,
            title='Disk Storage Forecast',
            ylabel='Usage (GB)'
        )
        
        # Create anomaly plot if anomalies detected
        if anomalies is not None and anomalies.any():
            fig2 = self.visualizer.plot_anomalies(
                data, anomalies,
                title='Anomaly Detection'
            )
        
        # Create dashboard
        metrics = {'placeholder': 0}  # Would be actual metrics
        fig3 = create_dashboard(data, forecast, metrics)
        
        # Save figures if configured
        if self.config.get('save_plots', False):
            output_dir = self.config.get('output_dir', 'output')
            fig1.savefig(f'{output_dir}/forecast.png')
            logger.info(f"Plots saved to {output_dir}")
        
        # Show plots if configured
        if self.config.get('show_plots', True):
            plt.show()
    
    def estimate_capacity_breach(self, 
                                data: pd.Series,
                                forecast: pd.DataFrame,
                                threshold: float) -> Optional[datetime]:
        """
        Estimate when storage will breach capacity threshold.
        
        Args:
            data: Historical data
            forecast: Forecast data
            threshold: Capacity threshold in GB
            
        Returns:
            Estimated breach date or None
        """
        # Combine historical and forecast
        combined = pd.concat([data, forecast['forecast']])
        
        breach_date = estimate_time_to_threshold(combined, threshold)
        
        if breach_date:
            logger.warning(f"Storage expected to reach {threshold}GB on {breach_date}")
        else:
            logger.info(f"Storage not expected to reach {threshold}GB in forecast period")
        
        return breach_date
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
            Dictionary with results
        """
        logger.info("Starting storage forecasting pipeline")
        
        try:
            # Collect data
            data = self.collect_data(
                source=self.config.get('data_source', 'csv')
            )
            logger.info(f"Collected {len(data)} data points")
            
            # Preprocess data
            data = self.preprocess_data(data)
            
            # Train model
            self.train_model(data)
            
            # Generate forecast
            forecast_steps = self.config.get('forecast_steps', 30)
            forecast = self.generate_forecast(forecast_steps)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(data)
            
            # Visualize results
            self.visualize_results(data, forecast, anomalies)
            
            # Check capacity breach
            threshold = self.config.get('capacity_threshold', 1000)
            breach_date = self.estimate_capacity_breach(data, forecast, threshold)
            
            # Prepare results
            results = {
                'data_points': len(data),
                'forecast_periods': forecast_steps,
                'anomalies_detected': anomalies.sum() if anomalies is not None else 0,
                'breach_date': breach_date,
                'model_order': self.forecaster.order,
                'forecast': forecast
            }
            
            logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        
        finally:
            # Cleanup
            if self.db_manager:
                self.db_manager.close()
            if self.monitor and self.monitor.is_running:
                self.monitor.stop_monitoring()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Storage Forecasting Pipeline'
    )
    
    # Data source arguments
    parser.add_argument(
        '--data-source', 
        choices=['database', 'monitor', 'csv'],
        default='csv',
        help='Data source for historical data'
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        default='data/disk_usage.csv',
        help='Path to CSV file (if using CSV source)'
    )
    
    # Database arguments
    parser.add_argument(
        '--db-host',
        type=str,
        default='localhost',
        help='InfluxDB host'
    )
    
    parser.add_argument(
        '--db-port',
        type=int,
        default=8086,
        help='InfluxDB port'
    )
    
    # Forecasting arguments
    parser.add_argument(
        '--forecast-steps',
        type=int,
        default=30,
        help='Number of periods to forecast'
    )
    
    parser.add_argument(
        '--seasonal',
        action='store_true',
        help='Enable seasonal ARIMA'
    )
    
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for prediction intervals'
    )
    
    # Output arguments
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--no-show-plots',
        action='store_true',
        help='Do not display plots'
    )
    
    # Threshold arguments
    parser.add_argument(
        '--capacity-threshold',
        type=float,
        default=1000,
        help='Storage capacity threshold in GB'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Build configuration from arguments
    config = {
        'data_source': args.data_source,
        'csv_path': args.csv_path,
        'use_database': args.data_source == 'database',
        'db_host': args.db_host,
        'db_port': args.db_port,
        'forecast_steps': args.forecast_steps,
        'seasonal': args.seasonal,
        'confidence_level': args.confidence_level,
        'save_plots': args.save_plots,
        'output_dir': args.output_dir,
        'show_plots': not args.no_show_plots,
        'capacity_threshold': args.capacity_threshold,
        'remove_outliers': True,
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'anomaly_window': 10
    }
    
    # Run pipeline
    pipeline = StorageForecastingPipeline(config)
    results = pipeline.run()
    
    # Print summary
    print("\n" + "="*50)
    print("STORAGE FORECASTING RESULTS")
    print("="*50)
    print(f"Data points analyzed: {results['data_points']}")
    print(f"Forecast periods: {results['forecast_periods']}")
    print(f"Anomalies detected: {results['anomalies_detected']}")
    print(f"Model order: {results['model_order']}")
    
    if results['breach_date']:
        print(f"⚠️  Capacity breach expected: {results['breach_date']}")
    else:
        print("✅ No capacity breach expected in forecast period")
    
    print("="*50)


if __name__ == '__main__':
    main()