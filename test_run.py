"""
Simple test runner for the storage forecasting system
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, 'src')

from src.forecasting.arima_model import ARIMAForecaster, AnomalyDetector
from src.utils.data_processing import handle_missing_values, detect_outliers, calculate_metrics
from src.visualization.plots import ForecastVisualizer

def test_forecasting_pipeline():
    print("="*50)
    print("STORAGE FORECASTING TEST RUN")
    print("="*50)
    
    try:
        # Load sample data
        print("Loading data...")
        data_path = 'data/sample_disk_usage.csv'
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        data = df['Usage']
        
        print(f"Data loaded: {len(data)} points")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Usage range: {data.min():.2f} - {data.max():.2f} GB")
        
        # Preprocess data
        print("\nPreprocessing data...")
        data = handle_missing_values(data, method='interpolate')
        
        # Detect outliers
        outliers, cleaned_data = detect_outliers(data, method='iqr', threshold=1.5)
        print(f"Outliers detected: {outliers.sum()}")
        
        # Split data for testing
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        print(f"Training data: {len(train_data)} points")
        print(f"Test data: {len(test_data)} points")
        
        # Train ARIMA model
        print("\nTraining ARIMA model...")
        forecaster = ARIMAForecaster()
        forecaster.auto_fit(train_data)
        
        # Get model diagnostics
        diagnostics = forecaster.get_model_diagnostics()
        print(f"Model order: {diagnostics['order']}")
        print(f"AIC: {diagnostics['aic']:.2f}")
        
        # Generate forecast
        print("\nGenerating forecast...")
        forecast_steps = 10
        forecast = forecaster.forecast(steps=forecast_steps, confidence_level=0.95)
        
        print("Forecast preview:")
        print(forecast.head())
        
        # Validate if we have test data
        if len(test_data) > 0:
            test_subset = test_data.iloc[:min(len(test_data), forecast_steps)]
            metrics = forecaster.validate_forecast(test_subset)
            print(f"\nValidation Metrics:")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"MAPE: {metrics['mape']:.2f}%")
        
        # Detect anomalies
        print("\nDetecting anomalies...")
        detector = AnomalyDetector(method='statistical', threshold=3.0)
        anomalies = detector.detect(data)
        print(f"Anomalies detected: {anomalies.sum()}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Set matplotlib backend for non-interactive plotting
        plt.switch_backend('Agg')
        
        visualizer = ForecastVisualizer()
        
        # Plot forecast
        fig1 = visualizer.plot_forecast(
            train_data, forecast,
            title='Disk Storage Forecast',
            ylabel='Usage (GB)'
        )
        fig1.savefig('output/forecast.png', dpi=150, bbox_inches='tight')
        print("Forecast plot saved to output/forecast.png")
        
        # Plot anomalies if any detected
        if anomalies.any():
            fig2 = visualizer.plot_anomalies(
                data, anomalies,
                title='Anomaly Detection'
            )
            fig2.savefig('output/anomalies.png', dpi=150, bbox_inches='tight')
            print("Anomaly plot saved to output/anomalies.png")
        
        # Calculate capacity breach estimation
        threshold = 520  # GB
        current_value = data.iloc[-1]
        growth_rate = data.diff().mean()
        
        if current_value < threshold and growth_rate > 0:
            days_to_threshold = (threshold - current_value) / growth_rate
            print(f"\nCapacity Analysis:")
            print(f"Current usage: {current_value:.2f} GB")
            print(f"Threshold: {threshold} GB")
            print(f"Growth rate: {growth_rate:.2f} GB/day")
            print(f"Days to reach threshold: {days_to_threshold:.1f}")
        
        print("\n" + "="*50)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_forecasting_pipeline()
    sys.exit(0 if success else 1)