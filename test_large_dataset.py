"""
Test runner for the storage forecasting system with large dataset (1 million records)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

from src.forecasting.arima_model import ARIMAForecaster, AnomalyDetector
from src.utils.data_processing import handle_missing_values, detect_outliers, calculate_metrics
from src.visualization.plots import ForecastVisualizer

def test_large_dataset_forecasting():
    print("="*60)
    print("STORAGE FORECASTING - LARGE DATASET (1M RECORDS)")
    print("="*60)
    
    try:
        # Load the 1 million record dataset
        print("\n1. Loading large dataset...")
        data_path = 'data/disk_usage_1million.csv'
        
        # Read the data
        print("   Reading CSV file...")
        df = pd.read_csv(data_path)
        print(f"   Total records loaded: {len(df):,}")
        
        # Convert Date to datetime
        print("   Converting dates...")
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Since we have hourly data, let's aggregate to daily for better ARIMA performance
        print("\n2. Aggregating hourly data to daily averages...")
        df['Day'] = df['Date'].dt.date
        daily_df = df.groupby('Day')['Usage'].mean().reset_index()
        daily_df.columns = ['Date', 'Usage']
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df.set_index('Date', inplace=True)
        
        print(f"   Daily records after aggregation: {len(daily_df):,}")
        print(f"   Date range: {daily_df.index.min()} to {daily_df.index.max()}")
        print(f"   Usage range: {daily_df['Usage'].min():.2f} - {daily_df['Usage'].max():.2f} GB")
        
        # Use only recent data for faster processing (last 365 days)
        print("\n3. Selecting recent data for analysis (last 365 days)...")
        recent_data = daily_df.iloc[-365:]
        data = recent_data['Usage']
        
        print(f"   Selected data points: {len(data)}")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        
        # Preprocess data
        print("\n4. Preprocessing data...")
        data = handle_missing_values(data, method='interpolate')
        
        # Detect outliers
        outliers, cleaned_data = detect_outliers(data, method='iqr', threshold=1.5)
        print(f"   Outliers detected: {outliers.sum()}")
        
        # Split data for testing (80/20 split)
        print("\n5. Splitting data for training and testing...")
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        print(f"   Training data: {len(train_data)} days")
        print(f"   Test data: {len(test_data)} days")
        
        # Create and fit ARIMA model
        print("\n6. Creating ARIMA forecasting model...")
        forecaster = ARIMAForecaster()
        
        print("   Fitting ARIMA model (this may take a moment)...")
        forecaster.auto_fit(train_data)
        
        # Make predictions
        print("\n7. Making predictions...")
        forecast_steps = len(test_data)
        forecast_df = forecaster.forecast(steps=forecast_steps)
        predictions = forecast_df['forecast'].values
        confidence_intervals = forecast_df[['lower_bound', 'upper_bound']].values
        
        # Calculate metrics
        print("\n8. Evaluating model performance...")
        # Convert to pandas Series for metrics calculation
        actual_series = pd.Series(test_data.values)
        pred_series = pd.Series(predictions[:len(test_data)])
        metrics = calculate_metrics(actual_series, pred_series)
        
        print("\n   Model Performance Metrics:")
        print(f"   - MAE:  {metrics['mae']:.2f} GB")
        print(f"   - RMSE: {metrics['rmse']:.2f} GB")
        print(f"   - MAPE: {metrics['mape']:.2f}%")
        if 'r2' in metrics:
            print(f"   - R²:   {metrics['r2']:.4f}")
        if 'directional_accuracy' in metrics:
            print(f"   - Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        # Anomaly detection
        print("\n9. Performing anomaly detection...")
        detector = AnomalyDetector(threshold=3.0)
        anomalies = detector.detect(data)
        print(f"   Anomalies detected: {anomalies.sum()} ({anomalies.sum()/len(data)*100:.2f}%)")
        
        # Future forecasting
        print("\n10. Generating future forecast (next 30 days)...")
        future_forecast_df = forecaster.forecast(steps=30)
        future_forecast = future_forecast_df['forecast'].values
        
        print(f"   Next 30 days forecast:")
        print(f"   - Mean predicted usage: {future_forecast.mean():.2f} GB")
        print(f"   - Max predicted usage:  {future_forecast.max():.2f} GB")
        print(f"   - Min predicted usage:  {future_forecast.min():.2f} GB")
        
        # Visualization
        print("\n11. Creating visualizations...")
        visualizer = ForecastVisualizer()
        
        # Create forecast DataFrame for plotting
        forecast_for_plot = pd.DataFrame({
            'forecast': predictions[:len(test_data)],
            'lower_bound': confidence_intervals[:len(test_data), 0] if len(confidence_intervals) > 0 else predictions[:len(test_data)] * 0.9,
            'upper_bound': confidence_intervals[:len(test_data), 1] if len(confidence_intervals) > 0 else predictions[:len(test_data)] * 1.1
        }, index=test_data.index)
        
        # Create forecast plot
        fig = visualizer.plot_forecast(
            historical_data=train_data,
            forecast=forecast_for_plot,
            actual_data=test_data,
            title="Disk Usage Forecast - Large Dataset",
            ylabel="Usage (GB)"
        )
        
        # Save plot
        output_path = 'output/large_dataset_forecast.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Forecast plot saved to: {output_path}")
        
        # Create anomaly plot
        if anomalies.sum() > 0:
            fig_anomaly = visualizer.plot_anomalies(
                data,
                anomalies,
                title="Anomaly Detection - Large Dataset"
            )
            anomaly_output = 'output/large_dataset_anomalies.png'
            plt.savefig(anomaly_output, dpi=150, bbox_inches='tight')
            print(f"   Anomaly plot saved to: {anomaly_output}")
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Dataset size: 1,000,000 hourly records")
        print(f"Aggregated to: {len(daily_df):,} daily records")
        print(f"Analysis period: Last 365 days")
        print(f"Storage growth rate: {(data.iloc[-1] - data.iloc[0])/data.iloc[0]*100:.2f}% over period")
        print(f"Average daily usage: {data.mean():.2f} GB")
        print(f"Peak usage: {data.max():.2f} GB on {data.idxmax().strftime('%Y-%m-%d')}")
        print(f"Minimum usage: {data.min():.2f} GB on {data.idxmin().strftime('%Y-%m-%d')}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Starting large dataset forecasting test...")
    success = test_large_dataset_forecasting()
    sys.exit(0 if success else 1)