"""
Simple ARIMA forecasting test without pmdarima
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import os

warnings.filterwarnings('ignore')

def run_simple_forecast():
    print("="*50)
    print("SIMPLE STORAGE FORECASTING TEST")
    print("="*50)
    
    try:
        # Load sample data
        print("Loading data...")
        df = pd.read_csv('data/sample_disk_usage.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        data = df['Usage']
        
        print(f"Data loaded: {len(data)} points")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Usage range: {data.min():.2f} - {data.max():.2f} GB")
        
        # Check stationarity
        print("\nChecking stationarity...")
        adf_result = adfuller(data)
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        
        if adf_result[1] > 0.05:
            print("Data is non-stationary, differencing...")
            data_diff = data.diff().dropna()
            adf_result_diff = adfuller(data_diff)
            print(f"After differencing - p-value: {adf_result_diff[1]:.4f}")
            d_order = 1
        else:
            print("Data is stationary")
            d_order = 0
        
        # Split data
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        print(f"\nTraining data: {len(train_data)} points")
        print(f"Test data: {len(test_data)} points")
        
        # Fit ARIMA model with simple order
        print("\nFitting ARIMA(1,1,1) model...")
        model = ARIMA(train_data, order=(1, d_order, 1))
        fitted_model = model.fit()
        
        print(f"Model AIC: {fitted_model.aic:.2f}")
        print(f"Model BIC: {fitted_model.bic:.2f}")
        
        # Generate forecast
        print("\nGenerating forecast...")
        forecast_steps = 10
        forecast_result = fitted_model.forecast(steps=forecast_steps)
        conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        # Create forecast DataFrame
        forecast_index = pd.date_range(
            start=train_data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_steps,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'forecast': forecast_result,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1]
        }, index=forecast_index)
        
        print("Forecast preview:")
        print(forecast_df.head())
        
        # Calculate validation metrics if test data available
        if len(test_data) > 0:
            n_compare = min(len(test_data), forecast_steps)
            actual_subset = test_data.iloc[:n_compare]
            forecast_subset = forecast_result[:n_compare]
            
            mae = np.mean(np.abs(actual_subset - forecast_subset))
            mse = np.mean((actual_subset - forecast_subset) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_subset - forecast_subset) / actual_subset)) * 100
            
            print(f"\nValidation Metrics:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAPE: {mape:.2f}%")
        
        # Simple anomaly detection using z-score
        print("\nDetecting anomalies...")
        z_scores = np.abs((data - data.mean()) / data.std())
        anomalies = z_scores > 3
        print(f"Anomalies detected: {anomalies.sum()}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        os.makedirs('output', exist_ok=True)
        
        # Set non-interactive backend
        plt.switch_backend('Agg')
        
        # Create forecast plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(train_data.index, train_data.values, 
               label='Historical', color='blue', linewidth=2)
        
        # Plot test data if available
        if len(test_data) > 0:
            ax.plot(test_data.index, test_data.values,
                   label='Actual', color='green', linewidth=2, marker='o')
        
        # Plot forecast
        ax.plot(forecast_df.index, forecast_df['forecast'].values,
               label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        ax.fill_between(forecast_df.index,
                       forecast_df['lower_bound'].values,
                       forecast_df['upper_bound'].values,
                       alpha=0.3, color='red',
                       label='Confidence Interval')
        
        ax.set_title('Disk Storage Forecast', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Usage (GB)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('output/simple_forecast.png', dpi=150, bbox_inches='tight')
        print("Forecast plot saved to output/simple_forecast.png")
        
        # Calculate capacity analysis
        print("\nCapacity Analysis:")
        current_value = data.iloc[-1]
        growth_rate = data.diff().mean()
        threshold = 520  # GB
        
        print(f"Current usage: {current_value:.2f} GB")
        print(f"Average growth rate: {growth_rate:.2f} GB/day")
        
        if growth_rate > 0:
            days_to_threshold = (threshold - current_value) / growth_rate
            print(f"Days to reach {threshold}GB: {days_to_threshold:.1f}")
            
            if days_to_threshold < 30:
                print("⚠️  WARNING: Capacity threshold may be reached soon!")
            else:
                print("✅ Capacity threshold is not expected to be reached soon.")
        
        print("\n" + "="*50)
        print("SIMPLE TEST COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import sys
    success = run_simple_forecast()
    sys.exit(0 if success else 1)