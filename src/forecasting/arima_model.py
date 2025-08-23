"""
ARIMA Forecasting Module

Implements ARIMA-based time series forecasting for disk storage prediction
with automatic parameter selection and anomaly detection.
"""

import warnings
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import itertools

warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA model for time series forecasting.
    
    Attributes:
        model: Fitted ARIMA model
        order: ARIMA order parameters (p, d, q)
        data: Original time series data
        predictions: Forecasted values
    """
    
    def __init__(self):
        """Initialize ARIMA forecaster."""
        self.model = None
        self.order = None
        self.data = None
        self.predictions = None
        self.confidence_intervals = None
    
    def check_stationarity(self, series: pd.Series, 
                         significance_level: float = 0.05) -> bool:
        """
        Check if a time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series data
            significance_level: Significance level for the test
            
        Returns:
            True if series is stationary, False otherwise
        """
        if len(series) < 3:
            return False
        
        result = adfuller(series.dropna())
        p_value = result[1]
        
        is_stationary = p_value <= significance_level
        
        print(f"ADF Test Statistics: {result[0]:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Series is {'stationary' if is_stationary else 'non-stationary'}")
        
        return is_stationary
    
    def make_stationary(self, series: pd.Series, 
                       max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Make a time series stationary through differencing.
        
        Args:
            series: Original time series
            max_diff: Maximum number of differencing operations
            
        Returns:
            Tuple of (stationary series, number of differences applied)
        """
        diff_count = 0
        current_series = series.copy()
        
        while diff_count < max_diff:
            if self.check_stationarity(current_series):
                break
            current_series = current_series.diff().dropna()
            diff_count += 1
        
        return current_series, diff_count
    
    def auto_fit(self, data: pd.Series, 
                 seasonal: bool = False,
                 seasonal_period: int = 12,
                 max_p: int = 3,
                 max_q: int = 3,
                 max_d: int = 2,
                 **kwargs) -> 'ARIMAForecaster':
        """
        Automatically fit ARIMA model with optimal parameters using grid search.
        
        Args:
            data: Time series data
            seasonal: Whether to consider seasonal ARIMA (not implemented)
            seasonal_period: Period for seasonal component (not used)
            max_p: Maximum AR order
            max_q: Maximum MA order  
            max_d: Maximum differencing order
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Self for method chaining
        """
        self.data = data.copy()
        
        print("Finding optimal ARIMA parameters using grid search...")
        
        # Determine optimal d by testing stationarity
        d_optimal = 0
        test_data = data.copy()
        
        for d in range(max_d + 1):
            if self.check_stationarity(test_data, significance_level=0.05):
                d_optimal = d
                break
            test_data = test_data.diff().dropna()
        
        print(f"Optimal d (differencing): {d_optimal}")
        
        # Grid search for p and q
        best_aic = float('inf')
        best_order = None
        
        p_range = range(0, max_p + 1)
        q_range = range(0, max_q + 1)
        
        for p, q in itertools.product(p_range, q_range):
            try:
                order = (p, d_optimal, q)
                model = ARIMA(data, order=order)
                fitted = model.fit()
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = order
                    
            except Exception:
                continue
        
        if best_order is None:
            # Fallback to simple model
            best_order = (1, d_optimal, 1)
            print("Using fallback order (1, d, 1)")
        
        self.order = best_order
        print(f"Optimal order: {self.order}")
        print(f"Best AIC: {best_aic:.2f}")
        
        # Fit the final model
        self.model = ARIMA(data, order=self.order)
        self.model = self.model.fit()
        
        return self
    
    def manual_fit(self, data: pd.Series, 
                  order: Tuple[int, int, int]) -> 'ARIMAForecaster':
        """
        Manually fit ARIMA model with specified parameters.
        
        Args:
            data: Time series data
            order: ARIMA order (p, d, q)
            
        Returns:
            Self for method chaining
        """
        self.data = data.copy()
        self.order = order
        
        print(f"Fitting ARIMA{order}...")
        self.model = ARIMA(data, order=order)
        self.model = self.model.fit()
        
        return self
    
    def forecast(self, steps: int, 
                confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Generate forecasts for future time periods.
        
        Args:
            steps: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Generate forecast
        forecast_result = self.model.forecast(steps=steps, alpha=1-confidence_level)
        
        # Get prediction intervals
        forecast_df = pd.DataFrame({
            'forecast': forecast_result
        })
        
        # Calculate confidence intervals
        forecast_summary = self.model.get_forecast(steps=steps)
        conf_int = forecast_summary.conf_int(alpha=1-confidence_level)
        
        forecast_df['lower_bound'] = conf_int.iloc[:, 0]
        forecast_df['upper_bound'] = conf_int.iloc[:, 1]
        
        self.predictions = forecast_df
        
        return forecast_df
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and performance metrics.
        
        Returns:
            Dictionary with diagnostic information
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        diagnostics = {
            'aic': self.model.aic,
            'bic': self.model.bic,
            'hqic': self.model.hqic,
            'log_likelihood': self.model.llf,
            'order': self.order
        }
        
        # Get residuals analysis
        residuals = self.model.resid
        diagnostics['residuals_mean'] = residuals.mean()
        diagnostics['residuals_std'] = residuals.std()
        diagnostics['residuals_skew'] = residuals.skew()
        diagnostics['residuals_kurtosis'] = residuals.kurtosis()
        
        return diagnostics
    
    def validate_forecast(self, test_data: pd.Series) -> Dict[str, float]:
        """
        Validate forecast against actual data.
        
        Args:
            test_data: Actual values to compare against forecast
            
        Returns:
            Dictionary with validation metrics
        """
        if self.predictions is None:
            raise ValueError("No predictions available")
        
        n = min(len(test_data), len(self.predictions))
        actual = test_data.iloc[:n].values
        predicted = self.predictions['forecast'].iloc[:n].values
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }


class AnomalyDetector:
    """
    Detect anomalies in time series data.
    
    Attributes:
        method (str): Anomaly detection method
        threshold (float): Threshold for anomaly detection
    """
    
    def __init__(self, method: str = 'statistical', threshold: float = 3.0):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method ('statistical', 'isolation_forest', 'moving_average')
            threshold: Threshold for anomaly detection
        """
        self.method = method
        self.threshold = threshold
    
    def detect_statistical_anomalies(self, data: pd.Series) -> pd.Series:
        """
        Detect anomalies using statistical method (z-score).
        
        Args:
            data: Time series data
            
        Returns:
            Boolean series indicating anomalies
        """
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        return z_scores > self.threshold
    
    def detect_moving_average_anomalies(self, data: pd.Series, 
                                       window: int = 10) -> pd.Series:
        """
        Detect anomalies using moving average method.
        
        Args:
            data: Time series data
            window: Window size for moving average
            
        Returns:
            Boolean series indicating anomalies
        """
        rolling_mean = data.rolling(window=window, center=True).mean()
        rolling_std = data.rolling(window=window, center=True).std()
        
        upper_bound = rolling_mean + (self.threshold * rolling_std)
        lower_bound = rolling_mean - (self.threshold * rolling_std)
        
        anomalies = (data > upper_bound) | (data < lower_bound)
        return anomalies
    
    def detect_trend_anomalies(self, data: pd.Series, 
                             window: int = 15) -> bool:
        """
        Detect anomalies in trend using linear regression.
        
        Args:
            data: Time series data
            window: Window size for trend calculation
            
        Returns:
            True if trend anomaly detected
        """
        if len(data) < window:
            return False
        
        recent_data = data.iloc[-window:].values
        x = np.arange(len(recent_data))
        
        # Calculate slope using linear regression
        x_mean = x.mean()
        y_mean = recent_data.mean()
        
        numerator = np.sum((x - x_mean) * (recent_data - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return False
        
        slope = numerator / denominator
        
        # Check if slope indicates anomaly
        historical_std = data.iloc[:-window].std()
        return abs(slope) > self.threshold * historical_std
    
    def detect(self, data: pd.Series, **kwargs) -> pd.Series:
        """
        Detect anomalies using specified method.
        
        Args:
            data: Time series data
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Boolean series indicating anomalies
        """
        if self.method == 'statistical':
            return self.detect_statistical_anomalies(data)
        elif self.method == 'moving_average':
            window = kwargs.get('window', 10)
            return self.detect_moving_average_anomalies(data, window)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def prepare_data_for_forecasting(df: pd.DataFrame, 
                                value_column: str,
                                date_column: Optional[str] = None) -> pd.Series:
    """
    Prepare DataFrame for time series forecasting.
    
    Args:
        df: Input DataFrame
        value_column: Column containing values to forecast
        date_column: Column containing dates (optional)
        
    Returns:
        Prepared time series
    """
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    
    # Ensure data is sorted by index
    df.sort_index(inplace=True)
    
    # Handle missing values
    series = df[value_column].fillna(method='ffill')
    
    return series


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100
    data = pd.Series(values, index=dates)
    
    # Create and fit model
    forecaster = ARIMAForecaster()
    forecaster.auto_fit(data)
    
    # Generate forecast
    forecast = forecaster.forecast(steps=10)
    print("\nForecast:")
    print(forecast)
    
    # Get diagnostics
    diagnostics = forecaster.get_model_diagnostics()
    print("\nModel Diagnostics:")
    for key, value in diagnostics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Detect anomalies
    detector = AnomalyDetector()
    anomalies = detector.detect(data)
    print(f"\nNumber of anomalies detected: {anomalies.sum()}")