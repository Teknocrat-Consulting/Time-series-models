"""
Unit tests for ARIMA forecasting module
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.forecasting.arima_model import (
    ARIMAForecaster,
    AnomalyDetector,
    prepare_data_for_forecasting
)


class TestARIMAForecaster(unittest.TestCase):
    """Test cases for ARIMAForecaster class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.cumsum(np.random.randn(100)) + 100
        self.test_data = pd.Series(values, index=dates, name='test_series')
        self.forecaster = ARIMAForecaster()
    
    def test_check_stationarity(self):
        """Test stationarity check."""
        # Create stationary data
        stationary_data = pd.Series(np.random.randn(100))
        result = self.forecaster.check_stationarity(stationary_data)
        self.assertIsInstance(result, bool)
    
    def test_make_stationary(self):
        """Test making series stationary."""
        stationary, diff_count = self.forecaster.make_stationary(self.test_data)
        self.assertIsInstance(stationary, pd.Series)
        self.assertIsInstance(diff_count, int)
        self.assertGreaterEqual(diff_count, 0)
    
    def test_auto_fit(self):
        """Test automatic model fitting."""
        self.forecaster.auto_fit(self.test_data)
        self.assertIsNotNone(self.forecaster.model)
        self.assertIsNotNone(self.forecaster.order)
    
    def test_manual_fit(self):
        """Test manual model fitting."""
        self.forecaster.manual_fit(self.test_data, order=(1, 1, 1))
        self.assertIsNotNone(self.forecaster.model)
        self.assertEqual(self.forecaster.order, (1, 1, 1))
    
    def test_forecast(self):
        """Test forecast generation."""
        self.forecaster.auto_fit(self.test_data)
        forecast = self.forecaster.forecast(steps=10)
        
        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertEqual(len(forecast), 10)
        self.assertIn('forecast', forecast.columns)
        self.assertIn('lower_bound', forecast.columns)
        self.assertIn('upper_bound', forecast.columns)
    
    def test_get_model_diagnostics(self):
        """Test model diagnostics."""
        self.forecaster.auto_fit(self.test_data)
        diagnostics = self.forecaster.get_model_diagnostics()
        
        self.assertIsInstance(diagnostics, dict)
        self.assertIn('aic', diagnostics)
        self.assertIn('bic', diagnostics)
        self.assertIn('order', diagnostics)
    
    def test_validate_forecast(self):
        """Test forecast validation."""
        # Split data
        train = self.test_data[:-10]
        test = self.test_data[-10:]
        
        # Fit and forecast
        self.forecaster.auto_fit(train)
        forecast = self.forecaster.forecast(steps=10)
        
        # Validate
        metrics = self.forecaster.validate_forecast(test)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.normal_data = pd.Series(np.random.randn(100))
        
        # Add anomalies
        self.anomaly_data = self.normal_data.copy()
        self.anomaly_data.iloc[25] = 10  # Outlier
        self.anomaly_data.iloc[75] = -10  # Outlier
        
        self.detector = AnomalyDetector()
    
    def test_detect_statistical_anomalies(self):
        """Test statistical anomaly detection."""
        anomalies = self.detector.detect_statistical_anomalies(self.anomaly_data)
        
        self.assertIsInstance(anomalies, pd.Series)
        self.assertEqual(anomalies.dtype, bool)
        self.assertTrue(anomalies.iloc[25])  # Should detect outlier
        self.assertTrue(anomalies.iloc[75])  # Should detect outlier
    
    def test_detect_moving_average_anomalies(self):
        """Test moving average anomaly detection."""
        anomalies = self.detector.detect_moving_average_anomalies(
            self.anomaly_data, window=10
        )
        
        self.assertIsInstance(anomalies, pd.Series)
        self.assertEqual(anomalies.dtype, bool)
    
    def test_detect_trend_anomalies(self):
        """Test trend anomaly detection."""
        # Create data with trend change
        data = pd.Series(np.concatenate([
            np.arange(50),  # Steady increase
            np.arange(50, 100) * 3  # Sudden steep increase
        ]))
        
        result = self.detector.detect_trend_anomalies(data, window=10)
        self.assertIsInstance(result, bool)
    
    def test_detect_method_dispatch(self):
        """Test detect method with different methods."""
        # Statistical method
        detector_stat = AnomalyDetector(method='statistical')
        anomalies_stat = detector_stat.detect(self.anomaly_data)
        self.assertIsInstance(anomalies_stat, pd.Series)
        
        # Moving average method
        detector_ma = AnomalyDetector(method='moving_average')
        anomalies_ma = detector_ma.detect(self.anomaly_data, window=10)
        self.assertIsInstance(anomalies_ma, pd.Series)


class TestDataPreparation(unittest.TestCase):
    """Test cases for data preparation functions."""
    
    def test_prepare_data_for_forecasting(self):
        """Test data preparation for forecasting."""
        # Create test DataFrame
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'value': np.random.randn(50) + 100
        })
        
        # Test with date column
        series = prepare_data_for_forecasting(df, 'value', 'date')
        self.assertIsInstance(series, pd.Series)
        self.assertIsInstance(series.index, pd.DatetimeIndex)
        
        # Test without date column (already indexed)
        df.set_index('date', inplace=True)
        series = prepare_data_for_forecasting(df, 'value')
        self.assertIsInstance(series, pd.Series)


if __name__ == '__main__':
    unittest.main()