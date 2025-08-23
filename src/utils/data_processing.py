"""
Data Processing Utilities

Helper functions for data preprocessing, transformation, and validation.
"""

from typing import Optional, Tuple, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_time_series_data(filepath: str, 
                         date_column: str = 'Date',
                         value_column: str = 'Usage',
                         parse_dates: bool = True) -> pd.DataFrame:
    """
    Load time series data from CSV file.
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        value_column: Name of value column
        parse_dates: Whether to parse dates
        
    Returns:
        DataFrame with time series data
    """
    df = pd.read_csv(filepath)
    
    if parse_dates and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    
    if value_column in df.columns:
        df = df[[value_column]]
    
    return df


def resample_time_series(data: pd.Series, 
                        freq: str = 'D',
                        method: str = 'mean') -> pd.Series:
    """
    Resample time series to specified frequency.
    
    Args:
        data: Time series data
        freq: Target frequency ('D', 'W', 'M', etc.)
        method: Aggregation method ('mean', 'sum', 'last')
        
    Returns:
        Resampled time series
    """
    resampler = data.resample(freq)
    
    if method == 'mean':
        return resampler.mean()
    elif method == 'sum':
        return resampler.sum()
    elif method == 'last':
        return resampler.last()
    else:
        raise ValueError(f"Unknown method: {method}")


def handle_missing_values(data: pd.Series, 
                        method: str = 'interpolate',
                        **kwargs) -> pd.Series:
    """
    Handle missing values in time series.
    
    Args:
        data: Time series with missing values
        method: Method to handle missing values
        **kwargs: Additional arguments for the method
        
    Returns:
        Series with missing values handled
    """
    if method == 'interpolate':
        return data.interpolate(**kwargs)
    elif method == 'forward_fill':
        return data.fillna(method='ffill')
    elif method == 'backward_fill':
        return data.fillna(method='bfill')
    elif method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_outliers(data: pd.Series, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    """
    Detect outliers in time series data.
    
    Args:
        data: Time series data
        method: Detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (outlier_mask, cleaned_data)
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    cleaned_data = data.copy()
    cleaned_data[outliers] = np.nan
    cleaned_data = handle_missing_values(cleaned_data, method='interpolate')
    
    return outliers, cleaned_data


def calculate_rolling_statistics(data: pd.Series, 
                                window: int = 7) -> pd.DataFrame:
    """
    Calculate rolling statistics for time series.
    
    Args:
        data: Time series data
        window: Window size for rolling calculations
        
    Returns:
        DataFrame with rolling statistics
    """
    stats = pd.DataFrame(index=data.index)
    
    stats['value'] = data
    stats['rolling_mean'] = data.rolling(window=window).mean()
    stats['rolling_std'] = data.rolling(window=window).std()
    stats['rolling_min'] = data.rolling(window=window).min()
    stats['rolling_max'] = data.rolling(window=window).max()
    stats['rolling_median'] = data.rolling(window=window).median()
    
    return stats


def normalize_data(data: pd.Series, 
                  method: str = 'minmax') -> Tuple[pd.Series, dict]:
    """
    Normalize time series data.
    
    Args:
        data: Time series to normalize
        method: Normalization method ('minmax', 'zscore')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == 'minmax':
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val, 'method': method}
        
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / std
        params = {'mean': mean, 'std': std, 'method': method}
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized, params


def denormalize_data(data: pd.Series, params: dict) -> pd.Series:
    """
    Denormalize data using stored parameters.
    
    Args:
        data: Normalized data
        params: Normalization parameters
        
    Returns:
        Denormalized data
    """
    method = params['method']
    
    if method == 'minmax':
        return data * (params['max'] - params['min']) + params['min']
    elif method == 'zscore':
        return data * params['std'] + params['mean']
    else:
        raise ValueError(f"Unknown method: {method}")


def create_lag_features(data: pd.Series, 
                       lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for time series.
    
    Args:
        data: Time series data
        lags: List of lag values
        
    Returns:
        DataFrame with lag features
    """
    df = pd.DataFrame(index=data.index)
    df['value'] = data
    
    for lag in lags:
        df[f'lag_{lag}'] = data.shift(lag)
    
    return df


def create_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create time-based features from datetime index.
    
    Args:
        index: DatetimeIndex
        
    Returns:
        DataFrame with time features
    """
    df = pd.DataFrame(index=index)
    
    df['year'] = index.year
    df['month'] = index.month
    df['day'] = index.day
    df['dayofweek'] = index.dayofweek
    df['quarter'] = index.quarter
    df['dayofyear'] = index.dayofyear
    df['weekofyear'] = index.isocalendar().week
    df['is_weekend'] = (index.dayofweek >= 5).astype(int)
    df['is_month_start'] = index.is_month_start.astype(int)
    df['is_month_end'] = index.is_month_end.astype(int)
    
    return df


def split_time_series(data: pd.Series, 
                     train_size: float = 0.8,
                     validation_size: Optional[float] = None) -> Union[Tuple[pd.Series, pd.Series], 
                                                                      Tuple[pd.Series, pd.Series, pd.Series]]:
    """
    Split time series into train/test or train/validation/test sets.
    
    Args:
        data: Time series data
        train_size: Proportion of data for training
        validation_size: Optional proportion for validation
        
    Returns:
        Tuple of split data
    """
    n = len(data)
    train_end = int(n * train_size)
    
    if validation_size is None:
        train = data.iloc[:train_end]
        test = data.iloc[train_end:]
        return train, test
    else:
        val_end = int(n * (train_size + validation_size))
        train = data.iloc[:train_end]
        validation = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]
        return train, validation, test


def calculate_metrics(actual: pd.Series, 
                     predicted: pd.Series) -> dict:
    """
    Calculate forecasting metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Directional accuracy
    actual_direction = (actual.diff() > 0).astype(int)
    predicted_direction = (predicted.diff() > 0).astype(int)
    da = (actual_direction == predicted_direction).mean() * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': da
    }


def convert_bytes_to_gb(bytes_value: Union[int, float, pd.Series]) -> Union[float, pd.Series]:
    """
    Convert bytes to gigabytes.
    
    Args:
        bytes_value: Value in bytes
        
    Returns:
        Value in gigabytes
    """
    return bytes_value / (1024 ** 3)


def estimate_time_to_threshold(data: pd.Series, 
                              threshold: float,
                              growth_rate: Optional[float] = None) -> Optional[datetime]:
    """
    Estimate when storage will reach a threshold.
    
    Args:
        data: Historical usage data
        threshold: Threshold value to reach
        growth_rate: Optional growth rate (calculated if not provided)
        
    Returns:
        Estimated datetime when threshold will be reached
    """
    current_value = data.iloc[-1]
    
    if current_value >= threshold:
        return data.index[-1]
    
    if growth_rate is None:
        # Calculate average growth rate
        growth_rate = data.diff().mean()
    
    if growth_rate <= 0:
        return None  # Never reaches threshold
    
    periods_to_threshold = (threshold - current_value) / growth_rate
    
    # Infer frequency
    freq = pd.infer_freq(data.index)
    if freq is None:
        # Assume daily if can't infer
        freq = 'D'
    
    # Calculate estimated date
    last_date = data.index[-1]
    if freq == 'D':
        estimated_date = last_date + timedelta(days=int(periods_to_threshold))
    elif freq == 'H':
        estimated_date = last_date + timedelta(hours=int(periods_to_threshold))
    else:
        # Default to days
        estimated_date = last_date + timedelta(days=int(periods_to_threshold))
    
    return estimated_date


if __name__ == '__main__':
    # Example usage
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100
    data = pd.Series(values, index=dates)
    
    # Calculate rolling statistics
    stats = calculate_rolling_statistics(data, window=7)
    print("Rolling statistics:")
    print(stats.head())
    
    # Detect outliers
    outliers, cleaned = detect_outliers(data, method='zscore', threshold=2)
    print(f"\nOutliers detected: {outliers.sum()}")
    
    # Split data
    train, test = split_time_series(data, train_size=0.8)
    print(f"\nTrain size: {len(train)}, Test size: {len(test)}")
    
    # Estimate time to threshold
    threshold_date = estimate_time_to_threshold(data, threshold=150)
    if threshold_date:
        print(f"\nEstimated to reach 150 GB on: {threshold_date}")