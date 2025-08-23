"""
Visualization Module

Provides plotting functionality for disk usage monitoring and forecasting results.
"""

from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from datetime import datetime


class ForecastVisualizer:
    """
    Visualizer for time series forecasts.
    
    Attributes:
        figsize (Tuple): Figure size for plots
        style (str): Matplotlib style to use
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), 
                 style: str = 'seaborn-v0_8'):
        """
        Initialize forecast visualizer.
        
        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style
        """
        self.figsize = figsize
        plt.style.use(style)
    
    def plot_forecast(self, 
                     historical_data: pd.Series,
                     forecast: pd.DataFrame,
                     actual_data: Optional[pd.Series] = None,
                     title: str = 'Time Series Forecast',
                     ylabel: str = 'Value',
                     xlabel: str = 'Date') -> plt.Figure:
        """
        Plot historical data with forecast.
        
        Args:
            historical_data: Historical time series
            forecast: DataFrame with forecast and confidence intervals
            actual_data: Optional actual values for comparison
            title: Plot title
            ylabel: Y-axis label
            xlabel: X-axis label
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data.values, 
               label='Historical', color='blue', linewidth=2)
        
        # Create forecast index
        last_date = historical_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date, 
            periods=len(forecast) + 1, 
            freq=pd.infer_freq(historical_data.index)
        )[1:]
        
        # Plot forecast
        ax.plot(forecast_dates, forecast['forecast'].values,
               label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        if 'lower_bound' in forecast.columns and 'upper_bound' in forecast.columns:
            ax.fill_between(forecast_dates,
                          forecast['lower_bound'].values,
                          forecast['upper_bound'].values,
                          alpha=0.3, color='red',
                          label='Confidence Interval')
        
        # Plot actual data if provided
        if actual_data is not None:
            ax.plot(actual_data.index, actual_data.values,
                   label='Actual', color='green', linewidth=2, marker='o')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, residuals: pd.Series) -> plt.Figure:
        """
        Plot residual diagnostics.
        
        Args:
            residuals: Model residuals
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1]*1.5))
        
        # Residuals over time
        axes[0, 0].plot(residuals.index, residuals.values, color='blue', alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals.values, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ACF of residuals
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals.values, lags=20, ax=axes[1, 1])
        axes[1, 1].set_title('Autocorrelation of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_anomalies(self, data: pd.Series, 
                      anomalies: pd.Series,
                      title: str = 'Anomaly Detection') -> plt.Figure:
        """
        Plot time series with highlighted anomalies.
        
        Args:
            data: Time series data
            anomalies: Boolean series indicating anomalies
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot normal data
        normal_mask = ~anomalies
        ax.plot(data.index[normal_mask], data.values[normal_mask],
               'bo-', label='Normal', markersize=4, alpha=0.7)
        
        # Highlight anomalies
        if anomalies.any():
            ax.plot(data.index[anomalies], data.values[anomalies],
                   'ro', label='Anomaly', markersize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class DiskUsageAnimator:
    """
    Animated visualization for real-time disk usage.
    
    Attributes:
        fig: Matplotlib figure
        ax: Matplotlib axis
        max_points (int): Maximum number of points to display
    """
    
    def __init__(self, max_points: int = 50):
        """
        Initialize disk usage animator.
        
        Args:
            max_points: Maximum number of points to display
        """
        self.max_points = max_points
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.timestamps = []
        self.values = []
    
    def update_plot(self, frame: int, data_func: callable) -> None:
        """
        Update plot with new data.
        
        Args:
            frame: Animation frame number
            data_func: Function that returns (timestamp, value)
        """
        # Get new data
        timestamp, value = data_func()
        
        self.timestamps.append(timestamp)
        self.values.append(value)
        
        # Keep only recent points
        if len(self.timestamps) > self.max_points:
            self.timestamps = self.timestamps[-self.max_points:]
            self.values = self.values[-self.max_points:]
        
        # Clear and redraw
        self.ax.clear()
        self.ax.plot(self.timestamps, self.values, 'b-', linewidth=2)
        self.ax.scatter(self.timestamps, self.values, color='red', s=30)
        
        # Format plot
        self.ax.set_title('Real-time Disk Usage', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Time', fontsize=12)
        self.ax.set_ylabel('Usage (GB)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    def start_animation(self, data_func: callable, 
                       interval: int = 1000) -> animation.FuncAnimation:
        """
        Start the animation.
        
        Args:
            data_func: Function that returns (timestamp, value)
            interval: Update interval in milliseconds
            
        Returns:
            Animation object
        """
        ani = animation.FuncAnimation(
            self.fig, self.update_plot, 
            fargs=(data_func,), 
            interval=interval,
            cache_frame_data=False
        )
        return ani


class ComparisonPlotter:
    """
    Plot comparisons between multiple forecasting models.
    
    Attributes:
        figsize (Tuple): Figure size for plots
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Initialize comparison plotter.
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
    
    def plot_model_comparison(self, 
                             actual: pd.Series,
                             predictions: Dict[str, pd.Series],
                             title: str = 'Model Comparison') -> plt.Figure:
        """
        Compare predictions from multiple models.
        
        Args:
            actual: Actual values
            predictions: Dictionary of model names to predictions
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Plot predictions
        ax1.plot(actual.index, actual.values, 'ko-', 
                label='Actual', linewidth=2, markersize=4)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for (name, pred), color in zip(predictions.items(), colors):
            ax1.plot(pred.index, pred.values, label=name, 
                    color=color, linewidth=1.5, alpha=0.7)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot errors
        for (name, pred), color in zip(predictions.items(), colors):
            errors = actual - pred
            ax2.plot(errors.index, errors.values, label=f'{name} Error',
                    color=color, linewidth=1.5, alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Prediction Errors', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Error', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, 
                              metrics: Dict[str, Dict[str, float]],
                              title: str = 'Model Metrics Comparison') -> plt.Figure:
        """
        Plot comparison of model metrics.
        
        Args:
            metrics: Dictionary of model names to metric dictionaries
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        models = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.ravel()
        
        for idx, metric in enumerate(metric_names[:4]):
            values = [metrics[model][metric] for model in models]
            
            ax = axes[idx]
            bars = ax.bar(models, values, color='steelblue', alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(models) > 3:
                ax.set_xticklabels(models, rotation=45, ha='right')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def create_dashboard(data: pd.DataFrame, 
                    forecast: pd.DataFrame,
                    metrics: Dict[str, float]) -> plt.Figure:
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        data: Historical data
        forecast: Forecast data
        metrics: Model metrics
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main forecast plot
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.plot(data.index, data.values, 'b-', label='Historical', linewidth=2)
    
    # Add forecast
    forecast_index = pd.date_range(
        start=data.index[-1], 
        periods=len(forecast) + 1,
        freq=pd.infer_freq(data.index)
    )[1:]
    
    ax1.plot(forecast_index, forecast['forecast'].values,
            'r--', label='Forecast', linewidth=2)
    
    if 'lower_bound' in forecast.columns:
        ax1.fill_between(forecast_index,
                        forecast['lower_bound'].values,
                        forecast['upper_bound'].values,
                        alpha=0.2, color='red')
    
    ax1.set_title('Storage Forecast Dashboard', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Usage (GB)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Metrics display
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axis('off')
    metrics_text = '\n'.join([f'{k.upper()}: {v:.3f}' 
                              for k, v in metrics.items()])
    ax2.text(0.1, 0.5, metrics_text, fontsize=12, 
            verticalalignment='center')
    ax2.set_title('Model Metrics', fontsize=12, fontweight='bold')
    
    # Growth rate
    ax3 = fig.add_subplot(gs[2, 1])
    growth_rate = data.diff().mean()
    ax3.bar(['Growth Rate'], [growth_rate], color='green', alpha=0.7)
    ax3.set_title('Average Growth Rate (GB/period)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('GB', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Forecast summary
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.axis('off')
    summary_text = f"Next Period: {forecast['forecast'].iloc[0]:.2f} GB\n"
    summary_text += f"Max Expected: {forecast['upper_bound'].max():.2f} GB\n"
    summary_text += f"Min Expected: {forecast['lower_bound'].min():.2f} GB"
    ax4.text(0.1, 0.5, summary_text, fontsize=12, 
            verticalalignment='center')
    ax4.set_title('Forecast Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('Disk Storage Forecasting Dashboard', 
                fontsize=18, fontweight='bold', y=1.02)
    
    return fig


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100
    data = pd.Series(values, index=dates, name='disk_usage')
    
    # Generate sample forecast
    forecast = pd.DataFrame({
        'forecast': np.cumsum(np.random.randn(10)) + values[-1],
        'lower_bound': np.cumsum(np.random.randn(10)) + values[-1] - 5,
        'upper_bound': np.cumsum(np.random.randn(10)) + values[-1] + 5
    })
    
    # Create visualizations
    viz = ForecastVisualizer()
    fig = viz.plot_forecast(data, forecast)
    plt.show()