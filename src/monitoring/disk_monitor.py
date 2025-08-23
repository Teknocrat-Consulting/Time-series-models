"""
Disk Monitoring Module

Real-time disk usage monitoring using psutil library with
data collection and visualization capabilities.
"""

import datetime as dt
from typing import List, Tuple, Optional
import pandas as pd
import psutil
import threading
import time


class DiskMonitor:
    """
    Monitor disk usage in real-time.
    
    Attributes:
        mount_point (str): Mount point to monitor
        interval (int): Monitoring interval in seconds
        data_points (List): List of collected data points
        is_running (bool): Flag indicating if monitoring is active
    """
    
    def __init__(self, mount_point: str = '/', interval: int = 1):
        """
        Initialize disk monitor.
        
        Args:
            mount_point: Mount point to monitor (default: root)
            interval: Monitoring interval in seconds
        """
        self.mount_point = mount_point
        self.interval = interval
        self.data_points = []
        self.is_running = False
        self._thread = None
    
    def get_current_usage(self) -> dict:
        """
        Get current disk usage statistics.
        
        Returns:
            Dictionary with disk usage information
        """
        usage = psutil.disk_usage(self.mount_point)
        
        return {
            'timestamp': dt.datetime.now(),
            'total_gb': usage.total / (1024**3),
            'used_gb': usage.used / (1024**3),
            'free_gb': usage.free / (1024**3),
            'percent': usage.percent,
            'mount_point': self.mount_point
        }
    
    def _monitor_loop(self) -> None:
        """Internal monitoring loop."""
        while self.is_running:
            data_point = self.get_current_usage()
            self.data_points.append(data_point)
            time.sleep(self.interval)
    
    def start_monitoring(self) -> None:
        """Start the monitoring process in a separate thread."""
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.daemon = True
            self._thread.start()
            print(f"Started monitoring disk usage at {self.mount_point}")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring process."""
        if self.is_running:
            self.is_running = False
            if self._thread:
                self._thread.join(timeout=self.interval + 1)
            print("Stopped monitoring")
    
    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Get collected data as a pandas DataFrame.
        
        Returns:
            DataFrame with disk usage data
        """
        if not self.data_points:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data_points)
        df.set_index('timestamp', inplace=True)
        return df
    
    def save_to_csv(self, filename: str = 'disk_usage.csv') -> None:
        """
        Save collected data to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        df = self.get_data_as_dataframe()
        if not df.empty:
            df.to_csv(filename)
            print(f"Data saved to {filename}")
        else:
            print("No data to save")
    
    def get_latest_readings(self, n: int = 10) -> List[dict]:
        """
        Get the latest n readings.
        
        Args:
            n: Number of latest readings to return
            
        Returns:
            List of latest data points
        """
        return self.data_points[-n:] if self.data_points else []
    
    def calculate_growth_rate(self, window: int = 10) -> Optional[float]:
        """
        Calculate disk usage growth rate.
        
        Args:
            window: Number of data points to use for calculation
            
        Returns:
            Growth rate in GB per interval, or None if insufficient data
        """
        if len(self.data_points) < window:
            return None
        
        recent_data = self.data_points[-window:]
        used_values = [d['used_gb'] for d in recent_data]
        
        # Simple linear regression
        n = len(used_values)
        if n < 2:
            return None
        
        x_mean = (n - 1) / 2
        y_mean = sum(used_values) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) 
                       for i, y in enumerate(used_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator


class MultiDiskMonitor:
    """
    Monitor multiple disk partitions simultaneously.
    
    Attributes:
        monitors (Dict): Dictionary of mount points to DiskMonitor instances
    """
    
    def __init__(self):
        """Initialize multi-disk monitor."""
        self.monitors = {}
    
    def add_mount_point(self, mount_point: str, interval: int = 1) -> None:
        """
        Add a mount point to monitor.
        
        Args:
            mount_point: Mount point path
            interval: Monitoring interval in seconds
        """
        if mount_point not in self.monitors:
            self.monitors[mount_point] = DiskMonitor(mount_point, interval)
            print(f"Added monitor for {mount_point}")
    
    def remove_mount_point(self, mount_point: str) -> None:
        """
        Remove a mount point from monitoring.
        
        Args:
            mount_point: Mount point path
        """
        if mount_point in self.monitors:
            self.monitors[mount_point].stop_monitoring()
            del self.monitors[mount_point]
            print(f"Removed monitor for {mount_point}")
    
    def start_all(self) -> None:
        """Start monitoring all configured mount points."""
        for mount_point, monitor in self.monitors.items():
            monitor.start_monitoring()
    
    def stop_all(self) -> None:
        """Stop monitoring all mount points."""
        for monitor in self.monitors.values():
            monitor.stop_monitoring()
    
    def get_all_current_usage(self) -> dict:
        """
        Get current usage for all monitored mount points.
        
        Returns:
            Dictionary with usage data for each mount point
        """
        return {
            mount_point: monitor.get_current_usage()
            for mount_point, monitor in self.monitors.items()
        }
    
    def save_all_to_csv(self, prefix: str = 'disk_') -> None:
        """
        Save data from all monitors to CSV files.
        
        Args:
            prefix: Prefix for output filenames
        """
        for mount_point, monitor in self.monitors.items():
            filename = f"{prefix}{mount_point.replace('/', '_')}.csv"
            monitor.save_to_csv(filename)


def get_all_disk_partitions() -> List[Tuple[str, str]]:
    """
    Get all available disk partitions.
    
    Returns:
        List of tuples (device, mount_point)
    """
    partitions = []
    for partition in psutil.disk_partitions():
        if partition.fstype and 'loop' not in partition.device:
            partitions.append((partition.device, partition.mountpoint))
    return partitions


if __name__ == '__main__':
    # Example usage
    monitor = DiskMonitor('/', interval=2)
    
    try:
        monitor.start_monitoring()
        print("Monitoring disk usage... Press Ctrl+C to stop")
        
        # Run for 10 seconds
        time.sleep(10)
        
        # Get latest readings
        latest = monitor.get_latest_readings(5)
        for reading in latest:
            print(f"{reading['timestamp']}: {reading['used_gb']:.2f} GB used")
        
        # Calculate growth rate
        growth = monitor.calculate_growth_rate()
        if growth:
            print(f"Growth rate: {growth:.4f} GB per interval")
        
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.stop_monitoring()
        monitor.save_to_csv('disk_usage_data.csv')