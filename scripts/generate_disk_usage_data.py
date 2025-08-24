#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_disk_usage_data(num_records=1000000):
    """
    Generate realistic disk usage data with:
    - Overall growth trend
    - Hourly records (to get 1M records in ~114 days)
    - Daily patterns (business hours vs night)
    - Weekly patterns (weekends vs weekdays)
    - Monthly patterns (end of month cleanups)
    - Seasonal patterns
    - Random spikes and drops (backups, cleanups, large file operations)
    """
    
    print(f"Generating {num_records:,} disk usage records...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Start date and initial usage (matching sample data range)
    start_date = datetime(2021, 1, 1)
    initial_usage = 450.0  # Starting at 450 GB (similar to sample)
    
    # Parameters for realistic patterns
    hourly_growth_rate = 0.00003  # Very small hourly growth (about 0.7% daily)
    daily_pattern_amplitude = 5.0  # GB variation for daily pattern
    weekly_pattern_amplitude = 10.0  # GB variation for weekly pattern
    monthly_pattern_amplitude = 15.0  # GB variation for monthly pattern
    seasonal_amplitude = 20.0  # GB variation for seasonal pattern
    noise_stddev = 0.8  # Standard deviation for random noise
    
    # Generate dates (hourly records to get more granular data)
    dates = [start_date + timedelta(hours=i) for i in range(num_records)]
    
    # Initialize usage array
    usage = np.zeros(num_records)
    current_usage = initial_usage
    
    for i, date in enumerate(dates):
        # Base growth trend (very small hourly growth)
        growth_amount = hourly_growth_rate * initial_usage * (1 + np.random.normal(0, 0.1))
        current_usage += growth_amount
        
        # Daily pattern (business hours vs night)
        hour = date.hour
        if 9 <= hour <= 17:  # Business hours
            daily_effect = daily_pattern_amplitude * np.random.uniform(0.3, 0.8)
        elif 22 <= hour or hour <= 5:  # Night hours
            daily_effect = -daily_pattern_amplitude * np.random.uniform(0.2, 0.6)
        else:
            daily_effect = daily_pattern_amplitude * np.random.uniform(-0.1, 0.3)
        
        # Weekly pattern (less usage on weekends)
        day_of_week = date.weekday()
        if day_of_week in [5, 6]:  # Saturday, Sunday
            weekly_effect = -weekly_pattern_amplitude * np.random.uniform(0.3, 0.8)
        else:  # Weekdays
            if day_of_week == 0:  # Monday - higher usage (after weekend)
                weekly_effect = weekly_pattern_amplitude * np.random.uniform(0.5, 1.0)
            elif day_of_week == 4:  # Friday - cleanup day
                weekly_effect = -weekly_pattern_amplitude * np.random.uniform(0.2, 0.5)
            else:
                weekly_effect = weekly_pattern_amplitude * np.random.uniform(-0.2, 0.4)
        
        # Monthly pattern (cleanups at month end)
        day_of_month = date.day
        if day_of_month >= 28:  # End of month cleanup
            monthly_effect = -monthly_pattern_amplitude * np.random.uniform(0.5, 1.5)
        elif day_of_month <= 5:  # Beginning of month increased activity
            monthly_effect = monthly_pattern_amplitude * np.random.uniform(0.3, 0.8)
        elif day_of_month in [15, 16]:  # Mid-month backups
            monthly_effect = monthly_pattern_amplitude * np.random.uniform(0.2, 0.6)
        else:
            monthly_effect = np.random.normal(0, 1.0)
        
        # Seasonal pattern (more usage in Q4, less in summer)
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        # Sinusoidal seasonal pattern
        seasonal_effect = seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365)
        
        # Additional monthly effects
        if month in [11, 12]:  # November-December - year-end higher usage
            seasonal_effect += seasonal_amplitude * np.random.uniform(0.3, 0.7)
        elif month in [7, 8]:  # July-August - vacation period, lower usage
            seasonal_effect -= seasonal_amplitude * np.random.uniform(0.2, 0.5)
        elif month == 1:  # January - new year cleanup
            if day_of_month <= 10:
                seasonal_effect -= seasonal_amplitude * np.random.uniform(0.3, 0.8)
        
        # Random events (different probabilities for different events)
        random_event = 0
        rand_val = np.random.random()
        
        if rand_val < 0.005:  # 0.5% chance of major cleanup
            random_event = -np.random.uniform(20, 50)
        elif rand_val < 0.015:  # 1% chance of large data import/backup
            random_event = np.random.uniform(15, 40)
        elif rand_val < 0.03:  # 1.5% chance of moderate cleanup
            random_event = -np.random.uniform(5, 15)
        elif rand_val < 0.05:  # 2% chance of moderate data addition
            random_event = np.random.uniform(5, 15)
        elif rand_val < 0.1:  # 5% chance of small fluctuation
            random_event = np.random.uniform(-3, 3)
        
        # Random noise (daily variations)
        noise = np.random.normal(0, noise_stddev)
        
        # Combine all effects
        usage[i] = current_usage + daily_effect + weekly_effect + monthly_effect + seasonal_effect + random_event + noise
        
        # Ensure usage doesn't go below a minimum threshold
        usage[i] = max(usage[i], 100.0)
        
        # Update current usage for next iteration (smooth the random events)
        current_usage = 0.85 * current_usage + 0.15 * usage[i]
        
        # Progress indicator
        if (i + 1) % 100000 == 0:
            print(f"  Generated {i + 1:,} records...")
    
    # Create DataFrame with formatted dates (datetime format)
    df = pd.DataFrame({
        'Date': [date.strftime('%Y-%m-%d %H:%M:%S') for date in dates],
        'Usage': np.round(usage, 1)  # Round to 1 decimal place like sample
    })
    
    print(f"\nSuccessfully generated {len(df):,} records")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"Usage range: {df['Usage'].min():.1f} GB to {df['Usage'].max():.1f} GB")
    print(f"Average usage: {df['Usage'].mean():.1f} GB")
    print(f"Median usage: {df['Usage'].median():.1f} GB")
    print(f"Standard deviation: {df['Usage'].std():.1f} GB")
    
    return df

if __name__ == "__main__":
    # Generate the data
    df = generate_disk_usage_data(1000000)
    
    # Save to CSV
    output_path = '/home/vineet/Desktop/projects/Arima/storage_forecasting_production/data/disk_usage_1million.csv'
    print(f"\nSaving data to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully!")
    
    # Show sample of the data
    print("\nFirst 10 records:")
    print(df.head(10))
    print("\nLast 10 records:")
    print(df.tail(10))
    
    # Show some statistics for verification
    print("\nDaily average usage (first 30 days):")
    df_temp = pd.DataFrame({
        'Date': pd.to_datetime(df['Date']),
        'Usage': df['Usage']
    })
    df_temp['Day'] = df_temp['Date'].dt.date
    daily_avg = df_temp.groupby('Day')['Usage'].mean().head(30)
    for day, avg in daily_avg.items():
        print(f"  {day}: {avg:.1f} GB")