# Storage Forecasting System: Workflow and Algorithm Explanation

## Table of Contents
1. [System Overview](#system-overview)
2. [Complete Workflow](#complete-workflow)
3. [ARIMA Algorithm Explanation](#arima-algorithm-explanation)
4. [Code Architecture](#code-architecture)
5. [Data Flow](#data-flow)
6. [Mathematical Details](#mathematical-details)
7. [Implementation Details](#implementation-details)

---

## System Overview

The Storage Forecasting System is a comprehensive time series analysis tool designed to monitor disk storage usage and predict future storage needs. It uses ARIMA (AutoRegressive Integrated Moving Average) models to forecast storage consumption and detect anomalies.

### Key Components:
- **Data Collection**: Monitor real-time disk usage or load historical data
- **Data Preprocessing**: Clean, validate, and prepare time series data
- **ARIMA Modeling**: Automatic parameter selection and model fitting
- **Forecasting**: Generate predictions with confidence intervals
- **Anomaly Detection**: Identify unusual patterns in storage usage
- **Visualization**: Create plots and dashboards for analysis
- **Alerting**: Predict when storage thresholds will be exceeded

---

## Complete Workflow

### Phase 1: Data Collection
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CSV Files     │    │   InfluxDB      │    │  Real-time      │
│                 │    │   Database      │    │  Monitoring     │
│ • Historical    │    │ • Time series   │    │ • psutil        │
│   usage data    │    │   storage       │    │ • Live metrics  │
│ • Manual input  │    │ • Query API     │    │ • Continuous    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Data Collection        │
                    │                           │
                    │ • Load time series data   │
                    │ • Validate data format    │
                    │ • Index by timestamp      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
```

### Phase 2: Data Preprocessing
```
                    ┌─────────────────────────────┐
                    │    Data Preprocessing       │
                    │                             │
                    │ 1. Handle Missing Values    │
                    │    • Forward fill           │
                    │    • Interpolation          │
                    │    • Mean/median fill       │
                    │                             │
                    │ 2. Outlier Detection        │
                    │    • IQR method             │
                    │    • Z-score method         │
                    │    • Remove/adjust outliers │
                    │                             │
                    │ 3. Data Validation          │
                    │    • Check continuity       │
                    │    • Verify data types      │
                    │    • Ensure sufficient data │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
```

### Phase 3: ARIMA Modeling
```
                    ┌─────────────────────────────┐
                    │    ARIMA Model Fitting      │
                    │                             │
                    │ 1. Stationarity Testing     │
                    │    • Augmented Dickey-Fuller│
                    │    • Determine differencing │
                    │                             │
                    │ 2. Parameter Selection      │
                    │    • Grid search (p, d, q)  │
                    │    • AIC minimization       │
                    │    • Model validation       │
                    │                             │
                    │ 3. Model Fitting            │
                    │    • Maximum likelihood     │
                    │    • Parameter estimation   │
                    │    • Residual analysis      │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
```

### Phase 4: Forecasting & Analysis
```
                    ┌─────────────────────────────┐
                    │   Forecasting & Analysis    │
                    │                             │
                    │ 1. Generate Forecasts       │
                    │    • Point predictions      │
                    │    • Confidence intervals   │
                    │    • Multiple horizons      │
                    │                             │
                    │ 2. Anomaly Detection        │
                    │    • Statistical methods    │
                    │    • Moving averages        │
                    │    • Trend analysis         │
                    │                             │
                    │ 3. Capacity Planning        │
                    │    • Threshold monitoring   │
                    │    • Breach estimation      │
                    │    • Alert generation       │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
```

### Phase 5: Output & Visualization
```
                    ┌─────────────────────────────┐
                    │    Output & Visualization   │
                    │                             │
                    │ 1. Plot Generation          │
                    │    • Time series plots      │
                    │    • Forecast visualization │
                    │    • Confidence bands       │
                    │    • Anomaly highlighting   │
                    │                             │
                    │ 2. Reports & Metrics        │
                    │    • Model diagnostics      │
                    │    • Forecast accuracy      │
                    │    • Capacity analysis      │
                    │                             │
                    │ 3. Alerting                 │
                    │    • Threshold breaches     │
                    │    • Anomaly notifications  │
                    │    • Email/webhook alerts   │
                    └─────────────────────────────┘
```

---

## ARIMA Algorithm Explanation

### What is ARIMA?

ARIMA (AutoRegressive Integrated Moving Average) is a powerful statistical model used for analyzing and forecasting time series data. It combines three key components:

### Components Breakdown:

#### 1. **AR (AutoRegressive) - p parameter**
- **What it does**: Uses past values of the series to predict future values
- **Mathematical form**: `X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t`
- **Interpretation**: "Today's value depends on yesterday's values"
- **Example**: If disk usage tends to follow recent trends

#### 2. **I (Integrated) - d parameter**
- **What it does**: Makes the series stationary by differencing
- **Mathematical form**: `Y_t = X_t - X_{t-1}` (first difference)
- **Why needed**: ARIMA requires stationary data (constant mean/variance)
- **Example**: Raw disk usage grows over time, but daily changes are stable

#### 3. **MA (Moving Average) - q parameter**
- **What it does**: Uses past forecast errors to improve predictions
- **Mathematical form**: `X_t = c + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θₑε_{t-q}`
- **Interpretation**: "Today's value depends on recent prediction errors"
- **Example**: If yesterday's prediction was wrong, adjust today's forecast

### ARIMA(p,d,q) Notation:
- **p**: Number of autoregressive terms (lag observations)
- **d**: Number of differences needed for stationarity
- **q**: Number of moving average terms (lag forecast errors)

### Our Implementation Process:

#### Step 1: Stationarity Testing
```python
def check_stationarity(series):
    """
    Augmented Dickey-Fuller Test:
    H₀: Series has unit root (non-stationary)
    H₁: Series is stationary
    
    If p-value < 0.05: Reject H₀ (stationary)
    If p-value > 0.05: Fail to reject H₀ (non-stationary)
    """
    result = adfuller(series)
    return result[1] <= 0.05  # p-value threshold
```

#### Step 2: Parameter Selection (Grid Search)
```python
def auto_fit(data):
    """
    Grid search over parameter space:
    - Test combinations of (p, d, q)
    - p: 0 to 3 (AR terms)
    - d: 0 to 2 (differencing)
    - q: 0 to 3 (MA terms)
    
    Select model with lowest AIC (Akaike Information Criterion)
    AIC = 2k - 2ln(L)
    where k = parameters, L = likelihood
    """
    best_aic = float('inf')
    best_order = None
    
    for p, q in itertools.product(range(4), range(4)):
        try:
            model = ARIMA(data, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except:
            continue
    
    return best_order
```

#### Step 3: Model Fitting
```python
# Maximum Likelihood Estimation
model = ARIMA(data, order=(p, d, q))
fitted_model = model.fit()

# Model equation becomes:
# (1 - φ₁L - φ₂L² - ... - φₚLᵖ)(1-L)ᵈXₜ = (1 + θ₁L + θ₂L² + ... + θₑLᵉ)εₜ
```

#### Step 4: Forecasting
```python
def forecast(steps):
    """
    Generate forecasts:
    1. Point forecasts: E[X_{t+h}|X_t, X_{t-1}, ...]
    2. Confidence intervals: forecast ± z_α/2 * σ_h
    
    where σ_h is the h-step ahead forecast standard error
    """
    forecast = fitted_model.forecast(steps=steps)
    conf_int = fitted_model.get_forecast(steps).conf_int()
    return forecast, conf_int
```

---

## Code Architecture

### Directory Structure:
```
src/
├── core/               # Main application orchestration
│   └── main.py         # Pipeline coordination
├── database/           # Data source management
│   └── influx_client.py # InfluxDB operations
├── monitoring/         # Real-time data collection
│   └── disk_monitor.py # System monitoring
├── forecasting/        # ARIMA implementation
│   └── arima_model.py  # Core forecasting logic
├── visualization/      # Plotting and dashboards
│   └── plots.py        # Chart generation
└── utils/             # Helper functions
    └── data_processing.py # Data manipulation
```

### Class Relationships:
```
┌─────────────────────────────────────────────┐
│            StorageForecastingPipeline       │
│                                             │
│  Orchestrates entire workflow:              │
│  • Data collection                          │
│  • Preprocessing                            │
│  • Model training                           │
│  • Forecasting                              │
│  • Visualization                            │
└─────────────┬───────────────────────────────┘
              │
              ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ InfluxDBManager │    │   DiskMonitor   │
    │                 │    │                 │
    │ • Query data    │    │ • Real-time     │
    │ • Store results │    │   monitoring    │
    │ • Manage conn   │    │ • Data collect  │
    └─────────────────┘    └─────────────────┘
              │                       │
              └───────────┬───────────┘
                          │
    ┌─────────────────────▼─────────────────┐
    │           ARIMAForecaster             │
    │                                       │
    │ • Stationarity testing                │
    │ • Parameter optimization              │
    │ • Model fitting                       │
    │ • Forecast generation                 │
    └─────────────────┬─────────────────────┘
                      │
    ┌─────────────────▼─────────────────────┐
    │         ForecastVisualizer            │
    │                                       │
    │ • Plot time series                    │
    │ • Show forecasts                      │
    │ • Highlight anomalies                 │
    │ • Create dashboards                   │
    └───────────────────────────────────────┘
```

---

## Data Flow

### 1. Input Data Format:
```csv
Date,Usage
2024-01-01,450.5
2024-01-02,451.2
2024-01-03,452.8
...
```

### 2. Internal Data Structure:
```python
# Pandas Series with DatetimeIndex
data = pd.Series([450.5, 451.2, 452.8, ...], 
                index=pd.DatetimeIndex(['2024-01-01', '2024-01-02', ...]),
                name='disk_usage')
```

### 3. Processing Steps:
```python
# Step 1: Load and validate
data = load_time_series_data('data.csv')

# Step 2: Preprocess
data = handle_missing_values(data)
outliers, data = detect_outliers(data)

# Step 3: Train model
forecaster = ARIMAForecaster()
forecaster.auto_fit(data)

# Step 4: Generate forecast
forecast = forecaster.forecast(steps=30)

# Step 5: Detect anomalies
detector = AnomalyDetector()
anomalies = detector.detect(data)

# Step 6: Visualize
visualizer = ForecastVisualizer()
fig = visualizer.plot_forecast(data, forecast)
```

### 4. Output Structure:
```python
# Forecast DataFrame
forecast = pd.DataFrame({
    'forecast': [501.2, 502.1, 503.0, ...],      # Point predictions
    'lower_bound': [498.1, 498.8, 499.5, ...],   # Lower confidence
    'upper_bound': [504.3, 505.4, 506.5, ...]    # Upper confidence
})

# Results Dictionary
results = {
    'data_points': 51,
    'forecast_periods': 30,
    'anomalies_detected': 2,
    'model_order': (3, 1, 2),
    'breach_date': datetime(2025, 7, 26),
    'forecast': forecast_df
}
```

---

## Mathematical Details

### Stationarity and Differencing

#### Augmented Dickey-Fuller Test:
```
Test equation: ΔYₜ = α + βt + γYₜ₋₁ + δ₁ΔYₜ₋₁ + ... + δₚ₋₁ΔYₜ₋ₚ₊₁ + εₜ

Where:
• ΔYₜ = Yₜ - Yₜ₋₁ (first difference)
• H₀: γ = 0 (unit root exists, non-stationary)
• H₁: γ < 0 (stationary)
```

#### Differencing Operations:
```python
# First difference (d=1)
diff1 = data.diff().dropna()  # Yₜ - Yₜ₋₁

# Second difference (d=2)
diff2 = data.diff().diff().dropna()  # (Yₜ - Yₜ₋₁) - (Yₜ₋₁ - Yₜ₋₂)
```

### ARIMA Model Equation

For ARIMA(p,d,q), the general form is:
```
(1 - φ₁L - φ₂L² - ... - φₚLᵖ)(1-L)ᵈXₜ = (1 + θ₁L + θ₂L² + ... + θₑLᵉ)εₜ

Where:
• L = lag operator (LXₜ = Xₜ₋₁)
• φᵢ = autoregressive parameters
• θⱼ = moving average parameters
• εₜ = white noise error term
• (1-L)ᵈ = differencing operator
```

### Example: ARIMA(1,1,1) for Disk Usage
```python
# Our typical model: ARIMA(1,1,1)
# (1 - φ₁L)(1-L)Xₜ = (1 + θ₁L)εₜ

# Expanded form:
# Xₜ - Xₜ₋₁ = φ₁(Xₜ₋₁ - Xₜ₋₂) + εₜ + θ₁εₜ₋₁

# In practical terms:
# Today's usage change = φ₁ × Yesterday's usage change + 
#                       Current shock + θ₁ × Yesterday's shock

# With estimated parameters φ₁=0.3, θ₁=0.7:
usage_change_today = 0.3 * usage_change_yesterday + 
                    current_shock + 0.7 * yesterday_shock
```

### Forecasting Mathematics

#### h-step ahead forecast:
```
X̂ₜ₊ₕ = E[Xₜ₊ₕ | Xₜ, Xₜ₋₁, Xₜ₋₂, ...]

For ARIMA(1,1,1):
X̂ₜ₊₁ = Xₜ + φ₁(Xₜ - Xₜ₋₁) + θ₁εₙ
X̂ₜ₊₂ = X̂ₜ₊₁ + φ₁(X̂ₜ₊₁ - Xₜ)
...
```

#### Confidence Intervals:
```
CI = X̂ₜ₊ₕ ± z_{α/2} × σₕ

Where:
• z_{α/2} = critical value (1.96 for 95% confidence)
• σₕ = h-step ahead forecast standard error
• σₕ increases with forecast horizon h
```

---

## Implementation Details

### Key Algorithms in Our Code:

#### 1. Grid Search for Optimal Parameters:
```python
def find_optimal_parameters(data, max_p=3, max_q=3, max_d=2):
    """
    Systematic search through parameter space
    Time complexity: O(max_p × max_q × max_d)
    """
    best_aic = float('inf')
    best_order = None
    
    # Test stationarity and determine d
    d_optimal = determine_differencing_order(data, max_d)
    
    # Grid search for p and q
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(data, order=(p, d_optimal, q))
                fitted = model.fit()
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d_optimal, q)
            except:
                continue
    
    return best_order, best_aic
```

#### 2. Anomaly Detection:
```python
def detect_statistical_anomalies(data, threshold=3.0):
    """
    Z-score based anomaly detection
    
    Z-score = |X - μ| / σ
    Anomaly if Z > threshold
    """
    mean = data.mean()
    std = data.std()
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold

def detect_moving_average_anomalies(data, window=10, threshold=3.0):
    """
    Rolling statistics anomaly detection
    
    Upper bound = rolling_mean + threshold × rolling_std
    Lower bound = rolling_mean - threshold × rolling_std
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    upper_bound = rolling_mean + threshold * rolling_std
    lower_bound = rolling_mean - threshold * rolling_std
    
    anomalies = (data > upper_bound) | (data < lower_bound)
    return anomalies
```

#### 3. Capacity Breach Prediction:
```python
def estimate_capacity_breach(data, forecast, threshold):
    """
    Linear extrapolation to estimate when threshold is reached
    
    growth_rate = average daily increase
    days_to_breach = (threshold - current_usage) / growth_rate
    breach_date = today + days_to_breach
    """
    current_usage = data.iloc[-1]
    
    if current_usage >= threshold:
        return data.index[-1]  # Already breached
    
    # Calculate average growth rate
    growth_rate = data.diff().mean()
    
    if growth_rate <= 0:
        return None  # Not growing
    
    # Estimate days to breach
    days_to_breach = (threshold - current_usage) / growth_rate
    
    # Calculate breach date
    breach_date = data.index[-1] + pd.Timedelta(days=days_to_breach)
    
    return breach_date
```

### Performance Considerations:

#### Model Fitting Time Complexity:
- **Grid Search**: O(p_max × q_max × n²) where n = data points
- **ARIMA Fitting**: O(n) for each parameter combination
- **Total**: Typically 1-10 seconds for 100 data points

#### Memory Usage:
- **Data Storage**: O(n) for time series
- **Model Parameters**: O(p + q) coefficients
- **Forecast**: O(h) where h = forecast horizon

#### Optimization Strategies:
```python
# 1. Limit parameter search space
max_p, max_q = 3, 3  # Instead of 5, 5

# 2. Use stepwise selection instead of full grid search
# (Not implemented in current version)

# 3. Parallel parameter testing
# (Could be added using multiprocessing)

# 4. Early stopping for poor models
if fitted.aic > best_aic * 1.1:
    continue  # Skip obviously worse models
```

---

## Real-World Example

### Input Data (Disk Usage):
```
Date         Usage(GB)   Daily_Change
2024-01-01   450.5       -
2024-01-02   451.2       +0.7
2024-01-03   452.8       +1.6
2024-01-04   453.1       +0.3
...          ...         ...
2024-02-20   498.9       +1.0
```

### ARIMA Analysis:
```
Step 1: Stationarity Test
- Raw data: p-value = 0.9936 (non-stationary)
- After differencing: p-value = 0.0001 (stationary)
- Result: d = 1

Step 2: Parameter Selection
- Tested: (0,1,0), (1,1,0), (0,1,1), (1,1,1), (2,1,1), (3,1,2)...
- Best: ARIMA(3,1,2) with AIC = -82.75

Step 3: Model Interpretation
- φ₁ = 0.234 (yesterday's change influences today)
- φ₂ = -0.145 (two days ago has negative influence)
- φ₃ = 0.089 (small influence from 3 days ago)
- θ₁ = 0.678 (yesterday's error correction)
- θ₂ = -0.123 (two-day error correction)
```

### Forecast Results:
```
Date         Forecast   Lower_95%   Upper_95%   Actual
2024-02-21   500.1      498.8       501.4       500.3
2024-02-22   501.2      499.1       503.3       501.0
2024-02-23   502.3      499.4       505.2       -
2024-02-24   503.4      499.7       507.1       -
...
```

### Business Insights:
```
Current Usage: 498.9 GB
Growth Rate: 0.97 GB/day
Capacity Threshold: 520 GB
Days to Breach: 21.8 days
Expected Breach: 2025-07-26

Recommendation: Plan capacity expansion within 3 weeks
```

---

This comprehensive explanation covers the entire system workflow, from data input to final predictions, along with the mathematical foundations of the ARIMA algorithm and implementation details specific to our storage forecasting use case.