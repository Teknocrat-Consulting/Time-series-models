# Quick Start Guide - Storage Forecasting

## 3-Step Quick Start 🚀

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Data
```bash
python scripts/generate_disk_usage_data.py
```

### Step 3: Run Forecast
```bash
python test_large_dataset.py
```

**That's it!** Check `output/large_dataset_forecast.png` for results.

---

## One-Line Setup

Copy and run this entire command:

```bash
pip install -r requirements.txt && python scripts/generate_disk_usage_data.py && python test_large_dataset.py
```

---

## What You'll See

```
Generating 1,000,000 disk usage records...
  ✓ Generated 1,000,000 records
  ✓ Data saved to disk_usage_1million.csv

Running forecasting...
  ✓ Model trained (ARIMA)
  ✓ Accuracy: MAPE ~1.66%
  ✓ Forecast generated for next 30 days
  ✓ Plot saved to output/

✅ Success! View your forecast at: output/large_dataset_forecast.png
```

---

## Files Created

- `data/disk_usage_1million.csv` - Your generated dataset
- `output/large_dataset_forecast.png` - Forecast visualization

---

## Next Steps

- View detailed documentation: `RUN_GUIDE.md`
- Customize parameters: Edit `test_large_dataset.py`
- Use your own data: Replace CSV file path

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Generate data | `python scripts/generate_disk_usage_data.py` |
| Run forecast | `python test_large_dataset.py` |
| Small test | `python test_run.py` |
| View results | `open output/large_dataset_forecast.png` |

---

## Troubleshooting

**Missing packages?**
```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

**Out of memory?**
Use smaller dataset in `test_large_dataset.py`:
```python
recent_data = daily_df.iloc[-100:]  # Use last 100 days only
```

**Need help?**
Check the full guide: `RUN_GUIDE.md`