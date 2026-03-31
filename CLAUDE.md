# Demand Forecasting Pipeline — CLAUDE.md

This codebase is a modular, platform-agnostic demand forecasting pipeline for
Godrej CP India (SKU × State, monthly frequency). It supports multiple deep
learning models and produces a unified comparison report.

## Project context

- **Business domain**: Consumer goods demand forecasting (Personal Wash, Fabric Care)
- **Granularity**: SKU × State level, monthly data
- **Target variable**: `Volume` (sales volume in cases/units)
- **Data source**: Azure Blob Storage CSV (Databricks) or local CSV (development)
- **Users**: Data scientists, not software engineers — keep interfaces simple

## Directory structure

```
demand_forecasting/
├── config/forecast_config.yaml   ← SINGLE source of truth for ALL settings
├── data/
│   ├── loader.py                 ← Platform-aware data loading (local vs Databricks)
│   ├── preprocessor.py           ← Outlier handling, imputation, log1p, segmentation
│   └── feature_engineering.py   ← Lags, calendar features, rolling stats
├── models/
│   ├── base.py                   ← Abstract BaseForecaster — every model implements this
│   ├── chronos_model.py          ← Amazon Chronos-2 (pre-trained, no training needed)
│   ├── tft_model.py              ← TemporalFusionTransformer (Lightning training)
│   ├── nhits_model.py            ← NHiTS (pytorch-forecasting)
│   └── deepar_model.py           ← DeepAR (pytorch-forecasting)
├── evaluation/
│   ├── metrics.py                ← Pure functions: wmape(), r2(), bias(), coverage()
│   └── comparison.py             ← compare_models() → summary_df + per_series_df
├── tracking/mlflow_logger.py     ← MLflow wrapper (no-ops when disabled)
├── utils/
│   ├── platform.py               ← ALL platform differences isolated here
│   └── date_utils.py             ← Backtest windows, M+1 latency logic
├── run_pipeline.py               ← CLI entry point
└── requirements.txt
```

## How to run

```bash
# From inside demand_forecasting/
pip install -r requirements.txt

# Full pipeline (all enabled models)
python run_pipeline.py --config config/forecast_config.yaml

# Specific models only
python run_pipeline.py --config config/forecast_config.yaml --models chronos tft

# Skip backtest (forward forecast only)
python run_pipeline.py --config config/forecast_config.yaml --skip-backtest
```

**From Databricks notebook cell:**
```python
import sys
sys.path.insert(0, "/Workspace/Users/<your-email>/demand_forecasting")
from run_pipeline import main
main("config/forecast_config.yaml")
```

## The config file is the only thing you should need to change between runs

`config/forecast_config.yaml` controls:
- `platform.mode`: `"local"` or `"databricks"` — the single toggle for environment
- `data.source_path` / `data.blob_url`: where to read data from
- `features.*`: which columns exist in your dataset
- `models.<name>.enabled`: turn individual models on/off
- `backtest.*`: number of rolling windows, M+1 latency, forecast horizon
- `evaluation.output_path`: where comparison results are saved

**Never hardcode column names or file paths in Python code.** All of these must
come from config.

## Key design rules

### Output schema contract
Every model's `predict()` and `backtest()` must return DataFrames with exactly:
```
predict() → [date_col, *grain_cols, "forecast", "q10", "q90", "model_name"]
backtest() → [date_col, *grain_cols, "forecast", "q10", "q90", "model_name", "actual", "series_segment"]
```
All downstream evaluation code depends on this schema. Never change column names
in model output without updating `evaluation/comparison.py`.

### Platform differences are isolated to two files only
- `utils/platform.py` — display, secrets, Spark session, path resolution
- `data/loader.py` — data reading

No other file should contain `if databricks` / `if local` logic.

### Metrics are pure functions
`evaluation/metrics.py` contains numpy-only functions with no side effects.
The WMAPE loss inside `tft_model.py` is intentionally separate — it operates
on tensors during training, not on numpy arrays for evaluation.

## M+1 backtest convention

The notebooks use M+1 latency: data is available up to month M but with 1 month
publication lag. So:
- Train cutoff: October 2025
- Latency skip: November 2025 (not forecasted)
- First forecast: December 2025 (2-step ahead)

This logic lives in `utils/date_utils.py`. `apply_latency_offset(cutoff, latency=1)`
returns the first forecastable date.

## Adding a new model

1. Create `models/<name>_model.py` — subclass `BaseForecaster`, implement
   `name`, `fit()`, `predict()`, `backtest()`
2. Add to `MODEL_REGISTRY` in `run_pipeline.py`:
   ```python
   "mymodel": ("models.mymodel_model", "MyModelForecaster"),
   ```
3. Add a `mymodel:` section to `config/forecast_config.yaml` under `models:`
4. Set `enabled: true` to include it in the next run

## Outputs

After a run, `outputs/` contains:
```
outputs/
├── comparison_results.csv       ← One row per model: wmape, r2, bias, best_series_count
├── per_series_metrics.csv       ← One row per (model × SKU × state × backtest window)
└── forecasts/
    ├── chronos_forecast.csv
    ├── tft_forecast.csv
    ├── nhits_forecast.csv
    └── deepar_forecast.csv
```

## Dependencies

| Library | Purpose |
|---|---|
| `chronos-forecasting` | Amazon Chronos-2 pre-trained model |
| `pytorch-forecasting` | TFT, NHiTS, DeepAR model classes |
| `lightning` | Training loop for all PyTorch models |
| `mlflow` | Optional experiment tracking |
| `pyspark` | Databricks data loading (pre-installed on cluster) |
| `python-dotenv` | Local `.env` secret management |

GPU note: change `accelerator: "cpu"` → `"gpu"` and `device: "cpu"` → `"cuda"`
in the config when running on a GPU cluster. No code changes needed.
