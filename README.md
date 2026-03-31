# Demand Forecasting Pipeline

Modular, platform-agnostic demand forecasting pipeline for SKU × State level monthly forecasting. Supports multiple deep learning models with a unified evaluation and comparison framework.

Runs identically on **Databricks** and **locally** — one config toggle switches between them.

---

## Models

| Model | Type | Library | Prediction Intervals |
|---|---|---|---|
| **Chronos** | Pre-trained foundation model | `chronos-forecasting` | Yes (Q10, Q50, Q90) |
| **TFT** | Temporal Fusion Transformer | `pytorch-forecasting` | Via QuantileLoss |
| **NHiTS** | Neural Hierarchical Interpolation | `pytorch-forecasting` | No (point forecast) |
| **DeepAR** | Probabilistic RNN | `pytorch-forecasting` | Via NormalDistributionLoss |

---

## Project Structure

```
demand_forecasting/
├── config/
│   └── forecast_config.yaml      # All settings live here — the only file you change between runs
│
├── data/
│   ├── loader.py                 # Reads data: local CSV or Azure Blob via Spark
│   ├── preprocessor.py           # Outlier clipping, imputation, log1p, head/tail tagging
│   └── feature_engineering.py   # Lag generation, calendar features, rolling stats
│
├── models/
│   ├── base.py                   # Abstract BaseForecaster (fit / predict / backtest interface)
│   ├── chronos_model.py
│   ├── tft_model.py
│   ├── nhits_model.py
│   └── deepar_model.py
│
├── evaluation/
│   ├── metrics.py                # wmape(), r_squared(), bias(), quantile_coverage()
│   └── comparison.py             # compare_models() → summary + per-series tables
│
├── tracking/
│   └── mlflow_logger.py          # MLflow wrapper — silently disabled when not needed
│
├── utils/
│   ├── platform.py               # Databricks vs local detection, display, secrets
│   └── date_utils.py             # Backtest window logic, M+1 latency offset
│
├── outputs/                      # Created at runtime
│   ├── comparison_results.csv
│   ├── per_series_metrics.csv
│   └── forecasts/
│
├── run_pipeline.py               # CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA separately:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure your dataset

Edit `config/forecast_config.yaml`:

```yaml
platform:
  mode: "local"              # change to "databricks" when running on cluster

data:
  source_path: "data/raw/sales.csv"   # your local CSV path
  date_col: "date"
  target_col: "Volume"
  grain_cols: ["sku", "state"]

features:
  static_categoricals: ["sku", "brand", "subcategory", "category", "state", "region"]
  # ... update to match your dataset columns
```

### 3. Run

```bash
# All enabled models
python run_pipeline.py --config config/forecast_config.yaml

# Specific models only
python run_pipeline.py --config config/forecast_config.yaml --models chronos tft

# Skip backtest — forward forecast only
python run_pipeline.py --config config/forecast_config.yaml --skip-backtest
```

---

## Running on Databricks

Change one line in `config/forecast_config.yaml`:

```yaml
platform:
  mode: "databricks"         # was "local"
```

Then call from a notebook cell:

```python
import sys
sys.path.insert(0, "/Workspace/Users/you@company.com/demand_forecasting")
from run_pipeline import main
main("config/forecast_config.yaml")
```

Data will be read from `data.blob_url` via Spark instead of a local CSV. MLflow tracking uses Databricks managed MLflow automatically. No other code changes needed.

For GPU acceleration on a GPU cluster, also set in config:
```yaml
models:
  tft:
    accelerator: "gpu"
  chronos:
    device: "cuda"
```

---

## Configuration Reference

All pipeline inputs are controlled by `config/forecast_config.yaml`. Key sections:

### `platform`
```yaml
platform:
  mode: "local"              # "local" | "databricks"
  use_mlflow: false          # Enable MLflow experiment tracking
  mlflow_experiment_name: "demand_forecast"
```

### `data`
```yaml
data:
  source_path: "data/raw/sales.csv"     # local mode
  blob_url: "https://..."               # databricks mode
  date_col: "date"
  target_col: "Volume"
  grain_cols: ["sku", "state"]          # columns that identify one time series
  freq: "MS"                            # Month-start (pandas offset)
```

### `features`
Define every column your dataset has. Models read only what they need — listing a column that doesn't exist will just be silently skipped.
```yaml
features:
  static_categoricals: [...]            # product/geography attributes
  time_varying_known: ["month", "qtr_bucket"]
  time_varying_unknown: ["sales_uoc"]   # only historical values available
  continuous:
    lag_cols: ["btl_lag1", "btl_lag2"]
    marketing: ["BTL%", "GRP", "SOV"]
    macro: ["GDP", "Inflation"]
    weather: ["Temperature", "humidity"]
```

### `backtest`
```yaml
backtest:
  n_rolling_windows: 3         # number of monthly rolling origins
  data_latency_months: 1       # M+1 convention: skip 1 month between train cutoff and first forecast
  forecast_horizon: 2          # steps per window
  forward_forecast_months: 12  # final forward forecast length
```

### `models`
Each model has an `enabled` flag and its own hyperparameters:
```yaml
models:
  chronos:
    enabled: true
    batch_size: 128
    context_length: 24
  tft:
    enabled: true
    hidden_size: 45
    max_epochs: 50
    accelerator: "cpu"        # "gpu" on cluster
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/comparison_results.csv` | One row per model: WMAPE, R², bias, best_series_count |
| `outputs/per_series_metrics.csv` | One row per (model × SKU × state × backtest window) |
| `outputs/forecasts/<model>_forecast.csv` | 12-month forward forecast per model |

### Example comparison output

| model_name | wmape_mean | r2_mean | bias_mean | best_series_count |
|---|---|---|---|---|
| tft | 0.154 | 0.81 | -0.008 | 73 |
| chronos | 0.182 | 0.74 | +0.031 | 47 |
| nhits | 0.196 | 0.71 | -0.012 | 22 |
| deepar | 0.201 | 0.69 | +0.018 | 18 |

`best_series_count` = number of SKU × state series where that model had the lowest WMAPE.

---

## Backtesting Convention (M+1)

The pipeline uses a 1-month data latency convention:

- Data is available up to month **M** but published with a 1-month lag
- Training cutoff: **October 2025** → first forecastable month: **December 2025**
- November 2025 is the latency gap (skipped, not forecast)

This matches the business reality where actuals from month M are only finalized in M+1.

---

## Adding a New Model

1. Create `models/<name>_model.py` subclassing `BaseForecaster`:

```python
from models.base import BaseForecaster

class MyModelForecaster(BaseForecaster):
    @property
    def name(self): return "mymodel"

    def fit(self, train_df): ...
    def predict(self, context_df, horizon): ...   # must return standard schema
    def backtest(self, df): ...
```

2. Register it in `run_pipeline.py`:
```python
MODEL_REGISTRY = {
    ...
    "mymodel": ("models.mymodel_model", "MyModelForecaster"),
}
```

3. Add config section in `forecast_config.yaml`:
```yaml
models:
  mymodel:
    enabled: true
    max_encoder_length: 24
    max_prediction_length: 2
```

The output schema that `predict()` must return:
```
[date_col, *grain_cols, "forecast", "q10", "q90", "model_name"]
```
Fill `q10`/`q90` with `NaN` if the model doesn't produce prediction intervals.

---

## Slash Commands (Claude Code)

If using Claude Code as your AI assistant, the following `/commands` are available:

| Command | Description |
|---|---|
| `/run [models]` | Run the pipeline and report results |
| `/validate-config` | Cross-check config columns against your CSV |
| `/results` | Show latest comparison output and worst-performing series |
| `/add-model <name>` | Step-by-step guide to add a new model |
| `/switch-env <local\|databricks>` | Update config for target environment |
| `/debug-model <name>` | Diagnose a failing or underperforming model |

---

## Dependencies

```
chronos-forecasting    Amazon Chronos-2 pre-trained model
pytorch-forecasting    TFT, NHiTS, DeepAR
lightning              PyTorch training loop abstraction
mlflow                 Experiment tracking (optional)
pandas, numpy          Data manipulation
PyYAML                 Config loading
python-dotenv          Local secret management (optional)
pyspark                Databricks data loading (pre-installed on cluster)
```
