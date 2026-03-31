Run the full forecasting pipeline using the current config.

Usage: /run [model_names...] [--skip-backtest]

Examples:
  /run                          → runs all enabled models with backtest + forecast
  /run chronos tft              → runs only Chronos and TFT
  /run chronos --skip-backtest  → forward forecast only, no backtest

Steps to follow:
1. Confirm which models are enabled in config/forecast_config.yaml (models.<name>.enabled)
2. Confirm platform.mode matches the current environment (local vs databricks)
3. From the demand_forecasting/ directory, run:
   `python run_pipeline.py --config config/forecast_config.yaml $ARGUMENTS`
4. Report the WMAPE and R² for each model from the console output
5. Show the path to the comparison_results.csv that was written

If $ARGUMENTS contains model names (not flags), pass them as: --models <names>
If $ARGUMENTS contains --skip-backtest, add that flag too.
