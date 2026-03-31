Validate the forecast_config.yaml against the actual dataset columns.

Steps to follow:
1. Read config/forecast_config.yaml
2. Check that all required top-level sections exist:
   platform, data, features, preprocessing, backtest, models, evaluation
3. Verify that data.source_path exists on disk (if platform.mode == "local")
4. If the CSV exists, load its header and cross-check:
   - data.date_col exists in the CSV
   - data.target_col exists in the CSV
   - data.grain_cols all exist in the CSV
   - features.static_categoricals — list which are present and which are missing
   - features.time_varying_known — same
   - features.time_varying_unknown — same
   - All continuous feature groups (lag_cols, marketing, macro, weather, etc.) — same
5. Check that each model under models: has max_prediction_length == backtest.forecast_horizon
6. Report a summary:
   - PASS items in green (or just list them)
   - WARN items: columns in config that aren't in the CSV (model will silently skip them)
   - ERROR items: missing required fields that would cause a crash

If the CSV is not available locally, skip step 4 and report "data validation skipped — file not found".
