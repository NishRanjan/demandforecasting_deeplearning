Debug a failing or underperforming model.

Usage: /debug-model <model_name>
Example: /debug-model tft

Steps to follow:
1. Read config/forecast_config.yaml to understand current settings for <model_name>
2. Check if outputs/per_series_metrics.csv exists — if so, identify which series
   the model performs worst on (highest wmape)
3. Run a quick diagnostic:
   - Does the model file exist? models/<model_name>_model.py
   - Is enabled: true in config?
   - Does max_prediction_length match backtest.forecast_horizon?
4. For training models (tft, nhits, deepar), check:
   - Is max_encoder_length + max_prediction_length <= minimum series length in the data?
   - Are num_workers: 0 on Windows? (common crash cause)
   - Is accelerator set correctly for the current hardware?
5. For Chronos specifically:
   - Is chronos-forecasting installed? (pip list | grep chronos)
   - Is context_length <= minimum series length?
6. Suggest specific config changes to try:
   - Reduce hidden_size if training is slow / OOM
   - Reduce batch_size if GPU memory error
   - Increase early_stopping_patience if training stops too early
   - Set max_epochs higher if val_loss was still improving at end of training
7. If the user reports a specific error message, diagnose it and propose a fix
