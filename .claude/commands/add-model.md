Add a new forecasting model to the pipeline.

Usage: /add-model <model_name> [description]
Example: /add-model prophet "Facebook Prophet seasonal decomposition model"

Steps to follow:
1. Ask the user (if not already provided):
   - What is the model name (short lowercase, e.g. "prophet")?
   - What library does it use?
   - Does it need training or is it pre-trained (like Chronos)?
   - Does it produce prediction intervals (q10/q90)?

2. Create models/<model_name>_model.py by copying the structure of the closest
   existing model:
   - Pre-trained / inference-only → copy chronos_model.py as template
   - Trainable with pytorch-forecasting → copy nhits_model.py as template (simpler than TFT)
   - Other library → use base.py as reference and implement from scratch

3. The new class MUST:
   - Subclass BaseForecaster from models/base.py
   - Implement the `name` property returning the model_name string
   - Implement fit(train_df), predict(context_df, horizon), backtest(df)
   - Use date_utils.get_backtest_cutoffs() and apply_latency_offset() in backtest()
   - Return DataFrames matching the output schema contract:
       predict()  → [date_col, *grain_cols, "forecast", "q10", "q90", "model_name"]
       backtest() → same + ["actual", "series_segment", "backtest_cutoff"]
   - Fill q10/q90 with NaN if the model doesn't produce intervals

4. Add the model to MODEL_REGISTRY in run_pipeline.py:
   "<model_name>": ("models.<model_name>_model", "<ClassName>"),

5. Add a config section to config/forecast_config.yaml under models:
   <model_name>:
     enabled: true
     max_encoder_length: 24
     max_prediction_length: 2
     # ... model-specific params

6. Test by running: python run_pipeline.py --models <model_name> --skip-backtest

7. Show the user what was created and what they may need to install (pip packages).
