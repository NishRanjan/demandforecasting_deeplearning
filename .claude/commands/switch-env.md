Switch the pipeline between local development and Databricks environments.

Usage: /switch-env <local|databricks>
Example: /switch-env databricks

Steps to follow:
1. Read config/forecast_config.yaml
2. Change platform.mode to the requested value ("local" or "databricks")
3. Remind the user of what else changes with each environment:

   When switching TO "databricks":
   - data.blob_url will be used for data loading (not source_path)
   - Ensure blob_url is filled in with the correct Azure Blob path
   - accelerator can be changed to "gpu" for all models
   - device: "cuda" for Chronos
   - num_workers can be increased (e.g. 4) for faster DataLoader
   - MLflow will use Databricks managed tracking automatically

   When switching TO "local":
   - data.source_path will be used (must point to a local CSV)
   - accelerator should be "cpu" unless you have a local GPU
   - device: "cpu" for Chronos (unless local GPU available)
   - num_workers: 0 on Windows to avoid multiprocessing issues
   - MLflow will use local server or can be disabled (use_mlflow: false)

4. Save the updated config
5. Show the user the relevant changed lines with a before/after diff
