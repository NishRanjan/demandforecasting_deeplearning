"""
data/loader.py
──────────────
Platform-aware data loading. Returns a single pandas DataFrame regardless
of whether we're running locally or on Databricks.

Only this module and utils/platform.py contain platform-conditional logic.
"""

from __future__ import annotations

import os

import pandas as pd

from utils.platform import get_platform_mode, get_spark, resolve_path


def load_data(config: dict) -> pd.DataFrame:
    """
    Load the raw dataset according to config settings.

    Local mode  : reads config['data']['source_path'] with pd.read_csv
    Databricks  : reads config['data']['blob_url'] via Spark, converts to pandas

    Returns a pandas DataFrame with the date column parsed as datetime.
    """
    mode = get_platform_mode(config)
    data_cfg = config["data"]
    date_col = data_cfg["date_col"]
    date_format = data_cfg.get("date_format", None)

    if mode == "databricks":
        df = _load_databricks(data_cfg)
    else:
        df = _load_local(data_cfg)

    # Parse date column
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    else:
        df[date_col] = pd.to_datetime(df[date_col])

    # Sort by grain + date for consistent downstream processing
    grain_cols = data_cfg.get("grain_cols", [])
    sort_cols = grain_cols + [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    print(f"[loader] Loaded {len(df):,} rows × {df.shape[1]} columns | mode={mode}")
    return df


def _load_local(data_cfg: dict) -> pd.DataFrame:
    source_path = data_cfg["source_path"]
    print(f"[loader] Reading local CSV: {source_path}")
    return pd.read_csv(source_path, low_memory=False)


def _load_databricks(data_cfg: dict) -> pd.DataFrame:
    blob_url = data_cfg.get("blob_url", "")
    if not blob_url or blob_url.startswith("${"):
        blob_url = os.environ.get("BLOB_URL", "")
    if not blob_url:
        raise ValueError("blob_url not set. Define BLOB_URL environment variable or set data.blob_url in config.")
    print(f"[loader] Reading from Blob via Spark: {blob_url}")
    spark = get_spark()
    sdf = spark.read.option("header", "true").option("inferSchema", "true").csv(blob_url)
    return sdf.toPandas()
