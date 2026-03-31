"""
utils/platform.py
─────────────────
Runtime environment detection and platform-specific utilities.
All Databricks vs. local differences are isolated here and in data/loader.py.
No other module should contain platform-conditional logic.
"""

import os
import warnings


def is_databricks() -> bool:
    """Returns True when running inside a Databricks cluster."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_platform_mode(config: dict) -> str:
    """
    Returns the effective platform mode and warns if the config value
    doesn't match the actual runtime environment.
    """
    configured = config.get("platform", {}).get("mode", "local")
    actual = "databricks" if is_databricks() else "local"
    if configured != actual:
        warnings.warn(
            f"Config platform.mode='{configured}' but actual runtime is '{actual}'. "
            f"Proceeding with configured mode. This may cause data loading to fail.",
            UserWarning,
            stacklevel=2,
        )
    return configured


def display_df(df, n: int = 20):
    """
    Display a DataFrame: uses Databricks display() on cluster, prints head locally.
    """
    if is_databricks():
        # display() is a Databricks built-in; injected into builtins at runtime
        try:
            display(df)  # noqa: F821 — available in Databricks scope
        except NameError:
            print(df.head(n).to_string())
    else:
        print(df.head(n).to_string())


def get_spark():
    """
    Returns the active SparkSession. Raises a clear error if called locally.
    Only call this from data/loader.py when platform.mode == 'databricks'.
    """
    if not is_databricks():
        raise EnvironmentError(
            "get_spark() called in local mode. Set platform.mode='databricks' "
            "or use source_path for local CSV loading."
        )
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession()
    except ImportError:
        raise ImportError(
            "pyspark is not installed. Install it or switch to platform.mode='local'."
        )


def get_secret(key: str) -> str:
    """
    Fetch a secret by key.
    - Databricks: uses dbutils.secrets (scope='forecast')
    - Local: reads from .env file via python-dotenv, then os.environ
    """
    if is_databricks():
        try:
            return dbutils.secrets.get(scope="forecast", key=key)  # noqa: F821
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch Databricks secret '{key}' from scope 'forecast': {e}"
            )
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # python-dotenv optional locally
        value = os.environ.get(key)
        if value is None:
            raise KeyError(
                f"Secret '{key}' not found in environment. "
                f"Set it in a .env file or as an environment variable."
            )
        return value


def resolve_path(path: str, platform_mode: str) -> str:
    """
    Resolve a file path for the given platform.
    - Databricks: prepends /dbfs/ if path is relative and doesn't start with /dbfs
    - Local: returns as-is
    """
    if platform_mode == "databricks":
        if not path.startswith("/dbfs") and not path.startswith("dbfs:"):
            return "/dbfs/" + path.lstrip("/")
    return path
