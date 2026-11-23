"""
Module 1: Dataset Management
Team 4: Smart Exam Monitoring & Integrity Analytics Platform
Handles loading, profiling, summary statistics, validation, and integration of all CSV datasets.
Supports local directory files, external production datasets, and custom user uploads.
"""

from pathlib import Path
from typing import Dict, Union, Tuple, Optional, Any
import io
import pandas as pd
import numpy as np

from src.data_integration import (
    get_integrated_datasets,
    ingest_uploaded_csvs,
    get_current_pipeline_status,
    DEFAULT_EXTERNAL_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_FALLBACK_DIR
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SUPPORTED_DATASETS = [
    "candidates.csv",
    "sessions.csv",
    "events.csv",
    "scores.csv",
    "risk.csv",
    "risk_data.csv",
    "ai_reports.csv",
    "ai_report_metadata.csv",
    "face_monitoring_ground_truth.csv"
]


def load_dataset(file_or_path: Union[str, Path, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """
    Load a CSV dataset from a file path or uploaded file stream.
    Handles encoding gracefully.
    """
    try:
        if isinstance(file_or_path, (str, Path)):
            path = Path(file_or_path)
            if not path.is_absolute():
                path = DATA_DIR / path
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return pd.read_csv(path)
        else:
            return pd.read_csv(file_or_path)
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")


def load_all_datasets(data_dir: Optional[Union[str, Path]] = None, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load all available datasets into a dictionary mapped by key name,
    using the automated integration and normalization pipeline.
    """
    datasets, _ = get_integrated_datasets(source_dir=data_dir, force_reload=force_reload)
    return datasets


def get_data_source_status() -> Dict[str, Any]:
    """
    Retrieves the current data source provenance, record counts, and last ingested timestamp.
    """
    return get_current_pipeline_status()


def get_dataset_summary(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
    """
    Generate dataset summary metrics:
    - rows and columns
    - numerical and categorical columns
    - missing values (count and percentage)
    - duplicate records count
    """
    if df.empty:
        return {
            "name": dataset_name,
            "rows": 0,
            "columns": 0,
            "numerical_cols": [],
            "categorical_cols": [],
            "total_missing": 0,
            "missing_pct": 0.0,
            "duplicate_rows": 0,
            "memory_usage_kb": 0.0
        }
        
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
    duplicate_count = int(df.duplicated().sum())
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    
    return {
        "name": dataset_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "total_missing": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_count,
        "memory_usage_kb": memory_kb
    }


def get_column_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return comprehensive profiling table for every column in the dataset:
    - Data Type
    - Non-Null Count
    - Missing Count & Missing %
    - Unique Count
    - Sample Values
    """
    if df.empty:
        return pd.DataFrame()
        
    profiles = []
    total_rows = len(df)
    
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        missing = int(series.isna().sum())
        missing_pct = round(missing / total_rows * 100, 2) if total_rows > 0 else 0.0
        unique_cnt = int(series.nunique(dropna=True))
        
        # Grab up to 3 non-null sample values
        samples = series.dropna().unique()[:3]
        samples_str = ", ".join([str(s) for s in samples]) if len(samples) > 0 else "None"
        
        profiles.append({
            "Column Name": col,
            "Data Type": dtype,
            "Type Category": "Numerical" if pd.api.types.is_numeric_dtype(series) else "Categorical/Text",
            "Non-Null Count": total_rows - missing,
            "Missing Count": missing,
            "Missing %": f"{missing_pct}%",
            "Unique Values": unique_cnt,
            "Sample Preview": samples_str
        })
        
    return pd.DataFrame(profiles)


def validate_uploaded_csv(uploaded_file) -> Tuple[bool, Union[pd.DataFrame, str]]:
    """
    Validates user uploaded CSV file.
    Returns (success_flag, DataFrame_or_error_message).
    """
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            return False, "Uploaded CSV file is empty."
        return True, df
    except Exception as e:
        return False, f"Failed to parse CSV file: {str(e)}"
