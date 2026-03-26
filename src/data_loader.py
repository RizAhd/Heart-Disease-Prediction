from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from src.config import DATASET_PATH, UCI_COLUMNS, UCI_FALLBACK_URL


ALT_COLUMN_MAP = {
    "Age": "age",
    "Sex": "sex",
    "Chest pain type": "cp",
    "BP": "trestbps",
    "Cholesterol": "chol",
    "FBS over 120": "fbs",
    "EKG results": "restecg",
    "Max HR": "thalach",
    "Exercise angina": "exang",
    "ST depression": "oldpeak",
    "Slope of ST": "slope",
    "Number of vessels fluro": "ca",
    "Thallium": "thal",
    "Heart Disease": "target",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "target" in df.columns and "age" in df.columns:
        return df

    if all(col in df.columns for col in ALT_COLUMN_MAP):
        df = df.rename(columns=ALT_COLUMN_MAP)
        return df

    return df


def _standardize_target(df: pd.DataFrame) -> pd.DataFrame:
    if "target" not in df.columns:
        return df

    if is_object_dtype(df["target"]) or is_string_dtype(df["target"]):
        normalized = df["target"].astype(str).str.strip().str.lower()
        mapping = {
            "presence": 1,
            "absence": 0,
            "present": 1,
            "absent": 0,
            "disease": 1,
            "no disease": 0,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }
        df["target"] = normalized.map(mapping)

    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    # If values are 0-4 style heart severity labels, collapse to binary.
    if df["target"].dropna().max() > 1:
        df["target"] = (df["target"] > 0).astype(int)

    return df


def _load_existing_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    df = _standardize_target(df)

    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [c for c in df.columns if c != "target"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "target" in df.columns:
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)

    return df


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if path.exists():
        return _load_existing_csv(path)

    # If a different CSV exists in data/, use it automatically.
    csv_files = sorted(path.parent.glob("*.csv"))
    if csv_files:
        return _load_existing_csv(csv_files[0])

    # Fallback: fetch the canonical UCI Cleveland dataset if local CSV is unavailable.
    df = pd.read_csv(UCI_FALLBACK_URL, header=None, names=UCI_COLUMNS)
    df = df.replace("?", pd.NA)

    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [c for c in df.columns if c != "target"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # UCI target ranges from 0-4; convert to binary (0 = no disease, 1 = disease).
    df["target"] = (df["target"] > 0).astype(int)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
