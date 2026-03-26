from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH, UCI_COLUMNS, UCI_FALLBACK_URL


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

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
