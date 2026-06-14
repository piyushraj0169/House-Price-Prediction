"""Utilities for loading and inspecting housing data."""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load the housing dataset from a CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Dataset file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("Dataset file could not be parsed as CSV.") from exc


def display_dataset_info(data: pd.DataFrame) -> None:
    """Print core dataset diagnostics."""
    print("\nDataset Shape:")
    print(data.shape)

    print("\nDataset Columns:")
    print(list(data.columns))

    print("\nDataset Info:")
    data.info()

    print("\nMissing Values:")
    print(data.isna().sum())

    print("\nDuplicate Rows:")
    print(data.duplicated().sum())

    print("\nStatistical Summary:")
    print(data.describe(include="all").T)
