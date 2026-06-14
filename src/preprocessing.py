"""Data cleaning, feature engineering, and preprocessing pipeline helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated rows from the dataset."""
    return data.drop_duplicates().reset_index(drop=True)


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using median for numeric and mode for categorical data."""
    cleaned = data.copy()
    numeric_columns = cleaned.select_dtypes(include=np.number).columns
    categorical_columns = cleaned.select_dtypes(exclude=np.number).columns

    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in categorical_columns:
        mode_value = cleaned[column].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned


def cap_outliers_iqr(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Cap numeric outliers using the IQR method."""
    capped = data.copy()
    target_columns = columns or list(capped.select_dtypes(include=np.number).columns)

    for column in target_columns:
        q1 = capped[column].quantile(0.25)
        q3 = capped[column].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        capped[column] = capped[column].clip(lower=lower_bound, upper=upper_bound)

    return capped


def clean_dataset(data: pd.DataFrame, target_column: str = "Price") -> pd.DataFrame:
    """Apply duplicate removal, missing-value handling, and outlier capping."""
    cleaned = remove_duplicates(data)
    cleaned = handle_missing_values(cleaned)
    numeric_features = [
        column
        for column in cleaned.select_dtypes(include=np.number).columns
        if column != target_column
    ]
    return cap_outliers_iqr(cleaned, columns=numeric_features)


def separate_features_target(
    data: pd.DataFrame,
    target_column: str = "Price",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target variable."""
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' is missing from the dataset.")

    features = data.drop(columns=[target_column])
    target = data[target_column]
    return features, target


def build_preprocessing_pipeline(features: pd.DataFrame) -> ColumnTransformer:
    """Create a reusable preprocessing pipeline for numeric and categorical columns."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
