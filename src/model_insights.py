"""Model insight utilities for trained house price pipelines."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def _clean_feature_name(feature_name: str) -> str:
    """Convert pipeline feature names into user-facing feature labels."""
    return feature_name.split("__", maxsplit=1)[-1].replace("_", " ").title()


def calculate_random_forest_feature_importance(
    model_path: str | Path,
) -> pd.DataFrame:
    """Calculate feature importance percentages from a Random Forest pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python -m src.train` first."
        )

    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError("The saved model must be a scikit-learn Pipeline.")

    regressor = model.named_steps.get("model")
    preprocessing = model.named_steps.get("preprocessing")

    if not isinstance(regressor, RandomForestRegressor):
        raise TypeError(
            "Feature importance requires a Random Forest Regressor model."
        )
    if preprocessing is None:
        raise TypeError("The saved pipeline is missing a preprocessing step.")

    feature_names = preprocessing.get_feature_names_out()
    importances = regressor.feature_importances_

    importance_table = pd.DataFrame(
        {
            "Feature": [_clean_feature_name(name) for name in feature_names],
            "Importance": importances,
        }
    )
    importance_table["Importance (%)"] = importance_table["Importance"] * 100

    return importance_table.sort_values(
        by="Importance (%)",
        ascending=False,
    ).reset_index(drop=True)
