"""Model training and comparison utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import CV_FOLDS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.data_loader import load_dataset
from src.preprocessing import (
    build_preprocessing_pipeline,
    clean_dataset,
    separate_features_target,
)
from src.utils import evaluate_regression_model


def get_regression_models() -> dict[str, Any]:
    """Return the regression models used for house price comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_house_price_models(
    data_path: str | Path,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Pipeline], str]:
    """Train candidate models and return their holdout evaluation results."""
    data = load_dataset(data_path)
    cleaned_data = clean_dataset(data, target_column=target_column)
    features, target = separate_features_target(
        cleaned_data,
        target_column=target_column,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    results: list[dict[str, float | str]] = []
    trained_pipelines: dict[str, Pipeline] = {}
    cross_validator = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for model_name, model in get_regression_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessing_pipeline(features)),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = evaluate_regression_model(y_test, predictions)
        cv_scores = cross_val_score(
            pipeline,
            features,
            target,
            cv=cross_validator,
            scoring="neg_root_mean_squared_error",
        )

        results.append(
            {
                "Model Name": model_name,
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "R2 Score": metrics["R2 Score"],
                "CV RMSE Mean": float((-cv_scores).mean()),
                "CV RMSE Std": float((-cv_scores).std()),
            }
        )
        trained_pipelines[model_name] = pipeline

    comparison_table = pd.DataFrame(results).sort_values(
        by=["RMSE", "MAE"],
        ascending=True,
    )
    best_model_name = str(comparison_table.iloc[0]["Model Name"])

    return comparison_table, trained_pipelines, best_model_name
