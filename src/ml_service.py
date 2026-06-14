"""Reusable ML service functions for the Streamlit app and scripts."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CONFIDENCE_MULTIPLIER,
    DATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_dataset
from src.predict import load_model, predict_house_price
from src.preprocessing import clean_dataset, separate_features_target
from src.utils import evaluate_regression_model


def load_cleaned_housing_data() -> pd.DataFrame:
    """Load and clean the configured housing dataset."""
    data = load_dataset(DATA_PATH)
    return clean_dataset(data, target_column=TARGET_COLUMN)


def get_holdout_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the project-standard train/test split."""
    cleaned_data = load_cleaned_housing_data()
    features, target = separate_features_target(
        cleaned_data,
        target_column=TARGET_COLUMN,
    )
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def evaluate_saved_model() -> dict[str, float]:
    """Evaluate the persisted model on the project holdout set."""
    _, x_test, _, y_test = get_holdout_split()
    model = load_model(MODEL_PATH)
    predictions = pd.Series(model.predict(x_test), index=y_test.index)
    return evaluate_regression_model(y_test, predictions)


def get_actual_predicted_prices() -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Return actual and predicted prices for the project holdout set."""
    _, x_test, _, y_test = get_holdout_split()
    model = load_model(MODEL_PATH)
    predictions = pd.Series(model.predict(x_test), index=y_test.index)
    metrics = evaluate_regression_model(y_test, predictions)
    return y_test, predictions, metrics


def calculate_residual_margin(
    confidence_multiplier: float = CONFIDENCE_MULTIPLIER,
) -> float:
    """Calculate a prediction interval margin from holdout residuals."""
    actual_prices, predicted_prices, _ = get_actual_predicted_prices()
    residuals = actual_prices - predicted_prices
    return float(confidence_multiplier * residuals.std(ddof=1))


def predict_with_interval(
    area: float,
    bedrooms: int,
    bathrooms: int,
    age: int,
    location: str,
) -> dict[str, float]:
    """Predict a house price and attach residual-based lower and upper bounds."""
    predicted_price = predict_house_price(area, bedrooms, bathrooms, age, location, MODEL_PATH)
    margin = calculate_residual_margin()
    return {
        "prediction": predicted_price,
        "lower_bound": max(0.0, predicted_price - margin),
        "upper_bound": predicted_price + margin,
        "margin": margin,
    }


def calculate_dataset_statistics() -> dict[str, float | int]:
    """Calculate key dataset summary statistics."""
    cleaned_data = load_cleaned_housing_data()
    feature_count = len([column for column in cleaned_data.columns if column != TARGET_COLUMN])

    return {
        "Total records": int(len(cleaned_data)),
        "Number of features": int(feature_count),
        "Average house price": float(cleaned_data[TARGET_COLUMN].mean()),
        "Average area": float(cleaned_data["Area"].mean()),
        "Average bedrooms": float(cleaned_data["Bedrooms"].mean()),
        "Average bathrooms": float(cleaned_data["Bathrooms"].mean()),
    }


def calculate_sidebar_summary() -> dict[str, Any]:
    """Calculate dataset and model summary details for the Streamlit sidebar."""
    cleaned_data = load_cleaned_housing_data()
    x_train, x_test, _, _ = get_holdout_split()
    metrics = evaluate_saved_model()
    model = load_model(MODEL_PATH)
    estimator = model.named_steps.get("model")
    model_name = estimator.__class__.__name__ if estimator is not None else "Unknown"

    return {
        "rows": int(cleaned_data.shape[0]),
        "columns": int(cleaned_data.shape[1]),
        "target_variable": TARGET_COLUMN,
        "model_name": model_name,
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "r2_score": float(metrics["R2 Score"]),
    }
