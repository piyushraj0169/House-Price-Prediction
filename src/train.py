"""Train and persist the best house price regression model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    SUMMARY_PATH,
    TARGET_COLUMN,
)
from src.data_loader import display_dataset_info, load_dataset
from src.model_evaluation import evaluate_house_price_models
from src.preprocessing import (
    clean_dataset,
)
from src.utils import (
    ensure_directory,
    generate_statistical_summary,
    save_eda_plots,
)


def train_models(data_path: str | Path = DATA_PATH) -> tuple[Pipeline, pd.DataFrame]:
    """Train candidate models, compare metrics, save the best pipeline, and return it."""
    data = load_dataset(data_path)
    display_dataset_info(data)

    cleaned_data = clean_dataset(data, target_column=TARGET_COLUMN)
    ensure_directory(PLOTS_DIR)
    save_eda_plots(cleaned_data, PLOTS_DIR)
    generate_statistical_summary(cleaned_data).to_csv(SUMMARY_PATH)

    comparison_table, trained_pipelines, best_model_name = evaluate_house_price_models(
        data_path=data_path,
        target_column=TARGET_COLUMN,
    )

    print("\nModel Comparison:")
    print(comparison_table.to_string(index=False))
    comparison_table.to_csv(MODEL_COMPARISON_PATH, index=False)

    best_pipeline = trained_pipelines[best_model_name]

    ensure_directory(MODEL_PATH.parent)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nBest Model: {best_model_name}")
    print(f"Saved model to: {MODEL_PATH}")

    return best_pipeline, comparison_table


if __name__ == "__main__":
    train_models()
