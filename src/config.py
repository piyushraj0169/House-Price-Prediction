"""Central project configuration."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "housing.csv"
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
SUMMARY_PATH = OUTPUT_DIR / "statistical_summary.csv"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"

TARGET_COLUMN = "Price"
FEATURE_COLUMNS = ["Area", "Bedrooms", "Bathrooms", "Age", "Location"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CONFIDENCE_MULTIPLIER = 1.96
