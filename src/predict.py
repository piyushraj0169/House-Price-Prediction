"""Reusable prediction module for house price inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.utils import format_indian_currency



BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"


def load_model(model_path: str | Path = MODEL_PATH):
    """Load a persisted house price model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def validate_inputs(area: float, bedrooms: int, bathrooms: int, age: int) -> None:
    """Validate prediction inputs."""
    if area <= 0:
        raise ValueError("Area must be greater than 0.")
    if bedrooms <= 0:
        raise ValueError("Bedrooms must be greater than 0.")
    if bathrooms <= 0:
        raise ValueError("Bathrooms must be greater than 0.")
    if age < 0:
        raise ValueError("Age cannot be negative.")


def predict_house_price(
    area: float,
    bedrooms: int,
    bathrooms: int,
    age: int,
    location: str,
    model_path: str | Path = MODEL_PATH,
) -> float:
    """Predict house price from area, bedrooms, bathrooms, age, and location."""
    validate_inputs(area, bedrooms, bathrooms, age)
    model = load_model(model_path)

    input_data = pd.DataFrame(
        [
            {
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Age": age,
                "Location": location,
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    return float(prediction)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    price = predict_house_price(area=1500, bedrooms=3, bathrooms=2, age=8, location="Delhi")
    print(f"Predicted House Price: {format_indian_currency(price)}")
