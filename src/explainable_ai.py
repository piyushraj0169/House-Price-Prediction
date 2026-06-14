"""Explainable AI (XAI) utility for local prediction explanations."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.config import DATA_PATH, TARGET_COLUMN
from src.preprocessing import clean_dataset
from src.predict import load_model


def explain_prediction(
    area: float,
    bedrooms: int,
    bathrooms: int,
    age: int,
    location: str,
    model_path: str | Path,
) -> dict[str, float | dict[str, float]]:
    """Calculate local feature contributions using the Marginal Contribution (LOO) method."""
    model = load_model(model_path)
    
    # Load and clean dataset to compute baselines (means/modes)
    data = pd.read_csv(DATA_PATH)
    cleaned_data = clean_dataset(data, target_column=TARGET_COLUMN)
    
    # Compute average baseline features
    mean_area = float(cleaned_data["Area"].mean())
    mean_bedrooms = float(cleaned_data["Bedrooms"].mean())
    mean_bathrooms = float(cleaned_data["Bathrooms"].mean())
    mean_age = float(cleaned_data["Age"].mean())
    mean_location = str(cleaned_data["Location"].mode().iloc[0])
    
    # Current input
    current_input = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": location
    }])
    
    # Baseline input (mean values)
    baseline_input = pd.DataFrame([{
        "Area": mean_area,
        "Bedrooms": mean_bedrooms,
        "Bathrooms": mean_bathrooms,
        "Age": mean_age,
        "Location": mean_location
    }])
    
    # Predictions
    current_pred = float(model.predict(current_input)[0])
    baseline_pred = float(model.predict(baseline_input)[0])
    
    # Leave-One-Out (LOO) marginal contributions
    # 1. Area contribution (varying Area from mean to current)
    input_varying_area = pd.DataFrame([{
        "Area": mean_area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": location
    }])
    contrib_area = current_pred - float(model.predict(input_varying_area)[0])
    
    # 2. Bedrooms contribution (varying Bedrooms from mean to current)
    input_varying_bedrooms = pd.DataFrame([{
        "Area": area,
        "Bedrooms": mean_bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": location
    }])
    contrib_bedrooms = current_pred - float(model.predict(input_varying_bedrooms)[0])
    
    # 3. Bathrooms contribution (varying Bathrooms from mean to current)
    input_varying_bathrooms = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": mean_bathrooms,
        "Age": age,
        "Location": location
    }])
    contrib_bathrooms = current_pred - float(model.predict(input_varying_bathrooms)[0])
    
    # 4. Age contribution (varying Age from mean to current)
    input_varying_age = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": mean_age,
        "Location": location
    }])
    contrib_age = current_pred - float(model.predict(input_varying_age)[0])
    
    # 5. Location contribution (varying Location from mode to current)
    input_varying_location = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": mean_location
    }])
    contrib_location = current_pred - float(model.predict(input_varying_location)[0])
    
    raw_contributions = {
        "Area": contrib_area,
        "Bedrooms": contrib_bedrooms,
        "Bathrooms": contrib_bathrooms,
        "Age": contrib_age,
        "Location": contrib_location
    }
    
    # Distribute the residual to match the exact difference from the baseline
    total_diff = current_pred - baseline_pred
    sum_raw = sum(abs(v) for v in raw_contributions.values())
    
    contributions = {}
    if sum_raw > 0:
        raw_sum = sum(raw_contributions.values())
        diff_residual = total_diff - raw_sum
        
        for k, v in raw_contributions.items():
            weight = abs(v) / sum_raw
            contributions[k] = v + (diff_residual * weight)
    else:
        for k in raw_contributions:
            contributions[k] = total_diff / 5.0
            
    return {
        "baseline_prediction": baseline_pred,
        "predicted_price": current_pred,
        "contributions": contributions,
        "feature_means": {
            "Area": mean_area,
            "Bedrooms": mean_bedrooms,
            "Bathrooms": mean_bathrooms,
            "Age": mean_age,
            "Location": mean_location
        }
    }
