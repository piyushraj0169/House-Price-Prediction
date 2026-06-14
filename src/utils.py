"""Shared utility functions for metrics, plotting, and filesystem handling."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / "matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_indian_currency(value: float | int, include_symbol: bool = True, decimals: int = 2) -> str:
    """Format a number according to the Indian numbering system (Lakhs, Crores)."""
    is_negative = value < 0
    value = abs(value)
    
    # Round value to specified decimals
    val_str = f"{value:.{decimals}f}"
    parts = val_str.split('.')
    int_part = parts[0]
    dec_part = parts[1] if len(parts) > 1 else ""
    
    if len(int_part) <= 3:
        formatted_int = int_part
    else:
        last_three = int_part[-3:]
        remaining = int_part[:-3]
        groups = []
        while len(remaining) > 0:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        groups.reverse()
        formatted_int = ",".join(groups) + "," + last_three
        
    res = formatted_int
    if decimals > 0:
        res = f"{formatted_int}.{dec_part}"
        
    if is_negative:
        res = "-" + res
        
    if include_symbol:
        res = "₹" + res
        
    return res



def generate_statistical_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return a complete statistical summary for the dataset."""
    return data.describe(include="all").T


def evaluate_regression_model(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse**0.5,
        "R2 Score": r2_score(y_true, y_pred),
    }


def save_eda_plots(data: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate and save EDA plots."""
    plots_dir = ensure_directory(output_dir)
    numeric_columns = data.select_dtypes(include="number").columns.tolist()

    sns.set_theme(style="whitegrid")

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, color="#2f6f73")
        plt.title(f"{column} Distribution")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{column.lower()}_histogram.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=data[column], color="#d99a3d")
        plt.title(f"{column} Boxplot")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{column.lower()}_boxplot.png")
        plt.close()

    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 7))
        correlation = data[numeric_columns].corr()
        sns.heatmap(correlation, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_heatmap.png")
        plt.close()

    if "Price" in data.columns:
        feature_columns = [column for column in numeric_columns if column != "Price"]
        for column in feature_columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=data, x=column, y="Price", color="#3b5b92")
            plt.title(f"{column} vs Price")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{column.lower()}_vs_price_scatter.png")
            plt.close()
