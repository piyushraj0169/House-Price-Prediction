"""Reusable exploratory data analysis visualizations."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / "matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.utils import format_indian_currency


def build_correlation_heatmap(data: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Build a Seaborn correlation heatmap from numeric dataset columns."""
    numeric_data = data.select_dtypes(include="number")
    if numeric_data.empty:
        raise ValueError("No numerical columns are available for correlation analysis.")

    correlation_matrix = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f172a")
    ax.set_facecolor("#111827")
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.7,
        linecolor="#334155",
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_title("Feature Correlation Heatmap", color="#f8fafc", fontsize=15, pad=14)
    ax.tick_params(axis="x", colors="#cbd5e1", rotation=35)
    ax.tick_params(axis="y", colors="#cbd5e1", rotation=0)

    colorbar = ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.ax.yaxis.label.set_color("#cbd5e1")
        colorbar.ax.tick_params(colors="#cbd5e1")

    fig.tight_layout()
    return fig, correlation_matrix


def build_price_distribution_plot(
    data: pd.DataFrame,
    target_column: str = "Price",
) -> tuple[plt.Figure, dict[str, float]]:
    """Build a house price histogram with KDE and summary statistics."""
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' is missing from the dataset.")

    prices = data[target_column].dropna()
    if prices.empty:
        raise ValueError("No house price values are available for distribution analysis.")

    statistics = {
        "mean": float(prices.mean()),
        "median": float(prices.median()),
        "skewness": float(prices.skew()),
    }

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f172a")
    ax.set_facecolor("#111827")
    sns.histplot(
        prices,
        kde=True,
        bins=14,
        color="#38bdf8",
        edgecolor="#0f172a",
        linewidth=0.8,
        alpha=0.75,
        ax=ax,
    )

    ax.axvline(
        statistics["mean"],
        color="#22c55e",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {format_indian_currency(statistics['mean'], decimals=0)}",
    )
    ax.axvline(
        statistics["median"],
        color="#facc15",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {format_indian_currency(statistics['median'], decimals=0)}",
    )

    ax.set_title("House Price Distribution", color="#f8fafc", fontsize=15, pad=14)
    ax.set_xlabel("House Price", color="#cbd5e1")
    ax.set_ylabel("Number of Houses", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, axis="y", color="#334155", alpha=0.6)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e5e7eb")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    fig.tight_layout()
    return fig, statistics
