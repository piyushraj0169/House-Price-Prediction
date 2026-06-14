"""Streamlit web application for house price prediction."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "output" / "matplotlib_cache"))

import matplotlib.pyplot as plt

from src.data_loader import load_dataset
from src.config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN
from src.utils import format_indian_currency
from src.pdf_generator import generate_prediction_pdf
from src.explainable_ai import explain_prediction
from src.real_estate_insights import calculate_real_estate_insights
from src.eda_visualizations import (
    build_correlation_heatmap,
    build_price_distribution_plot,
)
from src.model_insights import calculate_random_forest_feature_importance
from src.model_evaluation import evaluate_house_price_models
from src.ml_service import (
    calculate_dataset_statistics as service_calculate_dataset_statistics,
    calculate_residual_margin as service_calculate_residual_margin,
    calculate_sidebar_summary as service_calculate_sidebar_summary,
    evaluate_saved_model,
    get_actual_predicted_prices,
    load_cleaned_housing_data as service_load_cleaned_housing_data,
    predict_with_interval,
)


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="House",
    layout="centered",
)

@st.cache_data(show_spinner=False)
def calculate_model_metrics() -> dict[str, float]:
    """Calculate holdout metrics from the persisted trained model."""
    return evaluate_saved_model()


@st.cache_data(show_spinner=False)
def calculate_actual_predicted_prices() -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Return holdout actual and predicted prices for the saved trained model."""
    return get_actual_predicted_prices()


@st.cache_data(show_spinner=False)
def calculate_residual_margin(confidence_multiplier: float = 1.96) -> float:
    """Calculate a prediction range margin from holdout model residuals."""
    return service_calculate_residual_margin(confidence_multiplier)


@st.cache_data(show_spinner=False)
def load_cleaned_housing_data() -> pd.DataFrame:
    """Load and clean the housing dataset for EDA sections."""
    return service_load_cleaned_housing_data()


@st.cache_data(show_spinner=False)
def calculate_dataset_statistics() -> dict[str, float | int]:
    """Calculate key dataset summary statistics for dashboard cards."""
    return service_calculate_dataset_statistics()


@st.cache_data(show_spinner=False)
def calculate_sidebar_summary() -> dict[str, str | int | float]:
    """Calculate dataset and model details for the Streamlit sidebar."""
    return service_calculate_sidebar_summary()


@st.cache_data(show_spinner=False)
def calculate_model_comparison() -> tuple[pd.DataFrame, str]:
    """Train and compare regression models for the comparison dashboard."""
    comparison_table, _, best_model_name = evaluate_house_price_models(DATA_PATH)
    return comparison_table, best_model_name


@st.cache_data(show_spinner=False)
def calculate_feature_importance() -> pd.DataFrame:
    """Calculate Random Forest feature importance from the trained model."""
    return calculate_random_forest_feature_importance(MODEL_PATH)


def get_metric_status(metric_name: str, value: float) -> tuple[str, str]:
    """Return display status and CSS color class for a model metric."""
    if metric_name == "R2 Score":
        if value >= 0.90:
            return "Good", "good"
        if value >= 0.70:
            return "Moderate", "moderate"
        return "Poor", "poor"

    if metric_name == "MAE":
        if value <= 250000:
            return "Good", "good"
        if value <= 600000:
            return "Moderate", "moderate"
        return "Poor", "poor"

    if metric_name == "MSE":
        if value <= 100_000_000_000:
            return "Good", "good"
        if value <= 400_000_000_000:
            return "Moderate", "moderate"
        return "Poor", "poor"

    if value <= 350000:
        return "Good", "good"
    if value <= 650000:
        return "Moderate", "moderate"
    return "Poor", "poor"


def format_metric_value(metric_name: str, value: float) -> str:
    """Format metric values for the dashboard cards."""
    if metric_name == "R2 Score":
        return f"{value:.3f}"
    return format_indian_currency(value, include_symbol=True, decimals=2)


def render_metric_card(
    title: str,
    value: float,
    explanation: str,
) -> None:
    """Render a styled model-performance metric card."""
    status, status_class = get_metric_status(title, value)
    st.markdown(
        f"""
        <div class="metric-card {status_class}">
            <div class="metric-card-top">
                <span class="metric-title">{title}</span>
                <span class="metric-status">{status}</span>
            </div>
            <div class="metric-value">{format_metric_value(title, value)}</div>
            <p class="metric-explanation">{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_stat_value(stat_name: str, value: float | int) -> str:
    """Format dataset statistic values for display cards."""
    if stat_name in {"Average house price", "Mean price", "Median price"}:
        return format_indian_currency(value, include_symbol=True, decimals=2)
    if stat_name == "Average area":
        return f"{value:,.0f} sq ft"
    if stat_name in {"Average bedrooms", "Average bathrooms"}:
        return f"{value:.1f}"
    if stat_name == "Skewness":
        return f"{value:.2f}"
    return f"{value:,}"


def render_stat_card(title: str, value: float | int, icon: str) -> None:
    """Render a styled dataset statistic card."""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div>
                <div class="stat-title">{title}</div>
                <div class="stat-value">{format_stat_value(title, value)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(predicted_price: float, lower_bound: float, upper_bound: float) -> None:
    """Render a professional prediction result with confidence range."""
    formatted_pred = format_indian_currency(predicted_price, decimals=0)
    formatted_range = f"{format_indian_currency(lower_bound, decimals=0)} - {format_indian_currency(upper_bound, decimals=0)}"
    formatted_lower = format_indian_currency(lower_bound, decimals=0)
    formatted_upper = format_indian_currency(upper_bound, decimals=0)
    
    st.markdown(
        f"""<div class="prediction-result-card">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span class="prediction-label">Predicted Price</span>
<span class="metric-status good" style="font-size: 0.75rem; padding: 0.22rem 0.55rem; background: #22c55e; color: #0f172a; border-radius: 999px; font-weight: 800; text-transform: uppercase;">95% Confidence</span>
</div>
<div class="prediction-value" style="margin-top: 0;">{formatted_pred}</div>
<div class="prediction-range-label" style="margin-top: 1rem;">Confidence Interval (95%)</div>
<div class="prediction-range">{formatted_range}</div>
<div class="visual-indicator-container">
<div class="indicator-track">
<div class="indicator-fill" style="left: 0%; width: 100%;"></div>
<div class="indicator-point" style="left: 50%;"></div>
</div>
<div class="indicator-labels">
<span class="lower-label">{formatted_lower}<br><small style="font-size: 0.7rem; font-weight: normal; color: #64748b;">Lower Bound</small></span>
<span class="predicted-label">{formatted_pred}<br><small style="font-size: 0.7rem; font-weight: normal; color: #22c55e;">Predicted</small></span>
<span class="upper-label">{formatted_upper}<br><small style="font-size: 0.7rem; font-weight: normal; color: #64748b;">Upper Bound</small></span>
</div>
</div>
<div class="prediction-bounds" style="margin-top: 1.5rem;">
<div>
<span>Lower Bound</span>
<strong>{formatted_lower}</strong>
</div>
<div>
<span>Upper Bound</span>
<strong>{formatted_upper}</strong>
</div>
</div>
<p style="margin-top: 1.25rem;">
<strong>What does this mean?</strong><br>
Based on historical data and model performance, we are <strong>95% confident</strong> that the actual selling price for a property with these details will fall between <strong>{formatted_lower}</strong> and <strong>{formatted_upper}</strong>. 
The most likely point estimate is <strong>{formatted_pred}</strong>. This range accounts for standard market variations and errors observed in similar properties.
</p>
</div>""",
        unsafe_allow_html=True,
    )


def render_explanation_section(explanation: dict) -> None:
    """Render the local prediction explanation charts and text."""
    st.markdown("### Why was this price predicted?")
    st.caption(
        "Feature contributions show how each parameter moved the prediction "
        "away from the baseline (average house price)."
    )
    
    baseline = explanation["baseline_prediction"]
    feature_means = explanation["feature_means"]
    contributions = explanation["contributions"]
    
    # Create DataFrame for Altair chart
    chart_rows = []
    for feature, val in contributions.items():
        direction = "Positive (+)" if val >= 0 else "Negative (-)"
        chart_rows.append({
            "Feature": feature,
            "Contribution (₹)": val,
            "Formatted Contribution": format_indian_currency(val, include_symbol=True, decimals=0) if val >= 0 else f"-{format_indian_currency(abs(val), include_symbol=True, decimals=0)}",
            "Direction": direction
        })
    df_chart = pd.DataFrame(chart_rows)
    
    # Render explanation text cards
    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Baseline Price (Average House)</div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-top: 0.25rem;">{format_indian_currency(baseline, decimals=0)}</div>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem; margin-bottom: 0;">
                The average baseline represents the predicted price for a standard property in the dataset having:
                Area: {feature_means['Area']:,.0f} sq ft, Bedrooms: {feature_means['Bedrooms']:.1f}, Bathrooms: {feature_means['Bathrooms']:.1f}, Age: {feature_means['Age']:.1f} years.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Altair chart for local contributions
    chart = (
        alt.Chart(df_chart)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("Feature:N", sort=None, title="Property Parameter"),
            x=alt.X("Contribution (₹):Q", title="Price Impact (₹)"),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Positive (+)", "Negative (-)"],
                    range=["#22c55e", "#ef4444"]
                ),
                legend=alt.Legend(title="Impact Direction")
            ),
            tooltip=[
                alt.Tooltip("Feature:N", title="Parameter"),
                alt.Tooltip("Formatted Contribution:N", title="Impact"),
            ]
        )
        .properties(height=180)
        .configure_axis(labelColor="#cbd5e1", titleColor="#e5e7eb", gridColor="#334155")
        .configure_view(strokeWidth=0)
    )
    
    st.altair_chart(chart, width="stretch")
    
    # Bullet points explanation
    st.markdown("**Detail Breakdown:**")
    for row in chart_rows:
        sign = "+" if row["Contribution (₹)"] >= 0 else "-"
        amt_str = format_indian_currency(abs(row["Contribution (₹)"]), include_symbol=True, decimals=0)
        color = "#22c55e" if row["Contribution (₹)"] >= 0 else "#ef4444"
        
        st.markdown(
            f"- **{row['Feature']} Impact**: "
            f"<span style='color: {color}; font-weight: bold;'>{sign} {amt_str}</span>",
            unsafe_allow_html=True
        )


def render_insights_section(insights: dict) -> None:
    """Render the real estate market insights cards and commentary."""
    st.markdown("### Real Estate Market Insights")
    st.caption(
        "Market positioning and investment analysis calculated from predicted property price "
        "and active market listings."
    )
    
    category = insights["category"]
    color = insights["color"]
    top_pct = insights["top_pct"]
    commentary = insights["commentary"]
    
    first_col, second_col = st.columns(2)
    
    with first_col:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #334155; border-left: 5px solid {color}; border-radius: 8px; padding: 1.1rem; min-height: 120px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Price Category</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {color}; margin-top: 0.35rem;">{category}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.35rem;">Property valuation tier assignment</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with second_col:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #334155; border-left: 5px solid #a855f7; border-radius: 8px; padding: 1.1rem; min-height: 120px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Market Position</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: #a855f7; margin-top: 0.35rem;">Top {top_pct:.0f}%</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.35rem;">Compared to active market listings</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem; margin-top: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 0.5rem;">Investment Commentary</div>
            <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 0;">
                {commentary}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar() -> None:
    """Render the application sidebar with dataset, model, and stack details."""
    st.sidebar.title("Project Summary")

    try:
        summary = calculate_sidebar_summary()
    except Exception as exc:
        st.sidebar.error("Unable to load sidebar details.")
        st.sidebar.exception(exc)
        return

    with st.sidebar.expander("Dataset Information", expanded=True):
        st.write(f"**Number of rows:** {summary['rows']}")
        st.write(f"**Number of columns:** {summary['columns']}")
        st.write(f"**Target variable:** {summary['target_variable']}")

    with st.sidebar.expander("Model Information", expanded=True):
        st.write(f"**Model name:** {summary['model_name']}")
        st.write(f"**Training samples:** {summary['training_samples']}")
        st.write(f"**Test samples:** {summary['test_samples']}")
        st.write(f"**R2 score:** {summary['r2_score']:.3f}")

    with st.sidebar.expander("Technology Stack", expanded=False):
        st.write("Python")
        st.write("Pandas")
        st.write("NumPy")
        st.write("Scikit-Learn")
        st.write("Streamlit")


def render_dataset_statistics() -> None:
    """Display professional dataset summary statistics."""
    st.markdown("## Dataset Statistics")
    st.caption("Key summary values calculated from the cleaned housing dataset.")

    try:
        statistics = calculate_dataset_statistics()
    except Exception as exc:
        st.error("Unable to calculate dataset statistics.")
        st.exception(exc)
        return

    stat_cards = [
        ("Total records", statistics["Total records"], "&#35;"),
        ("Number of features", statistics["Number of features"], "&#9881;"),
        ("Average house price", statistics["Average house price"], "₹"),
        ("Average area", statistics["Average area"], "&#9633;"),
        ("Average bedrooms", statistics["Average bedrooms"], "BD"),
        ("Average bathrooms", statistics["Average bathrooms"], "BA"),
    ]

    for start in range(0, len(stat_cards), 3):
        columns = st.columns(3)
        for column, (title, value, icon) in zip(columns, stat_cards[start : start + 3]):
            with column:
                render_stat_card(title, value, icon)


def render_performance_dashboard() -> None:
    """Display the model performance dashboard."""
    st.markdown("## Model Performance Dashboard")
    st.caption("Metrics are calculated from the saved trained model on the holdout set.")

    try:
        metrics = calculate_model_metrics()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error("Unable to calculate model performance metrics.")
        st.exception(exc)
        return

    explanations = {
        "R2 Score": "Shows how much price variation the model explains. Higher is better.",
        "MAE": "Average absolute prediction error in Rupees (₹). Lower is better.",
        "MSE": "Average squared prediction error in Rupees (₹) squared. Lower is better.",
        "RMSE": "Typical prediction error in Rupees (₹), in the same unit as price. Lower is better.",
    }

    first_row = st.columns(2)
    with first_row[0]:
        render_metric_card("R2 Score", metrics["R2 Score"], explanations["R2 Score"])
    with first_row[1]:
        render_metric_card("MAE", metrics["MAE"], explanations["MAE"])

    second_row = st.columns(2)
    with second_row[0]:
        render_metric_card("MSE", metrics["MSE"], explanations["MSE"])
    with second_row[1]:
        render_metric_card("RMSE", metrics["RMSE"], explanations["RMSE"])


def format_comparison_table(comparison_table: pd.DataFrame) -> pd.DataFrame:
    """Format model comparison results for display."""
    display_table = comparison_table.copy()
    display_table["MAE"] = display_table["MAE"].map(lambda value: format_indian_currency(value, include_symbol=True, decimals=2))
    display_table["RMSE"] = display_table["RMSE"].map(lambda value: format_indian_currency(value, include_symbol=True, decimals=2))
    display_table["R2 Score"] = display_table["R2 Score"].map(
        lambda value: f"{value:.3f}"
    )
    return display_table


def render_model_comparison() -> None:
    """Display model comparison table, chart, and best-model summary."""
    st.markdown("## Model Comparison")
    st.caption(
        "Linear Regression, Decision Tree, and Random Forest are retrained and "
        "evaluated on the same holdout split."
    )

    try:
        comparison_table, best_model_name = calculate_model_comparison()
    except Exception as exc:
        st.error("Unable to calculate model comparison results.")
        st.exception(exc)
        return

    best_row = comparison_table.loc[
        comparison_table["Model Name"] == best_model_name
    ].iloc[0]

    formatted_rmse = format_indian_currency(best_row["RMSE"], include_symbol=True, decimals=2)
    st.markdown(
        f"""
        <div class="best-model-card">
            <span class="best-model-label">Best Model</span>
            <div class="best-model-name">{best_model_name}</div>
            <p>
                Selected by lowest RMSE, with MAE used as the tie breaker.
                Current RMSE: {formatted_rmse}.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_table = comparison_table.copy()
    display_table.insert(
        1,
        "Best Model",
        display_table["Model Name"].eq(best_model_name).map({True: "Yes", False: ""}),
    )

    st.dataframe(
        format_comparison_table(display_table),
        hide_index=True,
        width="stretch",
    )

    selected_metric = st.selectbox(
        "Chart metric",
        options=["RMSE", "MAE", "R2 Score"],
        index=0,
    )

    chart_data = comparison_table.copy()
    chart_data["Is Best Model"] = chart_data["Model Name"].eq(best_model_name)

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Model Name:N", sort=None, title="Model"),
            y=alt.Y(f"{selected_metric}:Q", title=selected_metric),
            color=alt.Color(
                "Is Best Model:N",
                scale=alt.Scale(
                    domain=[True, False],
                    range=["#22c55e", "#64748b"],
                ),
                legend=alt.Legend(title="Best Model"),
            ),
            tooltip=[
                alt.Tooltip("Model Name:N", title="Model"),
                alt.Tooltip("MAE:Q", title="MAE (₹)", format=",.2f"),
                alt.Tooltip("RMSE:Q", title="RMSE (₹)", format=",.2f"),
                alt.Tooltip("R2 Score:Q", title="R2 Score", format=".3f"),
            ],
        )
        .properties(height=330)
        .configure_axis(labelColor="#cbd5e1", titleColor="#e5e7eb", gridColor="#334155")
        .configure_view(strokeWidth=0)
    )

    st.altair_chart(chart, width="stretch")


def get_quality_interpretation(r2_score: float, rmse: float) -> tuple[str, str]:
    """Return a short quality label and interpretation for model fit."""
    formatted_rmse = format_indian_currency(rmse, include_symbol=True, decimals=0)
    if r2_score >= 0.90:
        return (
            "Strong model fit",
            f"The model explains {r2_score:.1%} of price variation, and the "
            f"typical prediction error is about {formatted_rmse}.",
        )
    if r2_score >= 0.70:
        return (
            "Moderate model fit",
            f"The model captures the main pricing pattern, but the typical "
            f"prediction error is still about {formatted_rmse}.",
        )
    return (
        "Weak model fit",
        f"The predictions are not closely aligned with real prices yet. The "
        f"typical prediction error is about {formatted_rmse}.",
    )


def build_actual_vs_predicted_chart(
    actual_prices: pd.Series,
    predicted_prices: pd.Series,
) -> plt.Figure:
    """Build an Actual vs Predicted prices Matplotlib figure."""
    min_price = min(actual_prices.min(), predicted_prices.min())
    max_price = max(actual_prices.max(), predicted_prices.max())

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f172a")
    ax.set_facecolor("#111827")
    ax.scatter(
        actual_prices,
        predicted_prices,
        color="#38bdf8",
        edgecolor="#e0f2fe",
        linewidth=0.7,
        alpha=0.9,
        s=72,
        label="Predictions",
    )
    ax.plot(
        [min_price, max_price],
        [min_price, max_price],
        color="#22c55e",
        linewidth=2.4,
        linestyle="--",
        label="Ideal prediction",
    )

    ax.set_title("Actual vs Predicted Prices", color="#f8fafc", fontsize=15, pad=14)
    ax.set_xlabel("Actual Prices", color="#cbd5e1")
    ax.set_ylabel("Predicted Prices", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, color="#334155", alpha=0.6)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e5e7eb")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    fig.tight_layout()
    return fig


def render_actual_vs_predicted_section() -> None:
    """Display the Actual vs Predicted model visualization section."""
    st.markdown("## Actual vs Predicted Prices")
    st.caption("This chart compares real holdout prices with model predictions.")

    try:
        actual_prices, predicted_prices, metrics = calculate_actual_predicted_prices()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error("Unable to generate the Actual vs Predicted chart.")
        st.exception(exc)
        return

    fig = build_actual_vs_predicted_chart(actual_prices, predicted_prices)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown(
        """
        The X-axis shows the actual house prices from the holdout data, while the
        Y-axis shows the prices predicted by the trained model. Points close to
        the green dashed line are accurate predictions. Points above the line are
        overpredictions, and points below the line are underpredictions.
        """
    )

    quality_label, quality_text = get_quality_interpretation(
        metrics["R2 Score"],
        metrics["RMSE"],
    )
    st.info(f"{quality_label}: {quality_text}")


def render_eda_section() -> None:
    """Display exploratory data analysis visualizations."""
    st.markdown("## Exploratory Data Analysis")
    st.caption("Correlation heatmap for numerical dataset columns.")

    try:
        cleaned_data = load_cleaned_housing_data()
        fig, _ = build_correlation_heatmap(cleaned_data)
    except ValueError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error("Unable to generate the EDA heatmap.")
        st.exception(exc)
        return

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown(
        """
        The heatmap shows how strongly numerical features move together.
        A value close to `1.00` indicates a strong positive correlation, meaning
        both variables tend to increase together. A value close to `-1.00`
        indicates a strong negative correlation, meaning one variable tends to
        decrease as the other increases. Values near `0.00` indicate weak
        correlation, meaning there is little linear relationship.
        """
    )


def build_feature_importance_chart(importance_table: pd.DataFrame) -> plt.Figure:
    """Build a horizontal feature importance bar chart."""
    plot_data = importance_table.sort_values("Importance (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f172a")
    ax.set_facecolor("#111827")
    bars = ax.barh(
        plot_data["Feature"],
        plot_data["Importance (%)"],
        color="#38bdf8",
        edgecolor="#e0f2fe",
        linewidth=0.7,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            color="#e5e7eb",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Random Forest Feature Importance", color="#f8fafc", fontsize=15, pad=14)
    ax.set_xlabel("Importance (%)", color="#cbd5e1")
    ax.set_ylabel("Feature", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, axis="x", color="#334155", alpha=0.6)
    ax.set_xlim(0, max(plot_data["Importance (%)"].max() * 1.18, 5))

    for spine in ax.spines.values():
        spine.set_color("#334155")

    fig.tight_layout()
    return fig


def render_feature_importance_section() -> None:
    """Display Random Forest feature importance insights."""
    st.markdown("## Feature Importance")
    st.caption(
        "Feature rankings are calculated from the trained Random Forest Regressor."
    )

    try:
        importance_table = calculate_feature_importance()
    except Exception as exc:
        st.error("Unable to calculate feature importance.")
        st.exception(exc)
        return

    fig = build_feature_importance_chart(importance_table)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    top_feature = importance_table.iloc[0]
    st.markdown(
        f"""
        The model relies most heavily on **{top_feature["Feature"]}**, which
        contributes **{top_feature["Importance (%)"]:.1f}%** of the Random
        Forest's split-based importance. Higher-ranked features have more
        influence on the model's price predictions.
        """
    )

    st.markdown("**Ranked insights**")
    for index, row in importance_table.iterrows():
        st.markdown(
            f"{index + 1}. **{row['Feature']}**: "
            f"{row['Importance (%)']:.1f}% importance"
        )


def interpret_skewness(skewness: float) -> str:
    """Return a short interpretation of distribution skewness."""
    if skewness > 0.5:
        return (
            "The distribution is right-skewed, meaning most houses are priced "
            "below the upper-end values and a smaller number of expensive houses "
            "pull the tail to the right."
        )
    if skewness < -0.5:
        return (
            "The distribution is left-skewed, meaning higher-priced houses are "
            "more common and lower-priced houses create a tail to the left."
        )
    return (
        "The distribution is fairly balanced, meaning prices are spread with "
        "limited skew around the center."
    )


def render_data_distribution_section() -> None:
    """Display the house price distribution chart and summary interpretation."""
    st.markdown("## Data Distribution")
    st.caption("House price histogram with KDE curve and central tendency markers.")

    try:
        cleaned_data = load_cleaned_housing_data()
        fig, statistics = build_price_distribution_plot(
            cleaned_data,
            target_column=TARGET_COLUMN,
        )
    except Exception as exc:
        st.error("Unable to generate the house price distribution chart.")
        st.exception(exc)
        return

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    columns = st.columns(3)
    with columns[0]:
        render_stat_card("Mean price", statistics["mean"], "&#8721;")
    with columns[1]:
        render_stat_card("Median price", statistics["median"], "M")
    with columns[2]:
        render_stat_card("Skewness", statistics["skewness"], "S")

    st.markdown(
        f"""
        The histogram shows how house prices are distributed across the dataset,
        while the KDE curve smooths the same pattern to make the overall shape
        easier to read. The **mean** is the arithmetic average price, the
        **median** is the middle price, and **skewness** describes whether the
        distribution has a longer tail on one side.

        **Modeling insight:** {interpret_skewness(statistics["skewness"])}
        """
    )


st.markdown(
    """
    <style>
        .stApp {
            background: #0f172a;
            color: #e5e7eb;
        }

        .metric-card {
            min-height: 190px;
            padding: 1.15rem;
            border: 1px solid #334155;
            border-radius: 8px;
            background: #111827;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
            margin-bottom: 1rem;
        }

        .metric-card.good {
            border-left: 6px solid #22c55e;
        }

        .metric-card.moderate {
            border-left: 6px solid #facc15;
        }

        .metric-card.poor {
            border-left: 6px solid #ef4444;
        }

        .metric-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .metric-title {
            color: #cbd5e1;
            font-size: 0.93rem;
            font-weight: 700;
        }

        .metric-status {
            border-radius: 999px;
            color: #0f172a;
            font-size: 0.72rem;
            font-weight: 800;
            padding: 0.22rem 0.55rem;
            text-transform: uppercase;
        }

        .good .metric-status {
            background: #22c55e;
        }

        .moderate .metric-status {
            background: #facc15;
        }

        .poor .metric-status {
            background: #ef4444;
            color: #ffffff;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.2;
            margin-top: 0.8rem;
            overflow-wrap: anywhere;
        }

        .metric-explanation {
            color: #94a3b8;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 0;
            margin-top: 0.8rem;
        }

        .stat-card {
            min-height: 124px;
            display: flex;
            align-items: center;
            gap: 0.95rem;
            border: 1px solid #334155;
            border-radius: 8px;
            background: #111827;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
            margin-bottom: 1rem;
            padding: 1.05rem;
        }

        .stat-icon {
            min-width: 44px;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #38bdf8;
            border-radius: 8px;
            background: #0f2742;
            color: #7dd3fc;
            font-size: 1.05rem;
            font-weight: 800;
        }

        .stat-title {
            color: #94a3b8;
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1.3;
        }

        .stat-value {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 800;
            line-height: 1.25;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }

        .prediction-result-card {
            border: 1px solid #22c55e;
            border-left: 6px solid #22c55e;
            border-radius: 8px;
            background: #111827;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
            margin: 1rem 0;
            padding: 1.25rem;
        }

        .prediction-label,
        .prediction-range-label {
            color: #94a3b8;
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .prediction-value {
            color: #f8fafc;
            font-size: 2.35rem;
            font-weight: 900;
            line-height: 1.15;
            margin: 0.35rem 0 1rem;
            overflow-wrap: anywhere;
        }

        .prediction-range {
            color: #7dd3fc;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.35rem;
            overflow-wrap: anywhere;
        }

        .prediction-bounds {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-top: 1rem;
        }

        .prediction-bounds div {
            border: 1px solid #334155;
            border-radius: 8px;
            background: #0f172a;
            padding: 0.75rem;
        }

        .prediction-bounds span {
            color: #94a3b8;
            display: block;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .prediction-bounds strong {
            color: #f8fafc;
            font-size: 1.05rem;
        }

        .prediction-result-card p {
            color: #94a3b8;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-bottom: 0;
            margin-top: 1rem;
        }

        .best-model-card {
            border: 1px solid #22c55e;
            border-left: 6px solid #22c55e;
            border-radius: 8px;
            background: #111827;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
            margin: 1rem 0;
            padding: 1.15rem;
        }

        .best-model-label {
            color: #22c55e;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .best-model-name {
            color: #f8fafc;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.35rem;
        }

        .best-model-card p {
            color: #94a3b8;
            margin-bottom: 0;
            margin-top: 0.6rem;
        }

        .visual-indicator-container {
            margin: 1.5rem 0;
            padding: 0.5rem 0;
        }

        .indicator-track {
            position: relative;
            height: 8px;
            background: #334155;
            border-radius: 4px;
            margin-bottom: 0.75rem;
        }

        .indicator-fill {
            position: absolute;
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #22c55e, #38bdf8);
            border-radius: 4px;
        }

        .indicator-point {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 18px;
            height: 18px;
            background: #ffffff;
            border: 3px solid #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.6);
            z-index: 2;
        }

        .indicator-labels {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
            color: #94a3b8;
        }

        .indicator-labels .lower-label {
            text-align: left;
            font-weight: 700;
            line-height: 1.2;
        }

        .indicator-labels .predicted-label {
            text-align: center;
            color: #22c55e;
            font-weight: 800;
            line-height: 1.2;
        }

        .indicator-labels .upper-label {
            text-align: right;
            font-weight: 700;
            line-height: 1.2;
        }

        .workflow-step {
            border: 1px solid #334155;
            border-left: 5px solid #38bdf8;
            border-radius: 8px;
            background: #111827;
            padding: 1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .workflow-step-title {
            color: #f8fafc;
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }

        .workflow-step-desc {
            color: #94a3b8;
            font-size: 0.88rem;
            line-height: 1.4;
        }

        .flow-arrow {
            text-align: center;
            font-size: 1.6rem;
            color: #38bdf8;
            margin: 0.5rem 0;
            font-weight: 900;
            animation: bounce 2s infinite;
        }

        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-5px); }
            60% { transform: translateY(-3px); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_workflow_section() -> None:
    """Display an interactive, beautifully designed ML workflow diagram."""
    st.markdown("## Machine Learning Workflow")
    st.caption("The step-by-step pipeline executed to build, train, evaluate, and run the house price predictor.")
    
    steps = [
        ("📁", "Dataset Acquisition", "Acquires historical housing records (Price) along with specifications (Area, Bedrooms, Bathrooms, Age) from local registry data sources."),
        ("🧹", "Data Cleaning", "Identifies and handles missing data, cleans duplicates, and caps extreme pricing outliers using the IQR method to ensure model stability."),
        ("📈", "Exploratory Data Analysis (EDA)", "Visualizes feature patterns, correlation matrices, and distribution shapes to discover relationships between house specifications and selling prices."),
        ("⚙️", "Feature Engineering & Preprocessing", "Prepares data using Scikit-Learn transformers (StandardScaler to normalize dimensions and OneHotEncoder for categorical labels) bound inside an inference Pipeline."),
        ("🔀", "Train-Test Split", "Splits the cleaned dataset into training samples (80%) to fit model estimators and holdout test samples (20%) for final performance validation."),
        ("🤖", "Model Training", "Trains multiple candidate regression models (Linear Regression, Decision Tree, and Random Forest) to learn pricing patterns from specifications."),
        ("📊", "Model Evaluation & Calibration", "Compares test performance using R² score, MAE, and RMSE metrics, run with K-Fold Cross-Validation for robustness checks."),
        ("🎯", "Inference & Prediction", "Deploys the optimal pipeline to estimate new property prices and calculates statistical confidence intervals from model residuals.")
    ]
    
    with st.container():
        for i, (icon, title, desc) in enumerate(steps):
            st.markdown(
                f"""
                <div class="workflow-step">
                    <div class="workflow-step-title">{icon} {title}</div>
                    <div class="workflow-step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if i < len(steps) - 1:
                st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)


render_sidebar()

st.title("House Price Prediction")
st.write("Enter property details to estimate the expected house price.")

render_dataset_statistics()

with st.form("prediction_form"):
    location = st.selectbox("Location", options=["Delhi", "Mumbai", "Bangalore", "Pune", "Hyderabad", "Chandigarh"], index=0)
    area = st.number_input("Area (sq ft)", min_value=1.0, value=1500.0, step=50.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, value=2, step=1)
    age = st.number_input("Age of Property (years)", min_value=0, value=5, step=1)
    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        prediction_result = predict_with_interval(
            area=float(area),
            bedrooms=int(bedrooms),
            bathrooms=int(bathrooms),
            age=int(age),
            location=location,
        )
        st.session_state["prediction_result"] = prediction_result
        st.session_state["inputs"] = {
            "area": float(area),
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "age": int(age),
            "location": location,
        }
        
        # Calculate Explainable AI contribution details
        explanation = explain_prediction(
            area=float(area),
            bedrooms=int(bedrooms),
            bathrooms=int(bathrooms),
            age=int(age),
            location=location,
            model_path=MODEL_PATH,
        )
        st.session_state["prediction_explanation"] = explanation
        
        # Calculate Real Estate Insights
        insights = calculate_real_estate_insights(prediction_result["prediction"])
        st.session_state["prediction_insights"] = insights
        
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first by running: python -m src.train")
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("An unexpected error occurred while making the prediction.")
        st.exception(exc)

if "prediction_result" in st.session_state:
    inputs = st.session_state["inputs"]
    prediction_result = st.session_state["prediction_result"]
    explanation = st.session_state.get("prediction_explanation")
    insights = st.session_state.get("prediction_insights")
    
    render_prediction_result(
        prediction_result["prediction"],
        prediction_result["lower_bound"],
        prediction_result["upper_bound"],
    )
    
    if explanation:
        st.write("")
        render_explanation_section(explanation)
        st.write("")
        
    if insights:
        st.write("")
        render_insights_section(insights)
        st.write("")
        
    try:
        summary = calculate_sidebar_summary()
        metrics = calculate_model_metrics()
        
        pdf_bytes = generate_prediction_pdf(
            area=inputs["area"],
            bedrooms=inputs["bedrooms"],
            bathrooms=inputs["bathrooms"],
            age=inputs["age"],
            location=inputs["location"],
            predicted_price=prediction_result["prediction"],
            lower_bound=prediction_result["lower_bound"],
            upper_bound=prediction_result["upper_bound"],
            model_name=summary["model_name"],
            r2_score=summary["r2_score"],
            mae=metrics["MAE"],
            rmse=metrics["RMSE"],
        )
        
        st.download_button(
            label="Download Prediction Report",
            data=pdf_bytes,
            file_name="house_price_prediction_report.pdf",
            mime="application/pdf",
        )
    except Exception as exc:
        st.error("Unable to generate the PDF report.")
        st.exception(exc)

st.divider()
render_performance_dashboard()

st.divider()
render_model_comparison()

st.divider()
render_actual_vs_predicted_section()

st.divider()
render_eda_section()

st.divider()
render_feature_importance_section()

st.divider()
render_data_distribution_section()

st.divider()
render_workflow_section()
