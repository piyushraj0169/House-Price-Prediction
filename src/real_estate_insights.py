"""Real Estate Insights helper for predicted properties."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.config import DATA_PATH, TARGET_COLUMN
from src.preprocessing import clean_dataset


def calculate_real_estate_insights(predicted_price: float) -> dict[str, str | float]:
    """Calculate price category, market percentile position, and investment commentary."""
    # Load and clean dataset
    data = pd.read_csv(DATA_PATH)
    cleaned_data = clean_dataset(data, target_column=TARGET_COLUMN)
    prices = cleaned_data[TARGET_COLUMN]
    
    # Percentiles
    p25 = float(prices.quantile(0.25))
    p75 = float(prices.quantile(0.75))
    p95 = float(prices.quantile(0.95))
    
    # Category
    if predicted_price <= p25:
        category = "Affordable"
        color = "#38bdf8" # Sky Blue
    elif predicted_price <= p75:
        category = "Mid-Range"
        color = "#eab308" # Yellow
    elif predicted_price <= p95:
        category = "Premium"
        color = "#3b82f6" # Royal Blue
    else:
        category = "Luxury"
        color = "#a855f7" # Purple
        
    # Market position
    total_records = len(prices)
    cheaper_count = sum(prices < predicted_price)
    cheaper_pct = (cheaper_count / total_records) * 100
    top_pct = 100 - cheaper_pct
    
    # Ensure reasonable boundaries
    if top_pct < 1.0:
        top_pct = 1.0
    elif top_pct > 99.0:
        top_pct = 99.0
        
    # Investment commentary
    if category == "Affordable":
        commentary = (
            "This property sits in the affordable tier of the local market. It presents a low-barrier entry point "
            "for first-time homebuyers or buy-to-let investors looking for strong rental yields. Capital downside "
            "risk is typically low in this segment, though appreciation may track general inflation."
        )
    elif category == "Mid-Range":
        commentary = (
            "Valued in the mid-range tier, this property represents the core of local market demand. Mid-market homes "
            "offer excellent resale liquidity and consistent demand from families. This makes it a "
            "stable and safe investment with moderate, reliable appreciation potential."
        )
    elif category == "Premium":
        commentary = (
            f"Classified as a premium property, this asset ranks in the top {top_pct:.0f}% of the market. Premium homes "
            "feature superior specifications and attract high-earning professionals. They enjoy robust capital "
            "appreciation during growth phases, though they may take slightly longer to sell than mid-market listings."
        )
    else:
        commentary = (
            f"This is a highly exclusive luxury tier asset in the top {top_pct:.0f}% of the market. Luxury listings serve "
            "as premium wealth-preservation vehicles. Resale timelines are extended due to the niche buyer pool, "
            "but unique location characteristics and design exclusivity can command exceptional capital premiums over time."
        )
        
    return {
        "category": category,
        "color": color,
        "top_pct": float(top_pct),
        "commentary": commentary,
        "p25": p25,
        "p75": p75,
        "p95": p95
    }
