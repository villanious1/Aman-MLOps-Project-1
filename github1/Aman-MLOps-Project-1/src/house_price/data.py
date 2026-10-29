"""Create deterministic synthetic housing data for offline demos and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "median_income", "house_age", "rooms", "bedrooms",
    "population", "households", "latitude", "longitude",
]
TARGET = "price_100k"


def make_housing_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Return synthetic housing records; target is in units of $100,000."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    rng = np.random.default_rng(random_state)
    income = rng.uniform(1.0, 10.0, n_samples)
    age = rng.uniform(1.0, 52.0, n_samples)
    rooms = rng.normal(1800, 550, n_samples).clip(400, 5000)
    bedrooms = (rooms / rng.uniform(3.5, 6.5, n_samples)).clip(1, 1200)
    population = rng.normal(1200, 650, n_samples).clip(50, 6000)
    households = (population / rng.uniform(2.0, 4.5, n_samples)).clip(20, 2000)
    latitude = rng.uniform(32.5, 41.9, n_samples)
    longitude = rng.uniform(-124.3, -114.1, n_samples)
    noise = rng.normal(0, 0.35, n_samples)
    price = (
        0.35 + 0.42 * income + 0.003 * age
        + 0.00012 * rooms - 0.00008 * bedrooms
        - 0.000025 * population + 0.00004 * households
        + 0.018 * (latitude - 37.0)
        + 0.012 * (longitude + 119.0) + noise
    ).clip(0.4, 5.0)
    return pd.DataFrame({
        "median_income": income,
        "house_age": age,
        "rooms": rooms,
        "bedrooms": bedrooms,
        "population": population,
        "households": households,
        "latitude": latitude,
        "longitude": longitude,
        "price_100k": price,
    })
