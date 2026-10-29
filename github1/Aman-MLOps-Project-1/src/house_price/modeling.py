"""Training, model comparison, and artifact persistence."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.house_price.data import FEATURES, TARGET, make_housing_data

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "house_price_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def build_candidates(random_state: int = 42) -> dict[str, object]:
    """Build a simple baseline and two candidate regression pipelines."""
    preprocessing = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "baseline": DummyRegressor(strategy="median"),
        "linear_regression": Pipeline(preprocessing + [("model", LinearRegression())]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=120, max_depth=12, min_samples_leaf=2,
                random_state=random_state, n_jobs=-1,
            )),
        ]),
    }


def train_and_evaluate(n_samples: int = 1200, random_state: int = 42) -> dict:
    """Train candidates, select lowest MAE, save model and metrics."""
    data = make_housing_data(n_samples=n_samples, random_state=random_state)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    results = {}
    fitted = {}
    for name, estimator in build_candidates(random_state).items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        results[name] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }
        fitted[name] = estimator

    winner = min(results, key=lambda name: results[name]["mae"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[winner], "features": FEATURES, "target": TARGET},
                MODEL_PATH)
    report = {
        "dataset": "deterministic synthetic housing data",
        "n_samples": n_samples,
        "random_state": random_state,
        "selected_model": winner,
        "selection_metric": "lowest test MAE",
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = train_and_evaluate()
    print(json.dumps(report, indent=2))
