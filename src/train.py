import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.database import get_connection, init_db
from src.preprocess import preprocess_data
from src.logging_utils import save_json


MODEL_PATH = "models/demand_model.pkl"
FEATURES_PATH = "models/model_features.pkl"

METRICS_PATH = "artifacts/model_metrics.csv"
MODEL_COMPARISON_PATH = "artifacts/model_comparison.csv"
TRAINING_SUMMARY_PATH = "artifacts/training_summary.json"


FEATURE_COLUMNS = [
    "product_encoded",
    "category_encoded",
    "store_id",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "promotion",
    "temperature",
    "unit_price",
    "expiry_days",
    "short_expiry",
    "promotion_active",
]

TARGET_COLUMN = "units_sold"


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def train_model():
    init_db()

    if not os.path.exists("data/processed_food_sales.csv"):
        df = preprocess_data()
    else:
        df = pd.read_csv("data/processed_food_sales.csv")

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required feature columns: {missing_features}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    y = pd.to_numeric(
        y,
        errors="coerce"
    ).fillna(
        y.median()
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    candidate_models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }

    if LIGHTGBM_AVAILABLE:
        candidate_models["LightGBMRegressor"] = LGBMRegressor(
            random_state=42,
            verbose=-1
        )
    else:
        print(
            "LightGBM is not installed. "
            "Only RandomForestRegressor will be trained."
        )

    comparison_rows = []

    best_model_name = None
    best_result = None

    for model_name, model in candidate_models.items():

        print(f"\nTraining model: {model_name}")

        result = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        comparison_rows.append({
            "model_name": model_name,
            "mae": round(result["mae"], 4),
            "rmse": round(result["rmse"], 4),
            "r2": round(result["r2"], 4),
            "training_rows": int(len(df)),
            "created_at": datetime.now().isoformat()
        })

        if (
            best_result is None
            or result["rmse"] < best_result["rmse"]
        ):
            best_result = result
            best_model_name = model_name

    best_model = best_result["model"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    versioned_model_path = f"models/demand_model_{timestamp}.pkl"

    joblib.dump(best_model, versioned_model_path)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)

    pd.DataFrame(comparison_rows).to_csv(
        MODEL_COMPARISON_PATH,
        index=False
    )

    metrics = {
        "model_name": best_model_name,
        "mae": round(best_result["mae"], 4),
        "rmse": round(best_result["rmse"], 4),
        "r2": round(best_result["r2"], 4),
        "training_rows": int(len(df)),
        "target_column": TARGET_COLUMN,
        "model_path": versioned_model_path,
        "latest_model_path": MODEL_PATH,
        "features_path": FEATURES_PATH,
        "created_at": datetime.now().isoformat()
    }

    pd.DataFrame([metrics]).to_csv(
        METRICS_PATH,
        index=False
    )

    training_summary = {
        **metrics,
        "features_used": FEATURE_COLUMNS,
        "models_compared": list(candidate_models.keys()),
        "purpose": (
            "AI-driven retail demand forecasting with model comparison, "
            "drift monitoring, retraining, and inventory decision support."
        )
    }

    save_json(TRAINING_SUMMARY_PATH, training_summary)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_registry (
            model_name,
            model_path,
            mae,
            rmse,
            r2,
            training_rows,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        best_model_name,
        versioned_model_path,
        best_result["mae"],
        best_result["rmse"],
        best_result["r2"],
        int(len(df)),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    print("\nModel training completed.")
    print("Models compared:", list(candidate_models.keys()))
    print("Best model:", best_model_name)
    print("MAE:", round(best_result["mae"], 4))
    print("RMSE:", round(best_result["rmse"], 4))
    print("R2:", round(best_result["r2"], 4))
    print("Versioned model saved to:", versioned_model_path)
    print("Latest model saved to:", MODEL_PATH)
    print("Model comparison saved to:", MODEL_COMPARISON_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print("Training summary saved to:", TRAINING_SUMMARY_PATH)

    return metrics


if __name__ == "__main__":
    train_model()