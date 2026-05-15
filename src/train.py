import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from database import get_connection, init_db
from preprocess import preprocess_data
from logging_utils import save_json


MODEL_PATH = "models/demand_model.pkl"
FEATURES_PATH = "models/model_features.pkl"
METRICS_PATH = "artifacts/model_metrics.csv"
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
    "current_stock",
    "unit_price",
    "expiry_days",
    "waste_quantity",
    "stock_to_sales_ratio",
    "is_high_stock",
    "is_low_demand",
    "short_expiry",
    "promotion_active",
    "waste_risk_score",
]

TARGET_COLUMN = "units_sold"


def train_model():
    init_db()

    if not os.path.exists("data/processed_food_sales.csv"):
        df = preprocess_data()
    else:
        df = pd.read_csv("data/processed_food_sales.csv")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.to_numeric(y, errors="coerce").fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)

    metrics = {
        "model_name": "RandomForestRegressor",
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "training_rows": int(len(df)),
        "target_column": TARGET_COLUMN,
        "model_path": MODEL_PATH,
        "features_path": FEATURES_PATH,
        "created_at": datetime.now().isoformat()
    }

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    training_summary = {
        **metrics,
        "features_used": FEATURE_COLUMNS,
        "purpose": "Demand forecasting model for food waste prediction and inventory optimization"
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
        "RandomForestRegressor",
        MODEL_PATH,
        float(mae),
        float(rmse),
        float(r2),
        int(len(df)),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    print("Model training completed.")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 2))
    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print("Training summary saved to:", TRAINING_SUMMARY_PATH)

    return metrics


if __name__ == "__main__":
    train_model()