import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from database import get_connection, init_db
from preprocess import preprocess_data
from train import train_model, MODEL_PATH, FEATURE_COLUMNS


app = FastAPI(
    title="AI-Powered Food Waste Prediction API",
    description="Retail demand forecasting, food waste risk prediction, monitoring, and retraining API.",
    version="1.0.0"
)


class SalesEvent(BaseModel):
    product_name: str = "Dept_1"
    category: str = "A"
    store_id: int = 1
    date: str = "2026-05-14"
    day_of_week: int = 3
    is_weekend: int = 0
    is_holiday: int = 0
    promotion: int = 0
    temperature: float = 20.0
    current_stock: int = 100
    units_sold: int = 60
    unit_price: float = 10.0
    expiry_days: int = 5
    waste_quantity: int = 20
    source: str = "api_event"


class PredictionRequest(BaseModel):
    product_name: str = "Dept_1"
    category: str = "A"
    store_id: int = 1
    day_of_week: int = 3
    is_weekend: int = 0
    is_holiday: int = 0
    promotion: int = 0
    temperature: float = 20.0
    current_stock: int = 100
    unit_price: float = 10.0
    expiry_days: int = 5
    waste_quantity: int = 20


@app.on_event("startup")
def startup_event():
    init_db()


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def encode_product(product_name):
    if product_name.startswith("Dept_"):
        try:
            return int(product_name.replace("Dept_", ""))
        except ValueError:
            return 0
    return 0


def encode_category(category):
    category_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "General": 3
    }
    return category_map.get(category, 0)


def build_prediction_features(data):
    estimated_units_sold = max(1, data.current_stock - data.waste_quantity)

    stock_to_sales_ratio = data.current_stock / (estimated_units_sold + 1)

    is_high_stock = 1 if data.current_stock > 100 else 0
    is_low_demand = 1 if estimated_units_sold < 50 else 0
    short_expiry = 1 if data.expiry_days <= 3 else 0
    promotion_active = data.promotion

    waste_risk_score = (
        stock_to_sales_ratio * 0.4
        + is_high_stock * 0.2
        + is_low_demand * 0.2
        + short_expiry * 0.2
    )

    row = {
        "product_encoded": encode_product(data.product_name),
        "category_encoded": encode_category(data.category),
        "store_id": data.store_id,
        "day_of_week": data.day_of_week,
        "is_weekend": data.is_weekend,
        "is_holiday": data.is_holiday,
        "promotion": data.promotion,
        "temperature": data.temperature,
        "current_stock": data.current_stock,
        "unit_price": data.unit_price,
        "expiry_days": data.expiry_days,
        "waste_quantity": data.waste_quantity,
        "stock_to_sales_ratio": stock_to_sales_ratio,
        "is_high_stock": is_high_stock,
        "is_low_demand": is_low_demand,
        "short_expiry": short_expiry,
        "promotion_active": promotion_active,
        "waste_risk_score": waste_risk_score
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def classify_waste_risk(predicted_demand, current_stock, expiry_days):
    overstock = current_stock - predicted_demand

    if overstock <= 0:
        return "Low"

    overstock_ratio = overstock / max(current_stock, 1)

    if overstock_ratio >= 0.40 or expiry_days <= 2:
        return "High"
    elif overstock_ratio >= 0.20 or expiry_days <= 5:
        return "Medium"
    else:
        return "Low"


def make_recommendation(predicted_demand, current_stock, waste_risk):
    safety_stock = 10
    recommended_order_quantity = max(
        0,
        int(predicted_demand + safety_stock - current_stock)
    )

    if waste_risk == "High":
        recommendation = "Reduce next order and consider discount or promotion."
    elif waste_risk == "Medium":
        recommendation = "Monitor inventory carefully and avoid over-ordering."
    else:
        recommendation = "Inventory level looks acceptable. Keep normal order plan."

    return recommended_order_quantity, recommendation


@app.get("/")
def root():
    return {
        "message": "Food Waste Prediction and Smart Inventory Optimization API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/sales-event")
def sales_event(event: SalesEvent):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO raw_food_sales (
            product_name,
            category,
            store_id,
            date,
            day_of_week,
            is_weekend,
            is_holiday,
            promotion,
            temperature,
            current_stock,
            units_sold,
            unit_price,
            expiry_days,
            waste_quantity,
            source,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event.product_name,
        event.category,
        event.store_id,
        event.date,
        event.day_of_week,
        event.is_weekend,
        event.is_holiday,
        event.promotion,
        event.temperature,
        event.current_stock,
        event.units_sold,
        event.unit_price,
        event.expiry_days,
        event.waste_quantity,
        event.source,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    return {
        "message": "Sales event inserted successfully through API.",
        "event": event.dict()
    }


@app.get("/latest-sales")
def latest_sales(limit: int = 10):
    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM raw_food_sales ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,)
    )

    conn.close()

    return df.to_dict(orient="records")


@app.post("/run-pipeline")
def run_pipeline():
    processed_df = preprocess_data()
    metrics = train_model()

    return {
        "message": "Preprocessing and training completed successfully.",
        "processed_rows": len(processed_df),
        "metrics": metrics
    }


@app.post("/predict-demand")
def predict_demand(data: PredictionRequest):
    model = load_model()

    if model is None:
        return {
            "error": "Model not found. Please run python src/train.py first."
        }

    input_df = build_prediction_features(data)

    predicted_demand = float(model.predict(input_df)[0])

    waste_risk = classify_waste_risk(
        predicted_demand,
        data.current_stock,
        data.expiry_days
    )

    recommended_order_quantity, recommendation = make_recommendation(
        predicted_demand,
        data.current_stock,
        waste_risk
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_logs (
            predicted_demand,
            current_stock,
            waste_risk,
            created_at
        )
        VALUES (?, ?, ?, ?)
    """, (
        predicted_demand,
        data.current_stock,
        waste_risk,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    return {
        "predicted_demand": round(predicted_demand, 2),
        "current_stock": data.current_stock,
        "waste_risk": waste_risk,
        "recommended_order_quantity": recommended_order_quantity,
        "recommendation": recommendation
    }


@app.get("/prediction-logs")
def prediction_logs(limit: int = 20):
    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,)
    )

    conn.close()

    return df.to_dict(orient="records")


@app.get("/model-info")
def model_info():
    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM model_registry ORDER BY id DESC LIMIT 1",
        conn
    )

    conn.close()

    if df.empty:
        return {
            "message": "No model information found."
        }

    return df.to_dict(orient="records")[0]


@app.post("/drift-check")
def drift_check():
    from drift_detection import run_drift_check
    return run_drift_check()


@app.get("/drift-reports")
def drift_reports(limit: int = 20):
    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM drift_reports ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,)
    )

    conn.close()

    return df.to_dict(orient="records")


@app.post("/retrain")
def retrain():
    processed_df = preprocess_data()
    metrics = train_model()

    return {
        "message": "Manual retraining completed successfully.",
        "processed_rows": len(processed_df),
        "metrics": metrics
    }


@app.post("/auto-retrain")
def auto_retrain_endpoint():
    from auto_retrain import auto_retrain
    return auto_retrain()