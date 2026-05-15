# M6 – Data Engineering and Machine Learning Operations in Business
# AI-Powered Food Waste Prediction and Smart Inventory Optimization for Retail
Submitted by: Anzuman Ara
Student ID: 20241266


# Overview
This project implements an end-to-end MLOps pipeline for food waste prediction and smart inventory optimization in retail supermarkets.

The system simulates a production-like environment where:
* External systems send supermarket inventory and sales data via API
* Data is stored in a database
* Preprocessing and feature engineering are applied
* A machine learning model is trained and versioned
* Demand predictions are generated via API
* Prediction outputs are logged for monitoring
* Data drift is detected
* The model is retrained automatically if drift is detected


# How to Run the Project
# Step 1: Install dependencies
pip install -r requirements.txt


# Step 2: Initialize the database
python -m src.database

This creates:
* raw_food_sales
* prediction_logs
* drift_reports
* model_registry


# Step 3: Run the FastAPI backend
uvicorn src.main:app --reload

Open browser:
http://127.0.0.1:8000/docs


# Step 4: Simulate external live data ingestion
python src/external_client.py
Flow:
external_client → POST /sales-event → Database

# Step 5: Check latest sales records
GET /latest-sales


# Step 6: Run full ML pipeline
POST /run-pipeline

Flow:
Raw Data
→ Preprocessing
→ Feature Engineering
→ Model Training
→ Artifact Saving
→ Model Registry

# Step 7: Make prediction
POST /predict-demand

Example JSON:
```json
{
  "product_name": "Milk",
  "category": "Dairy",
  "store_id": 1,
  "day_of_week": 4,
  "is_weekend": 0,
  "is_holiday": 0,
  "promotion": 1,
  "temperature": 30,
  "current_stock": 120,
  "unit_price": 5.5,
  "expiry_days": 3,
  "waste_quantity": 10
}

Output:
* predicted_demand
* waste_risk
* inventory_gap
* recommendation


# Step 8: View prediction logs
GET /prediction-logs

# Step 9: View model info
GET /model-info

Metrics include:
* MAE
* RMSE
* R² Score


# Step 10: Run drift detection
POST /drift-check

Uses features:
* units_sold
* current_stock
* temperature
* waste_quantity


# Step 11: View drift report
GET /drift-reports

# Step 12: Manual retraining
POST /retrain


# Step 13: Automatic retraining
POST /auto-retrain

If drift is detected → model retrains automatically.

# API Endpoints

| Endpoint         | Method | Purpose            |
| ---------------- | ------ | ------------------ |
| /                | GET    | API status         |
| /health          | GET    | Health check       |
| /sales-event     | POST   | Insert sales event |
| /latest-sales    | GET    | View latest sales  |
| /run-pipeline    | POST   | Run ML pipeline    |
| /predict-demand  | POST   | Predict demand     |
| /prediction-logs | GET    | View predictions   |
| /model-info      | GET    | View model metrics |
| /drift-check     | POST   | Drift detection    |
| /drift-reports   | GET    | View drift reports |
| /retrain         | POST   | Manual retraining  |
| /auto-retrain    | POST   | Auto retraining    |


# Pipeline Flow
External Client
→ API (/sales-event)
→ Database
→ Preprocessing
→ Feature Engineering
→ Model Training
→ Model Registry
→ Prediction
→ Logging
→ Drift Detection
→ Retraining

# Feature Engineering

* stock_to_sales_ratio → overstock detection
* is_high_stock → excessive inventory detection
* is_low_demand → weak demand detection
* short_expiry → near-expiration products
* promotion_active → promotional impact
* waste_risk_score → food waste risk estimation


# Model Metrics

* MAE → prediction error
* RMSE → error magnitude
* R² Score → model accuracy

# Artifacts
Stored in `artifacts/`:
* model_metrics.csv
* drift_summary.csv
* training_summary.json


# Logs
Stored in `logs/`:
* ingestion_log.csv
* retraining_log.csv


# Database Tables
* raw_food_sales
* prediction_logs
* drift_reports
* model_registry


# Production Simulation
This project demonstrates:
* API-based ingestion
* ML pipeline automation
* Model monitoring
* Drift detection
* Automatic retraining
* Production-style MLOps workflow


# GitHub Note
Large generated datasets and trained model binaries are excluded from GitHub due to GitHub storage limitations.

The complete pipeline can regenerate these files automatically.


# Conclusion
This project focuses on building a reproducible, scalable, and production-like MLOps system for intelligent inventory optimization and food waste reduction in retail supermarkets.
