# AI-Powered Food Waste Prediction and Smart Inventory Optimization

## Project Overview

This project is an MLOps-based machine learning system for supermarket food waste reduction.

The system simulates supermarket sales and inventory data, stores it in a SQLite database, preprocesses the data, trains a demand forecasting model, predicts future product demand, calculates waste risk, gives inventory recommendations, logs predictions, detects data drift, and supports retraining.

## Research Question

How can AI-driven demand forecasting reduce food waste and improve inventory planning in supermarkets?

## Project Workflow

```text
database.py
   ↓
live_ingestion.py
   ↓
external_client.py
   ↓
preprocess.py
   ↓
train.py
   ↓
main.py
   ↓
drift_detection.py
   ↓
auto_retrain.py
   ↓
frontend.py
```

## Files

| File | Purpose |
|---|---|
| database.py | Creates SQLite database and tables |
| live_ingestion.py | Generates simulated supermarket sales data |
| external_client.py | Sends sales events to FastAPI |
| preprocess.py | Cleans and prepares data |
| train.py | Trains demand forecasting model |
| main.py | FastAPI backend |
| drift_detection.py | Checks data drift |
| auto_retrain.py | Retrains model if drift is detected |
| frontend.py | Streamlit dashboard |

## Database Tables

| Table | Purpose |
|---|---|
| raw_food_sales | Stores raw supermarket sales and inventory data |
| demand_prediction_logs | Stores prediction results |
| drift_reports | Stores drift detection results |
| model_registry | Stores model metrics and model path |

## Installation

```bash
pip install -r requirements.txt
```

## Step 1: Initialize Database

```bash
python database.py
```

## Step 2: Generate Live Data

```bash
python live_ingestion.py
```

## Step 3: Preprocess Data

```bash
python preprocess.py
```

## Step 4: Train Model

```bash
python train.py
```

## Step 5: Run FastAPI

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Step 6: Run Streamlit Dashboard

Open another terminal:

```bash
streamlit run frontend.py
```

## Important API Endpoints

| Endpoint | Purpose |
|---|---|
| GET / | API welcome |
| GET /health | API health check |
| POST /sales-event | Insert new sales event |
| POST /ingest-live | Generate live data |
| GET /latest-sales | View latest sales |
| POST /run-pipeline | Preprocess + train model |
| POST /predict-demand | Predict demand and waste risk |
| GET /prediction-logs | View prediction logs |
| GET /model-info | View latest model metrics |
| POST /drift-check | Run drift detection |
| GET /drift-reports | View drift reports |
| POST /retrain | Manual retraining |
| POST /auto-retrain | Retrain only if drift is detected |

## Business Logic

The model predicts product demand.

Then the system compares predicted demand with current stock.

If current stock is much higher than predicted demand, the product has higher waste risk.

Example:

```text
Current stock = 100
Predicted demand = 55
Overstock = 45
Waste risk = High
Recommendation = Reduce next order and consider discount/promotion
```

## MLOps Features

- Automated data ingestion
- Database storage
- Preprocessing pipeline
- Model training
- Model registry
- Prediction API
- Prediction logging
- Drift detection
- Auto retraining
- Streamlit dashboard

## Conclusion

This project shows how AI and MLOps can support supermarkets in reducing food waste and improving inventory planning.
