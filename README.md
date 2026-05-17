
# Module: M6-MSc BDS – Data Engineering and Machine Learning Operations in Business
# Submitted by: Anzuman Ara

# AI-Driven Retail Demand Forecasting and Smart Inventory Optimization

# Overview

This project is an end-to-end MLOps pipeline for retail demand forecasting and smart inventory optimization. The system simulates a production-style workflow where live retail sales events are generated, sent through FastAPI APIs, stored in a SQLite database, processed through a preprocessing and feature engineering pipeline, used for demand prediction, monitored for drift, and automatically retrained when drift is detected.

The project demonstrates:
* Live API-based data ingestion
* Data preprocessing and feature engineering
* Machine learning model training and comparison
* Prediction serving through FastAPI
* Drift detection and monitoring
* Automatic retraining pipeline
* Model artifact tracking and versioning
* Streamlit dashboard frontend
* Docker-ready deployment


# Project Architecture

```text
External Data Generator
        ↓
FastAPI API Ingestion (/sales-event)
        ↓
SQLite Database Storage
        ↓
Preprocessing Pipeline
        ↓
Feature Engineering
        ↓
Model Training & Comparison
        ↓
Prediction Pipeline
        ↓
Drift Detection
        ↓
Automatic Retraining
        ↓
Streamlit Monitoring Dashboard
```


# Features

## API-Based Data Ingestion

The project simulates live retail sales events using an external client. Events are generated and sent to the FastAPI backend through REST APIs.

Endpoint:

```text
POST /sales-event
```

# Data Preprocessing
The preprocessing pipeline performs:
* Missing value handling
* Data type conversion
* Feature generation
* Feature selection
* Processed dataset creation


# Feature Engineering
The system creates engineered features including:
* product_encoded
* category_encoded
* day_of_week
* is_weekend
* promotion_active
* short_expiry
* temperature
* unit_price
* expiry_days

These features are used for retail demand forecasting and inventory risk estimation.


# Model Training and Comparison
The project trains and compares two machine learning models:
* RandomForestRegressor
* LightGBMRegressor
The best-performing model is selected automatically based on RMSE.

Tracked metrics:
* MAE
* RMSE
* R² Score


# Prediction Pipeline
The API prediction endpoint:
```text
POST /predict-demand
```

Pipeline steps:
1. Receive prediction request
2. Build engineered features
3. Run model inference
4. Estimate waste risk
5. Generate inventory recommendation
6. Store prediction logs


# Drift Detection
The system monitors feature drift using relative mean shift analysis between reference and current data windows to identify significant changes in incoming data patterns.

Endpoint:
```text
POST /drift-check
```


# Automatic Retraining
When significant drift is detected, the pipeline automatically retrains the model and saves a new versioned artifact.

Endpoint:
```text
POST /auto-retrain
```

# Model Versioning
Each retrained model is stored with a timestamp-based filename.

Example:
```text
models/demand_model_20260516_220712.pkl
```

The latest serving model is also maintained:
```text
models/demand_model.pkl
```


# Streamlit Dashboard

The frontend dashboard provides:
* Predict Demand page
* Latest Sales page
* Model Comparison page
* Model Information page
* Drift Reports page
* Pipeline Actions page


# Project Structure
```text
project/
│
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── train.py
│   ├── drift_detection.py
│   ├── auto_retrain.py
│   ├── external_client.py
│   ├── frontend.py
│   ├── database.py
│   └── kaggle_loader.py
│
├── models/
├── artifacts/
├── logs/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```


# 1. Install Dependencies
```bash
pip install -r requirements.txt
```

# 2. Initialize Database
```bash
python -m src.database
```


# 3. Start FastAPI Backend
```bash
uvicorn src.main:app --reload
```

Swagger Documentation:
```text
http://127.0.0.1:8000/docs
```

## 4. Generate Live Sales Events
Normal live data:
```bash
python -m src.external_client --mode normal --events 20 --delay 1
```

Shifted drift data:
```bash
python -m src.external_client --mode shifted --events 300 --delay 0.1
```


# 5. Start Streamlit Dashboard
```bash
streamlit run src/frontend.py
```

Dashboard:
```text
http://localhost:8501
```


# API Endpoints
| Endpoint          | Method | Description                    |
| ----------------- | ------ | ------------------------------ |
| /sales-event      | POST   | Insert live sales event        |
| /latest-sales     | GET    | Fetch latest sales records     |
| /run-pipeline     | POST   | Run preprocessing and training |
| /predict-demand   | POST   | Predict retail demand          |
| /prediction-logs  | GET    | View prediction history        |
| /model-info       | GET    | Latest model metadata          |
| /model-comparison | GET    | Compare trained models         |
| /drift-check      | POST   | Run drift detection            |
| /drift-reports    | GET    | View drift reports             |
| /retrain          | POST   | Manual retraining              |
| /auto-retrain     | POST   | Automatic retraining           |
| /health           | GET    | Health check                   |


# Example Prediction Request
```json
{
  "product_name": "Dept_1",
  "category": "A",
  "store_id": 1,
  "day_of_week": 2,
  "is_weekend": 0,
  "is_holiday": 0,
  "promotion": 1,
  "temperature": 65,
  "current_stock": 120,
  "unit_price": 10,
  "expiry_days": 5
}


# Testing
Run tests:

```bash
pytest tests/ -v
```

# Docker Deployment
Build and run:
```bash
docker-compose up --build
```

# Reproducibility
The project supports reproducibility through:
* Versioned model artifacts
* Stored metrics
* Drift reports
* Prediction logs
* Requirements freeze
* Docker configuration

# Conclusion
The project uses the Walmart Store Sales Forecasting dataset as the primary retail demand dataset.Additional inventory and waste-related fields are synthetically generated to simulate inventory optimization and waste-risk analysis workflows in a realistic MLOps environment.