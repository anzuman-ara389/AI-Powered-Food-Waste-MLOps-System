# Report Draft: AI-Powered Food Waste Prediction and Smart Inventory Optimization

## 1. Introduction

Food waste is a major business and sustainability problem in the retail sector. Supermarkets often overstock perishable products because demand is difficult to predict accurately. Overstocking can lead to expired products, financial losses, and unnecessary environmental impact.

This project investigates how artificial intelligence can be used to forecast supermarket product demand and support smarter inventory planning.

The research question is:

**How can AI-driven demand forecasting reduce food waste and improve inventory planning in supermarkets?**

## 2. Pipeline Diagram

```text
Supermarket Sales Events
        ↓
FastAPI Sales Event Endpoint
        ↓
SQLite Database
        ↓
Preprocessing Pipeline
        ↓
Demand Forecasting Model
        ↓
Prediction API
        ↓
Waste Risk Calculation
        ↓
Inventory Recommendation
        ↓
Prediction Logs
        ↓
Drift Detection
        ↓
Auto Retraining
        ↓
Streamlit Dashboard
```

## 3. Technical Details

The project uses simulated supermarket sales and inventory data. Each record includes product name, category, store ID, promotion status, temperature, current stock, units sold, unit price, expiry days, and waste quantity.

The data is stored in a SQLite database. The main raw table is called `raw_food_sales`.

The preprocessing pipeline cleans the data and creates new features such as:

- stock-to-sales ratio
- high stock flag
- low demand flag
- short expiry flag
- promotion active flag
- waste risk score

These features are used to train a machine learning model.

## 4. Model Training and Evaluation

The target variable is `units_sold`, which represents product demand.

A Random Forest Regressor is used to predict product demand.

The model is evaluated using:

- MAE
- RMSE
- R2 score

MAE and RMSE measure prediction error. Lower values mean the model predicts demand more accurately.

The trained model is saved in the `models` folder, and model metrics are stored in the `model_registry` table.

## 5. Prediction and Business Logic

After predicting demand, the system compares predicted demand with current inventory.

If current stock is much higher than predicted demand, the system identifies overstock risk. Overstock risk is then converted into a waste risk level:

- Low
- Medium
- High

The system also gives an inventory recommendation, such as:

- keep normal order quantity
- monitor inventory
- reduce next order and consider discount/promotion

## 6. Monitoring and Drift Detection

The system includes a drift detection component.

The drift detection script monitors changes in:

- units sold
- current stock
- temperature
- waste quantity

If the current data distribution changes significantly compared to reference data, drift is detected.

This is important because supermarket demand can change due to seasonality, weather, holidays, promotions, or consumer behavior.

## 7. Auto Retraining

The auto retraining script checks whether drift has occurred.

If drift is detected, the system automatically runs preprocessing and retrains the model.

If no drift is detected, retraining is skipped.

This supports a basic MLOps lifecycle.

## 8. Frontend Dashboard

The Streamlit dashboard allows users to:

- enter product and inventory information
- predict demand
- view waste risk level
- receive inventory recommendations
- view model metrics
- view prediction logs
- view drift reports
- run pipeline actions

This makes the system more understandable for supermarket managers.

## 9. Governance and Ethics

The system should support human decision-making, not replace it completely. Wrong predictions could cause understocking or overstocking. Therefore, supermarket managers should review the recommendations before making final inventory decisions.

The project also supports sustainability by helping reduce unnecessary food waste.

Model monitoring and retraining are important governance practices because they help maintain reliability over time.

## 10. Conclusion

This project demonstrates how AI-driven demand forecasting can support food waste reduction and smarter inventory planning in supermarkets.

By combining data ingestion, preprocessing, machine learning, prediction logging, drift detection, auto retraining, and a dashboard, the project shows a complete MLOps workflow for a real-world retail sustainability problem.
