import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from database import get_connection, init_db


PROCESSED_PATH = "data/processed_food_sales.csv"


def preprocess_data():

    init_db()

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM raw_food_sales",
        conn
    )

    conn.close()

    if df.empty:
        raise ValueError(
            "No raw data found. Please run python src/kaggle_loader.py first."
        )

    df = df.drop_duplicates()

    numeric_columns = [
        "store_id",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "promotion",
        "temperature",
        "current_stock",
        "units_sold",
        "unit_price",
        "expiry_days",
        "waste_quantity",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = df[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(
                df[col].median()
            )

    df["product_name"] = df["product_name"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")

    df["product_name"] = df["product_name"].astype(str)
    df["category"] = df["category"].astype(str)

    product_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    df["product_encoded"] = product_encoder.fit_transform(
        df["product_name"]
    )

    df["category_encoded"] = category_encoder.fit_transform(
        df["category"]
    )

    # Feature Engineering

    df["stock_to_sales_ratio"] = (
        df["current_stock"] / (df["units_sold"] + 1)
    )

    df["stock_to_sales_ratio"] = df[
        "stock_to_sales_ratio"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df["stock_to_sales_ratio"] = df[
        "stock_to_sales_ratio"
    ].fillna(0)

    df["stock_to_sales_ratio"] = df[
        "stock_to_sales_ratio"
    ].clip(
        lower=0,
        upper=100
    )

    stock_median = df["current_stock"].median()
    sales_median = df["units_sold"].median()

    df["is_high_stock"] = (
        df["current_stock"] > stock_median
    ).astype(int)

    df["is_low_demand"] = (
        df["units_sold"] < sales_median
    ).astype(int)

    df["short_expiry"] = (
        df["expiry_days"] <= 3
    ).astype(int)

    df["promotion_active"] = df["promotion"].astype(int)

    df["waste_risk_score"] = (
        (df["stock_to_sales_ratio"] * 0.4)
        + (df["is_high_stock"] * 0.2)
        + (df["is_low_demand"] * 0.2)
        + (df["short_expiry"] * 0.2)
    )

    df["waste_risk_score"] = df[
        "waste_risk_score"
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df["waste_risk_score"] = df[
        "waste_risk_score"
    ].fillna(0)

    df["waste_risk_score"] = df[
        "waste_risk_score"
    ].clip(
        lower=0,
        upper=100
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    os.makedirs("data", exist_ok=True)

    df.to_csv(
        PROCESSED_PATH,
        index=False
    )

    print("Preprocessing completed.")
    print(f"Processed file saved to {PROCESSED_PATH}")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


if __name__ == "__main__":
    preprocess_data()