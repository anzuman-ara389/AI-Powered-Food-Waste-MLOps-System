import os
from datetime import datetime

import pandas as pd

from src.database import get_connection, init_db


TRAIN_PATH = "data/train.csv"
FEATURES_PATH = "data/features.csv"
STORES_PATH = "data/stores.csv"


def load_walmart_data():
    init_db()

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError("Missing data/train.csv")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Missing data/features.csv")

    if not os.path.exists(STORES_PATH):
        raise FileNotFoundError("Missing data/stores.csv")

    train_df = pd.read_csv(TRAIN_PATH)
    features_df = pd.read_csv(FEATURES_PATH)
    stores_df = pd.read_csv(STORES_PATH)

    df = train_df.merge(
        features_df,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    df = df.merge(
        stores_df,
        on="Store",
        how="left"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df["day_of_week"] = df["Date"].dt.dayofweek

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["product_name"] = "Dept_" + df["Dept"].astype(str)

    df["category"] = df["Type"].astype(str)

    df["store_id"] = df["Store"].astype(int)

    df["date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df["is_holiday"] = df["IsHoliday"].astype(int)

    markdown_columns = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    for col in markdown_columns:
        if col not in df.columns:
            df[col] = 0

        df[col] = df[col].fillna(0)

    df["promotion"] = (
        (df["MarkDown1"] > 0)
        | (df["MarkDown2"] > 0)
        | (df["MarkDown3"] > 0)
        | (df["MarkDown4"] > 0)
        | (df["MarkDown5"] > 0)
    ).astype(int)

    df["temperature"] = df["Temperature"].fillna(
        df["Temperature"].median()
    )

    df["units_sold"] = (
        df["Weekly_Sales"]
        .clip(lower=0) / 10
    ).round().astype(int)

    df["unit_price"] = 10.0

    df["expiry_days"] = (
        10 - (df["Dept"].astype(int) % 10)
    ).clip(lower=1, upper=10)

    stock_buffer = (
        30
        + df["is_holiday"] * 20
        + df["promotion"] * 15
        + df["is_weekend"] * 10
    )

    df["current_stock"] = (
        df["units_sold"] + stock_buffer
    ).round().astype(int)

    overstock = (
        df["current_stock"] - df["units_sold"]
    ).clip(lower=0)

    df["waste_quantity"] = 0

    df.loc[df["expiry_days"] <= 2, "waste_quantity"] = (
        overstock * 0.40
    )

    df.loc[
        (df["expiry_days"] > 2) & (df["expiry_days"] <= 5),
        "waste_quantity"
    ] = (
        overstock * 0.20
    )

    df.loc[df["expiry_days"] > 5, "waste_quantity"] = (
        overstock * 0.10
    )

    df["waste_quantity"] = df["waste_quantity"].round().astype(int)

    df["source"] = "walmart_kaggle_dataset_with_synthetic_inventory_features"

    df["created_at"] = datetime.now().isoformat()

    final_df = df[
        [
            "product_name",
            "category",
            "store_id",
            "date",
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
            "source",
            "created_at"
        ]
    ].copy()

    conn = get_connection()

    conn.execute("DELETE FROM raw_food_sales")

    final_df.to_sql(
        "raw_food_sales",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()
    conn.close()

    print("Walmart Kaggle data loaded successfully.")
    print("Rows inserted:", len(final_df))
    print("Synthetic inventory and waste features generated successfully.")


if __name__ == "__main__":
    load_walmart_data()