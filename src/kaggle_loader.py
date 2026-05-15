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

    df["date"] = df["Date"].astype(str)

    df["is_holiday"] = df["IsHoliday"].astype(int)

    df["promotion"] = (
        (df["MarkDown1"].fillna(0) > 0)
        | (df["MarkDown2"].fillna(0) > 0)
        | (df["MarkDown3"].fillna(0) > 0)
        | (df["MarkDown4"].fillna(0) > 0)
        | (df["MarkDown5"].fillna(0) > 0)
    ).astype(int)

    df["temperature"] = df["Temperature"].fillna(
        df["Temperature"].median()
    )

    df["units_sold"] = (
        df["Weekly_Sales"]
        .clip(lower=0) / 10
    ).round().astype(int)

    df["current_stock"] = df["units_sold"] + 20

    df["unit_price"] = 10.0

    df["expiry_days"] = 5

    df["waste_quantity"] = (
        df["current_stock"] - df["units_sold"]
    ).clip(lower=0)

    df["source"] = "walmart_kaggle_dataset"

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


if __name__ == "__main__":
    load_walmart_data()