import numpy as np
import pandas as pd
from datetime import datetime

from src.database import get_connection, init_db
from src.logging_utils import append_csv_log


DRIFT_FEATURES = [
    "units_sold",
    "current_stock",
    "temperature",
    "waste_quantity"
]

DRIFT_THRESHOLD = 0.15
REFERENCE_WINDOW_SIZE = 1000
CURRENT_WINDOW_SIZE = 300
DRIFT_SUMMARY_PATH = "artifacts/drift_summary.csv"


def calculate_drift_score(reference_mean, current_mean):
    if reference_mean == 0:
        return 0

    return abs(current_mean - reference_mean) / abs(reference_mean)


def run_drift_check():
    init_db()

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM raw_food_sales ORDER BY id ASC",
        conn
    )

    if len(df) < REFERENCE_WINDOW_SIZE + CURRENT_WINDOW_SIZE:
        conn.close()

        return {
            "message": (
                "Not enough data for drift detection. "
                f"Need at least {REFERENCE_WINDOW_SIZE + CURRENT_WINDOW_SIZE} rows."
            ),
            "drift_detected": False
        }

    reference_df = df.iloc[-(REFERENCE_WINDOW_SIZE + CURRENT_WINDOW_SIZE):-CURRENT_WINDOW_SIZE]
    current_df = df.iloc[-CURRENT_WINDOW_SIZE:]

    drift_results = []
    overall_drift_detected = False

    cursor = conn.cursor()

    for feature in DRIFT_FEATURES:
        reference_mean = pd.to_numeric(
            reference_df[feature],
            errors="coerce"
        ).mean()

        current_mean = pd.to_numeric(
            current_df[feature],
            errors="coerce"
        ).mean()

        reference_mean = float(np.nan_to_num(reference_mean))
        current_mean = float(np.nan_to_num(current_mean))

        drift_score = calculate_drift_score(
            reference_mean,
            current_mean
        )

        feature_drift_detected = drift_score > DRIFT_THRESHOLD

        if feature_drift_detected:
            overall_drift_detected = True

        result = {
            "feature_name": feature,
            "reference_mean": round(reference_mean, 4),
            "current_mean": round(current_mean, 4),
            "drift_score": round(drift_score, 4),
            "drift_detected": feature_drift_detected,
            "threshold": DRIFT_THRESHOLD,
            "reference_window_size": REFERENCE_WINDOW_SIZE,
            "current_window_size": CURRENT_WINDOW_SIZE,
            "created_at": datetime.now().isoformat()
        }

        drift_results.append(result)

        cursor.execute("""
            INSERT INTO drift_reports (
                feature_name,
                reference_mean,
                current_mean,
                drift_score,
                drift_detected,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            feature,
            reference_mean,
            current_mean,
            drift_score,
            int(feature_drift_detected),
            datetime.now().isoformat()
        ))

        append_csv_log(DRIFT_SUMMARY_PATH, result)

    conn.commit()
    conn.close()

    return {
        "drift_detected": overall_drift_detected,
        "threshold": DRIFT_THRESHOLD,
        "reference_window_size": REFERENCE_WINDOW_SIZE,
        "current_window_size": CURRENT_WINDOW_SIZE,
        "drift_results": drift_results
    }


if __name__ == "__main__":
    result = run_drift_check()
    print(result)