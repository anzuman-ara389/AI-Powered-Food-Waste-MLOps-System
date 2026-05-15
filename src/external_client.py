import time
from datetime import datetime

import requests

from live_ingestion import generate_food_sales_record
from logging_utils import append_csv_log


API_URL = "http://127.0.0.1:8000/sales-event"
INGESTION_LOG_PATH = "logs/ingestion_log.csv"


def send_sales_event():
    record = generate_food_sales_record()

    try:
        response = requests.post(
            API_URL,
            json=record,
            timeout=30
        )

        log_row = {
            "timestamp": datetime.now().isoformat(),
            "status_code": response.status_code,
            "product_name": record["product_name"],
            "category": record["category"],
            "store_id": record["store_id"],
            "current_stock": record["current_stock"],
            "units_sold": record["units_sold"],
            "waste_quantity": record["waste_quantity"],
            "source": record["source"]
        }

        append_csv_log(INGESTION_LOG_PATH, log_row)

        print("Status Code:", response.status_code)
        print("Response:", response.json())

    except requests.exceptions.RequestException as e:
        print("API connection error:", e)


def stream_live_events(total_events=5, delay_seconds=2):
    print(f"Streaming {total_events} live events...")

    for i in range(total_events):
        print(f"\nSending event {i + 1}")
        send_sales_event()
        time.sleep(delay_seconds)

    print("\nLive event streaming completed.")


if __name__ == "__main__":
    stream_live_events(total_events=5, delay_seconds=2)