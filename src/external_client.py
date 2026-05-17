import argparse
import time
from datetime import datetime

import requests

from src.live_ingestion import (
    generate_food_sales_record,
    generate_shifted_food_sales_record,
)
from src.logging_utils import append_csv_log


API_URL = "http://127.0.0.1:8000/sales-event"
INGESTION_LOG_PATH = "logs/ingestion_log.csv"


def post_record(record):
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


def send_sales_event():
    record = generate_food_sales_record()
    post_record(record)


def send_shifted_sales_event():
    record = generate_shifted_food_sales_record()
    post_record(record)


def stream_live_events(total_events=None, delay_seconds=5):
    print("Starting continuous normal live API event streaming...")

    count = 0

    while total_events is None or count < total_events:
        count += 1
        print(f"\nSending normal event {count}")
        send_sales_event()
        time.sleep(delay_seconds)


def stream_shifted_events(total_events=300, delay_seconds=0.2):
    print(f"Starting shifted drift event streaming: {total_events} events...")

    for i in range(total_events):
        print(f"\nSending shifted event {i + 1}")
        send_shifted_sales_event()
        time.sleep(delay_seconds)

    print("\nShifted drift event streaming completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send normal or shifted sales events to the FastAPI ingestion endpoint."
    )

    parser.add_argument(
        "--mode",
        choices=["normal", "shifted"],
        default="normal",
        help="normal = live realistic data, shifted = abnormal data for drift testing"
    )

    parser.add_argument(
        "--events",
        type=int,
        default=None,
        help="Number of events to send. If omitted in normal mode, streaming continues until stopped."
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=5,
        help="Delay in seconds between events."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "shifted":
        event_count = args.events if args.events is not None else 300
        stream_shifted_events(
            total_events=event_count,
            delay_seconds=args.delay
        )
    else:
        stream_live_events(
            total_events=args.events,
            delay_seconds=args.delay
        )