from datetime import datetime

from src.drift_detection import run_drift_check
from src.train import train_model
from src.logging_utils import append_csv_log


RETRAINING_LOG_PATH = "logs/retraining_log.csv"


def auto_retrain():
    drift_result = run_drift_check()

    if drift_result.get("drift_detected") is True:
        print("Drift detected. Starting automatic retraining...")

        metrics = train_model()

        log_row = {
            "timestamp": datetime.now().isoformat(),
            "status": "retrained",
            "reason": "drift_detected",
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "training_rows": metrics["training_rows"],
            "model_path": metrics["model_path"]
        }

        append_csv_log(RETRAINING_LOG_PATH, log_row)

        return {
            "status": "retrained",
            "reason": "drift_detected",
            "metrics": metrics,
            "drift_result": drift_result
        }

    print("No drift detected. Retraining skipped.")

    log_row = {
        "timestamp": datetime.now().isoformat(),
        "status": "skipped",
        "reason": "no_drift_detected",
        "mae": None,
        "rmse": None,
        "r2": None,
        "training_rows": None,
        "model_path": None
    }

    append_csv_log(RETRAINING_LOG_PATH, log_row)

    return {
        "status": "skipped",
        "reason": "no_drift_detected",
        "drift_result": drift_result
    }


if __name__ == "__main__":
    result = auto_retrain()
    print(result)