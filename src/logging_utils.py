import os
import json
import pandas as pd


def append_csv_log(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame([row])

    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)