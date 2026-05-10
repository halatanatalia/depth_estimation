import csv
import os

METRIC_FIELDS = [
    "image_id",
    "RMSE",
    "MAE",
    "AbsRel",
    "SqRel",
    "δ<1.25",
    "δ<1.25²",
    "δ<1.25³",
    "scale",
]


def init_metrics_file(csv_path: str, overwrite: bool = True):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    mode = "w" if overwrite else "a"
    need_header = overwrite or (not os.path.exists(csv_path))

    with open(csv_path, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if need_header:
            writer.writeheader()



def append_metrics(csv_path: str, image_id: str, metrics_dict: dict):
    row = {"image_id": image_id}
    row.update(metrics_dict)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writerow(row)
