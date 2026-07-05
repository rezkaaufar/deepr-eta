import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def compute_error_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    abs_error = (pd.Series(y_true).reset_index(drop=True) - pd.Series(y_pred).reset_index(drop=True)).abs()

    return {
        "mae": abs_error.mean(),
        "p50": np.percentile(abs_error, 50),
        "p95": np.percentile(abs_error, 95),
    }


def relative_improvement(candidate_mae: float, baseline_mae: float) -> float:
    return (baseline_mae - candidate_mae) / baseline_mae


def evaluate_model(
    y_true: pd.Series, y_pred: pd.Series, baseline_pred: pd.Series, model_name: str
) -> dict:
    metrics = compute_error_metrics(y_true, y_pred)
    baseline_metrics = compute_error_metrics(y_true, baseline_pred)

    return {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
        "relative_improvement": relative_improvement(metrics["mae"], baseline_metrics["mae"]),
    }


def log_metrics(record: dict, csv_path=None, json_path=None) -> None:
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([record])
        row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "a") as f:
            f.write(json.dumps(record) + "\n")
