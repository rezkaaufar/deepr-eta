import json

import pandas as pd

from deepr_eta.evaluate import compute_error_metrics, evaluate_model, log_metrics, relative_improvement


def test_compute_error_metrics_computes_mae_p50_p95_on_absolute_error():
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([110, 190, 340])
    # abs errors = [10, 10, 40]
    # mae = (10 + 10 + 40) / 3 = 20.0
    # p50 (numpy linear interpolation on sorted [10, 10, 40], index (3-1)*0.5=1) = 10.0
    # p95 (index (3-1)*0.95=1.9 -> 10 + 0.9*(40-10) = 37.0)

    result = compute_error_metrics(y_true, y_pred)

    assert abs(result["mae"] - 20.0) < 1e-9
    assert abs(result["p50"] - 10.0) < 1e-9
    assert abs(result["p95"] - 37.0) < 1e-9


def test_relative_improvement_is_fractional_gain_over_baseline_mae():
    # (100 - 80) / 100 = 0.2
    assert abs(relative_improvement(80, 100) - 0.2) < 1e-9
    # no improvement when equal to baseline
    assert abs(relative_improvement(100, 100) - 0.0) < 1e-9
    # candidate worse than baseline -> negative improvement: (100-120)/100 = -0.2
    assert abs(relative_improvement(120, 100) - (-0.2)) < 1e-9


def test_evaluate_model_combines_error_metrics_and_relative_improvement_vs_baseline():
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([110, 190, 340])  # candidate abs errors [10, 10, 40] -> mae 20.0 (as above)
    baseline_pred = pd.Series([150, 150, 150])
    # baseline abs errors = [50, 50, 150] -> baseline mae = (50+50+150)/3 = 83.333...
    # relative_improvement = (83.333... - 20) / 83.333... = 190/250 = 0.76 exactly

    record = evaluate_model(y_true, y_pred, baseline_pred, model_name="candidate")

    assert abs(record["mae"] - 20.0) < 1e-9
    assert abs(record["p50"] - 10.0) < 1e-9
    assert abs(record["p95"] - 37.0) < 1e-9
    assert abs(record["relative_improvement"] - 0.76) < 1e-9
    assert record["model_name"] == "candidate"
    assert "timestamp" in record and isinstance(record["timestamp"], str) and record["timestamp"]


def test_log_metrics_appends_rows_to_csv_creating_header_if_absent(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    record1 = {"model_name": "naive", "mae": 20.0, "p50": 10.0, "p95": 37.0, "relative_improvement": 0.0}
    record2 = {"model_name": "xgboost", "mae": 15.0, "p50": 8.0, "p95": 30.0, "relative_improvement": 0.25}

    log_metrics(record1, csv_path=csv_path)
    log_metrics(record2, csv_path=csv_path)

    written = pd.read_csv(csv_path)

    assert len(written) == 2
    assert list(written["model_name"]) == ["naive", "xgboost"]
    assert abs(written.iloc[1]["mae"] - 15.0) < 1e-9


def test_log_metrics_appends_records_as_json_lines(tmp_path):
    json_path = tmp_path / "metrics.json"
    record1 = {"model_name": "naive", "mae": 20.0, "p50": 10.0, "p95": 37.0, "relative_improvement": 0.0}
    record2 = {"model_name": "xgboost", "mae": 15.0, "p50": 8.0, "p95": 30.0, "relative_improvement": 0.25}

    log_metrics(record1, json_path=json_path)
    log_metrics(record2, json_path=json_path)

    lines = json_path.read_text().strip().split("\n")

    assert len(lines) == 2
    assert json.loads(lines[0]) == record1
    assert json.loads(lines[1]) == record2
