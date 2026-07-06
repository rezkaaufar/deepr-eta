"""Run the naive ETA baseline through the evaluation harness end-to-end.

Downloads (or reuses a cached copy of) NYC yellow taxi trip data for March 2016,
cleans it, splits it chronologically into train/val/test, fits the hour-of-day
average speed on the train split only, predicts the naive ETA on the held-out
test split, and logs the resulting baseline metrics report to
results/metrics.csv and results/metrics.json.

For this first baseline run, the naive ETA prediction is used as both the
candidate and the baseline prediction, so relative_improvement is expected to
be 0.0 -- that's the correct, expected value for the baseline evaluating
itself.

Usage:
    python scripts/run_naive_baseline.py [--sample-size N]

If network egress or the full ~2GB March 2016 file is unavailable, pass
--sample-size with a synthetic fallback size to still exercise the harness
end-to-end (see `_synthetic_trips` below).
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from deepr_eta.data import clean_trips, download_month, sequential_split, subsample
from deepr_eta.evaluate import evaluate_model, log_metrics
from deepr_eta.naive_eta import fit_hourly_avg_speed, predict_naive_eta

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results"

REQUIRED_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}


def _synthetic_trips(n: int, seed: int = 0) -> pd.DataFrame:
    """Fallback synthetic trip data used only if the real download fails.

    Produces plausible NYC-bounded trips spread across ~2 months so that
    sequential_split has enough distinct days to carve out val/test windows.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2016-03-01")
    pickup_offsets = rng.integers(0, 60 * 24 * 60, size=n)  # minutes into a 60-day window
    pickup_times = start + pd.to_timedelta(pickup_offsets, unit="m")

    pickup_lat = rng.uniform(40.6, 40.9, size=n)
    pickup_lon = rng.uniform(-74.05, -73.9, size=n)
    dropoff_lat = pickup_lat + rng.uniform(-0.05, 0.05, size=n)
    dropoff_lon = pickup_lon + rng.uniform(-0.05, 0.05, size=n)

    duration_seconds = rng.uniform(120, 3000, size=n)
    dropoff_times = pickup_times + pd.to_timedelta(duration_seconds, unit="s")

    return pd.DataFrame({
        "tpep_pickup_datetime": pickup_times,
        "tpep_dropoff_datetime": dropoff_times,
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lon,
        "dropoff_latitude": dropoff_lat,
        "dropoff_longitude": dropoff_lon,
    })


def _load_trip_data(sample_size) -> Tuple[pd.DataFrame, str]:
    """Returns (raw trips df, note describing the data source actually used)."""
    try:
        path = download_month(2016, 3, cache_dir=CACHE_DIR)
        df = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"downloaded yellow_tripdata_2016-03.parquet is missing required "
                f"lat/lon columns {sorted(missing)} -- TLC has since republished this "
                f"file using PULocationID/DOLocationID instead of raw coordinates, so "
                f"the historical lat/lon schema this pipeline expects is no longer "
                f"served at the standard URL for this month."
            )
        if sample_size is not None:
            df = subsample(df, sample_size=sample_size)
        return df, f"real March 2016 TLC data ({len(df)} rows after any subsampling)"
    except Exception as exc:  # network unavailable, file missing, schema drift, etc.
        n = sample_size or 20000
        print(f"Real data download/load failed ({exc!r}); falling back to synthetic data (n={n}).")
        return _synthetic_trips(n), (
            f"SYNTHETIC fallback data (n={n} rows) -- real download/load failed: {exc!r}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Optional row subsample size applied to the raw downloaded data.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, data_note = _load_trip_data(args.sample_size)
    print(f"Data source: {data_note}")

    cleaned = clean_trips(raw_df)
    print(f"Cleaned trips: {len(cleaned)} (from {len(raw_df)} raw rows)")

    train_df, val_df, test_df = sequential_split(cleaned)
    print(f"Split sizes -- train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    hourly_speed = fit_hourly_avg_speed(train_df)
    print("Fitted hourly average speeds (mph) from train split:")
    print(hourly_speed)

    naive_pred = predict_naive_eta(test_df, hourly_speed)
    y_true = (
        test_df["tpep_dropoff_datetime"] - test_df["tpep_pickup_datetime"]
    ).dt.total_seconds()

    # First baseline report: the naive ETA is evaluated against itself, so
    # relative_improvement is expected to be 0.0.
    record = evaluate_model(
        y_true=y_true,
        y_pred=naive_pred,
        baseline_pred=naive_pred,
        model_name="naive_eta_baseline",
    )
    record["data_source"] = data_note
    print("Baseline metrics record:")
    print(record)

    log_metrics(
        record,
        csv_path=RESULTS_DIR / "metrics.csv",
        json_path=RESULTS_DIR / "metrics.json",
    )
    print(f"Logged metrics to {RESULTS_DIR / 'metrics.csv'} and {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
