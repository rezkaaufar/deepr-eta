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
from deepr_eta.zones import (
    _extract_shapefile,
    download_zone_shapefile,
    load_zone_centroids,
    map_location_ids_to_latlon,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "raw"
ZONE_CACHE_DIR = REPO_ROOT / "data" / "zones"
RESULTS_DIR = REPO_ROOT / "results"

REQUIRED_DATETIME_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
}
REQUIRED_LATLON_COLUMNS = {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}
LOCATION_ID_COLUMNS = {"PULocationID", "DOLocationID"}


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


def _load_via_zone_centroid_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Maps PULocationID/DOLocationID to lat/lon via TLC's zone shapefile centroids.

    TLC has retroactively republished March 2016 (and other historical months)
    using taxi zone IDs instead of raw lat/lon. There is no trustworthy
    zero-friction vendored lat/lon-schema copy of this data (checked TLC's own
    CDN, the AWS Registry of Open Data, and archive.org; BigQuery's and
    Kaggle's copies have the right columns but need external account
    credentials). Instead we recover an approximate but real coordinate signal
    by mapping each zone ID to that zone's polygon centroid.

    Accepted tradeoff: many trips have pickup and dropoff in the same zone;
    after centroid-mapping those collapse to distance == 0 and are dropped by
    clean_trips' existing valid_distance filter. This is expected.
    """
    zone_zip_path = download_zone_shapefile(cache_dir=ZONE_CACHE_DIR)
    shp_path = _extract_shapefile(zone_zip_path, ZONE_CACHE_DIR / "taxi_zones")
    centroids = load_zone_centroids(shp_path)

    # TLC's zone lookup includes a few placeholder location IDs with no actual
    # zone geometry (264 = "Unknown", 265 = "Outside of NYC"), so they have no
    # centroid to join. Drop trips referencing them here -- this is a known,
    # documented TLC data quirk, not a bug in the centroid mapping itself.
    known_zone_ids = centroids.index
    has_known_zones = df["PULocationID"].isin(known_zone_ids) & df["DOLocationID"].isin(known_zone_ids)
    n_dropped = int((~has_known_zones).sum())
    if n_dropped:
        print(
            f"Dropping {n_dropped} trips referencing PULocationID/DOLocationID values "
            f"with no zone geometry (e.g. TLC's placeholder 'Unknown'/'Outside of NYC' "
            f"IDs 264/265), which have no centroid to map to."
        )
    df = df[has_known_zones].reset_index(drop=True)

    return map_location_ids_to_latlon(df, centroids)


def _load_trip_data(sample_size) -> Tuple[pd.DataFrame, str]:
    """Returns (raw trips df, note describing the data source actually used)."""
    try:
        path = download_month(2016, 3, cache_dir=CACHE_DIR)
        df = pd.read_parquet(path)

        missing_datetime = REQUIRED_DATETIME_COLUMNS - set(df.columns)
        if missing_datetime:
            raise ValueError(
                f"downloaded yellow_tripdata_2016-03.parquet is missing required "
                f"datetime columns {sorted(missing_datetime)}"
            )

        has_latlon = REQUIRED_LATLON_COLUMNS <= set(df.columns)
        has_location_ids = LOCATION_ID_COLUMNS <= set(df.columns)

        if has_latlon:
            if sample_size is not None:
                df = subsample(df, sample_size=sample_size)
            return df, (
                f"real March 2016 TLC data with native lat/lon columns "
                f"({len(df)} rows after any subsampling)"
            )

        if has_location_ids:
            print(
                "Downloaded parquet uses PULocationID/DOLocationID (TLC's schema "
                "change) instead of raw lat/lon -- mapping zone IDs to lat/lon via "
                "TLC's official zone shapefile centroids."
            )
            try:
                df = _load_via_zone_centroid_mapping(df)
            except Exception as zone_exc:
                raise ValueError(
                    f"PULocationID/DOLocationID present but zone-centroid mapping "
                    f"failed: {zone_exc!r}"
                ) from zone_exc

            if sample_size is not None:
                df = subsample(df, sample_size=sample_size)
            return df, (
                f"real March 2016 TLC data ({len(df)} rows after any subsampling) -- "
                f"TLC republished this file using PULocationID/DOLocationID instead of "
                f"raw coordinates, so pickup/dropoff lat/lon here are ZONE-CENTROID-"
                f"DERIVED (via TLC's official taxi_zones.zip shapefile reprojected "
                f"from EPSG:2263 to WGS84), not raw GPS points. Trips with pickup and "
                f"dropoff in the same zone collapse to distance == 0 and are dropped "
                f"by clean_trips' valid_distance filter -- expected, not a bug."
            )

        raise ValueError(
            f"downloaded yellow_tripdata_2016-03.parquet has neither the raw "
            f"lat/lon schema {sorted(REQUIRED_LATLON_COLUMNS)} nor the zone-ID "
            f"schema {sorted(LOCATION_ID_COLUMNS)} this pipeline can use; columns "
            f"present: {sorted(df.columns)}"
        )
    except Exception as exc:  # network unavailable, file missing, schema unusable, etc.
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
