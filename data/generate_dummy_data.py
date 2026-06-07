"""Generate synthetic ETA dataset for deepr-eta development and testing.

Features are drawn to roughly match the schema described in Table 1 of the
DeepETA paper (arXiv:2206.02127).  This data is entirely synthetic and is
intended only for smoke-testing pipelines — it has no predictive validity.

Usage
-----
    python data/generate_dummy_data.py                         # defaults
    python data/generate_dummy_data.py --rows 50000 --seed 7
    python data/generate_dummy_data.py --output /tmp/test.csv --rows 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bounding box: San Francisco
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 37.70, 37.82
LON_MIN, LON_MAX = -122.52, -122.38


def generate(rows: int, seed: int) -> pd.DataFrame:
    """Return a DataFrame with ``rows`` synthetic ETA records."""
    rng = np.random.default_rng(seed)

    # --- Geospatial -----------------------------------------------------------
    pickup_lat = rng.uniform(LAT_MIN, LAT_MAX, rows)
    pickup_lon = rng.uniform(LON_MIN, LON_MAX, rows)
    dropoff_lat = rng.uniform(LAT_MIN, LAT_MAX, rows)
    dropoff_lon = rng.uniform(LON_MIN, LON_MAX, rows)

    # --- Temporal -------------------------------------------------------------
    request_hour = rng.integers(0, 24, rows)           # 0–23
    request_dow = rng.integers(0, 7, rows)             # 0–6
    request_month = rng.integers(1, 13, rows)          # 1–12

    # --- Categorical ----------------------------------------------------------
    segment_types = ["UberX", "UberPool", "UberBlack", "Eats"]
    segment_type = rng.choice(segment_types, rows)

    city_id = rng.integers(1, 101, rows)               # 100 distinct cities

    weather_conditions = ["clear", "rain", "snow", "fog"]
    # clear is most common
    weather_probs = [0.60, 0.25, 0.08, 0.07]
    weather_condition = rng.choice(weather_conditions, rows, p=weather_probs)

    platforms = ["ios", "android", "web"]
    platform_probs = [0.45, 0.45, 0.10]
    platform = rng.choice(platforms, rows, p=platform_probs)

    # --- Routing engine ETA (log-normal, clamped to [60, 7200] s) ------------
    # Target: mean ≈ 600 s, std ≈ 180 s  →  log-normal params:
    #   μ_ln = log(mean²/sqrt(mean²+std²))
    #   σ_ln = sqrt(log(1 + std²/mean²))
    _mean, _std = 600.0, 180.0
    mu_ln = np.log(_mean ** 2 / np.sqrt(_mean ** 2 + _std ** 2))
    sigma_ln = np.sqrt(np.log(1.0 + (_std / _mean) ** 2))
    routing_engine_eta = rng.lognormal(mu_ln, sigma_ln, rows)
    routing_engine_eta = np.clip(routing_engine_eta, 60.0, 7200.0)

    # --- Routing engine distance (≈ 1.2× eta, with noise) -------------------
    routing_engine_distance = routing_engine_eta * 1.2 * rng.uniform(0.85, 1.15, rows)

    # --- Traffic --------------------------------------------------------------
    traffic_speed = rng.uniform(1.0, 15.0, rows)           # m/s
    traffic_congestion_index = rng.uniform(0.0, 1.0, rows) # 0–1

    # --- Trip metadata --------------------------------------------------------
    surge_multiplier = rng.uniform(1.0, 3.0, rows)
    driver_rating = rng.uniform(3.5, 5.0, rows)
    num_stops = rng.integers(0, 4, rows)   # 0–3

    # --- Target: residual ETA = actual_eta - routing_engine_eta --------------
    # Base: Normal(0, 60).  Inject ~5% outliers in [-600, 600].
    residual_eta = rng.normal(0.0, 60.0, rows)
    outlier_mask = rng.random(rows) < 0.05
    residual_eta[outlier_mask] = rng.uniform(-600.0, 600.0, outlier_mask.sum())
    residual_eta = np.clip(residual_eta, -600.0, 600.0)

    df = pd.DataFrame(
        {
            "pickup_lat": pickup_lat.astype(np.float32),
            "pickup_lon": pickup_lon.astype(np.float32),
            "dropoff_lat": dropoff_lat.astype(np.float32),
            "dropoff_lon": dropoff_lon.astype(np.float32),
            "request_hour": request_hour.astype(np.int16),
            "request_dow": request_dow.astype(np.int8),
            "request_month": request_month.astype(np.int8),
            "segment_type": segment_type,
            "city_id": city_id.astype(np.int16),
            "routing_engine_eta": routing_engine_eta.astype(np.float32),
            "routing_engine_distance": routing_engine_distance.astype(np.float32),
            "traffic_speed": traffic_speed.astype(np.float32),
            "traffic_congestion_index": traffic_congestion_index.astype(np.float32),
            "weather_condition": weather_condition,
            "surge_multiplier": surge_multiplier.astype(np.float32),
            "driver_rating": driver_rating.astype(np.float32),
            "num_stops": num_stops.astype(np.int8),
            "platform": platform,
            "residual_eta": residual_eta.astype(np.float32),
        }
    )
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic ETA dataset (deepr-eta)."
    )
    parser.add_argument(
        "--output",
        default="data/dummy_data.csv",
        help="Output CSV path (default: data/dummy_data.csv)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="Number of rows to generate (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows:,} rows with seed={args.seed} …")
    df = generate(rows=args.rows, seed=args.seed)

    # --- CSV -----------------------------------------------------------------
    df.to_csv(out_path, index=False)
    print(f"CSV written  → {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    # --- Parquet -------------------------------------------------------------
    parquet_path = out_path.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Parquet written → {parquet_path}  ({parquet_path.stat().st_size / 1024:.1f} KB)")

    print(f"\nSchema ({len(df.columns)} columns, {len(df):,} rows):")
    print(df.dtypes.to_string())
    print("\nSample (first 3 rows):")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
