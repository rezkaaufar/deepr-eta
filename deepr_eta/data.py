from pathlib import Path

import numpy as np
import pandas as pd
import requests

NYC_LAT_BOUNDS = (40.5, 41.0)
NYC_LON_BOUNDS = (-74.3, -73.7)
EARTH_RADIUS_MILES = 3958.8
MAX_PLAUSIBLE_MPH = 100
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def _http_fetch(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def _within_nyc_bounds(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return (
        lat.between(*NYC_LAT_BOUNDS)
        & lon.between(*NYC_LON_BOUNDS)
    )


def haversine_miles(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    valid_duration = (duration > 0) & (duration <= 3 * 3600)

    valid_bounds = _within_nyc_bounds(
        df["pickup_latitude"], df["pickup_longitude"]
    ) & _within_nyc_bounds(df["dropoff_latitude"], df["dropoff_longitude"])

    distance = haversine_miles(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    valid_distance = distance > 0

    speed_mph = distance / (duration / 3600)
    valid_speed = speed_mph <= MAX_PLAUSIBLE_MPH

    return df[valid_duration & valid_bounds & valid_distance & valid_speed].reset_index(drop=True)


def sequential_split(
    df: pd.DataFrame,
    pickup_col: str = "tpep_pickup_datetime",
    test_days: int = 7,
    val_frac: float = 0.1,
):
    df_sorted = df.sort_values(pickup_col).reset_index(drop=True)
    dates = df_sorted[pickup_col].dt.normalize()

    unique_dates = sorted(dates.unique())
    test_dates = set(unique_dates[-test_days:]) if test_days > 0 else set()

    train_val_dates = [d for d in unique_dates if d not in test_dates]
    n_val_days = round(len(train_val_dates) * val_frac)
    val_dates = set(train_val_dates[len(train_val_dates) - n_val_days:]) if n_val_days > 0 else set()

    is_test = dates.isin(test_dates)
    is_val = dates.isin(val_dates)
    is_train = ~is_test & ~is_val

    train_df = df_sorted[is_train].reset_index(drop=True)
    val_df = df_sorted[is_val].reset_index(drop=True)
    test_df = df_sorted[is_test].reset_index(drop=True)

    return train_df, val_df, test_df


def subsample(df: pd.DataFrame, sample_size, seed: int = 0) -> pd.DataFrame:
    if sample_size is None:
        return df
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def download_month(year: int, month: int, cache_dir, fetch_fn=_http_fetch) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_path = cache_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"

    if dest_path.exists():
        return dest_path

    url = f"{TLC_BASE_URL}/yellow_tripdata_{year}-{month:02d}.parquet"
    fetch_fn(url, dest_path)
    return dest_path
