import pandas as pd

from deepr_eta.data import haversine_miles


def fit_hourly_avg_speed(train_df: pd.DataFrame) -> pd.Series:
    hour = train_df["tpep_pickup_datetime"].dt.hour
    duration_hours = (
        train_df["tpep_dropoff_datetime"] - train_df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 3600
    distance = haversine_miles(
        train_df["pickup_latitude"], train_df["pickup_longitude"],
        train_df["dropoff_latitude"], train_df["dropoff_longitude"],
    )
    speed_mph = distance / duration_hours

    return speed_mph.groupby(hour).mean().reindex(range(24))


def predict_naive_eta(df: pd.DataFrame, hourly_speed: pd.Series) -> pd.Series:
    hour = df["tpep_pickup_datetime"].dt.hour
    speed = hour.map(hourly_speed)

    if speed.isna().any():
        missing_hours = sorted(hour[speed.isna()].unique().tolist())
        raise ValueError(
            f"hourly_speed has no entry for hour(s): {missing_hours}"
        )

    distance = haversine_miles(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    return distance / speed * 3600
