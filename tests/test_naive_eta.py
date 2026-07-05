import pandas as pd
import pytest

from deepr_eta.naive_eta import fit_hourly_avg_speed, predict_naive_eta


def _base_row(**overrides):
    row = {
        "tpep_pickup_datetime": pd.Timestamp("2016-03-01 08:00:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2016-03-01 08:30:00"),
        "pickup_latitude": 40.0,
        "pickup_longitude": -74.0,
        "dropoff_latitude": 40.1,
        "dropoff_longitude": -74.0,
    }
    row.update(overrides)
    return row


def test_fit_hourly_avg_speed_averages_speed_within_hour_and_covers_full_day():
    # Trip A: hour 8, dlat=0.1 deg -> haversine (same longitude) = 3958.8 * (0.1*pi/180) = 6.909409442795 mi,
    # duration 30 min = 0.5h -> speed = 6.909409442795 / 0.5 = 13.818818885590 mph
    trip_a = _base_row()

    # Trip B: hour 8, dlat=0.05 deg -> distance = 6.909409442795 / 2 = 3.4547047213975 mi,
    # duration 30 min = 0.5h -> speed = 3.4547047213975 / 0.5 = 6.909409442795 mph
    trip_b = _base_row(dropoff_latitude=40.05)

    # Trip C: hour 14, dlat=0.1 deg -> distance = 6.909409442795 mi, duration 6 min = 0.1h
    # -> speed = 6.909409442795 / 0.1 = 69.09409442795 mph
    trip_c = _base_row(
        tpep_pickup_datetime=pd.Timestamp("2016-03-01 14:00:00"),
        tpep_dropoff_datetime=pd.Timestamp("2016-03-01 14:06:00"),
    )

    df = pd.DataFrame([trip_a, trip_b, trip_c])

    result = fit_hourly_avg_speed(df)

    assert list(result.index) == list(range(24))
    # hour 8 average of 13.818818885590 and 6.909409442795
    assert abs(result[8] - 10.3641141641925) < 1e-6
    assert abs(result[14] - 69.09409442795) < 1e-6
    assert pd.isna(result[3])


def test_fit_hourly_avg_speed_only_reflects_the_frame_it_is_given():
    # Trip A (hour 8, speed 13.818818885590 mph) and Trip B (hour 8, speed 6.909409442795 mph)
    # as computed above. Fitting on each frame separately must produce that frame's own
    # hour-8 speed, not a value blended with data the function was never given -- there is
    # no global state, so the only way leakage could occur is via the caller passing the
    # wrong frame in.
    trip_a = _base_row()
    trip_b = _base_row(dropoff_latitude=40.05)

    result_a_only = fit_hourly_avg_speed(pd.DataFrame([trip_a]))
    result_b_only = fit_hourly_avg_speed(pd.DataFrame([trip_b]))

    assert abs(result_a_only[8] - 13.818818885590) < 1e-6
    assert abs(result_b_only[8] - 6.909409442795) < 1e-6


def test_predict_naive_eta_computes_seconds_from_distance_and_hourly_speed():
    # Row 1: hour 8, dlat=0.1 deg -> distance = 6.909409442795 mi, hourly_speed[8] = 10.0 mph
    # -> predicted seconds = 6.909409442795 / 10.0 * 3600 = 2487.3873993862
    row1 = _base_row()

    # Row 2: hour 14, dlat=0.05 deg -> distance = 3.4547047213975 mi, hourly_speed[14] = 20.0 mph
    # -> predicted seconds = 3.4547047213975 / 20.0 * 3600 = 621.84684985155
    row2 = _base_row(
        tpep_pickup_datetime=pd.Timestamp("2016-03-01 14:00:00"),
        tpep_dropoff_datetime=pd.Timestamp("2016-03-01 14:06:00"),
        dropoff_latitude=40.05,
    )

    df = pd.DataFrame([row1, row2])
    hourly_speed = pd.Series({8: 10.0, 14: 20.0})

    result = predict_naive_eta(df, hourly_speed)

    assert abs(result.iloc[0] - 2487.3873993862) < 1e-4
    assert abs(result.iloc[1] - 621.84684985155) < 1e-4


def test_predict_naive_eta_raises_when_hour_missing_from_hourly_speed():
    row = _base_row()  # hour 8
    df = pd.DataFrame([row])
    hourly_speed = pd.Series({9: 10.0})  # no entry for hour 8

    with pytest.raises(ValueError):
        predict_naive_eta(df, hourly_speed)
