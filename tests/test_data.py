import pandas as pd

from deepr_eta.data import clean_trips, download_month, sequential_split, subsample


def _base_row(**overrides):
    row = {
        "tpep_pickup_datetime": pd.Timestamp("2016-03-01 08:00:00"),
        "tpep_dropoff_datetime": pd.Timestamp("2016-03-01 08:10:00"),
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.99,
        "dropoff_latitude": 40.76,
    }
    row.update(overrides)
    return row


def test_clean_trips_drops_nonpositive_and_overlong_duration():
    df = pd.DataFrame(
        [
            _base_row(),  # 10 min - valid
            _base_row(tpep_dropoff_datetime=pd.Timestamp("2016-03-01 08:00:00")),  # 0s - invalid
            _base_row(tpep_dropoff_datetime=pd.Timestamp("2016-03-01 07:59:00")),  # negative - invalid
            _base_row(tpep_dropoff_datetime=pd.Timestamp("2016-03-01 11:30:00")),  # 3.5h - invalid
        ]
    )

    cleaned = clean_trips(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["tpep_dropoff_datetime"] == pd.Timestamp("2016-03-01 08:10:00")


def test_clean_trips_drops_trips_outside_nyc_bounding_box():
    df = pd.DataFrame(
        [
            _base_row(),  # inside box - valid
            _base_row(pickup_latitude=42.0),  # pickup lat north of box
            _base_row(pickup_longitude=-75.0),  # pickup lon west of box
            _base_row(dropoff_latitude=39.0),  # dropoff lat south of box
            _base_row(dropoff_longitude=-73.0),  # dropoff lon east of box
        ]
    )

    cleaned = clean_trips(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["pickup_latitude"] == 40.75


def test_clean_trips_drops_zero_distance_with_nonzero_duration():
    df = pd.DataFrame(
        [
            _base_row(),  # nonzero distance - valid
            _base_row(dropoff_longitude=-73.98, dropoff_latitude=40.75),  # pickup == dropoff - invalid
        ]
    )

    cleaned = clean_trips(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["dropoff_latitude"] == 40.76


def test_clean_trips_drops_implausible_speed():
    df = pd.DataFrame(
        [
            _base_row(),  # ~5 mph over 10 min - valid
            _base_row(
                tpep_dropoff_datetime=pd.Timestamp("2016-03-01 08:00:01")
            ),  # same distance covered in 1s - implausible speed - invalid
        ]
    )

    cleaned = clean_trips(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["tpep_dropoff_datetime"] == pd.Timestamp("2016-03-01 08:10:00")


def test_sequential_split_produces_chronological_train_val_test():
    dates = pd.date_range("2016-03-01", periods=10, freq="D")
    df = pd.DataFrame({"tpep_pickup_datetime": dates, "value": range(10)})

    train_df, val_df, test_df = sequential_split(df, test_days=2, val_frac=0.25)

    assert list(test_df["value"]) == [8, 9]
    assert list(val_df["value"]) == [6, 7]
    assert list(train_df["value"]) == [0, 1, 2, 3, 4, 5]


def test_sequential_split_has_no_date_overlap_across_splits():
    dates = []
    for day in pd.date_range("2016-03-01", periods=5, freq="D"):
        dates.extend([day + pd.Timedelta(hours=h) for h in (1, 12, 23)])
    df = pd.DataFrame({"tpep_pickup_datetime": dates, "value": range(len(dates))})

    train_df, val_df, test_df = sequential_split(df, test_days=2, val_frac=0.2)

    train_dates = train_df["tpep_pickup_datetime"].dt.normalize()
    val_dates = val_df["tpep_pickup_datetime"].dt.normalize()
    test_dates = test_df["tpep_pickup_datetime"].dt.normalize()

    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_subsample_returns_full_dataframe_when_sample_size_is_none():
    df = pd.DataFrame({"value": range(20)})

    result = subsample(df, sample_size=None, seed=42)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))


def test_subsample_returns_reproducible_sample_of_requested_size():
    df = pd.DataFrame({"value": range(1000)})

    result_a = subsample(df, sample_size=50, seed=42)
    result_b = subsample(df, sample_size=50, seed=42)

    assert len(result_a) == 50
    pd.testing.assert_frame_equal(
        result_a.reset_index(drop=True), result_b.reset_index(drop=True)
    )


def test_download_month_fetches_and_caches_when_not_present(tmp_path):
    calls = []

    def fake_fetch(url, dest_path):
        calls.append((url, dest_path))
        dest_path.write_bytes(b"fake parquet content")

    result_path = download_month(2016, 3, cache_dir=tmp_path, fetch_fn=fake_fetch)

    assert result_path.exists()
    assert result_path.read_bytes() == b"fake parquet content"
    assert len(calls) == 1


def test_download_month_skips_fetch_when_already_cached(tmp_path):
    calls = []

    def fake_fetch(url, dest_path):
        calls.append((url, dest_path))
        dest_path.write_bytes(b"first content")

    download_month(2016, 3, cache_dir=tmp_path, fetch_fn=fake_fetch)
    result_path = download_month(2016, 3, cache_dir=tmp_path, fetch_fn=fake_fetch)

    assert len(calls) == 1
    assert result_path.read_bytes() == b"first content"
