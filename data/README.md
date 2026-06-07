# data/

This directory contains data utilities for deepr-eta.

## Generating Dummy Data

The script `generate_dummy_data.py` creates a fully synthetic dataset that
mirrors the feature schema described in Table 1 of the DeepETA paper
(arXiv:2206.02127).  It is intended for smoke-testing the pipeline only — the
data has no real predictive validity.

### Quick start

```bash
# From the repo root — defaults: 10,000 rows, seed 42, output data/dummy_data.csv
python data/generate_dummy_data.py

# Custom size and path
python data/generate_dummy_data.py --rows 50000 --seed 7 --output /tmp/eta_dev.csv
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `data/dummy_data.csv` | Output CSV path (a matching `.parquet` is also written) |
| `--rows` | `10000` | Number of rows to generate |
| `--seed` | `42` | Random seed for reproducibility |

### Output files

Both `data/dummy_data.csv` and `data/dummy_data.parquet` are listed in
`.gitignore` — they should not be committed.

### Feature schema

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `pickup_lat` | float32 | 37.70–37.82 | Pickup latitude (SF bounding box) |
| `pickup_lon` | float32 | -122.52–-122.38 | Pickup longitude |
| `dropoff_lat` | float32 | 37.70–37.82 | Dropoff latitude |
| `dropoff_lon` | float32 | -122.52–-122.38 | Dropoff longitude |
| `request_hour` | int16 | 0–23 | Hour of the trip request |
| `request_dow` | int8 | 0–6 | Day of week (0=Mon) |
| `request_month` | int8 | 1–12 | Month of the trip request |
| `segment_type` | str | UberX / UberPool / UberBlack / Eats | Product segment |
| `city_id` | int16 | 1–100 | City identifier (100 distinct cities) |
| `routing_engine_eta` | float32 | 60–7200 s | Routing engine ETA (log-normal) |
| `routing_engine_distance` | float32 | — m | Estimated trip distance (≈ 1.2× ETA) |
| `traffic_speed` | float32 | 1–15 m/s | Real-time traffic speed |
| `traffic_congestion_index` | float32 | 0.0–1.0 | Congestion level |
| `weather_condition` | str | clear / rain / snow / fog | Weather at request time |
| `surge_multiplier` | float32 | 1.0–3.0 | Demand surge multiplier |
| `driver_rating` | float32 | 3.5–5.0 | Driver's average rating |
| `num_stops` | int8 | 0–3 | Number of intermediate stops |
| `platform` | str | ios / android / web | Rider platform |
| `residual_eta` | float32 | -600–600 s | **Target label**: actual_eta − routing_engine_eta |
