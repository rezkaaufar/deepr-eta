import shapefile
import pandas as pd
import pytest

from deepr_eta.zones import (
    download_zone_shapefile,
    load_zone_centroids,
    map_location_ids_to_latlon,
)


def _write_rectangle_shapefile(path, rectangles):
    """Writes a tiny synthetic shapefile of rectangular polygons in EPSG:2263.

    `rectangles` is a list of (location_id, center_x, center_y, half_width) in
    US survey feet. Each rectangle is `2*half_width` feet on a side, centered
    exactly on (center_x, center_y), so its centroid is exactly the center
    point by construction -- no need to re-derive the centroid formula.
    """
    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    writer.field("LocationID", "N")
    for location_id, cx, cy, half in rectangles:
        ring = [
            [cx - half, cy - half],
            [cx - half, cy + half],
            [cx + half, cy + half],
            [cx + half, cy - half],
            [cx - half, cy - half],
        ]
        writer.poly([ring])
        writer.record(LocationID=location_id)
    writer.close()


def test_download_zone_shapefile_fetches_and_caches_when_not_present(tmp_path):
    calls = []

    def fake_fetch(url, dest_path):
        calls.append((url, dest_path))
        dest_path.write_bytes(b"fake zip content")

    result_path = download_zone_shapefile(cache_dir=tmp_path, fetch_fn=fake_fetch)

    assert result_path.exists()
    assert result_path.read_bytes() == b"fake zip content"
    assert len(calls) == 1


def test_download_zone_shapefile_skips_fetch_when_already_cached(tmp_path):
    calls = []

    def fake_fetch(url, dest_path):
        calls.append((url, dest_path))
        dest_path.write_bytes(b"first content")

    download_zone_shapefile(cache_dir=tmp_path, fetch_fn=fake_fetch)
    result_path = download_zone_shapefile(cache_dir=tmp_path, fetch_fn=fake_fetch)

    assert len(calls) == 1
    assert result_path.read_bytes() == b"first content"


def test_load_zone_centroids_reprojects_epsg2263_to_wgs84(tmp_path):
    # Two rectangles, each centered exactly on a documented NYC landmark's
    # EPSG:2263 (NAD83 / NY Long Island, US survey feet) coordinate,
    # independently obtained from NOAA NGS's official NCAT converter
    # (https://geodesy.noaa.gov/api/ncat/llh) -- not derived from this
    # codebase's own pyproj call.
    #
    # Empire State Building: WGS84 (40.748817, -73.985428)
    #   -> NCAT NAD83(2011) NY Long Island (ftUS): Easting 988287.578, Northing 212091.217
    # Times Square: WGS84 (40.758896, -73.985130)
    #   -> NCAT NAD83(2011) NY Long Island (ftUS): Easting 988369.523, Northing 215763.338
    #
    # A rectangle centered exactly on a point has that point as its centroid
    # by construction, so the expected centroid coordinates are these
    # documented lat/lon values, not a re-derivation of this module's formula.
    shp_path = tmp_path / "zones.shp"
    _write_rectangle_shapefile(
        tmp_path / "zones",
        rectangles=[
            (1, 988287.578, 212091.217, 500.0),  # Empire State Building
            (2, 988369.523, 215763.338, 500.0),  # Times Square
        ],
    )

    centroids = load_zone_centroids(shp_path)

    assert list(centroids.index.sort_values()) == [1, 2]
    assert centroids.loc[1, "centroid_lat"] == pytest.approx(40.748817, abs=1e-4)
    assert centroids.loc[1, "centroid_lon"] == pytest.approx(-73.985428, abs=1e-4)
    assert centroids.loc[2, "centroid_lat"] == pytest.approx(40.758896, abs=1e-4)
    assert centroids.loc[2, "centroid_lon"] == pytest.approx(-73.985130, abs=1e-4)


def test_map_location_ids_to_latlon_joins_pickup_and_dropoff_coordinates():
    centroids = pd.DataFrame(
        {
            "centroid_lat": [40.75, 40.76, 40.77],
            "centroid_lon": [-73.99, -73.98, -73.97],
        },
        index=pd.Index([10, 20, 30], name="LocationID"),
    )
    df = pd.DataFrame(
        {
            "PULocationID": [10, 20],
            "DOLocationID": [30, 10],
            "trip_id": ["a", "b"],
        }
    )

    result = map_location_ids_to_latlon(df, centroids)

    assert list(result.columns) == [
        "PULocationID",
        "DOLocationID",
        "trip_id",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    # Row 0: pickup zone 10 -> (40.75, -73.99), dropoff zone 30 -> (40.77, -73.97)
    assert result.loc[0, "pickup_latitude"] == 40.75
    assert result.loc[0, "pickup_longitude"] == -73.99
    assert result.loc[0, "dropoff_latitude"] == 40.77
    assert result.loc[0, "dropoff_longitude"] == -73.97

    # Row 1: pickup zone 20 -> (40.76, -73.98), dropoff zone 10 -> (40.75, -73.99)
    assert result.loc[1, "pickup_latitude"] == 40.76
    assert result.loc[1, "pickup_longitude"] == -73.98
    assert result.loc[1, "dropoff_latitude"] == 40.75
    assert result.loc[1, "dropoff_longitude"] == -73.99

    # original df is untouched (function returns a copy)
    assert "pickup_latitude" not in df.columns
