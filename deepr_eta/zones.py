import zipfile
from pathlib import Path

import pandas as pd
import shapefile
from pyproj import Transformer
from shapely.geometry import shape

from deepr_eta.data import _http_fetch

TLC_ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# TLC's official zone shapefile ships in NY State Plane Long Island, US survey
# feet (see taxi_zones.prj inside the archive) -- not lat/lon.
ZONE_SHAPEFILE_CRS = "EPSG:2263"
WGS84_CRS = "EPSG:4326"


def download_zone_shapefile(cache_dir, fetch_fn=_http_fetch) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_path = cache_dir / "taxi_zones.zip"

    if dest_path.exists():
        return dest_path

    fetch_fn(TLC_ZONE_SHAPEFILE_URL, dest_path)
    return dest_path


def _extract_shapefile(zip_path, extract_dir) -> Path:
    """Unzips `zip_path` (as returned by `download_zone_shapefile`) into
    `extract_dir` and returns the path to the `.shp` file inside, skipping
    re-extraction if a `.shp` is already present. Not one of the confirmed
    seams -- covered indirectly by running the end-to-end pipeline, not a
    dedicated unit test.
    """
    extract_dir = Path(extract_dir)
    existing = list(extract_dir.rglob("*.shp"))
    if existing:
        return existing[0]

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    extracted = list(extract_dir.rglob("*.shp"))
    if not extracted:
        raise FileNotFoundError(
            f"no .shp file found in {zip_path} after extracting to {extract_dir}"
        )
    return extracted[0]


def load_zone_centroids(shapefile_path) -> pd.DataFrame:
    transformer = Transformer.from_crs(ZONE_SHAPEFILE_CRS, WGS84_CRS, always_xy=True)

    reader = shapefile.Reader(str(shapefile_path))
    records = []
    for shape_record in reader.shapeRecords():
        geom = shape(shape_record.shape.__geo_interface__)
        centroid = geom.centroid
        lon, lat = transformer.transform(centroid.x, centroid.y)
        records.append(
            {
                "LocationID": shape_record.record["LocationID"],
                "centroid_lat": lat,
                "centroid_lon": lon,
            }
        )

    return pd.DataFrame.from_records(records).set_index("LocationID")


def map_location_ids_to_latlon(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    pu_col: str = "PULocationID",
    do_col: str = "DOLocationID",
) -> pd.DataFrame:
    result = df.copy()

    pickup = centroids.loc[df[pu_col]]
    dropoff = centroids.loc[df[do_col]]

    result["pickup_latitude"] = pickup["centroid_lat"].to_numpy()
    result["pickup_longitude"] = pickup["centroid_lon"].to_numpy()
    result["dropoff_latitude"] = dropoff["centroid_lat"].to_numpy()
    result["dropoff_longitude"] = dropoff["centroid_lon"].to_numpy()

    return result
