# ETADataset: PyTorch Dataset for DeeprETA; loads rows, applies all preprocessing.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.geohash_utils import GeohashEncoder, MultipleFeatureHasher
from data.preprocessing import QuantileBucketizer


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ConfigLike = Dict[str, Any]
BucketizerMap = Dict[str, QuantileBucketizer]


# ---------------------------------------------------------------------------
# Default feature lists (aligned with config/default_config.yaml)
# ---------------------------------------------------------------------------
_DEFAULT_CONTINUOUS: Tuple[str, ...] = (
    "minute_of_day",
    "re_eta",
    "estimated_distance",
    "realtime_speed",
    "historical_speed",
)

_DEFAULT_CATEGORICAL: Tuple[str, ...] = (
    "day_of_week",
    "minute_of_week",
    "trip_type",
    "route_type",
    "request_type",
    "country_id",
    "region_id",
    "city_id",
)

_DEFAULT_GEO_PAIRS: Dict[str, Tuple[str, str]] = {
    "origin": ("origin_lat", "origin_lng"),
    "destination": ("destination_lat", "destination_lng"),
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cfg(config: ConfigLike, *keys: str, default: Any = None) -> Any:
    """Drill into a nested dict with dot-path keys; return default if missing."""
    node = config
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ETADataset(Dataset):
    """
    Loads CSV/Parquet ETA data, applies preprocessing, and returns per-row tensors.

    Returns a tuple (feature_dict, label) where:
      - feature_dict maps feature names to int64 index tensors (for embedding lookup)
      - label is a float32 scalar: residual = actual_travel_time - re_eta
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        config: ConfigLike,
        bucketizers: Optional[BucketizerMap] = None,
        train: bool = False,
        bucketizer_save_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Parameters
        ----------
        data_path:
            Path to a CSV or Parquet file.
        config:
            Config dict (loaded from default_config.yaml or equivalent).
        bucketizers:
            Pre-fitted {feature_name: QuantileBucketizer} map. If train=True and
            this is None, bucketizers are fitted from the loaded data.
        train:
            If True, fit bucketizers on this dataset's continuous features.
        bucketizer_save_dir:
            If provided (and train=True), save fitted bucketizers here.
        """
        self.config = config
        self.train = train

        # --- Load data ---------------------------------------------------
        data_path = Path(data_path)
        if data_path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        # --- Filter outliers ---------------------------------------------
        max_ata = _cfg(config, "data", "max_actual_travel_time", default=7200)
        label_col = _cfg(config, "data", "label_col", default="actual_travel_time")
        re_eta_col = _cfg(config, "data", "re_eta_col", default="re_eta")

        df = df[df[label_col] <= max_ata].reset_index(drop=True)

        # --- Resolve feature lists from config ---------------------------
        self.continuous_features: List[str] = list(
            _cfg(config, "continuous_features", default=list(_DEFAULT_CONTINUOUS))
        )
        self.categorical_features: List[str] = list(
            _cfg(config, "categorical_features", default=list(_DEFAULT_CATEGORICAL))
        )
        geo_pairs_cfg = _cfg(config, "geospatial_feature_pairs", default=None)
        if geo_pairs_cfg:
            self.geo_pairs: Dict[str, Tuple[str, str]] = {
                k: tuple(v) for k, v in geo_pairs_cfg.items()  # type: ignore[misc]
            }
        else:
            self.geo_pairs = dict(_DEFAULT_GEO_PAIRS)

        # --- Geospatial helpers -----------------------------------------
        resolutions = _cfg(config, "geospatial", "geohash_resolutions", default=[4, 5, 6, 7])
        seeds = _cfg(config, "geospatial", "hash_seeds", default=[0, 42])
        bucket_size = _cfg(config, "embedding", "hash_bucket_size", default=100_000)
        self.geo_encoder = GeohashEncoder(resolutions=resolutions)
        self.feature_hasher = MultipleFeatureHasher(
            hash_bucket_size=bucket_size, seeds=seeds
        )

        # --- Quantile bucketizers ----------------------------------------
        n_buckets = _cfg(config, "quantile", "n_buckets", default=256)
        if train:
            bucketizers = {}
            for feat in self.continuous_features:
                bkt = QuantileBucketizer(n_buckets=n_buckets)
                bkt.fit(df[feat].values)
                bucketizers[feat] = bkt
                if bucketizer_save_dir is not None:
                    save_dir = Path(bucketizer_save_dir)
                    bkt.save(save_dir / f"{feat}.joblib")
        elif bucketizers is None:
            raise ValueError(
                "bucketizers must be provided when train=False. "
                "Pass pre-fitted bucketizers or set train=True."
            )
        self.bucketizers: BucketizerMap = bucketizers  # type: ignore[assignment]

        # --- Pre-compute labels -----------------------------------------
        self.labels: np.ndarray = (
            df[label_col].values - df[re_eta_col].values
        ).astype(np.float32)

        # --- Cache raw arrays for fast __getitem__ ----------------------
        self._cont: Dict[str, np.ndarray] = {
            f: df[f].values.astype(np.float64) for f in self.continuous_features
        }
        self._cat: Dict[str, np.ndarray] = {
            f: df[f].values.astype(np.int64) for f in self.categorical_features
        }
        self._geo_lat: Dict[str, np.ndarray] = {
            ctx: df[cols[0]].values.astype(np.float64)
            for ctx, cols in self.geo_pairs.items()
        }
        self._geo_lng: Dict[str, np.ndarray] = {
            ctx: df[cols[1]].values.astype(np.float64)
            for ctx, cols in self.geo_pairs.items()
        }
        self._n = len(df)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}

        # --- Continuous features → quantile bin indices ----------------
        for feat in self.continuous_features:
            raw_val = np.array([self._cont[feat][idx]])
            bin_idx = self.bucketizers[feat].transform(raw_val)[0]
            features[feat] = torch.tensor(int(bin_idx), dtype=torch.long)

        # --- Categorical features → embedding lookup index -------------
        for feat in self.categorical_features:
            features[feat] = torch.tensor(int(self._cat[feat][idx]), dtype=torch.long)

        # --- Geospatial features → multi-res geohash → MurmurHash3 ----
        # For each context (origin, destination, pair):
        #   For each resolution:
        #     For each seed: one bucket index
        # Key naming: geo_{context}_{resolution}_s{seed_idx}
        geo_hashes: Dict[str, List[str]] = {}
        for ctx in self.geo_pairs:
            lat = float(self._geo_lat[ctx][idx])
            lng = float(self._geo_lng[ctx][idx])
            geo_hashes[ctx] = self.geo_encoder.encode(lat, lng)

        resolutions = self.geo_encoder.resolutions
        n_seeds = len(self.feature_hasher.seeds)

        for r_idx, res in enumerate(resolutions):
            for ctx in self.geo_pairs:
                gh = geo_hashes[ctx][r_idx]
                bucket_indices = self.feature_hasher.hash(gh)
                for s_idx in range(n_seeds):
                    key = f"geo_{ctx}_r{res}_s{s_idx}"
                    features[key] = torch.tensor(bucket_indices[s_idx], dtype=torch.long)

            # Origin-destination pair context
            if "origin" in geo_hashes and "destination" in geo_hashes:
                gh_o = geo_hashes["origin"][r_idx]
                gh_d = geo_hashes["destination"][r_idx]
                pair_indices = self.feature_hasher.hash_pair(gh_o, gh_d)
                for s_idx in range(n_seeds):
                    key = f"geo_od_r{res}_s{s_idx}"
                    features[key] = torch.tensor(pair_indices[s_idx], dtype=torch.long)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label
