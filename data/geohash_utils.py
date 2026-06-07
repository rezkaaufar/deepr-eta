# GeohashEncoder and MultipleFeatureHasher for geospatial feature encoding.
from __future__ import annotations

from typing import List, Sequence, Tuple

import mmh3
import pygeohash as pgh


class GeohashEncoder:
    """Encodes lat/lng pairs as geohash strings at multiple precisions."""

    DEFAULT_RESOLUTIONS: Tuple[int, ...] = (4, 5, 6, 7)

    def __init__(self, resolutions: Sequence[int] = DEFAULT_RESOLUTIONS) -> None:
        self.resolutions = tuple(resolutions)

    def encode(self, lat: float, lng: float) -> List[str]:
        """Return one geohash string per configured resolution."""
        return [pgh.encode(lat, lng, precision=r) for r in self.resolutions]

    def __repr__(self) -> str:
        return f"GeohashEncoder(resolutions={self.resolutions})"


class MultipleFeatureHasher:
    """Hashes geohash strings to bucket indices using two independent MurmurHash3 seeds."""

    DEFAULT_SEEDS: Tuple[int, int] = (0, 42)

    def __init__(
        self,
        hash_bucket_size: int = 100_000,
        seeds: Sequence[int] = DEFAULT_SEEDS,
    ) -> None:
        self.hash_bucket_size = hash_bucket_size
        self.seeds = tuple(seeds)

    def hash(self, geohash_str: str) -> List[int]:
        """Return one non-negative bucket index per configured seed."""
        return [
            abs(mmh3.hash(geohash_str, seed=s)) % self.hash_bucket_size
            for s in self.seeds
        ]

    def hash_pair(self, geohash_a: str, geohash_b: str) -> List[int]:
        """Hash the concatenation of two geohash strings (origin–destination pair)."""
        return self.hash(geohash_a + geohash_b)

    def __repr__(self) -> str:
        return (
            f"MultipleFeatureHasher("
            f"hash_bucket_size={self.hash_bucket_size}, seeds={self.seeds})"
        )
