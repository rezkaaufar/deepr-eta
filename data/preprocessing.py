# QuantileBucketizer: fit quantile thresholds on training data, transform to bin indices.
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import joblib
import numpy as np


class QuantileBucketizer:
    """Discretizes continuous values into quantile bucket indices."""

    def __init__(self, n_buckets: int = 256) -> None:
        self.n_buckets = n_buckets
        self.thresholds_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, values: np.ndarray) -> "QuantileBucketizer":
        """Compute quantile thresholds from training values."""
        flat = np.asarray(values, dtype=np.float64).ravel()
        # n_buckets + 1 quantile points define n_buckets intervals
        quantiles = np.linspace(0.0, 1.0, self.n_buckets + 1)
        self.thresholds_ = np.quantile(flat, quantiles)
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map values to integer bin indices in [0, n_buckets - 1]."""
        if self.thresholds_ is None:
            raise RuntimeError("QuantileBucketizer has not been fitted yet.")
        flat = np.asarray(values, dtype=np.float64).ravel()
        # searchsorted with 'right': bin 0 for values <= thresholds_[0]
        indices = np.searchsorted(self.thresholds_, flat, side="right")
        indices = indices.clip(0, self.n_buckets - 1).astype(np.int64)
        return indices.reshape(np.asarray(values).shape)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Fit and immediately transform."""
        return self.fit(values).transform(values)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted bucketizer (thresholds) to disk."""
        if self.thresholds_ is None:
            raise RuntimeError("Cannot save an unfitted QuantileBucketizer.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"n_buckets": self.n_buckets, "thresholds": self.thresholds_}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QuantileBucketizer":
        """Load a previously saved QuantileBucketizer from disk."""
        data = joblib.load(Path(path))
        obj = cls(n_buckets=data["n_buckets"])
        obj.thresholds_ = data["thresholds"]
        return obj

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.thresholds_ is not None
        return f"QuantileBucketizer(n_buckets={self.n_buckets}, fitted={fitted})"
