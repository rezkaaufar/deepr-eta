# DeeprETA Implementation Plan

## Overview

This document outlines a comprehensive implementation plan for **DeeprETA** — Uber's deep residual ETA post-processing network, described in the paper *"DeeprETA: An ETA Post-processing System at Scale"* (Hu et al., KDD 2022, arXiv:2206.02127) and the accompanying engineering blog post.

DeeprETA is a hybrid ETA prediction system: it treats the output of a routing engine as a noisy estimate of the true arrival time and trains a deep neural network to predict the residual correction. The final predicted ETA is:

```
ŷ = ŷ₀ + r̂
```

where `ŷ₀` is the routing engine ETA and `r̂` is the learned residual.

---

## Architecture Summary

DeeprETA consists of three major components:

```
Raw Features
     │
     ▼
┌─────────────────────────────────────────┐
│           Embedding Module              │
│  • Categorical → nn.Embedding lookup    │
│  • Continuous → Quantile bins → Embed   │
│  • Geospatial → Multi-res Geohash +     │
│    Multiple Feature Hashing → Embed     │
└─────────────────────────────────────────┘
     │  sequence of K embedding vectors [K × d]
     ▼
┌─────────────────────────────────────────┐
│    Linear Self-Attention Layer          │
│  • Feature map φ(x) = elu(x) + 1       │
│  • O(Kd²) complexity (not O(K²d))      │
│  • Residual connection                  │
│  • Key/Query/Value dim = 4              │
│  • No positional encoding               │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│    Fully Connected Decoder              │
│  • hidden_size = 2048, ReLU            │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│    Calibration (Bias) Layer             │
│  • Per-segment learned scalar bias      │
│  • Segments: delivery/rides × pickup/   │
│    dropoff                              │
└─────────────────────────────────────────┘
     │
     ▼
  Residual r̂  →  ŷ = ReLU(ŷ₀ + r̂)  → clamp to [0, 7200s]
```

---

## Input Features

Based on Table 1 of the paper and the engineering blog post. The dataset is not yet available; implementation assumes it will contain exactly these features.

| Category | Feature Name | Type | Processing | Notes |
|---|---|---|---|---|
| **Temporal** | `minute_of_day` | continuous | quantile bins | 0–1439 |
| **Temporal** | `day_of_week` | categorical | embedding lookup | 0–6 |
| **Temporal** | `minute_of_week` | categorical | embedding lookup | vocab=10080, emb_dim=8 per paper |
| **Geospatial** | `origin_lat` | float | geohash + feature hash | combined with origin_lng |
| **Geospatial** | `origin_lng` | float | geohash + feature hash | combined with origin_lat |
| **Geospatial** | `destination_lat` | float | geohash + feature hash | combined with destination_lng |
| **Geospatial** | `destination_lng` | float | geohash + feature hash | combined with destination_lat |
| **Trip** | `trip_type` | categorical | embedding lookup | e.g., ride-hailing vs food delivery |
| **Trip** | `route_type` | categorical | embedding lookup | type of route |
| **Trip** | `request_type` | categorical | embedding lookup | pickup vs dropoff |
| **Routing Engine** | `re_eta` | continuous | quantile bins | routing engine ETA in seconds |
| **Routing Engine** | `estimated_distance` | continuous | quantile bins | route distance in meters |
| **Traffic** | `realtime_speed` | continuous | quantile bins | 256 buckets per paper |
| **Traffic** | `historical_speed` | continuous | quantile bins | historical average |
| **Context** | `country_id` | categorical | embedding lookup | country identifier |
| **Context** | `region_id` | categorical | embedding lookup | region identifier |
| **Context** | `city_id` | categorical | embedding lookup | city identifier |
| **Label** | `actual_travel_time` | continuous | — | ATA in seconds (training target) |

---

## Embedding Module (Detailed)

All features are mapped to embedding vectors of the same dimension `d` before entering the interaction layer.

### 1. Categorical Feature Embedding

```python
e_α = nn.Embedding(vocab_size, d)[x_α]
```

Example: `minute_of_week` → `nn.Embedding(10080, 8)`

### 2. Continuous Feature Embedding (Quantile Bucketization)

Continuous features are discretized into `n_buckets` quantile bins, then embedded:

```python
# Fit on training data:
thresholds = np.quantile(train_values, np.linspace(0, 1, n_buckets + 1))

# At runtime:
bin_index = np.searchsorted(thresholds, x_β, side='right').clip(0, n_buckets - 1)
e_β = nn.Embedding(n_buckets, d)[bin_index]
```

Quantile bins preferred over equal-width bins: for a fixed number of bins, quantile bins convey maximum information about the original distribution in bits.

### 3. Geospatial Feature Embedding

This is the most novel part of the architecture. The pipeline:

**Step 1 — Geohash encoding** at 4 resolutions `u ∈ {4, 5, 6, 7}`:

| Precision | Cell size (approx.) |
|---|---|
| 4 | ~39 km × 20 km |
| 5 | ~4.9 km × 4.9 km |
| 6 | ~1.2 km × 610 m |
| 7 | ~153 m × 153 m |

```python
import pygeohash as pgh
geohash_str = pgh.encode(lat, lng, precision=u)
```

**Step 2 — Multiple Feature Hashing** (Algorithm 1 from the paper):

For each geohash string, apply two independent MurmurHash3 functions with different seeds to obtain two bucket indices:

```python
import mmh3
h1 = mmh3.hash(geohash_str, seed=0)  % hash_bucket_size
h2 = mmh3.hash(geohash_str, seed=42) % hash_bucket_size
```

Two independent hash functions mitigate the hash collision problem of single feature hashing.

**Step 3 — Three spatial contexts per resolution**:
- Origin alone (`h_o`)
- Destination alone (`h_d`)
- Origin-destination pair (`h_od`): hash the concatenation of origin and destination geohash strings

**Step 4 — Separate embedding tables per hash function**:
```python
emb_h1 = nn.Embedding(hash_bucket_size, d)
emb_h2 = nn.Embedding(hash_bucket_size, d)
e_geo = emb_h1[h1] + emb_h2[h2]
```

Total geospatial embedding lookups: `4 resolutions × 2 hash functions × 3 spatial contexts = 24 lookups`

---

## Two-Layer Module (Detailed)

### Layer 1: Linear Self-Attention (Interaction Layer)

Standard softmax self-attention has O(K²d) complexity (K = number of feature vectors). DeeprETA uses linear attention that reduces this to O(Kd²) by replacing softmax with an ELU+1 feature map:

```python
class LinearSelfAttention(nn.Module):
    def __init__(self, d_model, d_head=4):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.out_proj = nn.Linear(d_head, d_model, bias=False)

    def forward(self, X):
        # X: [batch, K, d_model]
        Q = F.elu(self.W_q(X)) + 1   # [batch, K, d_head]
        K = F.elu(self.W_k(X)) + 1   # [batch, K, d_head]
        V = self.W_v(X)               # [batch, K, d_head]

        # Efficient: compute KV once, then apply per-query
        KV = torch.einsum('bkh,bkv->bhv', K, V)     # [batch, d_head, d_head]
        K_sum = K.sum(dim=1)                          # [batch, d_head]

        QKV = torch.einsum('bkh,bhv->bkv', Q, KV)   # [batch, K, d_head]
        QK = torch.einsum('bkh,bh->bk', Q, K_sum)   # [batch, K]

        V_prime = QKV / (QK.unsqueeze(-1) + 1e-6)   # [batch, K, d_head]
        V_prime = self.out_proj(V_prime)              # [batch, K, d_model]
        return V_prime + X                            # residual
```

Key design choices:
- Feature map `φ(x) = elu(x) + 1` ensures non-negative attention weights (approximates softmax)
- Key/Query/Value dimension `d_head = 4` (from paper)
- No positional encoding (feature order is irrelevant for tabular data)
- Residual connection (Eq. 9–11 in paper)

### Layer 2: Fully Connected Decoder

```python
class FCDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [batch, K * d_model] — flatten the K feature vectors
        return self.net(x)
```

Hidden size = 2048 per paper.

### Layer 3: Calibration (Bias Adjustment) Layer

A per-segment learned scalar bias corrects mean-shift differences across request types (delivery vs rides, pickup vs dropoff):

```python
class CalibrationLayer(nn.Module):
    def __init__(self, n_segments):
        super().__init__()
        self.bias = nn.Embedding(n_segments, 1)

    def forward(self, raw_pred, segment_id):
        # raw_pred: [batch, 1]
        # segment_id: [batch] — integer index into segment lookup
        return raw_pred + self.bias(segment_id)
```

Segment IDs are derived from `request_type` (pickup/dropoff) and `trip_type` (rides/delivery). The calibration layer is deliberately simple (a single bias per segment, not a full FC branch) to minimize serving latency.

---

## Full Model

```python
class DeeprETANet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_module = EmbeddingModule(config)
        self.attention = LinearSelfAttention(config.d_model, d_head=4)
        self.decoder = FCDecoder(config.n_features * config.d_model, hidden_dim=2048)
        self.calibration = CalibrationLayer(config.n_segments)
        self.max_eta = config.max_eta  # e.g., 7200 seconds

    def forward(self, features, re_eta, segment_id):
        # features: dict of feature tensors
        X_emb = self.embedding_module(features)   # [batch, K, d_model]
        X_int = self.attention(X_emb)             # [batch, K, d_model]
        X_flat = X_int.flatten(1)                 # [batch, K * d_model]
        residual = self.decoder(X_flat)            # [batch, 1]
        residual = self.calibration(residual, segment_id)
        eta = F.relu(re_eta.unsqueeze(-1) + residual)
        eta = eta.clamp(max=self.max_eta)
        return eta.squeeze(-1)
```

---

## Loss Function: Asymmetric Huber Loss

```python
def asymmetric_huber_loss(y_pred, y_true, delta=1.0, omega=0.4):
    """
    delta: transition point between squared-error and absolute-error regimes
    omega: weight for overprediction (y < y_hat); (1-omega) for underprediction
    """
    error = y_true - y_pred
    abs_error = torch.abs(error)
    huber = torch.where(
        abs_error < delta,
        0.5 * error.pow(2),
        delta * abs_error - 0.5 * delta ** 2,
    )
    # Overprediction: y_true < y_pred → model predicted too high
    weight = torch.where(y_true < y_pred,
                         torch.full_like(y_true, omega),
                         torch.full_like(y_true, 1.0 - omega))
    return (weight * huber).mean()
```

Parameters `delta` and `omega` are configurable:
- `delta` smoothly interpolates between squared error (small `delta`) and absolute error (large `delta`)
- `omega < 0.5` → penalize late arrivals more than early ones

---

## Probabilistic Extension: Weibull Loss (DoorDash-Inspired)

### Motivation

Point-estimate models (including DeeprETA with Asymmetric Huber Loss) produce a single ETA value but say nothing about uncertainty. For food-delivery use cases, delivery times follow a **long-tail, right-skewed distribution** that cannot be adequately modeled by a Gaussian or exponential. DoorDash's engineering team ("Improving ETAs with multi-task models, deep learning, and probabilistic forecasts", 2023) demonstrated that the **3-parameter Weibull distribution** is a natural fit:

- Flexible shape: can represent early-peak, symmetric, or heavy-tailed distributions by varying `k`
- Long right tail: captures occasional very-late deliveries without inflating the mean
- Closed-form survival and quantile functions: efficient at inference time
- Enables **uncertainty quantification**: instead of a point ETA, the model exposes an interval, e.g. "ETA is 28–42 min at 90% confidence"

Although DeeprETA's primary domain is ride/delivery routing-engine post-processing, the same distribution assumption applies whenever delivery time distributions are long-tailed, and the probabilistic head can be stacked alongside or substituted for the existing AsymmetricHuberLoss head.

---

### Model Output Changes

In point-estimate mode the `FCDecoder` outputs a single scalar residual. In probabilistic mode the final linear layer is replaced to output **3 parameters** per prediction: shape `k`, scale `λ`, and location `γ`.

```python
class FCDecoderWeibull(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # Three separate output heads
        self.head_k     = nn.Linear(hidden_dim, 1)  # shape
        self.head_lam   = nn.Linear(hidden_dim, 1)  # scale
        self.head_gamma = nn.Linear(hidden_dim, 1)  # location

    def forward(self, x):
        h = self.shared(x)
        # softplus ensures k > 0 and λ > 0; γ ≥ 0 via softplus
        k     = F.softplus(self.head_k(h))      # [batch, 1]
        lam   = F.softplus(self.head_lam(h))    # [batch, 1]
        gamma = F.softplus(self.head_gamma(h))  # [batch, 1]
        return k, lam, gamma
```

`softplus(x) = log(1 + exp(x))` is a smooth, always-positive alternative to `ReLU` that avoids hard zeros in the parameter outputs.

---

### Loss Functions

#### Point NLL (exact observation)

Given a delivery time observation `t`, the Weibull PDF is:

```
f(t; k, λ, γ) = (k / λ) · ((t − γ) / λ)^(k−1) · exp(−((t − γ) / λ)^k)
```

The negative log-likelihood (NLL) per sample is:

```
NLL(t) = −log f(t)
       = −log(k) + log(λ) − (k−1)·log((t−γ)/λ) + ((t−γ)/λ)^k
```

#### Interval-censored NLL (the DoorDash training approach)

Instead of treating each delivery as an exact observation, DoorDash groups deliveries with similar features and bins delivery times into **6-minute buckets**. Given that a delivery fell in bucket `[a, b]`, the probability is:

```
P(a ≤ t ≤ b) = S(a) − S(b)
             = exp(−((a−γ)/λ)^k) − exp(−((b−γ)/λ)^k)
```

where `S(t) = exp(−((t−γ)/λ)^k)` is the Weibull survival function. The interval-censored NLL loss is:

```
Loss_interval = −log(S(a) − S(b))
              = −log[ exp(−((a−γ)/λ)^k) − exp(−((b−γ)/λ)^k) ]
```

Numerically stable implementation uses `torch.logaddexp` or computes in log-space to avoid underflow when `a` and `b` are large.

---

### Python Pseudocode

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeibullNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for the 3-parameter Weibull distribution.

    Args:
        interval_censored (bool): If True, expects bucket boundaries (a, b)
            instead of exact observations t.
        eps (float): Small constant to prevent log(0).
    """

    def __init__(self, interval_censored: bool = False, eps: float = 1e-6):
        super().__init__()
        self.interval_censored = interval_censored
        self.eps = eps

    def forward(self, k, lam, gamma, t_or_a, b=None):
        """
        Args:
            k, lam, gamma: Weibull parameters, each shape [batch, 1]
            t_or_a: exact observation t (point mode) or lower bucket edge a (interval mode)
            b: upper bucket edge (interval mode only)
        Returns:
            Scalar mean NLL loss.
        """
        if self.interval_censored:
            assert b is not None, "b (upper bucket edge) required for interval mode"
            a = t_or_a
            # Clamp to avoid (t - gamma) <= 0
            za = ((a - gamma) / lam).clamp(min=self.eps)
            zb = ((b - gamma) / lam).clamp(min=self.eps)
            log_sa = -(za ** k)                   # log S(a)
            log_sb = -(zb ** k)                   # log S(b)
            # log(S(a) - S(b)) via log-sum-exp trick
            log_prob = torch.logaddexp(log_sa, log_sb + torch.log(
                1 - torch.exp((log_sb - log_sa).clamp(max=0))
            ))
            # Simpler but less numerically stable alternative:
            # log_prob = torch.log((log_sa.exp() - log_sb.exp()).clamp(min=self.eps))
            return -log_prob.mean()
        else:
            t = t_or_a
            z = ((t - gamma) / lam).clamp(min=self.eps)  # (t − γ) / λ
            nll = (
                -torch.log(k + self.eps)
                + torch.log(lam + self.eps)
                - (k - 1) * torch.log(z + self.eps)
                + z ** k
            )
            return nll.mean()
```

---

### Point Estimate at Inference Time

After training, extract a scalar ETA from the distribution parameters:

| Estimator | Formula | Notes |
|---|---|---|
| Mean | `γ + λ · Γ(1 + 1/k)` | Minimizes MSE; can be slow for high-variance distributions |
| Mode | `γ + λ · ((k−1)/k)^(1/k)` | Only valid when `k > 1`; minimizes point estimate for peaked distributions |
| p-th Quantile | `γ + λ · (−log(1−p))^(1/k)` | e.g. p=0.5 for median, p=0.9 for 90th-percentile ETA |

```python
import torch
from torch.special import gammaln   # log-gamma function


class WeibullPointEstimate:
    """Compute scalar point estimates from Weibull parameters."""

    @staticmethod
    def mean(k, lam, gamma):
        # Γ(1 + 1/k) = exp(lgamma(1 + 1/k))
        log_gamma_val = gammaln(1.0 + 1.0 / k)
        return gamma + lam * log_gamma_val.exp()

    @staticmethod
    def mode(k, lam, gamma):
        # Only meaningful for k > 1
        valid = k > 1
        m = gamma + lam * ((k - 1) / k) ** (1.0 / k)
        return torch.where(valid, m, gamma)  # fallback to γ when k ≤ 1

    @staticmethod
    def quantile(k, lam, gamma, p: float = 0.5):
        return gamma + lam * (-torch.log(torch.tensor(1.0 - p))) ** (1.0 / k)
```

---

### Multi-Task Extension

DoorDash's architecture adds **separate prediction heads** for sub-stage ETAs (restaurant preparation time, Dasher pickup time, and overall delivery time) on top of a shared encoder trunk. Within DeeprETA's framework, this maps naturally to:

```
EmbeddingModule → LinearSelfAttention → flatten
        │
        ├──→ WeibullHead (overall ETA)
        ├──→ WeibullHead (restaurant prep time)    [optional auxiliary task]
        └──→ WeibullHead (Dasher pickup time)       [optional auxiliary task]
```

Each `WeibullHead` is an `FCDecoderWeibull` (described above). During training, the auxiliary-task losses are summed with configurable weights:

```python
loss_total = loss_eta + alpha * loss_prep + beta * loss_pickup
```

Auxiliary tasks act as **regularizers** on the shared encoder, improving generalization on the primary ETA head. This is the same Mixture-of-Experts (MoE) pattern used by DoorDash (DeepNet + CrossNet + Transformer specialized encoders), simplified here to a single shared DeeprETA encoder.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| LR scheduler | Cosine annealing (relative) |
| Activation | ReLU |
| FC hidden size | 2048 |
| Attention d_head | 4 |
| Quantile buckets (continuous) | 256 (default) |
| Quantile buckets (minute_of_week) | vocab=10080 (categorical, not binned) |
| Geohash resolutions | {4, 5, 6, 7} |
| Hash bucket size | configurable (e.g., 100,000) |
| Embedding dimension d | configurable (e.g., 16 or 32) |
| Loss delta | 1.0 (tunable) |
| Loss omega | 0.4 (tunable) |
| Outlier removal | drop ATA > 7200s |
| Train/val split | 90%/10% sequential |
| Test period | separate held-out period |
| Hardware | GPU (NVIDIA Quadro RTX 5000 used in paper) |

---

## Evaluation Metrics

```python
import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def p50_error(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))

def p95_error(y_true, y_pred):
    return np.percentile(np.abs(y_true - y_pred), 95)

def relative_improvement(baseline_error, model_error):
    return (baseline_error - model_error) / baseline_error
```

Primary comparison baseline: routing engine ETA (`re_eta`) directly.

---

## Proposed Project Structure

```
deepr-eta/
├── PLAN.md                        # This file
├── README.md
├── requirements.txt
├── config/
│   └── default_config.yaml        # All hyperparameters and feature specs
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset; loads rows, applies preprocessing
│   ├── preprocessing.py           # QuantileBucketizer (fit/transform/save)
│   └── geohash_utils.py           # Geohash encoding + MurmurHash3 feature hashing
├── models/
│   ├── __init__.py
│   ├── deepreta.py                # DeeprETANet: top-level model
│   ├── embedding_module.py        # EmbeddingModule: all embedding tables + lookups
│   ├── linear_attention.py        # LinearSelfAttention layer
│   └── loss.py                    # AsymmetricHuberLoss
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # Training loop, checkpointing, LR schedule
│   └── metrics.py                 # MAE, p50, p95, relative improvement
├── serving/
│   ├── __init__.py
│   └── predictor.py               # Inference wrapper: load checkpoint, preprocess, predict
└── notebooks/
    └── 01_eda.ipynb               # EDA placeholder for when dataset arrives
```

---

## Implementation Phases

### Phase 1: Data Infrastructure
- [ ] `data/preprocessing.py` — `QuantileBucketizer`: fit on training split, serialize thresholds to disk
- [ ] `data/geohash_utils.py` — `GeohashEncoder` (wraps `pygeohash`), `MultipleFeatureHasher` (wraps `mmh3`)
- [ ] `data/dataset.py` — `ETADataset`: loads rows, applies all preprocessing, returns feature dict + label
- [ ] `config/default_config.yaml` — document all vocab sizes, n_buckets, embedding dims, hash bucket sizes

### Phase 2: Model
- [ ] `models/embedding_module.py` — `EmbeddingModule` with three sub-modules (categorical, continuous, geospatial)
- [ ] `models/linear_attention.py` — `LinearSelfAttention` with ELU+1 feature map, efficient KV trick, residual
- [ ] `models/deepreta.py` — `DeeprETANet` wiring: embed → attention → flatten → FC → calibration → output
- [ ] `models/loss.py` — `AsymmetricHuberLoss` as `nn.Module`

### Phase 3: Training
- [ ] `training/metrics.py` — MAE, p50, p95, relative improvement functions
- [ ] `training/trainer.py` — Adam + cosine LR, gradient clipping, periodic validation, checkpointing
- [ ] `train.py` — CLI entry point: reads config, instantiates dataset + model + trainer, runs training

### Phase 4: Serving
- [ ] `serving/predictor.py` — `Predictor`: loads trained checkpoint + quantile thresholds, runs end-to-end inference
- [ ] Measure inference latency; target p95 < 4ms (per online deployment results in paper)

### Phase 5: Ablation Validation
- [ ] DeeprETANet without calibration layer → should decrease accuracy (Table 2)
- [ ] DeeprETANet without feature hashing (exact geohash indexing) → should decrease MAE (Table 2)
- [ ] Compare against XGBoost baseline on the same feature set

### Phase 6: Probabilistic Extension
- [ ] `models/loss.py` — `WeibullNLLLoss`: implement point NLL and interval-censored NLL variants
- [ ] `models/decoder.py` — `FCDecoderWeibull`: modify FCDecoder to output `[k, λ, γ]` with softplus activations
- [ ] `models/weibull_utils.py` — `WeibullPointEstimate`: mean, mode, and quantile extractors
- [ ] `models/deepreta.py` — add `WeibullHead` as an optional replace-or-stack alongside the existing AsymmetricHuberLoss head
- [ ] `config/default_config.yaml` — add `loss_mode` toggle: `"point_estimate"` (current, AsymmetricHuberLoss) vs `"probabilistic"` (WeibullNLLLoss)

---

## Key Design Decisions & Trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| Residual prediction | Predict `r̂ = ATA − RE-ETA` | RE-ETA is a strong prior; residual is smoother and smaller in magnitude |
| Feature discretization | Quantile buckets (not equal-width) | Maximizes information per bin; validated empirically in the paper |
| Geohash resolutions | {4, 5, 6, 7} | Fine grids are more accurate but sparse; coarse grids provide coverage |
| Two hash functions | MurmurHash3 ×2 | Single hashing suffers collisions; exact indexing too memory-intensive at fine resolutions |
| Attention | Linear (ELU+1 kernel) | O(Kd²) vs O(K²d); paper benchmarked 7 architectures and chose this for speed+accuracy |
| No positional encoding | Omitted | Feature order is arbitrary for tabular data; paper explicitly notes this |
| Calibration | Bias layer per segment | A full multi-branch decoder failed latency requirements; a scalar bias is negligible |
| Loss | Asymmetric Huber | Handles skewed residual distribution + business asymmetry (late > early cost) |
| Loss (probabilistic mode) | Weibull NLL | Models full delivery time distribution; enables uncertainty quantification |
| Embedding dim > model compute | ~99% params in embeddings | Large tables do O(1) lookup — effectively free at serving time |

---

## Assumptions (Dataset Not Yet Available)

1. **Format**: CSV or Parquet, one row per ETA request
2. **Column names**: Will match the Feature Table above (a mapping config will handle any differences)
3. **RE-ETA column**: Routing engine ETA is a pre-computed column in the dataset
4. **Geospatial columns**: Raw float `latitude`/`longitude` pairs for origin and destination
5. **Traffic columns**: Pre-aggregated per-request speed values, not raw road-segment data
6. **Label column**: `actual_travel_time` in seconds
7. **Scale**: ~1.4B training rows per the paper; local development will use a sampled subset
8. **Outliers**: Rows with ATA > 7200s (2 hours) are dropped at training time

---

## References

- Hu, X., Binaykiya, T., Frank, E., & Cirit, O. (2022). *DeeprETA: An ETA Post-processing System at Scale*. KDD 2022. [arXiv:2206.02127](https://arxiv.org/abs/2206.02127)
- Uber Engineering Blog: *DeepETA: How Uber Predicts Arrival Times Using Deep Learning* (2022).
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML 2020.
- Weinberger, K. et al. (2009). *Feature Hashing for Large Scale Multitask Learning*. ICML 2009.
- Saberian, M., Delgado, P., & Raimond, Y. (2019). *Gradient Boosted Decision Tree Neural Network* (Hammock). arXiv:1910.09340.
- DoorDash Engineering Blog: *Improving ETAs with multi-task models, deep learning, and probabilistic forecasts* (2023).
