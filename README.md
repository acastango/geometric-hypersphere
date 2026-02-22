# Geometric Hypersphere — ROE

**Readout via Orthogonal Eigenprojection**

ROE is a post-hoc geometric classification system. It takes frozen pretrained features (e.g., ResNet50 2048d vectors), compresses them onto a unit hypersphere, and classifies using quadratic energy wells instead of linear heads.

**Core claim:** Curved readout surfaces on compressed features recover classification information that linear probes miss, because class structure in embedding spaces is naturally curved, not flat.

ROE is not a model. It doesn't train a backbone or learn features. It reorganizes the geometry of existing features. ROE is a lens, not a camera.

---

## Pipeline

```
Raw Images
    │
    ▼
[1] Feature Extraction ─── Frozen backbone (ResNet50), run once
    ℝ^(H×W×3) → ℝ^2048
    │
    ▼
[2] Spectral Compression ─── Learned whitening + recoloring + projection
    SpectralCompressor: ℝ^2048 → S^(d-1)
    │
    ▼
[3] Sovereign Training ─── Unsupervised geometry formation
    GeometricCore: EMA basin prototypes on S^(d-1)
    Loss: contrastive + isotropy + separation + PR target
    NO class labels. Basins form from geometry alone.
    │
    ▼
[4] Scalar Field Training ─── Supervised classification
    TokenScalarField: one quadratic well per class
    f_c(z) = ‖V_c^T z‖² + b_c · z
    │
    ▼
[5] Inference
    Direct: ŷ = argmax_c f_c(z)
    Stepped: walk along ∇_S f_ŷ, re-evaluate (optional)
```

Training is two-phase: **sovereign** (unsupervised geometry, no labels) then **field** (supervised classification on frozen geometry). The backbone is never trained.

---

## Results

### Caltech-256 (257 classes, 30,607 images)

| Dim | Direct Top-1 | Stepped Top-1 | PR | Inter-cos (δ) |
|-----|-------------|---------------|-----|----------------|
| 64  | 97.7%       | 100%          | 44  | 0.55           |
| 128 | 100%        | 99.9%         | 66  | 0.53           |
| 256 | ~oscillating| ~oscillating  | 65-85 | 0.52         |

128d is the sweet spot. 64d over-sharpens. 256d underdetermined for 257 classes.

### ImageNet-1k (1000 classes, 50K val)

| Readout | Dims | Top-1 | Top-5 |
|---------|------|-------|-------|
| Linear probe (ridge) | 2048d | 76.0% | 92.7% |
| Linear probe (ridge) | 128d  | 66.4% | —     |
| ROE direct           | 128d  | 67.3% | 86.1% |
| ROE stepped (oracle) | 128d  | 100%  | 100%  |

ROE beats 128d linear probe by +0.9 points. The 9-point gap to the full-dimensional probe is compression loss, not readout failure. Stepped 100% confirms perfect topology — every class has a reachable well.

### Dims/Class Scaling Law

```
dims/class │ direct acc │ dataset
───────────┼────────────┼──────────────────────
0.13       │ 67%        │ ImageNet 128d/1000c
0.64       │ 77%        │ CIFAR-100 64d/100c
1.28       │ 87%        │ CIFAR-100 128d/100c
2.56       │ 98%        │ CIFAR-100 256d/100c
```

This is an **information-theoretic packing law**. Classification accuracy is determined by how much angular room each class gets on the sphere. Prediction: ImageNet at 1024d (dims/class ~ 1.0) should hit ~87%.

---

## Architecture

### Core Modules

| File | Purpose |
|------|---------|
| `roe_engine.py` | Foundation — GeometricCore, SpectralCompressor, VisionEncoder, CoreLossStack, SpectralRegulator, sovereign training loop |
| `roe_scalar_field.py` | Classification — TokenScalarField, SharedBasisScalarField, ScalarFieldLoss, field training loop |
| `roe_hippocampus_continuous.py` | Episodic memory — drift-triggered Merkle-chained storage on the sphere (validated, not yet integrated) |
| `locus_core.py` | Theoretical substrate — geometric framework for embedding consciousness (conceptual) |

### Key Components

**GeometricCore** — `n_basins` EMA-updated prototypes on S^(d-1). Sparse top-k softmax activation. Tracks utilization entropy, intra-cohesion, inter-prototype cosines.

**SpectralCompressor** — 4-stage pipeline: center → whiten → recolor → project → L2-normalize. Running covariance via low-rank SVD. Returns `(z, spectral_penalty)` tuple.

**TokenScalarField** — Each class c gets V_c (rank-r quadratic) and b_c (linear bias). Field value: f_c(z) = ‖V_c^T z‖² + b_c · z. Supports intrinsic spherical gradient: ∇_S f = (I - zz^T)(2V(V^T z) + b).

**SpectralRegulator** — Adaptive agent that reads spectral metrics every epoch and adjusts loss weights to enforce health targets (PR > 15, mean_cos < 0.5, top_eig < 0.15).

**ContinuousHippocampus** — Drift-triggered episodic memory. Stores snapshots when sphere position drifts beyond threshold. Merkle-chained for integrity. Field-aware retrieval.

### Supporting Modules

| File | Status | Purpose |
|------|--------|---------|
| `roe_mode.py` | Conceptual | Operating modes (thinking/encoding/sleep) |
| `roe_loop.py` | Conceptual | Strange loop / recursive attention |
| `roe_place.py` | Conceptual | Place cells on the manifold |
| `roe_entorhinal.py` | Conceptual | LLM-to-ROE translation layer |
| `roe_sleep.py` | Conceptual | Wake-sleep consolidation |
| `roe_continuous.py` | Historical | Earlier continuous geometry experiments |
| `roe_context.py` | Historical | Context integration experiments |
| `roe_hippocampus.py` | Deprecated | Older discrete hippocampus (superseded by continuous) |
| `roe_token_gradient.py` | Historical | Early token experiments |

---

## Health Metrics

| Metric | Meaning | Healthy Range |
|--------|---------|---------------|
| Participation Ratio (PR) | Effective dimensionality of point cloud | > d/2 |
| Mean Inter-Prototype Cosine (δ) | Basin separation | 0.45 – 0.55 |
| Utilization Entropy | Basin usage uniformity | > 0.95 |
| Active Basins | Basins with > 0.1% traffic | = n_basins |
| Intra-Cohesion | Within-basin tightness | > 0.8 |

Health metrics matter more than accuracy during sovereign training. If the geometry is healthy, classification follows.

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- timm

---

## Implementation Notes

1. `SpectralCompressor` returns a tuple: always unpack `z, spectral_penalty = compressor(x)`
2. `field_value_all` OOMs on 8GB GPUs with many classes — chunk to batches of ~200
3. `multi_step` uses ground truth labels; for inference, use self-guided stepping (predict → walk → re-predict)
4. Sovereign training is fully unsupervised — basins form from geometry alone
5. Whitening stats are tied to training data — don't mix compressors across datasets
6. Use `torch.manual_seed(42)` for reproducible train/test splits

---

## Design Philosophy

- Don't add capacity, add structure
- Frozen backbone — don't retrain, reinterpret
- Compression before classification — remove noise, keep geometry
- Quadratic fields exploit curvature that linear heads waste
- Health metrics over accuracy during geometry formation
- The sphere concentrates structure; the field exploits it
- Everything has scaling laws; design with them, not against them
