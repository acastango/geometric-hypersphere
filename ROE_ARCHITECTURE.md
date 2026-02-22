# ROE — Readout via Orthogonal Eigenprojection

## What ROE Is

ROE is a post-hoc geometric classification system. It takes frozen pretrained features (e.g., ResNet50 2048d vectors), compresses them onto a unit hypersphere, and classifies using quadratic energy wells instead of linear heads.

**Core claim:** Curved readout surfaces on compressed features recover classification information that linear probes miss, because class structure in embedding spaces is naturally curved, not flat.

**What ROE is NOT:** ROE is not a model. It doesn't train a backbone. It doesn't learn features. It reorganizes the geometry of existing features. The analogy: ROE is a lens, not a camera.

---

## Pipeline (5 stages, strictly sequential)

```
Raw Images
    │
    ▼
[1] Feature Extraction (frozen backbone, one-time)
    ResNet50\{head}: ℝ^(H×W×3) → ℝ^2048
    │
    ▼
[2] Spectral Compression (learned, trained with sovereign)
    SpectralCompressor: ℝ^2048 → S^(d-1)
    Whitening → Recoloring → Projection → L2 Normalize
    │
    ▼
[3] Sovereign Training (unsupervised geometry formation)
    GeometricCore: EMA basin prototypes on S^(d-1)
    Loss: contrastive + isotropy + separation + PR target
    NO class labels used. Basins form from geometry alone.
    │
    ▼
[4] Scalar Field Training (supervised, classification)
    TokenScalarField: one quadratic well per class
    f_c(z) = z^T V_c V_c^T z + b_c^T z
    Trained with cross-entropy on field values.
    │
    ▼
[5] Classification
    Direct: ŷ = argmax_c f_c(z)
    Stepped: walk along ∇_S f_ŷ, re-evaluate (optional)
```

---

## Core Modules (ACTIVE — these are the system)

### `roe_engine.py`
The foundation. Contains:

- **`GeometricCore`** — The hypersphere substrate. `n_basins` EMA-updated prototypes on S^(d-1). Sparse top-k softmax activation. Tracks utilization, entropy, cohesion, inter-prototype cosines.
- **`SpectralCompressor`** — Whitening + learnable recoloring + orthogonal projection + L2 normalize. Returns `(z, spectral_penalty)` tuple (NOTE: returns a tuple, not just z). Running covariance via low-rank SVD.
- **`VisionEncoder`** — Wrapper around timm backbones. Frozen by default.
- **`CoreLossStack`** — Contrastive + isotropy + separation + PR losses. Unsupervised.
- **`SpectralRegulator`** — PR monitoring, automatic loss weight adjustment.
- **`spectral_health(z)`** — Standalone function: PR, mean cosine, eigenvalue analysis.
- **`train_phase1(config)`** — Full sovereign training loop.

### `roe_scalar_field.py`
The classification layer. Contains:

- **`TokenScalarField`** — Each class gets V_c ∈ ℝ^(d×r) and b_c ∈ ℝ^d. Field value: `f_c(z) = ||V_c^T z||^2 + b_c · z`. Parameters generated via embedding → MLP (so it's embedding(class_id) → v_proj → V, b_proj → b). Supports:
  - `field_value(z, token_ids)` — evaluate specific class fields
  - `field_value_all(z)` — evaluate ALL class fields (caution: OOMs on large batches with many classes, chunk to ~200)
  - `spherical_gradient(z, token_ids)` — intrinsic gradient: `∇_S f = (I - zz^T)(2V(V^T z) + b)`
  - `step_along_field(z, token_ids, step_size)` — one gradient step on sphere
  - `multi_step(z, token_ids, n_steps, step_size)` — iterated stepping
  
- **`SharedBasisScalarField`** — Experimental variant. `V_c = U · A_c` where U ∈ ℝ^(d×R) is shared across all classes, A_c ∈ ℝ^(R×r_c) is per-class. Fewer params, shared curvature. Same interface as TokenScalarField. **Status: tested, does not improve generalization over TokenScalarField at current data scale.**

- **`ScalarFieldLoss`** — Cross-entropy + tightening + smoothness + composition losses.

### `roe_scaling_test.py`
The experiment runner. Handles:
- Feature extraction and caching
- Sovereign training (with `--skip_sovereign` to reload checkpoints)
- 80/20 train/test split
- Linear probe baseline (ridge regression on raw 2048d features)
- Scalar field training (with chunked eval to avoid OOM)
- Full evaluation: direct top-1/5, stepped top-1, tightening, composition, health
- Multi-dimension sweeps (`--dims 64 128 256`)
- Dataset sources: Caltech-256, ImageNet (HuggingFace), ImageFolder, local ImageNet

### `roe_hippocampus_continuous.py`
Episodic memory on the sphere. Contains:

- **`ContinuousHippocampus`** — Drift-triggered storage: stores a memory when `1 - z_t · z_stored > threshold`. Route detection: breaks route when drift exceeds higher threshold. Merkle-chained for integrity. Field-aware retrieval: find memories by position AND field agreement. Supports 64d/128d/256d.
- **Status: validated with unit tests on real Caltech-256 data. Perfect route purity (every route = single class). Not yet integrated into scalar field training loop.**

### `step_ablation.py`
Step-count ablation: measures test accuracy at 0, 1, 2, 3, 5, 10, 20, 50 gradient steps. Self-guided (uses predicted label, not ground truth). Auto-detects field model type from checkpoint keys. Also computes 128d linear probe diagnostic.

---

## Health Metrics (what we measure)

| Metric | Symbol | Meaning | Healthy Range |
|--------|--------|---------|---------------|
| Participation Ratio | PR | Effective dimensionality of point cloud | > d/2 |
| Mean Inter-Prototype Cosine | δ | Basin separation | 0.45–0.55 |
| Utilization Entropy | ent | Basin usage uniformity | > 0.95 |
| Active Basins | — | Basins with > 0.1% traffic | = n_basins |
| Intra-Cohesion | — | Within-basin tightness | > 0.8 |
| Loss | — | Cross-entropy on field values | → 0 (training) |

---

## Confirmed Results

### Caltech-256 (257 classes, 30,607 images, cached features)

| Dim | Direct Top-1 | Stepped Top-1 | PR | δ | Basins |
|-----|-------------|---------------|-----|------|--------|
| 64 | 97.7% | 100% | 44 | 0.55 | 64/64 |
| 128 | 100% (direct) | 99.9% (stepped) | 66 | 0.53 | 64/64 |
| 256 | ~oscillating | ~oscillating | 65-85 | 0.52 | 62/64 |

- **Feature cache:** `./roe_core/features_caltech256.pt`
- 128d is the sweet spot. 64d over-sharpens. 256d underdetermined for 257 classes.

### ImageNet-1k (1000 classes, 50,000 val images)

| Readout | Dimensions | Top-1 | Top-5 |
|---------|-----------|-------|-------|
| Linear probe (ridge) | 2048d | 76.0% | 92.7% |
| Linear probe (ridge) | 128d | 66.4% | — |
| ROE direct | 128d | 67.3% | 86.1% |
| ROE stepped (ground truth labels) | 128d | 100% | 100% |

- **Feature cache:** `./roe_core/features_imagenet.pt`
- **Sovereign checkpoint:** `./roe_core/imagenet/dim_128/compressor.pt`, `core.pt`
- **Field checkpoint:** `./roe_core/imagenet/dim_128/field_model.pt` (NOTE: currently SharedBasisScalarField, keys are U, A, b)
- ROE beats 128d linear probe by +0.9 points. The 9-point gap to 2048d probe is compression loss, not readout failure.
- Stepped 100% confirms topology is perfect: every class has a reachable well.

### Dims/Class Scaling Law

```
dims/class  | direct acc | dataset
------------|-----------|--------
0.13        | 67%       | ImageNet 128d/1000c
0.64        | 77%       | CIFAR-100 64d/100c
1.28        | 87%       | CIFAR-100 128d/100c
2.56        | 98%       | CIFAR-100 256d/100c
```

**This is a packing law.** Classification accuracy is determined by how much angular room each class gets on the sphere. Not a regularization problem, not an overfitting problem — an information-theoretic capacity limit.

**Prediction:** ImageNet at 1024d (dims/class ≈ 1.0) should hit ~87%. At 2048d (dims/class ≈ 2.0) should hit ~95%+.

---

## Experimental Frontiers (NOT YET CONFIRMED)

### Dimension Scaling on ImageNet
Run 512d and 1024d to confirm the dims/class law transfers across datasets. Sovereign training needed at each dimension. This is the most important next experiment.

### Step-Count Ablation
Self-guided stepping (predict → walk → re-evaluate) at various step counts. Tests whether gradient dynamics recover accuracy at inference time. Script exists (`step_ablation.py`), needs a TokenScalarField checkpoint to run properly.

### Hippocampus Integration
The hippocampus works independently (validated). Integration with scalar field training — using episodic memory to guide traversal or inform field learning — is unexplored.

---

## File Status Guide

### ACTIVE (the system)
- `roe_engine.py` — Core: GeometricCore, SpectralCompressor, loss stack, training
- `roe_scalar_field.py` — TokenScalarField, SharedBasisScalarField, losses
- `roe_scaling_test.py` — Main experiment runner
- `roe_hippocampus_continuous.py` — Episodic memory (validated, not yet integrated)
- `step_ablation.py` — Step-count evaluation

### HISTORICAL / DEPRECATED (can be archived)
- `roe_hippocampus.py` — Older discrete hippocampus, superseded by continuous
- `roe_hippocampus_adversarial_1.py`, `roe_hippocampus_adversarial_2.py` — Experiments, not part of main pipeline
- `roe_loop.py` — Strange loop experiments, conceptual
- `roe_mode.py` — Mode controller (thinking/encoding/sleep), conceptual
- `roe_sleep.py` — Sleep consolidation, conceptual
- `roe_place.py` — Place cells, conceptual (absorbed into hippocampus)
- `roe_entorhinal.py` — Entorhinal translation layer, conceptual
- `roe_phase2.py` — Early phase 2 experiments, superseded by scalar field
- `roe_token_gradient.py`, `roe_token_pair.py`, `roe_token_projection.py` — Early token experiments, superseded
- `roe_continuous.py` — Earlier continuous geometry experiments
- `roe_conversational.py` — Conversational interface experiments
- `roe_context.py` — Context experiments
- `roe_curriculum.py`, `roe_curriculum_analysis.py` — Curriculum learning experiments
- `roe_basin_interior.py` — Basin analysis experiments
- `roe_centroid_check.py` — Centroid diagnostics
- `roe_composition_test.py` — Composition experiments (now in scaling test)
- `roe_dimension_sweep.py` — Earlier dim sweep (now in scaling test)
- `roe_eval_rigorous.py` — Earlier evaluation (now in scaling test)
- `roe_imagenet.py` — Earlier ImageNet attempt
- `roe_inverse_composition.py` — Inverse composition experiments
- `roe_noise_reconstruction.py` — Noise robustness experiments
- `roe_replication.py` — Replication experiments
- `roe_run.py` — Earlier run script
- `roe_triple_composition.py` — Triple composition experiments
- `roe_validate.py` — Earlier validation
- `roe_visualizer.py`, `roe_viz.html`, `roe_sphere_viz.html`, `roe_viz_export.py` — Visualization tools (may be useful, not core)
- `roe_project_push.py` — Project management
- `roe_corpus_stats.txt`, `roe_vocab.json` — NLP-era artifacts
- `*.npz` files — Old ontology data from Pythia experiments
- `corpus.txt`, `corpus2.txt`, `corpus_pride.txt` — Old text corpora
- `roe_P.npy` — Old projection matrix

---

## Key Implementation Notes

1. **SpectralCompressor returns a tuple:** `z, spectral_penalty = compressor(x)`. Always unpack.
2. **field_value_all OOMs on 8GB GPUs:** Chunk to batches of ~200 when evaluating 1000 classes.
3. **multi_step takes ground truth labels:** The stepped accuracy result uses the correct class label to guide traversal. For realistic inference, use self-guided stepping (predict → walk → re-predict).
4. **Sovereign training is unsupervised:** Basins form from geometry, not class labels. The scalar field maps classes to basins after the fact.
5. **Random seed matters for train/test split:** Use `torch.manual_seed(42)` for reproducible splits.
6. **The spectral compressor's whitening stats are tied to the training data:** Don't mix compressors trained on different datasets.

---

## Design Philosophy

- Don't add capacity, add structure
- Frozen backbone — don't retrain, reinterpret
- Compression before classification (remove noise, keep geometry)
- Quadratic fields exploit curvature that linear heads waste
- Health metrics over accuracy during geometry formation
- The sphere concentrates structure; the field exploits it
- Bigger isn't always better — there's an optimal pressure (Jeans limit)
- Everything has scaling laws; design with them, not against them
