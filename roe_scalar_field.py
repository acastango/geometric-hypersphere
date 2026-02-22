"""
ROE Phase 2f — Token as Scalar Field
=======================================

A token defines a SCALAR FIELD on the sphere.
  f("tiger", z) → compatibility score at point z

Everything follows from f:
  - Value:     f(z) = how compatible is this point with "tiger"
  - Gradient:  ∇_S f(z) = direction of steepest ascent ON the sphere
  - Level sets: {z : f(z) = c} = constraint contours
  - Classification: argmax_token f_token(z)

Projection and gradient were never two operations.
They're both views of the same scalar field.

The gradient is intrinsic to the sphere — no "push then renormalize."
∇_S f = (I - zz^T)(∂f/∂z) — project out the radial component.

Usage:
    python roe_scalar_field.py --max_steps 10000
    python roe_scalar_field.py --max_steps 10000 --field_rank 16
"""

import os
import sys
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roe_engine import (
    GeometricCore, SpectralCompressor, spectral_health,
)


# ════════════════════════════════════════════
# SCALAR FIELD ON THE SPHERE
# ════════════════════════════════════════════

class TokenScalarField(nn.Module):
    """
    Each token defines a scalar field on the 64d unit sphere.

    f_token(z) = z^T W_token z + b_token · z

    where W_token = V_token V_token^T (low-rank symmetric, PSD)
          b_token = learned bias direction

    This gives:
      - Quadratic + linear field on the sphere
      - Curved level sets (not just hyperplanes)
      - Rich structure with few parameters

    The spherical gradient:
      ∇_S f = (I - zz^T) · ∂f/∂z
      ∂f/∂z = 2 W z + b
      ∇_S f = (I - zz^T)(2 W z + b)

    This is the INTRINSIC gradient — it lives in the tangent plane.
    No renormalization. No projection hack. Native to the sphere.
    """
    def __init__(self, n_tokens, core_dim=64, field_rank=16, hidden_dim=128):
        super().__init__()
        self.core_dim = core_dim
        self.field_rank = field_rank
        self.n_tokens = n_tokens

        # Token embedding
        self.embedding = nn.Embedding(n_tokens, hidden_dim)

        # Field parameters: V (for W = VV^T) and b
        self.v_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, core_dim * field_rank),
        )
        self.b_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, core_dim),
        )

        # Initialize small — fields start near-uniform
        with torch.no_grad():
            self.v_proj[-1].weight.mul_(0.01)
            self.v_proj[-1].bias.zero_()
            self.b_proj[-1].weight.mul_(0.01)
            self.b_proj[-1].bias.zero_()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  TokenScalarField: {n_tokens} tokens → "
              f"rank-{field_rank} quadratic field in {core_dim}d "
              f"({n_params:,} params)")

    def get_field_params(self, token_ids):
        """
        token_ids: [B]
        Returns:
          V: [B, d, rank] — for W = VV^T
          b: [B, d] — linear term
        """
        emb = self.embedding(token_ids)
        V = self.v_proj(emb).view(-1, self.core_dim, self.field_rank)
        b = self.b_proj(emb)
        return V, b

    def field_value(self, z, token_ids):
        """
        Evaluate f_token(z) = z^T W z + b · z

        z: [B, d] — points on sphere
        token_ids: [B]
        Returns: [B] — scalar field values
        """
        V, b = self.get_field_params(token_ids)

        # z^T W z = z^T V V^T z = ||V^T z||^2
        Vt_z = torch.bmm(V.transpose(1, 2),
                         z.unsqueeze(-1)).squeeze(-1)  # [B, rank]
        quadratic = (Vt_z ** 2).sum(dim=-1)  # [B]

        # b · z
        linear = (b * z).sum(dim=-1)  # [B]

        return quadratic + linear

    def field_value_all(self, z, n_tokens=None):
        """
        Evaluate ALL token fields at point z.
        z: [B, d]
        Returns: [B, n_tokens] — compatibility with every token
        """
        if n_tokens is None:
            n_tokens = self.n_tokens
        B = z.shape[0]

        all_ids = torch.arange(n_tokens, device=z.device)
        V_all, b_all = self.get_field_params(all_ids)
        # V_all: [n_tokens, d, rank], b_all: [n_tokens, d]

        # For each point, evaluate all fields
        # z: [B, d] → [B, 1, d, 1]
        # V_all: [1, n_tokens, d, rank]
        z_exp = z.unsqueeze(1).unsqueeze(-1)      # [B, 1, d, 1]
        V_exp = V_all.unsqueeze(0)                  # [1, T, d, rank]

        Vt_z = torch.matmul(
            V_exp.transpose(-1, -2), z_exp)         # [B, T, rank, 1]
        quadratic = (Vt_z.squeeze(-1) ** 2).sum(-1)  # [B, T]

        b_exp = b_all.unsqueeze(0)                  # [1, T, d]
        z_exp2 = z.unsqueeze(1)                     # [B, 1, d]
        linear = (b_exp * z_exp2).sum(-1)            # [B, T]

        return quadratic + linear

    def spherical_gradient(self, z, token_ids):
        """
        Compute ∇_S f — the intrinsic gradient on the sphere.

        ∇_S f = (I - zz^T) · ∂f/∂z
        ∂f/∂z = 2Wz + b = 2V(V^T z) + b

        z: [B, d]
        token_ids: [B]
        Returns: [B, d] — tangent vector (lives on sphere, no renorm needed)
        """
        V, b = self.get_field_params(token_ids)

        # ∂f/∂z = 2V(V^T z) + b
        Vt_z = torch.bmm(V.transpose(1, 2),
                         z.unsqueeze(-1))              # [B, rank, 1]
        euclidean_grad = 2 * torch.bmm(V, Vt_z).squeeze(-1) + b  # [B, d]

        # Project to tangent plane: (I - zz^T) g = g - (g·z)z
        radial = (euclidean_grad * z).sum(dim=-1, keepdim=True)
        sphere_grad = euclidean_grad - radial * z

        return sphere_grad

    def step_along_field(self, z, token_ids, step_size=0.1):
        """
        Move z along the spherical gradient of f_token.
        This is INTRINSIC — stays on the sphere naturally.

        z: [B, d]
        token_ids: [B]
        step_size: how far to move
        Returns: [B, d] — new point on sphere
        """
        grad = self.spherical_gradient(z, token_ids)
        z_new = z + step_size * grad
        z_new = F.normalize(z_new, dim=-1)
        return z_new

    def multi_step(self, z, token_ids, n_steps=3, step_size=0.1):
        """
        Follow the field gradient for multiple steps.
        Like gradient ascent on the compatibility surface.
        """
        z_current = z
        for _ in range(n_steps):
            z_current = self.step_along_field(
                z_current, token_ids, step_size)
        return z_current


class SharedBasisScalarField(nn.Module):
    """
    Shared-basis energy landscape on the sphere.

    E_c(z) = -(z^T U) A_c A_c^T (U^T z) - b_c^T z

    U ∈ ℝ^(d × R)    — shared basis (global manifold structure)
    A_c ∈ ℝ^(R × r_c) — per-class coefficients (local well shape)
    b_c ∈ ℝ^d          — linear term (primary discrimination)

    Decision boundary between c, c':
        z^T U(A_c A_c^T - A_{c'} A_{c'}^T)U^T z + (b_c - b_{c'})^T z = 0
        Quadratic part is rank-2r_c in projected space.

    Wells share curvature (U) but carve individual basins (A_c).
    Linear terms do the heavy lifting; quadratic adds gentle basins.
    """
    def __init__(self, n_tokens, core_dim=128, shared_rank=64,
                 class_rank=4, hidden_dim=128):
        super().__init__()
        self.core_dim = core_dim
        self.shared_rank = shared_rank
        self.class_rank = class_rank
        self.n_tokens = n_tokens
        self.field_rank = class_rank  # compatibility with training loop

        # Shared basis U — orthogonally initialized
        self.U = nn.Parameter(torch.empty(core_dim, shared_rank))
        nn.init.orthogonal_(self.U)

        # Per-class coefficients A_c: [n_tokens, shared_rank, class_rank]
        self.A = nn.Parameter(
            torch.randn(n_tokens, shared_rank, class_rank) * 0.01)

        # Per-class linear term b_c: [n_tokens, core_dim]
        self.b = nn.Parameter(torch.randn(n_tokens, core_dim) * 0.01)

        n_params = sum(p.numel() for p in self.parameters())
        params_shared = core_dim * shared_rank
        params_per_class = shared_rank * class_rank + core_dim
        print(f"  SharedBasisScalarField: {n_tokens} tokens")
        print(f"    Shared basis U: {core_dim}×{shared_rank} = {params_shared:,}")
        print(f"    Per-class A_c:  {shared_rank}×{class_rank} = "
              f"{shared_rank * class_rank} + b: {core_dim} = {params_per_class}/class")
        print(f"    Total: {n_params:,} params")

    def field_value(self, z, token_ids):
        """
        f_c(z) = ||A_c^T U^T z||^2 + b_c · z

        z: [B, d], token_ids: [B]
        Returns: [B]
        """
        # U^T z: [B, R]
        Ut_z = z @ self.U  # [B, shared_rank]

        # A_c for each sample: [B, shared_rank, class_rank]
        A_batch = self.A[token_ids]

        # A_c^T (U^T z): [B, class_rank]
        At_Ut_z = torch.bmm(
            A_batch.transpose(1, 2),
            Ut_z.unsqueeze(-1)).squeeze(-1)  # [B, class_rank]

        quadratic = (At_Ut_z ** 2).sum(dim=-1)  # [B]

        # b_c · z
        b_batch = self.b[token_ids]  # [B, d]
        linear = (b_batch * z).sum(dim=-1)  # [B]

        return quadratic + linear

    def field_value_all(self, z, n_tokens=None):
        """
        Evaluate all fields at each point.
        z: [B, d]
        Returns: [B, n_tokens]
        """
        if n_tokens is None:
            n_tokens = self.n_tokens
        B = z.shape[0]

        # U^T z: [B, R]
        Ut_z = z @ self.U  # [B, shared_rank]

        # For all classes: A^T (U^T z)
        # A: [T, R, r_c], Ut_z: [B, R]
        # Want: [B, T, r_c]
        # Ut_z: [B, 1, R] @ A: [T, R, r_c] → need broadcast
        Ut_z_exp = Ut_z.unsqueeze(1)  # [B, 1, R]
        A_all = self.A[:n_tokens]  # [T, R, r_c]

        # [B, T, r_c] via einsum
        At_Ut_z = torch.einsum('br,trk->btk', Ut_z, A_all)
        quadratic = (At_Ut_z ** 2).sum(dim=-1)  # [B, T]

        # Linear: b · z for all classes
        b_all = self.b[:n_tokens]  # [T, d]
        linear = z @ b_all.T  # [B, T]

        return quadratic + linear

    def spherical_gradient(self, z, token_ids):
        """
        ∇_S f = (I - zz^T) ∂f/∂z
        ∂f/∂z = 2 U A_c A_c^T U^T z + b_c
        """
        Ut_z = z @ self.U  # [B, R]
        A_batch = self.A[token_ids]  # [B, R, r_c]

        At_Ut_z = torch.bmm(
            A_batch.transpose(1, 2),
            Ut_z.unsqueeze(-1)).squeeze(-1)  # [B, r_c]

        # U A_c (A_c^T U^T z): [B, d]
        UA = torch.matmul(self.U, A_batch)  # [B, d, r_c]
        euclidean_grad = 2 * torch.bmm(
            UA, At_Ut_z.unsqueeze(-1)).squeeze(-1)  # [B, d]

        b_batch = self.b[token_ids]
        euclidean_grad = euclidean_grad + b_batch

        # Project to tangent plane
        radial = (euclidean_grad * z).sum(dim=-1, keepdim=True)
        sphere_grad = euclidean_grad - radial * z

        return sphere_grad

    def step_along_field(self, z, token_ids, step_size=0.1):
        grad = self.spherical_gradient(z, token_ids)
        z_new = z + step_size * grad
        z_new = F.normalize(z_new, dim=-1)
        return z_new

    def multi_step(self, z, token_ids, n_steps=3, step_size=0.1):
        z_current = z
        for _ in range(n_steps):
            z_current = self.step_along_field(
                z_current, token_ids, step_size)
        return z_current


# ════════════════════════════════════════════
# LOSS
# ════════════════════════════════════════════

class ScalarFieldLoss:
    """
    The scalar field gives us a natural loss:
      - Correct token should have HIGH field value at z
      - Wrong tokens should have LOW field value at z
      - This is just a classification loss on field values

    Plus:
      - Following the gradient should tighten clusters
      - Field should have non-trivial structure (not flat)
    """
    def __init__(self, lambda_classify=1.0, lambda_tighten=0.5,
                 lambda_structure=0.1, step_size=0.1, n_steps=3,
                 temperature=0.1):
        self.lambda_classify = lambda_classify
        self.lambda_tighten = lambda_tighten
        self.lambda_structure = lambda_structure
        self.step_size = step_size
        self.n_steps = n_steps
        self.temperature = temperature

    def compute(self, z_v, field_model, labels):
        """
        z_v: [B, d] — vision points on sphere
        field_model: TokenScalarField
        labels: [B] — correct token indices
        """
        B = z_v.shape[0]
        device = z_v.device

        # 1. CLASSIFICATION via field values
        # f_all: [B, n_tokens] — value of every field at every point
        f_all = field_model.field_value_all(z_v)

        # Cross-entropy: correct token should have highest value
        classify_loss = F.cross_entropy(
            f_all / self.temperature, labels)

        # Top-1 accuracy for logging
        preds = f_all.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()

        # 2. TIGHTENING: following correct field should improve coherence
        z_stepped = field_model.multi_step(
            z_v, labels, self.n_steps, self.step_size)

        unique_labels = labels.unique()
        tighten = torch.zeros(B, device=device)
        for c in unique_labels:
            mask = labels == c
            if mask.sum() < 2:
                continue
            centroid_orig = F.normalize(z_v[mask].mean(dim=0), dim=0)
            centroid_step = F.normalize(z_stepped[mask].mean(dim=0), dim=0)
            tighten[mask] = ((z_stepped[mask] * centroid_step).sum(-1) -
                             (z_v[mask] * centroid_orig).sum(-1))

        tighten_loss = -tighten.mean()

        # 3. STRUCTURE: gradients should be non-trivial
        grad = field_model.spherical_gradient(z_v, labels)
        grad_mag = grad.norm(dim=-1).mean()
        # Want moderate magnitude
        structure_loss = ((grad_mag - 0.5) ** 2)

        total = (self.lambda_classify * classify_loss +
                 self.lambda_tighten * tighten_loss +
                 self.lambda_structure * structure_loss)

        return total, {
            "total": total.item(),
            "classify": classify_loss.item(),
            "tighten": tighten_loss.item(),
            "structure": structure_loss.item(),
            "accuracy": accuracy.item(),
            "improvement": tighten.mean().item(),
            "grad_mag": grad_mag.item(),
            "f_correct": f_all[torch.arange(B), labels].mean().item(),
            "f_mean": f_all.mean().item(),
        }


# ════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════

class ScalarFieldDataset(Dataset):
    def __init__(self, z_features, labels):
        self.z = z_features
        self.labels = labels

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.labels[idx]


# ════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════

@torch.no_grad()
def scalar_field_metrics(z_v, field_model, prototypes, labels,
                         n_classes, device, step_size=0.1, n_steps=3):
    B = z_v.shape[0]

    # === CLASSIFICATION (field value) ===
    f_all = field_model.field_value_all(z_v)
    preds = f_all.argmax(dim=-1)
    field_accuracy = (preds == labels).float().mean().item()

    # === CLASSIFICATION (after gradient steps) ===
    z_stepped = field_model.multi_step(z_v, labels, n_steps, step_size)
    f_stepped = field_model.field_value_all(z_stepped)
    preds_stepped = f_stepped.argmax(dim=-1)
    stepped_accuracy = (preds_stepped == labels).float().mean().item()

    # === TIGHTENING ===
    tighten_values = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() < 5:
            continue
        z_class = z_v[mask][:50]
        tokens = torch.full((len(z_class),), c, device=device)
        z_s = field_model.multi_step(z_class, tokens, n_steps, step_size)

        c_orig = F.normalize(z_class.mean(dim=0), dim=0)
        c_step = F.normalize(z_s.mean(dim=0), dim=0)
        coh_orig = (z_class * c_orig).sum(-1).mean().item()
        coh_step = (z_s * c_step).sum(-1).mean().item()
        tighten_values.append(coh_step - coh_orig)

    mean_tighten = np.mean(tighten_values) if tighten_values else 0

    # === FIELD DIVERSITY ===
    # Sample field values at random points for different tokens
    z_sample = z_v[:20]
    token_a = torch.zeros(20, dtype=torch.long, device=device)
    token_b = torch.ones(20, dtype=torch.long, device=device) * 50
    f_a = field_model.field_value(z_sample, token_a)
    f_b = field_model.field_value(z_sample, token_b)
    field_diversity = (f_a - f_b).abs().mean().item()

    # === GRADIENT MAGNITUDE STATS ===
    grad = field_model.spherical_gradient(z_v[:200], labels[:200])
    grad_mag_mean = grad.norm(dim=-1).mean().item()
    grad_mag_std = grad.norm(dim=-1).std().item()

    return {
        "field_accuracy": field_accuracy,
        "stepped_accuracy": stepped_accuracy,
        "step_improvement": stepped_accuracy - field_accuracy,
        "tightening": mean_tighten,
        "field_diversity": field_diversity,
        "grad_mag_mean": grad_mag_mean,
        "grad_mag_std": grad_mag_std,
    }


# ════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load core
    core_path = config.get("core_checkpoint") or "./roe_core/checkpoint_sovereign"
    if not os.path.exists(core_path):
        for alt in ["./roe_core/curriculum/stage_2",
                     "./roe_core/curriculum/stage_1",
                     "./roe_core/step_20000"]:
            if os.path.exists(alt):
                core_path = alt
                break

    if not os.path.exists(core_path):
        print(f"  ERROR: No checkpoint found.")
        return

    core = GeometricCore(core_dim=64, n_basins=32, ema_alpha=0.99).to(device)
    core_state = torch.load(os.path.join(core_path, "core.pt"),
                            weights_only=True)
    core.prototypes.copy_(core_state["prototypes"])

    vision_compressor = SpectralCompressor(input_dim=2048, core_dim=64).to(device)
    vc_state = torch.load(os.path.join(core_path, "compressor.pt"),
                          weights_only=True)
    vision_compressor.load_state_dict(vc_state)
    vision_compressor.eval()
    for p in vision_compressor.parameters():
        p.requires_grad = False

    print(f"  Core: {core_path}")

    # Load + project
    v_cache = "./roe_core/features_cifar100.pt"
    if not os.path.exists(v_cache):
        print(f"  ERROR: No features at {v_cache}")
        return
    cached = torch.load(v_cache, weights_only=True)
    v_features = cached["features"]
    v_labels = cached["labels"]
    n_classes = v_labels.unique().numel()

    print(f"  Projecting to core space...")
    all_z = []
    with torch.no_grad():
        for i in range(0, len(v_features), 512):
            batch = v_features[i:i+512].to(device)
            z, _ = vision_compressor(batch)
            all_z.append(z.cpu())
    z_all = torch.cat(all_z, dim=0)
    print(f"  Data: {z_all.shape}")

    # Scalar field model
    field_rank = config.get("field_rank", 16)
    step_size = config.get("step_size", 0.1)
    n_steps = config.get("n_steps", 3)

    field_model = TokenScalarField(
        n_tokens=n_classes, core_dim=64, field_rank=field_rank,
        hidden_dim=config.get("hidden_dim", 128)).to(device)

    # Dataset
    dataset = ScalarFieldDataset(z_all, v_labels)
    dataloader = DataLoader(
        dataset, batch_size=config.get("batch_size", 256),
        shuffle=True, num_workers=0, drop_last=True)

    loss_fn = ScalarFieldLoss(
        step_size=step_size, n_steps=n_steps,
        temperature=config.get("temperature", 0.1))
    optimizer = torch.optim.AdamW(
        field_model.parameters(), lr=config.get("lr", 1e-3),
        weight_decay=0.01)

    max_steps = config.get("max_steps", 10000)
    log_every = config.get("log_every", 200)
    eval_every = config.get("eval_every", 1000)

    print(f"\n{'='*60}")
    print(f"  Token as Scalar Field on the Sphere")
    print(f"  f(z) = z^T W z + b·z")
    print(f"  Rank: {field_rank}  Steps: {n_steps}  Step size: {step_size}")
    print(f"  Max steps: {max_steps}")
    print(f"  Baselines: gradient=51%, projection=47%")
    print(f"{'='*60}\n")

    field_model.train()
    step = 0
    running = defaultdict(float)
    running_count = 0
    data_iter = iter(dataloader)
    start_time = time.time()

    while step < max_steps:
        try:
            z_batch, labels_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            z_batch, labels_batch = next(data_iter)

        z_batch = z_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        total_loss, loss_dict = loss_fn.compute(
            z_batch, field_model, labels_batch)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(field_model.parameters(), 1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            running[k] += v
        running_count += 1
        step += 1

        if step % log_every == 0:
            avg = {k: v / running_count for k, v in running.items()}
            elapsed = time.time() - start_time
            print(f"  step={step:5d}  "
                  f"loss={avg['total']:.3f}  "
                  f"cls={avg['classify']:.3f}  "
                  f"acc={avg['accuracy']:.1%}  "
                  f"imp={avg['improvement']:.4f}  "
                  f"gmag={avg['grad_mag']:.3f}  "
                  f"f_c={avg['f_correct']:.2f}  "
                  f"f_m={avg['f_mean']:.2f}  "
                  f"[{step/elapsed:.0f} step/s]")
            running = defaultdict(float)
            running_count = 0

        if step % eval_every == 0:
            field_model.eval()

            n_eval = 2048
            idx = torch.randperm(len(z_all))[:n_eval]
            z_eval = z_all[idx].to(device)
            l_eval = v_labels[idx].to(device)

            metrics = scalar_field_metrics(
                z_eval, field_model, core.prototypes,
                l_eval, n_classes, device,
                step_size, n_steps)

            print(f"\n    ── Scalar Field Report @ Step {step} ──")
            print(f"    Field accuracy (direct):  {metrics['field_accuracy']:.1%}")
            print(f"    Field accuracy (stepped):  {metrics['stepped_accuracy']:.1%}")
            print(f"    Step improvement:          {metrics['step_improvement']:+.1%}")
            print(f"    Cluster tightening:        {metrics['tightening']:.4f}")
            print(f"    Field diversity:           {metrics['field_diversity']:.4f}")
            print(f"    Gradient magnitude:        {metrics['grad_mag_mean']:.3f} ± {metrics['grad_mag_std']:.3f}")

            checks = []
            if metrics['field_accuracy'] > 0.51:
                checks.append(f"✓ beats gradient baseline (51%)")
            else:
                checks.append(f"~ field acc {metrics['field_accuracy']:.1%} vs 51%")
            if metrics['step_improvement'] > 0.01:
                checks.append(f"✓ steps help (+{metrics['step_improvement']:.1%})")
            elif metrics['step_improvement'] > -0.01:
                checks.append(f"~ steps neutral")
            else:
                checks.append(f"✗ steps hurt")
            if metrics['tightening'] > 0:
                checks.append(f"✓ tightening clusters")
            else:
                checks.append(f"✗ not tightening")

            for c in checks:
                print(f"    {c}")

            if metrics['field_accuracy'] > 0.55:
                print(f"\n    ★ SCALAR FIELD WINS — one function, all behavior ★")
            print()

            field_model.train()

    # Save
    output_dir = config.get("output_dir") or "./roe_core/scalar_field"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(field_model.state_dict(),
              os.path.join(output_dir, "field_model.pt"))

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Complete — {step} steps in {elapsed:.0f}s")
    print(f"  Saved to {output_dir}")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--field_rank", type=int, default=16,
                       help="Rank of quadratic form (complexity of field)")
    parser.add_argument("--step_size", type=float, default=0.1,
                       help="Step size for gradient ascent on field")
    parser.add_argument("--n_steps", type=int, default=3,
                       help="Number of gradient ascent steps")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for classification loss")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--core_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  ROE — Token as Scalar Field")
    print("  f(z): one function, all behavior.")
    print("  The gradient IS the geometry.")
    print("=" * 60)

    train(vars(args))


if __name__ == "__main__":
    main()
