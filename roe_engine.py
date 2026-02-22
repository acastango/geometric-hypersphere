"""
ROE Phase 1 — Pure Visual Geometric Core
==========================================

The geometry is primary. Vision is the first sensor.

Architecture:
  Vision Encoder → Spectral Compression → 64D Hyperspherical Core
  
Core modules (verified, imported):
  - Hippocampus (Merkle-indexed episodic memory)
  - Spectral monitor (PR, mean cosine, eigenvalue tracking)

New modules:
  - GeometricCore: 64D hypersphere with EMA prototypes
  - SpectralCompressor: whitening + recoloring + projection
  - VisionEncoder: ViT/CNN backbone (frozen or fine-tuned)

Loss stack (geometry health, NOT classification):
  A) Contrastive / SSL (semantic grouping)
  B) Local isotropy regularization (prevent top eigen dominance)
  C) Density contrast (intra-cohesion, inter-separation)
  D) Basin separation margin (push prototypes apart)
  E) Participation ratio target (maintain PR floor)

Success criteria:
  - Mean cosine < 0.5
  - PR > 10, stable across epochs
  - Top eigenvalue ratio < 0.15
  - Clusters convex (intra > 0.8, inter < 0.3)
  - Rotation perturbation robustness
  - Drift-trigger storage shows meaningful wells

Usage:
    python roe_engine.py --phase 1 --epochs 50 --batch_size 128
    python roe_engine.py --phase 1 --eval_only --checkpoint roe_core/checkpoint_10

Requirements:
    pip install torch torchvision timm
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ════════════════════════════════════════════
# GEOMETRIC CORE (the substrate)
# ════════════════════════════════════════════

class GeometricCore(nn.Module):
    """
    64D hyperspherical latent space.
    Unit-normalized vectors. EMA-updated basin prototypes.
    Density-aware clustering. Spectral regularization target.

    This is the space. Not a model. Not an encoder.
    The thing itself.
    """
    def __init__(self, core_dim=64, n_basins=32, ema_alpha=0.99,
                 softmax_temp=5.0, top_k=3):
        super().__init__()
        self.core_dim = core_dim
        self.n_basins = n_basins
        self.ema_alpha = ema_alpha
        self.softmax_temp = softmax_temp
        self.top_k = top_k

        # Basin prototypes on unit sphere
        prototypes = torch.randn(n_basins, core_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        self.register_buffer("prototypes", prototypes)

        # Tracking
        self.register_buffer("utilization", torch.zeros(n_basins))
        self.register_buffer("intra_cohesion", torch.zeros(n_basins))
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))

    def activate(self, z):
        """
        Given core-space vectors, compute basin activations.
        z: [B, core_dim] — must be on unit sphere already.
        Returns: activations [B, n_basins], assignments [B]
        """
        cos_sim = torch.mm(z, self.prototypes.T)  # [B, n_basins]

        # Sparse top-k softmax
        top_vals, top_idx = cos_sim.topk(self.top_k, dim=-1)
        sparse_mask = torch.zeros_like(cos_sim).scatter_(1, top_idx, 1.0)
        masked_sim = cos_sim * sparse_mask + (-1e9) * (1 - sparse_mask)
        activations = F.softmax(masked_sim / self.softmax_temp, dim=-1)

        assignments = top_idx[:, 0]  # primary basin

        # Track utilization
        with torch.no_grad():
            for idx in assignments:
                self.utilization[idx] += 1
            self.update_count += len(assignments)

        return activations, assignments

    @torch.no_grad()
    def update_prototypes(self, z, assignments):
        """EMA update. Prototypes drift toward their members."""
        for b in range(self.n_basins):
            mask = (assignments == b)
            if mask.sum() < 2:
                continue
            members = z[mask]
            centroid = F.normalize(members.mean(dim=0), dim=0)
            self.prototypes[b] = (self.ema_alpha * self.prototypes[b]
                                  + (1 - self.ema_alpha) * centroid)
            self.prototypes[b] = F.normalize(self.prototypes[b], dim=0)

            # Track cohesion
            cos_intra = (members @ centroid).mean()
            self.intra_cohesion[b] = (0.9 * self.intra_cohesion[b]
                                      + 0.1 * cos_intra)

    def health(self):
        """Basin health metrics."""
        if self.update_count == 0:
            return {}

        util = self.utilization / (self.update_count.float() + 1e-12)
        active = (util > 0.001).sum().item()

        # Utilization entropy
        util_safe = util.clamp(min=1e-12)
        entropy = -torch.sum(util_safe * torch.log(util_safe)).item()
        max_entropy = math.log(self.n_basins)

        # Inter-prototype cosines
        cos_matrix = torch.mm(self.prototypes, self.prototypes.T)
        mask = ~torch.eye(self.n_basins, dtype=torch.bool,
                         device=cos_matrix.device)
        inter_cos = cos_matrix[mask].mean().item()
        inter_max = cos_matrix[mask].max().item()

        # Mean intra-cohesion (only active basins)
        active_mask = util > 0.001
        mean_cohesion = (self.intra_cohesion[active_mask].mean().item()
                        if active_mask.any() else 0)

        return {
            "active_basins": active,
            "utilization_entropy": entropy / max_entropy if max_entropy > 0 else 0,
            "inter_prototype_cos_mean": inter_cos,
            "inter_prototype_cos_max": inter_max,
            "intra_cohesion_mean": mean_cohesion,
            "intra_inter_delta": mean_cohesion - inter_cos,
        }

    def reset_tracking(self):
        self.utilization.zero_()
        self.intra_cohesion.zero_()
        self.update_count.zero_()


# ════════════════════════════════════════════
# SPECTRAL COMPRESSOR
# ════════════════════════════════════════════

class SpectralCompressor(nn.Module):
    """
    Whitening → Recoloring → Projection → Hypersphere.

    Not a naive linear layer. Enforces spectral discipline:
    1. Batch whitening (removes dominant eigenstructure)
    2. Learnable recoloring (constrained to spread variance)
    3. Projection to core_dim
    4. L2 normalize to unit sphere

    The compressor ensures that whatever enters the core
    has healthy spectral properties regardless of encoder pathology.
    """
    def __init__(self, input_dim, core_dim=64, whiten_momentum=0.01,
                 spectral_penalty_weight=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.core_dim = core_dim
        self.spectral_penalty_weight = spectral_penalty_weight

        # Running whitening statistics
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_cov_eigvals",
                            torch.ones(min(input_dim, 512)))
        self.register_buffer("running_cov_eigvecs",
                            torch.eye(input_dim, min(input_dim, 512)))
        self.register_buffer("whiten_initialized", torch.tensor(False))
        self.whiten_momentum = whiten_momentum

        # Learnable recoloring: diagonal scaling in whitened space
        # Initialized to 1.0 = identity after whitening
        whiten_dim = min(input_dim, 512)
        self.recolor = nn.Parameter(torch.ones(whiten_dim))

        # Projection to core
        self.proj = nn.Linear(whiten_dim, core_dim, bias=False)

        # Initialize projection orthogonally
        nn.init.orthogonal_(self.proj.weight)

    def _update_whitening(self, x):
        """Update running whitening statistics."""
        with torch.no_grad():
            B = x.shape[0]
            if B < 4:
                return

            mean = x.mean(dim=0)
            centered = x - mean
            # Low-rank SVD for efficiency
            k = min(self.input_dim, 512, B)
            try:
                U, S, Vt = torch.linalg.svd(centered / math.sqrt(B - 1),
                                             full_matrices=False)
                eigvals = S[:k] ** 2
                eigvecs = Vt[:k].T  # [input_dim, k]

                if not self.whiten_initialized:
                    self.running_mean.copy_(mean)
                    self.running_cov_eigvals[:len(eigvals)] = eigvals
                    self.running_cov_eigvecs[:, :eigvecs.shape[1]] = eigvecs
                    self.whiten_initialized.fill_(True)
                else:
                    m = self.whiten_momentum
                    self.running_mean.mul_(1 - m).add_(mean * m)
                    n = min(len(eigvals), len(self.running_cov_eigvals))
                    self.running_cov_eigvals[:n] = (
                        (1 - m) * self.running_cov_eigvals[:n]
                        + m * eigvals[:n])
                    self.running_cov_eigvecs[:, :n] = (
                        (1 - m) * self.running_cov_eigvecs[:, :n]
                        + m * eigvecs[:, :n])
            except RuntimeError:
                pass  # SVD can fail on degenerate batches

    def forward(self, x):
        """
        x: [B, input_dim] — raw encoder features
        Returns: z [B, core_dim] on unit hypersphere, spectral_penalty scalar
        """
        if self.training:
            self._update_whitening(x)

        # 1. Center
        centered = x - self.running_mean

        # 2. Whiten: project onto eigenvectors, scale by 1/sqrt(eigenvalue)
        k = self.running_cov_eigvecs.shape[1]
        projected = centered @ self.running_cov_eigvecs  # [B, k]
        scale = 1.0 / (self.running_cov_eigvals[:k].sqrt() + 1e-6)
        whitened = projected * scale  # [B, k]

        # 3. Recolor: learnable diagonal scaling
        # Constrained: recolor values must be positive
        recolor_weights = F.softplus(self.recolor[:k])
        recolored = whitened * recolor_weights  # [B, k]

        # 4. Project to core dimension
        z = self.proj(recolored)  # [B, core_dim]

        # 5. Normalize to unit sphere
        z = F.normalize(z, dim=-1)

        # Spectral penalty: penalize if recoloring recreates dominance
        spectral_penalty = self._spectral_penalty(recolor_weights)

        return z, spectral_penalty

    def _spectral_penalty(self, weights):
        """
        Penalize top eigenvalue dominance in recoloring weights.
        Encourages variance spread across dimensions.
        """
        w_sorted = torch.sort(weights, descending=True)[0]
        top_ratio = w_sorted[0] / (w_sorted.sum() + 1e-12)
        # Penalty: top weight shouldn't dominate
        return self.spectral_penalty_weight * (top_ratio - 1.0 / len(weights)) ** 2


# ════════════════════════════════════════════
# VISION ENCODER
# ════════════════════════════════════════════

class VisionEncoder(nn.Module):
    """
    Standard vision backbone → feature vector.
    Frozen or fine-tuned depending on phase.
    """
    def __init__(self, backbone="resnet50", pretrained=True, freeze=True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            import torchvision.models as models
            base = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.feature_dim = base.fc.in_features  # 2048
            self.backbone = nn.Sequential(*list(base.children())[:-1],
                                          nn.Flatten())
        elif backbone == "vit_small":
            try:
                import timm
                base = timm.create_model("vit_small_patch16_224",
                                         pretrained=pretrained)
                self.feature_dim = base.head.in_features  # 384
                base.head = nn.Identity()
                self.backbone = base
            except ImportError:
                print("  timm not available, falling back to resnet50")
                return self.__init__("resnet50", pretrained, freeze)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"  Vision encoder: {backbone} (FROZEN, {self.feature_dim}d)")
        else:
            print(f"  Vision encoder: {backbone} (trainable, {self.feature_dim}d)")

    def forward(self, images):
        """images: [B, 3, H, W] → features [B, feature_dim]"""
        return self.backbone(images)


# ════════════════════════════════════════════
# CORE LOSS STACK
# ════════════════════════════════════════════

class CoreLossStack:
    """
    Five losses. All operate on core-space vectors.
    Optimizes GEOMETRY HEALTH. Not classification accuracy.
    """
    def __init__(self, config):
        self.lambda_contrastive = config.get("lambda_contrastive", 1.0)
        self.lambda_isotropy = config.get("lambda_isotropy", 0.5)
        self.lambda_density = config.get("lambda_density", 0.3)
        self.lambda_separation = config.get("lambda_separation", 0.5)
        self.lambda_pr = config.get("lambda_pr", 0.2)
        self.pr_target = config.get("pr_target", 15.0)
        self.separation_margin = config.get("separation_margin", 0.3)

    def contrastive_ssl(self, z1, z2, temperature=0.07):
        """
        A) SimCLR-style contrastive loss on two augmented views.
        z1, z2: [B, core_dim] — two views of same images, projected to core.
        Positives: (z1[i], z2[i]). Negatives: everything else.
        """
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2B, d]
        sim = torch.mm(z, z.T) / temperature  # [2B, 2B]

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2 * B),
                           torch.arange(0, B)]).to(z.device)

        # Mask out self-similarity
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, -1e9)

        loss = F.cross_entropy(sim, labels)
        return self.lambda_contrastive * loss

    def local_isotropy(self, z):
        """
        B) Penalize top eigenvalue dominance in core space.
        Prevents collapse to low-dimensional subspace.
        """
        B, d = z.shape
        if B < 4:
            return torch.tensor(0.0, device=z.device)

        centered = z - z.mean(dim=0, keepdim=True)

        # Use Gram matrix if B < d
        if B < d:
            gram = centered @ centered.T / (B - 1)
            eigvals = torch.linalg.eigvalsh(gram)
        else:
            cov = centered.T @ centered / (B - 1)
            eigvals = torch.linalg.eigvalsh(cov)

        eigvals = eigvals.clamp(min=1e-12)

        # Top eigenvalue ratio
        top_ratio = eigvals[-1] / eigvals.sum()

        # Penalty: top ratio should be ~1/d, penalize excess
        target_ratio = 1.0 / d
        penalty = (top_ratio - target_ratio).clamp(min=0) ** 2

        return self.lambda_isotropy * penalty

    def density_contrast(self, z, assignments, n_basins):
        """
        C) Intra-neighborhood cohesion vs inter-neighborhood separation.
        """
        z_norm = F.normalize(z, dim=-1)
        intra_sum = 0.0
        inter_parts = []
        n_active = 0

        centroids = []
        for b in range(n_basins):
            mask = (assignments == b)
            if mask.sum() < 2:
                continue
            members = z_norm[mask]
            centroid = F.normalize(members.mean(dim=0), dim=0)
            centroids.append(centroid)

            # Intra: mean cosine to centroid
            intra_cos = (members @ centroid).mean()
            intra_sum += intra_cos
            n_active += 1

        if n_active < 2:
            return torch.tensor(0.0, device=z.device)

        intra_mean = intra_sum / n_active

        # Inter: mean cosine between centroids
        centroids_t = torch.stack(centroids)
        cos_inter = centroids_t @ centroids_t.T
        n_c = len(centroids)
        mask = ~torch.eye(n_c, dtype=torch.bool, device=z.device)
        inter_mean = cos_inter[mask].mean()

        # Loss: maximize intra - inter
        loss = -(intra_mean - inter_mean)
        return self.lambda_density * loss

    def basin_separation(self, prototypes):
        """
        D) Push prototypes apart. Penalize any pair closer than margin.
        """
        cos = torch.mm(prototypes, prototypes.T)
        n = prototypes.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=prototypes.device)

        # Hinge: penalize pairs with cosine > margin
        violations = (cos[mask] - self.separation_margin).clamp(min=0)
        loss = violations.mean()
        return self.lambda_separation * loss

    def participation_ratio_loss(self, z):
        """
        E) Maintain PR above target threshold.
        """
        B, d = z.shape
        if B < 4:
            return torch.tensor(0.0, device=z.device), 0.0

        centered = z - z.mean(dim=0, keepdim=True)
        if B < d:
            gram = centered @ centered.T / (B - 1)
            eigvals = torch.linalg.eigvalsh(gram)
        else:
            cov = centered.T @ centered / (B - 1)
            eigvals = torch.linalg.eigvalsh(cov)

        eigvals = eigvals.clamp(min=1e-12)
        trace = eigvals.sum()
        pr = (trace ** 2) / (eigvals ** 2).sum()

        # Loss: penalize if PR below target
        pr_deficit = (self.pr_target - pr).clamp(min=0)
        loss = self.lambda_pr * (pr_deficit / self.pr_target)

        return loss, pr.item()

    def compute_all(self, z1, z2, assignments, prototypes, n_basins):
        """
        Compute full loss stack.
        z1, z2: two augmented views in core space
        Returns: total_loss, loss_dict
        """
        z_combined = torch.cat([z1, z2], dim=0)
        assign_combined = torch.cat([assignments, assignments], dim=0)

        l_con = self.contrastive_ssl(z1, z2)
        l_iso = self.local_isotropy(z_combined)
        l_dens = self.density_contrast(z_combined, assign_combined, n_basins)
        l_sep = self.basin_separation(prototypes)
        l_pr, pr_val = self.participation_ratio_loss(z_combined)

        total = l_con + l_iso + l_dens + l_sep + l_pr

        losses = {
            "contrastive": l_con.item(),
            "isotropy": l_iso.item(),
            "density_contrast": l_dens.item(),
            "separation": l_sep.item(),
            "pr_loss": l_pr.item(),
            "pr_value": pr_val,
            "total": total.item(),
        }

        return total, losses


# ════════════════════════════════════════════
# SPECTRAL MONITOR (carried from retrofit)
# ════════════════════════════════════════════

@torch.no_grad()
def spectral_health(z):
    """
    Full spectral health report on core-space vectors.
    z: [B, core_dim]
    """
    z_np = z.float().cpu().numpy()
    B, d = z_np.shape

    # Mean pairwise cosine
    normed = z_np / (np.linalg.norm(z_np, axis=1, keepdims=True) + 1e-12)
    cos_matrix = normed @ normed.T
    upper = cos_matrix[np.triu_indices(B, k=1)]
    mean_cos = float(np.mean(upper)) if len(upper) > 0 else 0

    # Participation ratio
    cov = np.cov(z_np.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]
    pr = float(np.sum(eigvals) ** 2 / np.sum(eigvals ** 2)) if len(eigvals) > 0 else 1.0

    # Top eigenvalue fraction
    top_frac = float(eigvals[-1] / np.sum(eigvals)) if len(eigvals) > 0 else 1.0

    # Effective dimensionality (entropy-based)
    eigvals_norm = eigvals / eigvals.sum()
    ent = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-12))
    eff_dim = float(np.exp(ent))

    return {
        "mean_cos": mean_cos,
        "participation_ratio": pr,
        "top_eigval_frac": top_frac,
        "effective_dim": eff_dim,
    }


# ════════════════════════════════════════════
# PRECOMPUTED FEATURES (eliminates ResNet bottleneck)
# ════════════════════════════════════════════

class PrecomputedFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset of precomputed vision features.
    Returns two noisy views per sample for contrastive learning.
    Noise serves the same role as augmentation: forces the compressor
    to learn invariant structure, not memorize individual features.
    """
    def __init__(self, features, labels, noise_std=0.1):
        self.features = features  # [N, feature_dim] tensor
        self.labels = labels      # [N] tensor
        self.noise_std = noise_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        # Two noisy views (contrastive pairs)
        view1 = feat + torch.randn_like(feat) * self.noise_std
        view2 = feat + torch.randn_like(feat) * self.noise_std
        return view1, view2, self.labels[idx]


def precompute_features(vision, dataset, device, batch_size=64):
    """
    Run frozen vision encoder once over entire dataset.
    Returns: features [N, feature_dim], labels [N]
    """
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Wrap dataset with transform
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base[idx]
            return self.transform(img), label

    transformed = TransformedDataset(dataset, transform)
    loader = DataLoader(transformed, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    all_features = []
    all_labels = []

    vision.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            features = vision(images)  # [B, feature_dim]
            all_features.append(features.cpu())
            all_labels.append(labels)
            if (i + 1) % 50 == 0:
                print(f"    Precomputing: {(i+1)*batch_size}/{len(dataset)}")

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ════════════════════════════════════════════
# DATA AUGMENTATION (for contrastive pairs)
# ════════════════════════════════════════════

def get_augmentations(image_size=224):
    """Two augmented views for contrastive learning."""
    import torchvision.transforms as T

    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                      p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class DualAugDataset(torch.utils.data.Dataset):
    """Wraps a dataset to return two augmented views per image."""
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, label


# ════════════════════════════════════════════
# SPECTRAL REGULATOR (Interpretation 1)
# ════════════════════════════════════════════

class SpectralRegulator:
    """
    Internal agent that monitors the eigenspectrum and adjusts
    loss weights to make high-PR convergence inevitable.

    Not optimization. Regulation.

    Reads geometry → detects what's stuck → adjusts pressures.
    Makes the deep basin the only basin.
    """
    def __init__(self, loss_stack, config=None):
        self.loss_stack = loss_stack
        self.history = []
        self.plateau_window = 5
        self.plateau_threshold = 0.5  # PR change < this = plateau

        # Targets
        self.pr_target = 15.0
        self.eig_target = 0.15

        # Bounds (don't let lambdas go insane)
        self.iso_bounds = (0.1, 2.0)
        self.sep_bounds = (0.1, 2.0)
        self.den_bounds = (0.1, 1.5)
        self.pr_bounds = (0.05, 1.0)

        # Track interventions
        self.interventions = []

    def regulate(self, epoch, spectral_metrics, basin_health):
        """
        Called once per epoch after spectral measurement.
        Reads the state. Adjusts the loss stack. Returns intervention log.
        """
        pr = spectral_metrics["participation_ratio"]
        top_eig = spectral_metrics["top_eigval_frac"]
        mean_cos = spectral_metrics["mean_cos"]
        delta = basin_health.get("intra_inter_delta", 0)
        active = basin_health.get("active_basins", 0)
        util_ent = basin_health.get("utilization_entropy", 0)

        self.history.append({
            "epoch": epoch, "pr": pr, "top_eig": top_eig,
            "delta": delta, "active": active,
        })

        actions = []

        # ── PR plateau detection ──
        if len(self.history) >= self.plateau_window:
            recent_pr = [h["pr"] for h in self.history[-self.plateau_window:]]
            pr_range = max(recent_pr) - min(recent_pr)
            pr_mean = sum(recent_pr) / len(recent_pr)

            if pr_range < self.plateau_threshold and pr_mean < self.pr_target:
                # PR is stuck below target. Spike isotropy pressure.
                old_iso = self.loss_stack.lambda_isotropy
                new_iso = min(old_iso * 1.5, self.iso_bounds[1])
                self.loss_stack.lambda_isotropy = new_iso
                actions.append(f"PR plateau ({pr_mean:.1f}): "
                              f"λ_iso {old_iso:.3f}→{new_iso:.3f}")

        # ── Top eigenvalue stuck ──
        if top_eig > self.eig_target * 1.5 and epoch > 10:
            # Top eigenvalue too dominant. Boost isotropy + PR loss.
            old_iso = self.loss_stack.lambda_isotropy
            new_iso = min(old_iso * 1.3, self.iso_bounds[1])
            old_pr = self.loss_stack.lambda_pr
            new_pr = min(old_pr * 1.3, self.pr_bounds[1])
            self.loss_stack.lambda_isotropy = new_iso
            self.loss_stack.lambda_pr = new_pr
            actions.append(f"top_eig high ({top_eig:.3f}): "
                          f"λ_iso→{new_iso:.3f}, λ_pr→{new_pr:.3f}")

        # ── Basin collapse ──
        if active < 20 and epoch > 5:
            old_den = self.loss_stack.lambda_density
            new_den = min(old_den * 1.5, self.den_bounds[1])
            self.loss_stack.lambda_density = new_den
            actions.append(f"basins collapsing ({active}): "
                          f"λ_density→{new_den:.3f}")

        # ── Delta shrinking (basins merging) ──
        if len(self.history) >= 3:
            recent_delta = [h["delta"] for h in self.history[-3:]]
            if all(d < 0.3 for d in recent_delta) and epoch > 10:
                old_sep = self.loss_stack.lambda_separation
                new_sep = min(old_sep * 1.3, self.sep_bounds[1])
                self.loss_stack.lambda_separation = new_sep
                actions.append(f"delta low ({delta:.3f}): "
                              f"λ_sep→{new_sep:.3f}")

        # ── Ease off when targets met (don't overshoot) ──
        if pr > self.pr_target * 1.2 and top_eig < self.eig_target:
            # Healthy. Gently reduce pressure to avoid instability.
            self.loss_stack.lambda_isotropy = max(
                self.loss_stack.lambda_isotropy * 0.9, self.iso_bounds[0])
            self.loss_stack.lambda_pr = max(
                self.loss_stack.lambda_pr * 0.9, self.pr_bounds[0])
            actions.append(f"healthy — easing pressure")

        if actions:
            self.interventions.append({"epoch": epoch, "actions": actions})

        return actions


# ════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════

def train_phase1(config):
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f}GB")

    # ── Vision encoder ──
    vision = VisionEncoder(
        backbone=config.get("vision_backbone", "resnet50"),
        pretrained=True,
        freeze=config.get("freeze_vision", True),
    ).to(device)

    # ── Spectral compressor ──
    compressor = SpectralCompressor(
        input_dim=vision.feature_dim,
        core_dim=config.get("core_dim", 64),
    ).to(device)

    # ── Geometric core ──
    core = GeometricCore(
        core_dim=config.get("core_dim", 64),
        n_basins=config.get("n_basins", 32),
        ema_alpha=config.get("ema_alpha", 0.99),
    ).to(device)

    # ── Loss stack ──
    loss_stack = CoreLossStack(config)

    # ── Spectral regulator (the agent inside the sphere) ──
    regulator = SpectralRegulator(loss_stack)

    # ── Hippocampus (verified module, imported) ──
    try:
        from roe_hippocampus import Hippocampus
        hippocampus = Hippocampus(drift_threshold=0.05)
        print(f"  Hippocampus: loaded (θ_store=0.05)")
    except ImportError:
        hippocampus = None
        print(f"  Hippocampus: not available (continuing without)")

    # ── Optimizer (compressor only — vision frozen, core is buffers) ──
    trainable_params = list(compressor.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get("lr", 3e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )
    n_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── LR scheduler ──
    max_steps = config.get("epochs", 50) * config.get("steps_per_epoch", 100)
    warmup = config.get("warmup_steps", 500)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Data ──
    print(f"\n  Loading dataset...")

    dataset_name = config.get("dataset", "cifar100")
    if dataset_name == "cifar100":
        base_dataset = torchvision.datasets.CIFAR100(
            root=config.get("data_dir", "./data"),
            train=True, download=True)
        n_classes = 100
    elif dataset_name == "imagenet_subset":
        base_dataset = torchvision.datasets.ImageFolder(
            config.get("data_dir", "./data/imagenet_subset"))
        n_classes = len(base_dataset.classes)
    else:
        base_dataset = torchvision.datasets.CIFAR10(
            root=config.get("data_dir", "./data"),
            train=True, download=True)
        n_classes = 10

    print(f"  Dataset: {dataset_name} ({len(base_dataset)} images, "
          f"{n_classes} classes)")

    # ── Precompute features (one-time ResNet pass) ──
    cache_path = os.path.join(config.get("output_dir", "./roe_core"),
                              f"features_{dataset_name}.pt")
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        all_features = cached["features"]
        all_labels = cached["labels"]
    else:
        print(f"  Precomputing ResNet50 features (one-time, ~2 min)...")
        all_features, all_labels = precompute_features(
            vision, base_dataset, device, batch_size=64)
        os.makedirs(config.get("output_dir", "./roe_core"), exist_ok=True)
        torch.save({"features": all_features, "labels": all_labels}, cache_path)
        print(f"  Cached to {cache_path} ({all_features.shape})")

    print(f"  Features: {all_features.shape} ({all_features.nbytes/1e6:.0f}MB)")

    # ── Create dataset from precomputed features ──
    feature_dataset = PrecomputedFeatureDataset(
        all_features, all_labels, noise_std=0.1)
    dataloader = DataLoader(
        feature_dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    print(f"  Batch size: {config.get('batch_size', 256)}")

    # ── Output ──
    output_dir = config.get("output_dir", "./roe_core")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Phase 1 — Pure Visual Geometric Core")
    print(f"  Epochs: {config.get('epochs', 50)}")
    print(f"  Core: {config.get('core_dim', 64)}D hypersphere, "
          f"{config.get('n_basins', 32)} basins")
    print(f"{'='*70}\n")

    global_step = 0
    best_pr = 0

    for epoch in range(config.get("epochs", 50)):
        epoch_losses = defaultdict(float)
        epoch_steps = 0

        compressor.train()
        core.reset_tracking()

        for batch_idx, (view1, view2, labels) in enumerate(dataloader):
            view1 = view1.to(device)
            view2 = view2.to(device)

            optimizer.zero_grad()

            # Features already precomputed — go straight to compressor
            z1, sp1 = compressor(view1)  # [B, core_dim], scalar
            z2, sp2 = compressor(view2)

            # Core activations (for density/separation losses)
            act1, assign1 = core.activate(z1.detach())
            act2, assign2 = core.activate(z2.detach())
            assignments = assign1  # use view1 assignments

            # Loss stack
            total_loss, loss_dict = loss_stack.compute_all(
                z1, z2, assignments, core.prototypes, core.n_basins)

            # Add spectral compression penalties
            total_loss = total_loss + sp1 + sp2

            # Backward + step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            # EMA prototype update
            with torch.no_grad():
                z_all = torch.cat([z1.detach(), z2.detach()], dim=0)
                a_all = torch.cat([assign1, assign2], dim=0)
                core.update_prototypes(z_all, a_all)

            # Hippocampal storage (sample from batch)
            if hippocampus is not None and batch_idx % 10 == 0:
                sample_z = z1[0].detach().cpu().numpy().astype(np.float64)
                sample_basins = [int(assign1[0])]
                sample_weights = [float(act1[0, assign1[0]])]
                hippocampus.maybe_store(
                    manifold_state=sample_z,
                    active_basins=sample_basins,
                    basin_weights=sample_weights,
                    entropy=loss_dict.get("contrastive", 0),
                    injection_strength=0.0,
                    token_position=global_step,
                )

            # Track
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            epoch_steps += 1
            global_step += 1

        # ── Epoch summary ──
        avg = {k: v / max(epoch_steps, 1) for k, v in epoch_losses.items()}
        health = core.health()
        spec = spectral_health(z_all)

        print(f"  Epoch {epoch+1:3d}  "
              f"loss={avg['total']:.4f}  "
              f"con={avg['contrastive']:.3f}  "
              f"iso={avg['isotropy']:.4f}  "
              f"dens={avg['density_contrast']:.3f}  "
              f"sep={avg['separation']:.3f}  "
              f"PR={spec['participation_ratio']:.1f}  "
              f"cos={spec['mean_cos']:.3f}  "
              f"top_eig={spec['top_eigval_frac']:.3f}  "
              f"basins={health.get('active_basins', 0)}")

        # ── Spectral regulator: read geometry, adjust pressures ──
        reg_actions = regulator.regulate(epoch + 1, spec, health)
        if reg_actions:
            for action in reg_actions:
                print(f"    ⚡ Regulator: {action}")

        # ── Detailed report every 5 epochs ──
        if (epoch + 1) % 5 == 0:
            print(f"\n    ── Detailed Report @ Epoch {epoch+1} ──")
            print(f"    Spectral: PR={spec['participation_ratio']:.2f}  "
                  f"mean_cos={spec['mean_cos']:.4f}  "
                  f"eff_dim={spec['effective_dim']:.1f}")
            print(f"    Basins: active={health.get('active_basins', 0)}  "
                  f"util_ent={health.get('utilization_entropy', 0):.3f}  "
                  f"inter={health.get('inter_prototype_cos_mean', 0):.3f}  "
                  f"intra={health.get('intra_cohesion_mean', 0):.3f}  "
                  f"delta={health.get('intra_inter_delta', 0):.3f}")
            if hippocampus is not None:
                hs = hippocampus.session_summary()
                print(f"    Hippo: records={hs['n_records']}  "
                      f"chain={'✓' if hs['chain_valid'] else '✗'}")

            # Check success criteria
            success = True
            checks = []
            if spec["mean_cos"] < 0.5:
                checks.append(f"✓ mean_cos={spec['mean_cos']:.3f} < 0.5")
            else:
                checks.append(f"✗ mean_cos={spec['mean_cos']:.3f} >= 0.5")
                success = False
            if spec["participation_ratio"] > 10:
                checks.append(f"✓ PR={spec['participation_ratio']:.1f} > 10")
            else:
                checks.append(f"✗ PR={spec['participation_ratio']:.1f} <= 10")
                success = False
            if spec["top_eigval_frac"] < 0.15:
                checks.append(f"✓ top_eig={spec['top_eigval_frac']:.3f} < 0.15")
            else:
                checks.append(f"✗ top_eig={spec['top_eigval_frac']:.3f} >= 0.15")
                success = False
            delta = health.get("intra_inter_delta", 0)
            if delta > 0.2:
                checks.append(f"✓ delta={delta:.3f} > 0.2")
            else:
                checks.append(f"✗ delta={delta:.3f} <= 0.2")
                success = False

            for c in checks:
                print(f"    {c}")

            if success:
                print(f"\n    ★ CORE IS SOVEREIGN — all criteria met ★")
                save_core(compressor, core, hippocampus, epoch + 1,
                         config, output_dir)
                break

            print()

        # ── Save best ──
        if spec["participation_ratio"] > best_pr:
            best_pr = spec["participation_ratio"]
            save_core(compressor, core, hippocampus, epoch + 1,
                     config, output_dir, tag="best")

        # ── Periodic save ──
        if (epoch + 1) % 10 == 0:
            save_core(compressor, core, hippocampus, epoch + 1,
                     config, output_dir)

    print(f"\n{'='*70}")
    print(f"  Phase 1 Complete")
    print(f"  Best PR: {best_pr:.2f}")
    print(f"{'='*70}")


def save_core(compressor, core, hippocampus, epoch, config, output_dir,
              tag=None):
    """Save core state."""
    name = f"checkpoint_{epoch}" if tag is None else f"checkpoint_{tag}"
    path = os.path.join(output_dir, name)
    os.makedirs(path, exist_ok=True)

    torch.save(compressor.state_dict(),
              os.path.join(path, "compressor.pt"))
    torch.save({
        "prototypes": core.prototypes,
        "config": {
            "core_dim": core.core_dim,
            "n_basins": core.n_basins,
        }
    }, os.path.join(path, "core.pt"))

    if hippocampus is not None:
        summary = hippocampus.session_summary()
        with open(os.path.join(path, "hippocampus_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"    Saved {name}")


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ROE Phase 1 — Visual Core")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--core_dim", type=int, default=64)
    parser.add_argument("--n_basins", type=int, default=32)
    parser.add_argument("--vision_backbone", type=str, default="resnet50")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--output_dir", type=str, default="./roe_core")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "core_dim": args.core_dim,
        "n_basins": args.n_basins,
        "vision_backbone": args.vision_backbone,
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "num_workers": args.num_workers,
        "freeze_vision": True,
        "ema_alpha": 0.99,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        # Loss weights
        "lambda_contrastive": 1.0,
        "lambda_isotropy": 0.5,
        "lambda_density": 0.3,
        "lambda_separation": 0.5,
        "lambda_pr": 0.2,
        "pr_target": 15.0,
        "separation_margin": 0.3,
    }

    print("=" * 70)
    print("  ROE Engine — Phase 1: Pure Visual Geometric Core")
    print(f"  Vision: {args.vision_backbone} (frozen)")
    print(f"  Core: {args.core_dim}D hypersphere, {args.n_basins} basins")
    print(f"  Dataset: {args.dataset}")
    print("=" * 70)

    train_phase1(config)


if __name__ == "__main__":
    main()
