"""
ROE Phase 2b — Token as Gradient
==================================

The key insight:
  A token is not a point on the sphere.
  A token is a FORCE on the sphere.
  
  Vision says "I am here" (position).
  Language says "move this way" (gradient).

Architecture:
  1. Vision → z_v (point on 64d sphere)          [frozen, works]
  2. Token → g_t (tangent vector in 64d)          [trainable]  
  3. z_modified = normalize(z_v + α * g_t)        [the push]
  4. z_modified should land in the RIGHT basin     [the signal]

Training:
  - Correct token gradient stabilizes the point (same basin)
  - Wrong token gradient destabilizes it (different basin)
  - The gradient carries meaning. The geometry receives force.

This solves the text-as-images collapse (PR=7.9).
Tokens don't need visual diversity. They need semantic direction.

Usage:
    python roe_token_gradient.py --max_steps 10000
    python roe_token_gradient.py --max_steps 10000 --alpha 0.3
"""

import os
import sys
import json
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
# TOKEN GRADIENT ENCODER
# ════════════════════════════════════════════

class TokenGradientEncoder(nn.Module):
    """
    Converts token indices into tangent vectors on the sphere.
    
    NOT an embedding that gives you a point.
    A projection that gives you a DIRECTION.
    
    The output is NOT normalized — it's a tangent vector,
    not a point. Its magnitude is the strength of the push.
    """
    def __init__(self, n_tokens, core_dim=64, hidden_dim=128):
        super().__init__()
        self.core_dim = core_dim
        
        # Token embedding (learnable)
        self.embedding = nn.Embedding(n_tokens, hidden_dim)
        
        # Project to tangent space
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, core_dim),
            # NO normalization — this is a direction, not a point
        )
        
        # Initialize small — gradients should be gentle pushes
        with torch.no_grad():
            self.proj[-1].weight.mul_(0.1)
            self.proj[-1].bias.mul_(0.1)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  TokenGradientEncoder: {n_tokens} tokens → {core_dim}d "
              f"tangent vectors ({n_params:,} params)")
    
    def forward(self, token_ids):
        """
        token_ids: [B] integer indices
        Returns: [B, core_dim] tangent vectors (NOT on sphere)
        """
        emb = self.embedding(token_ids)  # [B, hidden_dim]
        gradient = self.proj(emb)         # [B, core_dim]
        return gradient


# ════════════════════════════════════════════
# GRADIENT APPLICATION
# ════════════════════════════════════════════

def apply_gradient(z_v, g_t, alpha=0.2):
    """
    Push a point on the sphere by a tangent vector.
    
    z_v: [B, d] — points on unit sphere (vision)
    g_t: [B, d] — tangent vectors (token gradients)
    alpha: strength of push
    
    Returns: [B, d] — new points on unit sphere
    """
    # Project gradient onto tangent plane (remove radial component)
    # g_tangent = g_t - (g_t · z_v) * z_v
    radial = (g_t * z_v).sum(dim=-1, keepdim=True)
    g_tangent = g_t - radial * z_v
    
    # Apply and renormalize
    z_new = z_v + alpha * g_tangent
    z_new = F.normalize(z_new, dim=-1)
    
    return z_new


# ════════════════════════════════════════════
# GRADIENT LOSS
# ════════════════════════════════════════════

class GradientLoss:
    """
    Three losses for token-as-gradient:
    
    1. STABILIZATION: correct token should keep point in same basin
       or push it toward the class centroid
       
    2. DESTABILIZATION: wrong token should push point away
    
    3. GRADIENT MAGNITUDE: don't let gradients explode or vanish
    """
    def __init__(self, lambda_stable=1.0, lambda_unstable=0.5,
                 lambda_magnitude=0.1, alpha=0.2):
        self.lambda_stable = lambda_stable
        self.lambda_unstable = lambda_unstable
        self.lambda_magnitude = lambda_magnitude
        self.alpha = alpha
    
    def compute(self, z_v, g_correct, g_wrong, prototypes, labels):
        """
        z_v: [B, d] — vision points on sphere
        g_correct: [B, d] — gradient from correct token
        g_wrong: [B, d] — gradient from wrong token
        prototypes: [n_basins, d] — basin centers
        labels: [B] — class labels
        """
        B = z_v.shape[0]
        
        # Apply gradients
        z_correct = apply_gradient(z_v, g_correct, self.alpha)
        z_wrong = apply_gradient(z_v, g_wrong, self.alpha)
        
        # 1. STABILIZATION
        # Correct token push should increase cosine with original position
        # (or at least not decrease it much)
        cos_correct = (z_correct * z_v).sum(dim=-1)  # [B]
        
        # Also: correct push should increase similarity with class centroid
        # Compute per-class centroids on the fly
        unique_labels = labels.unique()
        centroid_sim_correct = torch.zeros(B, device=z_v.device)
        centroid_sim_original = torch.zeros(B, device=z_v.device)
        for c in unique_labels:
            mask = labels == c
            if mask.sum() < 2:
                continue
            centroid = F.normalize(z_v[mask].mean(dim=0), dim=0)
            centroid_sim_correct[mask] = (z_correct[mask] * centroid).sum(dim=-1)
            centroid_sim_original[mask] = (z_v[mask] * centroid).sum(dim=-1)
        
        # Correct gradient should improve centroid alignment
        improvement = centroid_sim_correct - centroid_sim_original
        stable_loss = -improvement.mean()  # maximize improvement
        
        # 2. DESTABILIZATION
        # Wrong token push should decrease cosine with original position
        cos_wrong = (z_wrong * z_v).sum(dim=-1)  # [B]
        
        # Margin: wrong push should be worse than correct push
        margin = 0.1
        destable_loss = F.relu(cos_wrong - cos_correct + margin).mean()
        
        # 3. MAGNITUDE REGULARIZATION
        # Gradients should be moderate — not zero, not huge
        mag_correct = g_correct.norm(dim=-1)
        mag_wrong = g_wrong.norm(dim=-1)
        # Target magnitude around 0.5
        mag_loss = ((mag_correct - 0.5) ** 2).mean() + \
                   ((mag_wrong - 0.5) ** 2).mean()
        
        total = (self.lambda_stable * stable_loss +
                 self.lambda_unstable * destable_loss +
                 self.lambda_magnitude * mag_loss)
        
        return total, {
            "stable": stable_loss.item(),
            "destable": destable_loss.item(),
            "magnitude": mag_loss.item(),
            "total": total.item(),
            "cos_correct": cos_correct.mean().item(),
            "cos_wrong": cos_wrong.mean().item(),
            "improvement": improvement.mean().item(),
            "grad_mag": mag_correct.mean().item(),
        }


# ════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════

class TokenGradientDataset(Dataset):
    """
    Triplets: (vision_features, correct_token, wrong_token)
    """
    def __init__(self, vision_features, labels, n_classes=100):
        self.features = vision_features
        self.labels = labels
        self.n_classes = n_classes
        
        # Index by class for fast negative sampling
        self.class_indices = defaultdict(list)
        for i, l in enumerate(labels.tolist()):
            self.class_indices[l].append(i)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx].item()
        
        # Correct token = class label
        correct = label
        
        # Wrong token = random different class
        wrong = label
        while wrong == label:
            wrong = torch.randint(self.n_classes, (1,)).item()
        
        return feat, correct, wrong, label


# ════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════

@torch.no_grad()
def gradient_metrics(z_v, token_encoder, prototypes, labels,
                     n_classes, alpha, device):
    """
    Measure how well token gradients steer the geometry.
    """
    B = z_v.shape[0]
    
    # Generate all class gradients
    all_tokens = torch.arange(n_classes, device=device)
    all_gradients = token_encoder(all_tokens)  # [n_classes, d]
    
    # For each vision point, apply its correct token gradient
    correct_tokens = labels.to(device)
    g_correct = token_encoder(correct_tokens)
    z_pushed = apply_gradient(z_v, g_correct, alpha)
    
    # Basin assignment: before and after push
    assign_before = torch.mm(z_v, prototypes.T).argmax(dim=-1)
    assign_after = torch.mm(z_pushed, prototypes.T).argmax(dim=-1)
    
    # Did the push keep it in the same basin?
    same_basin = (assign_before == assign_after).float().mean().item()
    
    # Centroid improvement
    improvements = []
    for c in labels.unique():
        mask = labels == c
        if mask.sum() < 2:
            continue
        centroid = F.normalize(z_v[mask].mean(dim=0), dim=0)
        sim_before = (z_v[mask] * centroid).sum(dim=-1).mean().item()
        sim_after = (z_pushed[mask] * centroid).sum(dim=-1).mean().item()
        improvements.append(sim_after - sim_before)
    
    mean_improvement = np.mean(improvements) if improvements else 0
    
    # Classification by gradient: for each point, which token gradient
    # produces the highest cosine with class centroid?
    # This tests if tokens are semantically meaningful forces
    class_centroids = torch.zeros(n_classes, z_v.shape[1], device=device)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_centroids[c] = F.normalize(z_v[mask].mean(dim=0), dim=0)
    
    # For a random subset, test all 100 token gradients
    n_test = min(500, B)
    idx = torch.randperm(B)[:n_test]
    z_test = z_v[idx]
    labels_test = labels[idx]
    
    correct_count = 0
    for i in range(n_test):
        z_i = z_test[i].unsqueeze(0).expand(n_classes, -1)
        z_pushed_all = apply_gradient(z_i, all_gradients, alpha)
        # Which push gives highest similarity to that token's centroid?
        sims = (z_pushed_all * class_centroids).sum(dim=-1)
        predicted = sims.argmax().item()
        if predicted == labels_test[i].item():
            correct_count += 1
    
    gradient_accuracy = correct_count / n_test
    
    # Gradient diversity: how spread are the gradients?
    g_normalized = F.normalize(all_gradients, dim=-1)
    g_cos = torch.mm(g_normalized, g_normalized.T)
    mask = ~torch.eye(n_classes, dtype=torch.bool, device=device)
    mean_gradient_cos = g_cos[mask].mean().item()
    
    return {
        "same_basin_rate": same_basin,
        "centroid_improvement": mean_improvement,
        "gradient_accuracy": gradient_accuracy,
        "mean_gradient_cos": mean_gradient_cos,
        "mean_gradient_mag": all_gradients.norm(dim=-1).mean().item(),
    }


# ════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    
    # Load sovereign core
    core_path = config.get("core_checkpoint") or "./roe_core/checkpoint_sovereign"
    if not os.path.exists(core_path):
        for alt in ["./roe_core/curriculum/stage_2",
                     "./roe_core/curriculum/stage_1",
                     "./roe_core/step_20000"]:
            if os.path.exists(alt):
                core_path = alt
                break
    
    if not os.path.exists(core_path):
        print(f"  ERROR: No checkpoint found. Tried {core_path}")
        print(f"  Run roe_curriculum.py first to create a checkpoint.")
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
    
    print(f"  Core: loaded from {core_path}")
    
    # Load vision features
    v_cache = "./roe_core/features_cifar100.pt"
    if not os.path.exists(v_cache):
        print(f"  ERROR: Features not found at {v_cache}")
        print(f"  Run roe_curriculum.py first to generate features.")
        return
    cached = torch.load(v_cache, weights_only=True)
    v_features = cached["features"]
    v_labels = cached["labels"]
    n_classes = v_labels.unique().numel()
    print(f"  Data: {len(v_features)} images, {n_classes} classes")
    
    # Pre-project all vision features to core space
    print(f"  Projecting to core space...")
    all_z = []
    with torch.no_grad():
        for i in range(0, len(v_features), 512):
            batch = v_features[i:i+512].to(device)
            z, _ = vision_compressor(batch)
            all_z.append(z.cpu())
    z_all = torch.cat(all_z, dim=0)
    print(f"  Projected: {z_all.shape}")
    
    # Token gradient encoder
    alpha = config.get("alpha", 0.2)
    token_encoder = TokenGradientEncoder(
        n_tokens=n_classes, core_dim=64,
        hidden_dim=config.get("hidden_dim", 128)).to(device)
    
    # Dataset
    dataset = TokenGradientDataset(z_all, v_labels, n_classes)
    dataloader = DataLoader(
        dataset, batch_size=config.get("batch_size", 256),
        shuffle=True, num_workers=0, drop_last=True)
    
    # Loss + optimizer
    loss_fn = GradientLoss(alpha=alpha)
    optimizer = torch.optim.AdamW(
        token_encoder.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=0.01)
    
    # Training loop
    max_steps = config.get("max_steps", 10000)
    log_every = config.get("log_every", 100)
    eval_every = config.get("eval_every", 500)
    
    print(f"\n{'='*60}")
    print(f"  Token as Gradient — Training")
    print(f"  Alpha (push strength): {alpha}")
    print(f"  Max steps: {max_steps}")
    print(f"{'='*60}\n")
    
    token_encoder.train()
    step = 0
    running = defaultdict(float)
    running_count = 0
    data_iter = iter(dataloader)
    start_time = time.time()
    
    while step < max_steps:
        try:
            z_batch, correct, wrong, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            z_batch, correct, wrong, labels = next(data_iter)
        
        z_batch = z_batch.to(device)
        correct = correct.to(device)
        wrong = wrong.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        g_correct = token_encoder(correct)
        g_wrong = token_encoder(wrong)
        
        total_loss, loss_dict = loss_fn.compute(
            z_batch, g_correct, g_wrong,
            core.prototypes, labels)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(token_encoder.parameters(), 1.0)
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
                  f"stable={avg['stable']:.3f}  "
                  f"destab={avg['destable']:.3f}  "
                  f"cos_c={avg['cos_correct']:.3f}  "
                  f"cos_w={avg['cos_wrong']:.3f}  "
                  f"imp={avg['improvement']:.4f}  "
                  f"mag={avg['grad_mag']:.3f}  "
                  f"[{step/elapsed:.0f} step/s]")
            running = defaultdict(float)
            running_count = 0
        
        if step % eval_every == 0:
            token_encoder.eval()
            
            # Eval on subset
            n_eval = 2048
            idx = torch.randperm(len(z_all))[:n_eval]
            z_eval = z_all[idx].to(device)
            l_eval = v_labels[idx].to(device)
            
            metrics = gradient_metrics(
                z_eval, token_encoder, core.prototypes,
                l_eval, n_classes, alpha, device)
            
            print(f"\n    ── Gradient Report @ Step {step} ──")
            print(f"    Same basin rate: {metrics['same_basin_rate']:.1%}")
            print(f"    Centroid improvement: {metrics['centroid_improvement']:.4f}")
            print(f"    Gradient accuracy: {metrics['gradient_accuracy']:.1%}")
            print(f"    Gradient diversity (cos): {metrics['mean_gradient_cos']:.3f}")
            print(f"    Gradient magnitude: {metrics['mean_gradient_mag']:.3f}")
            
            checks = []
            if metrics['centroid_improvement'] > 0:
                checks.append(f"✓ improving centroids")
            else:
                checks.append(f"✗ not improving centroids")
            if metrics['gradient_accuracy'] > 0.05:
                checks.append(f"✓ gradient_acc={metrics['gradient_accuracy']:.1%}")
            else:
                checks.append(f"✗ gradient_acc={metrics['gradient_accuracy']:.1%}")
            if metrics['mean_gradient_cos'] < 0.5:
                checks.append(f"✓ diverse gradients")
            else:
                checks.append(f"✗ collapsed gradients")
            
            for c in checks:
                print(f"    {c}")
            
            if all(c.startswith("✓") for c in checks):
                print(f"\n    ★ TOKENS ARE FORCES ★")
            print()
            
            token_encoder.train()
    
    # Save
    output_dir = config.get("output_dir", "./roe_core/token_gradient")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(token_encoder.state_dict(),
              os.path.join(output_dir, "token_encoder.pt"))
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Complete — {step} steps in {elapsed:.0f}s")
    print(f"  Saved to {output_dir}")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--core_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                       default="./roe_core/token_gradient")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  ROE — Token as Gradient")
    print("  A token is not a point. It's a force.")
    print("=" * 60)
    
    train(vars(args))


if __name__ == "__main__":
    main()
