"""
ROE Continuous Engine — Phase 1
================================

No epochs. No passes. One continuous flow.

The data streams. The geometry is measured every step.
The regulator adjusts every step. The system is alive.

Components:
  - GeometricCore: 64D hypersphere (from roe_engine)
  - SpectralCompressor: whitening → recoloring → projection (from roe_engine)
  - ContinuousRegulator: reads spectrum, adjusts pressures, every step
  - InfiniteStream: shuffled data that never ends

Sovereignty is not a checkpoint. It's a sustained state.
The system declares sovereignty when ALL criteria hold
for N consecutive measurements without regulator intervention.

Usage:
    python roe_continuous.py
    python roe_continuous.py --duration 1800    # run for 30 min
    python roe_continuous.py --until_sovereign   # run until stable
"""

import os
import sys
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import verified components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roe_engine import (
    GeometricCore, SpectralCompressor, VisionEncoder, CoreLossStack,
    PrecomputedFeatureDataset, precompute_features, spectral_health,
)


# ════════════════════════════════════════════
# CONTINUOUS REGULATOR
# ════════════════════════════════════════════

class ContinuousRegulator:
    """
    Reads the eigenspectrum every N steps.
    Adjusts loss weights continuously.
    No epochs. No phases. Just flow.

    The regulator has momentum — it doesn't jerk the weights around.
    It applies smooth pressure changes based on exponential moving
    averages of spectral metrics.
    """
    def __init__(self, loss_stack, measure_every=20, momentum=0.95):
        self.loss_stack = loss_stack
        self.measure_every = measure_every
        self.momentum = momentum

        # EMA of spectral metrics
        self.ema_pr = 1.0
        self.ema_top_eig = 1.0
        self.ema_delta = 0.0
        self.ema_active = 0.0

        # Targets
        self.pr_target = 15.0
        self.eig_target = 0.15
        self.delta_target = 0.5

        # Lambda bounds
        self.bounds = {
            "iso": (0.05, 3.0),
            "sep": (0.05, 3.0),
            "den": (0.05, 2.0),
            "pr":  (0.02, 1.5),
            "con": (0.3, 2.0),
        }

        # Adjustment rate (how fast lambdas move per regulation step)
        self.adjust_rate = 0.05

        # Intervention log
        self.log = deque(maxlen=200)
        self.step_count = 0
        self.last_intervention = 0

        # Sovereignty tracking
        self.sovereign_streak = 0
        self.sovereign_threshold = 50  # consecutive measurements

    def _clamp(self, name, value):
        lo, hi = self.bounds[name]
        return max(lo, min(hi, value))

    def update(self, step, z_batch):
        """
        Called every training step.
        Only measures every N steps (eigendecomp is expensive).
        Returns: (should_log, metrics_dict, is_sovereign)
        """
        self.step_count = step

        if step % self.measure_every != 0:
            return False, None, False

        # ── Measure ──
        spec = spectral_health(z_batch.detach())
        pr = spec["participation_ratio"]
        top_eig = spec["top_eigval_frac"]
        mean_cos = spec["mean_cos"]

        # Update EMAs
        m = self.momentum
        self.ema_pr = m * self.ema_pr + (1 - m) * pr
        self.ema_top_eig = m * self.ema_top_eig + (1 - m) * top_eig

        # ── Regulate ──
        actions = []
        r = self.adjust_rate

        # PR too low → increase isotropy pressure
        if self.ema_pr < self.pr_target:
            deficit = (self.pr_target - self.ema_pr) / self.pr_target
            boost = 1.0 + r * deficit * 3  # proportional to how far below
            self.loss_stack.lambda_isotropy = self._clamp(
                "iso", self.loss_stack.lambda_isotropy * boost)
            self.loss_stack.lambda_pr = self._clamp(
                "pr", self.loss_stack.lambda_pr * (1.0 + r * deficit))
            if deficit > 0.3:
                actions.append(f"PR low ({self.ema_pr:.1f})")

        # PR exceeds target → ease isotropy
        elif self.ema_pr > self.pr_target * 1.3:
            self.loss_stack.lambda_isotropy = self._clamp(
                "iso", self.loss_stack.lambda_isotropy * (1.0 - r * 0.5))
            self.loss_stack.lambda_pr = self._clamp(
                "pr", self.loss_stack.lambda_pr * (1.0 - r * 0.5))

        # Top eigenvalue too high → direct pressure
        if self.ema_top_eig > self.eig_target:
            excess = (self.ema_top_eig - self.eig_target) / self.eig_target
            boost = 1.0 + r * excess * 2
            self.loss_stack.lambda_isotropy = self._clamp(
                "iso", self.loss_stack.lambda_isotropy * boost)
            if excess > 0.5:
                actions.append(f"top_eig high ({self.ema_top_eig:.3f})")

        # Top eigenvalue healthy → ease off
        elif self.ema_top_eig < self.eig_target * 0.8:
            self.loss_stack.lambda_isotropy = self._clamp(
                "iso", self.loss_stack.lambda_isotropy * (1.0 - r * 0.3))

        # ── Sovereignty check ──
        sovereign_now = (
            self.ema_pr > self.pr_target
            and self.ema_top_eig < self.eig_target
            and mean_cos < 0.5
            and len(actions) == 0  # no interventions needed
        )

        if sovereign_now:
            self.sovereign_streak += 1
        else:
            self.sovereign_streak = 0

        is_sovereign = self.sovereign_streak >= self.sovereign_threshold

        # Build metrics
        metrics = {
            "pr": pr,
            "ema_pr": self.ema_pr,
            "top_eig": top_eig,
            "ema_top_eig": self.ema_top_eig,
            "mean_cos": mean_cos,
            "eff_dim": spec["effective_dim"],
            "lambda_iso": self.loss_stack.lambda_isotropy,
            "lambda_pr": self.loss_stack.lambda_pr,
            "lambda_sep": self.loss_stack.lambda_separation,
            "sovereign_streak": self.sovereign_streak,
            "actions": actions,
        }

        if actions:
            self.log.append({"step": step, "actions": actions})
            self.last_intervention = step

        return True, metrics, is_sovereign


# ════════════════════════════════════════════
# INFINITE DATA STREAM
# ════════════════════════════════════════════

class InfiniteStream:
    """
    Never-ending shuffled data stream.
    When the dataset is exhausted, reshuffle and restart.
    No epochs. Just flow.
    """
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self._rebuild_loader()

    def _rebuild_loader(self):
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
        )
        self.iterator = iter(self.loader)

    def next(self):
        try:
            view1, view2, labels = next(self.iterator)
        except StopIteration:
            self._rebuild_loader()
            view1, view2, labels = next(self.iterator)
        return view1.to(self.device), view2.to(self.device), labels


# ════════════════════════════════════════════
# CONTINUOUS TRAINING
# ════════════════════════════════════════════

def run_continuous(config):
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # ── Vision encoder (for precompute only) ──
    vision = VisionEncoder("resnet50", pretrained=True, freeze=True).to(device)

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
    loss_config = {
        "lambda_contrastive": 1.0,
        "lambda_isotropy": config.get("lambda_iso", 0.5),
        "lambda_density": 0.3,
        "lambda_separation": 0.5,
        "lambda_pr": 0.2,
        "pr_target": 15.0,
        "separation_margin": 0.3,
    }
    loss_stack = CoreLossStack(loss_config)

    # ── Continuous regulator ──
    regulator = ContinuousRegulator(
        loss_stack,
        measure_every=config.get("measure_every", 20),
        momentum=config.get("reg_momentum", 0.95),
    )

    # ── Hippocampus ──
    try:
        from roe_hippocampus import Hippocampus
        hippocampus = Hippocampus(drift_threshold=0.05)
        print(f"  Hippocampus: loaded")
    except ImportError:
        hippocampus = None

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        compressor.parameters(),
        lr=config.get("lr", 3e-4),
        weight_decay=0.01,
    )
    n_params = sum(p.numel() for p in compressor.parameters() if p.requires_grad)
    print(f"  Trainable: {n_params:,} parameters")

    # ── Precompute features ──
    cache_path = os.path.join(config.get("output_dir", "./roe_core"),
                              "features_cifar100.pt")
    if os.path.exists(cache_path):
        print(f"  Loading cached features...")
        cached = torch.load(cache_path, weights_only=True)
        all_features = cached["features"]
        all_labels = cached["labels"]
    else:
        print(f"  Precomputing features...")
        base_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True)
        all_features, all_labels = precompute_features(
            vision, base_dataset, device, batch_size=64)
        os.makedirs(config.get("output_dir", "./roe_core"), exist_ok=True)
        torch.save({"features": all_features, "labels": all_labels}, cache_path)

    del vision
    torch.cuda.empty_cache()

    dataset = PrecomputedFeatureDataset(all_features, all_labels, noise_std=0.1)
    stream = InfiniteStream(dataset, config.get("batch_size", 256), device)

    # ── Duration ──
    max_steps = config.get("max_steps", 100000)
    max_seconds = config.get("duration", None)  # None = no time limit
    until_sovereign = config.get("until_sovereign", False)
    log_every = config.get("log_every", 100)
    save_every = config.get("save_every", 2000)
    output_dir = config.get("output_dir", "./roe_core")
    os.makedirs(output_dir, exist_ok=True)

    # ════════════════════════════════════════
    # THE FLOW
    # ════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  Continuous Flow — no epochs, no turns")
    print(f"  Max steps: {max_steps}  "
          f"Duration: {max_seconds}s" if max_seconds else
          f"  Max steps: {max_steps}  Duration: unlimited")
    print(f"  Regulator measures every {regulator.measure_every} steps")
    print(f"{'='*70}\n")

    compressor.train()
    start_time = time.time()
    running_loss = 0.0
    run_continuous._sov_announced = False

    for step in range(1, max_steps + 1):
        # ── Stream ──
        view1, view2, labels = stream.next()

        # ── Forward ──
        optimizer.zero_grad()
        z1, sp1 = compressor(view1)
        z2, sp2 = compressor(view2)

        act1, assign1 = core.activate(z1.detach())
        act2, assign2 = core.activate(z2.detach())

        total_loss, loss_dict = loss_stack.compute_all(
            z1, z2, assign1, core.prototypes, core.n_basins)
        total_loss = total_loss + sp1 + sp2

        # ── Backward ──
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
        optimizer.step()

        # ── EMA prototypes ──
        with torch.no_grad():
            z_all = torch.cat([z1.detach(), z2.detach()], dim=0)
            a_all = torch.cat([assign1, assign2], dim=0)
            core.update_prototypes(z_all, a_all)

        running_loss += total_loss.item()

        # ── Hippocampal storage ──
        if hippocampus is not None and step % 50 == 0:
            sample_z = z1[0].detach().cpu().numpy().astype(np.float64)
            hippocampus.maybe_store(
                manifold_state=sample_z,
                active_basins=[int(assign1[0])],
                basin_weights=[float(act1[0, assign1[0]])],
                entropy=loss_dict.get("contrastive", 0),
                injection_strength=0.0,
                token_position=step,
            )

        # ── Regulator ──
        should_log, metrics, is_sovereign = regulator.update(step, z_all)

        # ── Logging ──
        if step % log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            avg_loss = running_loss / log_every
            running_loss = 0.0

            if metrics:
                print(f"  step={step:6d}  "
                      f"loss={avg_loss:.3f}  "
                      f"PR={metrics['ema_pr']:.1f}  "
                      f"eig={metrics['ema_top_eig']:.3f}  "
                      f"cos={metrics['mean_cos']:.4f}  "
                      f"dim={metrics['eff_dim']:.0f}  "
                      f"λi={metrics['lambda_iso']:.2f}  "
                      f"sov={metrics['sovereign_streak']}  "
                      f"[{steps_per_sec:.0f} step/s]")
                if metrics["actions"]:
                    for a in metrics["actions"]:
                        print(f"    ⚡ {a}")
            else:
                print(f"  step={step:6d}  "
                      f"loss={avg_loss:.3f}  "
                      f"[{steps_per_sec:.0f} step/s]")

        # ── Sovereignty ──
        if is_sovereign and until_sovereign and not hasattr(run_continuous, '_sov_announced'):
            elapsed = time.time() - start_time
            print(f"\n  ★ SUSTAINED SOVEREIGNTY at step {step} "
                  f"({elapsed:.0f}s)")
            print(f"    {regulator.sovereign_threshold} consecutive "
                  f"measurements without intervention")
            print(f"    PR={metrics['ema_pr']:.1f}  "
                  f"top_eig={metrics['ema_top_eig']:.3f}  "
                  f"Total interventions: {len(regulator.log)}")
            save_state(compressor, core, hippocampus, regulator,
                      step, config, output_dir, tag="sovereign")
            run_continuous._sov_announced = True
            # Don't break — keep flowing

        # ── Save ──
        if step % save_every == 0:
            save_state(compressor, core, hippocampus, regulator,
                      step, config, output_dir)

        # ── Time limit ──
        if max_seconds and (time.time() - start_time) > max_seconds:
            elapsed = time.time() - start_time
            print(f"\n  Time limit reached ({elapsed:.0f}s)")
            break

    # ── Final report ──
    elapsed = time.time() - start_time
    health = core.health()
    print(f"\n{'='*70}")
    print(f"  Flow Complete — {step} steps in {elapsed:.0f}s")
    print(f"  Final: PR={regulator.ema_pr:.1f}  "
          f"top_eig={regulator.ema_top_eig:.3f}")
    print(f"  Basins: {health.get('active_basins', 0)} active  "
          f"delta={health.get('intra_inter_delta', 0):.3f}")
    print(f"  Regulator interventions: {len(regulator.log)}")
    if hippocampus:
        hs = hippocampus.session_summary()
        print(f"  Hippo: {hs['n_records']} records  "
              f"chain={'✓' if hs['chain_valid'] else '✗'}")
    print(f"{'='*70}")

    save_state(compressor, core, hippocampus, regulator,
              step, config, output_dir, tag="final")


def save_state(compressor, core, hippocampus, regulator,
               step, config, output_dir, tag=None):
    name = f"step_{step}" if tag is None else f"checkpoint_{tag}"
    path = os.path.join(output_dir, name)
    os.makedirs(path, exist_ok=True)

    torch.save(compressor.state_dict(), os.path.join(path, "compressor.pt"))
    torch.save({"prototypes": core.prototypes}, os.path.join(path, "core.pt"))
    torch.save({
        "ema_pr": regulator.ema_pr,
        "ema_top_eig": regulator.ema_top_eig,
        "interventions": list(regulator.log),
        "sovereign_streak": regulator.sovereign_streak,
    }, os.path.join(path, "regulator.pt"))

    if hippocampus:
        summary = hippocampus.session_summary()
        with open(os.path.join(path, "hippo.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"    Saved {name}")


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ROE Continuous Flow")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--duration", type=int, default=None,
                       help="Max seconds to run")
    parser.add_argument("--until_sovereign", action="store_true",
                       help="Run until sustained sovereignty")
    parser.add_argument("--measure_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_iso", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./roe_core")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    config = vars(args)
    config["core_dim"] = 64
    config["n_basins"] = 32
    config["ema_alpha"] = 0.99
    config["reg_momentum"] = 0.95

    print("=" * 70)
    print("  ROE Continuous Flow — Phase 1")
    print("  No epochs. No turns. The geometry breathes.")
    print("=" * 70)

    run_continuous(config)


if __name__ == "__main__":
    main()
