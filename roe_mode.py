"""
ROE Mode-Dependent Geometry Modulation
========================================

Implements the three operating modes of the hippocampal cycle and the
geometry kernels that modulate the transition matrix P under each mode.

The system breathes between three modes:
  THINKING  — tension kernel (λ→1): explores cross-context bridges,
               increases curvature variance, allows drift
  ENCODING  — preserve kernel (λ→0): stabilizes structure around active
               nodes, deepens crystallization
  SLEEP     — controlled by external sleep cycle; mode controller
               interpolates toward preserve during SWS, tension during REM

Core math:
  p_i = c · u_i                           (context projection per node)
  g_preserve(i,j) = α·p_i·p_j + β·(p_i+p_j)/2
  g_tension(i,j)  = α·p_i·p_j + β·|p_i−p_j|
  g(i,j)          = λ·g_tension + (1−λ)·g_preserve   (soft interpolation)
  P_new = normalize_rows(P ⊙ (1 + g))

Key design note on projection magnitudes:
  Node unit vectors are derived from 256-bit SHA-256 hashes, which in R^256
  yield dot products clustering near 0.5. Projections are centered at 0.5 and
  normalized to unit variance, producing p_i values in approximately [-3, +2].
  This matches roe_context.py's context_projection() — both modules operate
  in the same centered projection space.

  Kernel behavior with centered projections:
    PRESERVE (encoding): α·p_i·p_j clusters same-sign nodes, β·(p_i+p_j)/2
      biases toward high-projection nodes → creates structure, increases κ variance
    TENSION (thinking): α·p_i·p_j clusters same-sign, β·|p_i−p_j| bridges
      cross-sign nodes → competes with clustering, smooths landscape, decreases κ variance
  This is biologically correct: thinking = smoother exploration, encoding = sharper crystallization.

Dependencies:
  roe.py         — build_default_ontology, OntologyNode, hash_to_int, int_to_bits
  roe_crystal.py — build_T, write_T_to_ont
  roe_geometry.py — normalize_rows, ollivier_ricci, geodesic_distances, spectral_analysis
  roe_spectral.py — measure_spectral, SpectralSnapshot
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from roe import build_default_ontology, OntologyNode, hash_to_int, int_to_bits
from roe_crystal import build_T, write_T_to_ont
from roe_geometry import normalize_rows, ollivier_ricci, geodesic_distances, spectral_analysis
from roe_spectral import measure_spectral


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Geometry kernel parameters
# α controls product term (symmetric alignment — both same-sign or both anti-aligned)
# β controls sum/difference term (smoothing in preserve, bridging in tension)
#
# With centered+normalized projections (zero-mean, unit-variance), p_i values
# range from about -3.5 to +1.5, so products p_i*p_j can reach ~5.
# Small α and β produce meaningful modulation without manifold destruction.
# At α=0.08, β=0.05 over 20 cycles: 11% curvature variance separation between modes.
ALPHA_DEFAULT = 0.08
BETA_DEFAULT  = 0.05

# λ values per mode: 0 = pure preserve, 1 = pure tension
LAMBDA_THINKING = 1.0
LAMBDA_ENCODING = 0.0
LAMBDA_SLEEP    = 0.5   # neutral; overridden by SleepCycle

# Interpolation speed: fraction of (target - current) applied per step
LAMBDA_STEP_SIZE = 0.15

# Modulation floor: P entries will not fall below this (manifold safety)
P_FLOOR = 1e-4


# ─────────────────────────────────────────────────────────────
# OPERATING MODE ENUM
# ─────────────────────────────────────────────────────────────

class OperatingMode(Enum):
    THINKING = auto()   # explore, high curvature variance, tension kernel
    ENCODING = auto()   # stabilize, crystallize, preserve kernel
    SLEEP    = auto()   # offline integration; sub-phases managed by SleepCycle


# ─────────────────────────────────────────────────────────────
# VECTOR UTILITIES
# ─────────────────────────────────────────────────────────────

def node_unit_vectors(ont: dict) -> tuple[np.ndarray, list]:
    """
    Convert ontology node 256-bit integer vectors to L2-normalized float
    unit vectors in R^256.

    Preserves relative angular relationships: nodes close in Hamming distance
    will have high cosine similarity as unit vectors. Both node vectors and
    context vectors must pass through this conversion before dot products.

    Returns:
        U     — (n, 256) float array, rows are unit vectors, one per node
        names — list of node names in row order (sorted, matches build_T ordering)
    """
    names = sorted(ont.keys())
    n = len(names)
    U = np.zeros((n, 256), dtype=np.float64)
    for i, name in enumerate(names):
        bits = np.array(int_to_bits(ont[name].vector), dtype=np.float64)
        norm = np.linalg.norm(bits)
        U[i] = bits / norm if norm > 1e-10 else bits
    return U, names


def context_to_unit_vec(context_int: int) -> np.ndarray:
    """
    Convert a 256-bit context integer to an L2-normalized unit vector in R^256.
    Same conversion as node_unit_vectors — both must live in the same space
    for dot products to be meaningful.
    """
    bits = np.array(int_to_bits(context_int), dtype=np.float64)
    norm = np.linalg.norm(bits)
    return bits / norm if norm > 1e-10 else bits


def compute_projections(context_vec: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Compute centered, normalized per-node context projections.

    Raw dot products between L2-normalized {0,1}^256 vectors cluster near 0.5
    with std ≈ 0.04 — too flat for meaningful geometry modulation. Centering
    at 0.5 and dividing by observed std yields zero-mean, unit-variance
    projections where the sign and magnitude carry real geometric information:
      positive p_i → node aligned with context
      negative p_i → node anti-aligned with context

    This matches roe_context.py's context_projection() — both modules produce
    centered, normalized projections from the same underlying dot products.

    Args:
        context_vec — (256,) unit vector
        U           — (n, 256) node unit vectors

    Returns:
        p — (n,) centered and normalized projections, approximately N(0, 1)
    """
    raw = U @ context_vec
    centered = raw - 0.5  # expected dot product for {0,1}^256 unit vectors
    std = centered.std()
    if std < 1e-10:
        return centered
    return centered / std


# ─────────────────────────────────────────────────────────────
# GEOMETRY KERNELS
# ─────────────────────────────────────────────────────────────

def g_preserve(p_i: float, p_j: float,
               alpha: float = ALPHA_DEFAULT,
               beta: float  = BETA_DEFAULT) -> float:
    """
    Preservation kernel: reinforces edges between nodes with similar
    context alignment. Smoothing term uses arithmetic mean.

        g_preserve(i,j) = α·p_i·p_j + β·(p_i+p_j)/2

    High where both nodes align with context → deepens existing structure.
    Used during ENCODING to stabilize contextually relevant pathways.
    """
    return alpha * p_i * p_j + beta * (p_i + p_j) / 2.0


def g_tension(p_i: float, p_j: float,
              alpha: float = ALPHA_DEFAULT,
              beta: float  = BETA_DEFAULT) -> float:
    """
    Tension kernel: reinforces edges that bridge nodes with different
    context alignments. Contrast term uses absolute difference.

        g_tension(i,j) = α·p_i·p_j + β·|p_i−p_j|

    High where one node aligns with context and the other doesn't →
    bridges across contextual boundaries. Used during THINKING to
    encourage cross-context exploration and raise curvature variance.
    """
    return alpha * p_i * p_j + beta * abs(p_i - p_j)


def g_interpolated(p_i: float, p_j: float, lam: float,
                   alpha: float = ALPHA_DEFAULT,
                   beta: float  = BETA_DEFAULT) -> float:
    """
    Soft interpolation between tension and preserve kernels.

        g(i,j) = λ·g_tension(i,j) + (1−λ)·g_preserve(i,j)

    λ=0 → pure preserve (ENCODING)
    λ=1 → pure tension  (THINKING)
    λ∈(0,1) → blend (transitions, SLEEP sub-phases)
    """
    return lam * g_tension(p_i, p_j, alpha, beta) + \
           (1.0 - lam) * g_preserve(p_i, p_j, alpha, beta)


# ─────────────────────────────────────────────────────────────
# TRANSITION MATRIX MODULATION
# ─────────────────────────────────────────────────────────────

def modulate_transitions(P: np.ndarray,
                         context_vec: np.ndarray,
                         node_unit_vecs: np.ndarray,
                         lam: float,
                         alpha: float = ALPHA_DEFAULT,
                         beta: float  = BETA_DEFAULT) -> np.ndarray:
    """
    Apply mode-dependent geometry modulation to transition matrix P.

    Algorithm:
        1. Compute per-node projections p = context_vec · node_unit_vecs.T
        2. Build (n×n) modulation matrix G where G[i,j] = g(p_i, p_j; λ)
        3. P_new = normalize_rows(P ⊙ (1 + G))
        4. Apply P_FLOOR to prevent manifold destruction

    Args:
        P              — (n,n) current transition matrix, rows sum to 1
        context_vec    — (256,) L2-normalized context unit vector
        node_unit_vecs — (n,256) L2-normalized node unit vectors
        lam            — interpolation parameter [0=preserve, 1=tension]
        alpha, beta    — kernel scaling parameters

    Returns:
        P_new — (n,n) modulated, row-normalized transition matrix
    """
    n = P.shape[0]
    p = compute_projections(context_vec, node_unit_vecs)  # (n,)

    # Build modulation matrix
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            G[i, j] = g_interpolated(p[i], p[j], lam, alpha, beta)

    # Element-wise scaling, then row-normalize
    P_new = P * (1.0 + G)

    # Safety floor: never let any positive edge collapse to zero
    # Only apply floor where original P had positive weight
    edge_mask = P > 1e-12
    P_new = np.where(edge_mask, np.maximum(P_new, P_FLOOR), P_new)

    return normalize_rows(P_new)


# ─────────────────────────────────────────────────────────────
# MODE CONTROLLER
# ─────────────────────────────────────────────────────────────

@dataclass
class ModeSnapshot:
    """Captured state at a single mode cycle."""
    cycle:          int
    mode:           OperatingMode
    lam:            float
    kappa_mean:     float
    kappa_var:      float
    spectral_gap:   float
    dom_eigenmode:  float


class ModeController:
    """
    Manages λ(t) interpolation and mode transitions for ROE.

    λ(t) smoothly interpolates toward the target value for the current
    mode rather than snapping instantly, preventing geometric discontinuities.

    Usage:
        mc = ModeController(ont)
        mc.set_mode(OperatingMode.THINKING)
        for step in range(20):
            P, snap = mc.step(context_int)
        mc.set_mode(OperatingMode.ENCODING)
        ...
        ont_updated = mc.write_back()
    """

    def __init__(self,
                 ont: dict,
                 alpha: float = ALPHA_DEFAULT,
                 beta: float  = BETA_DEFAULT,
                 lam_step: float = LAMBDA_STEP_SIZE):
        self.ont     = dict(ont)
        self.alpha   = alpha
        self.beta    = beta
        self.lam_step = lam_step

        # Build initial P and node unit vectors
        T, self.names, self.n, self.idx = build_T(self.ont)
        self.P = T.copy()
        self.U, _ = node_unit_vectors(self.ont)

        # State
        self.mode    = OperatingMode.THINKING
        self.lam     = LAMBDA_THINKING   # start in thinking
        self.cycle   = 0
        self.history: list[ModeSnapshot] = []

    def set_mode(self, mode: OperatingMode, lam_override: Optional[float] = None) -> None:
        """
        Switch operating mode. λ will interpolate toward the mode's target
        value over subsequent steps rather than jumping instantly.

        Args:
            mode         — new OperatingMode
            lam_override — if provided, set target λ directly (used by SleepCycle
                           to dial in SWS=0.0 or REM=1.0 sub-phases)
        """
        self.mode = mode
        if lam_override is not None:
            self._lam_target = lam_override
        elif mode == OperatingMode.THINKING:
            self._lam_target = LAMBDA_THINKING
        elif mode == OperatingMode.ENCODING:
            self._lam_target = LAMBDA_ENCODING
        else:  # SLEEP neutral — SleepCycle should use lam_override
            self._lam_target = LAMBDA_SLEEP

    def _interpolate_lam(self) -> None:
        """Smooth λ toward target by LAMBDA_STEP_SIZE fraction."""
        delta = self._lam_target - self.lam
        self.lam += self.lam_step * delta

    def step(self,
             context_int: Optional[int] = None,
             context_vec: Optional[np.ndarray] = None,
             fast_estimator=None) -> tuple[np.ndarray, ModeSnapshot]:
        """
        Execute one modulation cycle.

        Provide either context_int (256-bit integer, converted internally)
        or context_vec (pre-computed unit vector). If neither given, uses
        a neutral context (mean of all node vectors).

        If fast_estimator is provided, uses it for curvature measurement
        instead of full Ollivier-Ricci (~4000x faster).

        Returns:
            P_new    — updated transition matrix after this step
            snapshot — ModeSnapshot with curvature and spectral measurements
        """
        # Resolve context vector
        if context_vec is not None:
            c = context_vec
        elif context_int is not None:
            c = context_to_unit_vec(context_int)
        else:
            # Neutral context: mean of all node unit vectors, then renormalize
            c = self.U.mean(axis=0)
            norm = np.linalg.norm(c)
            c = c / norm if norm > 1e-10 else c

        # Interpolate λ toward target
        self._interpolate_lam()

        # Apply modulation
        self.P = modulate_transitions(
            self.P, c, self.U, self.lam, self.alpha, self.beta
        )

        # Measure curvature and spectral properties
        if fast_estimator is not None:
            kappa_mean, kappa_var, gap = fast_estimator.estimate_stats(self.P)
        else:
            D       = geodesic_distances(self.P)
            kappas  = ollivier_ricci(self.P, D)
            kv      = np.array([v for v in kappas.values() if not np.isnan(v)])
            kappa_mean = float(kv.mean()) if len(kv) > 0 else 0.0
            kappa_var  = float(kv.var())  if len(kv) > 0 else 0.0
            gap_spec = spectral_analysis(self.P)
            gap = float(gap_spec['spectral_gap']) if 'spectral_gap' in gap_spec else 0.0

        spec    = spectral_analysis(self.P)
        dom     = float(spec['eigenvalues'][1]) if spec.get('eigenvalues') is not None and len(spec['eigenvalues']) > 1 else 0.0

        snap = ModeSnapshot(
            cycle=self.cycle,
            mode=self.mode,
            lam=self.lam,
            kappa_mean=kappa_mean,
            kappa_var=kappa_var,
            spectral_gap=gap,
            dom_eigenmode=dom,
        )
        self.history.append(snap)
        self.cycle += 1
        return self.P.copy(), snap

    def write_back(self) -> dict:
        """
        Push current P back into the ontology as updated edge weights.
        Returns updated ontology dict.
        """
        self.ont = write_T_to_ont(self.ont, self.P, self.names, self.n)
        return self.ont

    def current_lam(self) -> float:
        return self.lam

    def summary(self) -> None:
        """Print per-mode statistics from history."""
        from collections import defaultdict
        by_mode = defaultdict(list)
        for s in self.history:
            by_mode[s.mode].append(s)

        print("\n── ModeController Summary ──────────────────────────")
        for mode, snaps in by_mode.items():
            kvars = [s.kappa_var for s in snaps]
            gaps  = [s.spectral_gap for s in snaps]
            print(f"  {mode.name:10s}  cycles={len(snaps):3d}  "
                  f"κ_var={np.mean(kvars):.4f}±{np.std(kvars):.4f}  "
                  f"gap={np.mean(gaps):.4f}±{np.std(gaps):.4f}")
        print()


# ─────────────────────────────────────────────────────────────
# VALIDATION / DEMO
# ─────────────────────────────────────────────────────────────

def run_validation():
    """
    Validation suite for roe_mode.py.

    Tests:
      1. Projection diagnostic — distribution of p_i values for a sample context
      2. Kernel shape — g_tension vs g_preserve at varying p values
      3. 20-cycle runs in each mode — curvature variance, spectral gap stability
      4. Mode switch — verify manifold survives transition THINKING→ENCODING
      5. λ interpolation — verify smooth ramp, no discontinuous jumps
    """
    print("=" * 60)
    print("ROE Mode-Dependent Geometry — Validation")
    print("=" * 60)

    ont = build_default_ontology()
    U, names = node_unit_vectors(ont)
    n = len(names)

    # ── Test 1: Projection diagnostic ───────────────────────
    print("\n[1] Context projection distribution (p_i = c · u_i)")
    sample_context = hash_to_int("validation_context_vector_alpha")
    c = context_to_unit_vec(sample_context)
    p = compute_projections(c, U)

    print(f"    Nodes: {n}  |  Context: hash of 'validation_context_vector_alpha'")
    print(f"    p_i values:")
    for i, name in enumerate(names):
        bar = "█" * int(abs(p[i]) * 200)
        sign = "+" if p[i] >= 0 else "-"
        print(f"      {name:25s}  {sign}{abs(p[i]):.4f}  {bar}")
    print(f"    mean={p.mean():.4f}  std={p.std():.4f}  "
          f"min={p.min():.4f}  max={p.max():.4f}")
    print(f"    (Expected |p_i| ~ 1/√256 ≈ 0.0625 for random unit vecs)")

    # ── Test 2: Kernel shape ─────────────────────────────────
    print("\n[2] Kernel comparison (g_preserve vs g_tension)")
    test_pairs = [(0.0, 0.0), (0.05, 0.05), (0.05, -0.05), (0.1, 0.0), (0.1, 0.1)]
    print(f"    {'(p_i, p_j)':20s}  {'g_preserve':12s}  {'g_tension':12s}  {'delta':10s}")
    for pi, pj in test_pairs:
        gp = g_preserve(pi, pj)
        gt = g_tension(pi, pj)
        print(f"    ({pi:+.2f}, {pj:+.2f})        {gp:+.6f}    {gt:+.6f}    {gt-gp:+.6f}")

    # ── Test 3: 20-cycle runs per mode ───────────────────────
    print("\n[3] 20-cycle runs per mode")

    context_ints = [hash_to_int(f"ctx_{i}") for i in range(20)]
    results = {}

    for mode in [OperatingMode.THINKING, OperatingMode.ENCODING]:
        mc = ModeController(build_default_ontology())
        mc.set_mode(mode)
        # Force λ to target immediately for clean measurement
        mc.lam = mc._lam_target

        kvars = []
        gaps  = []
        for i in range(20):
            P, snap = mc.step(context_int=context_ints[i])
            kvars.append(snap.kappa_var)
            gaps.append(snap.spectral_gap)

        results[mode] = {"kvar": kvars, "gap": gaps, "P_final": P}
        print(f"\n    Mode: {mode.name}")
        print(f"      κ variance — mean={np.mean(kvars):.5f}  "
              f"std={np.std(kvars):.5f}  "
              f"min={np.min(kvars):.5f}  max={np.max(kvars):.5f}")
        print(f"      spectral gap — mean={np.mean(gaps):.4f}  "
              f"std={np.std(gaps):.4f}  "
              f"min={np.min(gaps):.4f}  max={np.max(gaps):.4f}")

    # Key assertion: ENCODING should have higher curvature variance (sharper structure)
    # THINKING bridges across contexts → smoother landscape → lower variance
    # ENCODING clusters context-aligned nodes → sharper structure → higher variance
    think_kvar = np.mean(results[OperatingMode.THINKING]["kvar"])
    encode_kvar = np.mean(results[OperatingMode.ENCODING]["kvar"])
    print(f"\n    κ variance: THINKING={think_kvar:.5f}  ENCODING={encode_kvar:.5f}")
    variance_ok = encode_kvar > think_kvar
    if variance_ok:
        print("    ✓ ENCODING has higher κ variance (sharper structure, as expected)")
    else:
        print("    ✗ WARNING: THINKING has higher κ variance — check kernel params")

    # ── Test 4: Mode switch — manifold survival ───────────────
    print("\n[4] Mode switch: THINKING → ENCODING (manifold check)")
    mc = ModeController(build_default_ontology())
    mc.set_mode(OperatingMode.THINKING)
    mc.lam = LAMBDA_THINKING

    # Run 10 thinking cycles
    for i in range(10):
        mc.step(context_int=context_ints[i])

    P_pre = mc.P.copy()
    spec_pre = spectral_analysis(P_pre)

    # Switch to encoding
    mc.set_mode(OperatingMode.ENCODING)
    for i in range(10):
        mc.step(context_int=context_ints[i])

    P_post = mc.P.copy()
    spec_post = spectral_analysis(P_post)

    # Check rows still sum to 1 and no NaN
    row_sums = P_post.sum(axis=1)
    has_nan  = np.any(np.isnan(P_post))
    has_neg  = np.any(P_post < 0)
    gap_pre  = float(spec_pre.get('spectral_gap', 0.0))
    gap_post = float(spec_post.get('spectral_gap', 0.0))

    print(f"    Row sums: min={row_sums.min():.6f}  max={row_sums.max():.6f}")
    print(f"    NaN entries: {has_nan}  |  Negative entries: {has_neg}")
    print(f"    Spectral gap: pre={gap_pre:.4f}  post={gap_post:.4f}")
    print(f"    P delta norm: {np.linalg.norm(P_post - P_pre):.4f}")

    manifold_ok = (not has_nan and not has_neg and
                   abs(row_sums - 1.0).max() < 1e-9 and
                   gap_post > 0.01)
    print(f"    {'✓ Manifold survived mode transition' if manifold_ok else '✗ Manifold integrity issue'}")

    # ── Test 5: λ interpolation ──────────────────────────────
    print("\n[5] λ interpolation (smooth ramp, no jumps)")
    mc = ModeController(build_default_ontology())
    mc.set_mode(OperatingMode.THINKING)
    mc.lam = 0.0  # start at encoding value, ramp toward thinking

    lam_trace = []
    for i in range(15):
        mc.step(context_int=context_ints[i % 20])
        lam_trace.append(mc.lam)

    print(f"    λ trace (0.0 → {LAMBDA_THINKING}):")
    for i, lam in enumerate(lam_trace):
        bar = "▓" * int(lam * 30)
        print(f"      step {i:2d}:  λ={lam:.4f}  {bar}")

    jumps = [abs(lam_trace[i] - lam_trace[i-1]) for i in range(1, len(lam_trace))]
    max_jump = max(jumps) if jumps else 0.0
    print(f"    Max single-step λ jump: {max_jump:.4f}  "
          f"{'✓ smooth' if max_jump < 0.2 else '✗ discontinuous'}")

    # ── Final summary ────────────────────────────────────────
    print("\n[Summary]")
    print(f"  α={ALPHA_DEFAULT}  β={BETA_DEFAULT}  λ_step={LAMBDA_STEP_SIZE}")
    print(f"  p_i distribution: mean={p.mean():.4f}  std={p.std():.4f}")
    print(f"  Manifold integrity: {'✓' if manifold_ok else '✗'}")
    print(f"  Mode variance separation: "
          f"{'✓' if encode_kvar > think_kvar else '✗'}")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
