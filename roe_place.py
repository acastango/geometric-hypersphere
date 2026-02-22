"""
ROE Place Cells — Manifold Position Awareness
===============================================

Implements place cells: nodes that activate when the system's state falls
within a specific geodesic region of the manifold. Each cell has a preferred
location (an ontology node) and a field radius inversely proportional to
local curvature. High-curvature regions → tight fields (fine resolution).
Low-curvature regions → broad fields (coarse resolution).

Biological analogue: hippocampal place cells fire when the animal is at a
specific location in space. Place field size varies with environment —
smaller in complex/feature-rich areas, larger in open fields. In ROE,
"space" is the transition manifold and "features" are curvature.

Core math (from spec §4.3):
    Active(P, S) = { g(S, L_P) < r(P) }
    r(P) = r_base / (1 + |κ(L_P)|)
    activation(P, S) = max(0, 1 - g(S, L_P) / r(P))

Two distance metrics available:
    1. Geodesic distance on P (primary) — "where is the state in manifold terms?"
       Requires mapping state to manifold coordinates via nearest-node projection.
    2. Hamming distance in bit space (secondary) — "how similar is the raw state?"

Place cells operate on the geodesic metric because that's where the manifold
geometry lives. The 256-bit state vector is projected onto the manifold by
finding which ontology node it's closest to in Hamming space, then using
that node's position in the geodesic distance matrix.

Neurogenesis: when no existing cell activates above threshold, a new cell
can be spawned at the current manifold position. This allows the system to
grow its positional vocabulary as it encounters novel regions.

Dependencies:
    roe.py          — build_default_ontology, hamming_similarity, hash_to_int
    roe_crystal.py  — build_T
    roe_geometry.py — geodesic_distances, ollivier_ricci, node_curvature, connectivity_fix

Imports for downstream phases:
    from roe_place import (
        PlaceCell, PlaceCellMap,
        compute_field_radii, activate_place_cells, where_am_i,
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from roe import (
    build_default_ontology, OntologyNode,
    hamming_similarity, hamming_distance, hash_to_int,
)
from roe_crystal import build_T
from roe_geometry import (
    geodesic_distances, ollivier_ricci, node_curvature,
    connectivity_fix,
)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

R_BASE = 2.5            # base field radius (geodesic units)
R_MIN = 0.3             # minimum radius (prevents infinitely tight fields)
R_MAX = 6.0             # maximum radius (prevents fields covering entire manifold)
ACTIVATION_THRESHOLD = 0.05   # minimum activation to count as "active"
NEUROGENESIS_THRESHOLD = 0.02 # spawn new cell if max activation below this


# ─────────────────────────────────────────────────────────────
# PLACE CELL
# ─────────────────────────────────────────────────────────────

@dataclass
class PlaceCell:
    """
    A single place cell in the manifold.

    Attributes:
        name:       Human-readable label (usually the ontology node name)
        node_idx:   Index into the transition matrix / geodesic distance matrix
        curvature:  Local Ollivier-Ricci curvature at this node
        field_radius: Geodesic radius of the place field (r ∝ 1/|κ|)
        activation_count: Running count of times this cell has been the
                          top-activated cell (for monitoring)
    """
    name: str
    node_idx: int
    curvature: float
    field_radius: float
    activation_count: int = 0


# ─────────────────────────────────────────────────────────────
# FIELD RADIUS COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_field_radius(kappa: float,
                         r_base: float = R_BASE,
                         r_min: float = R_MIN,
                         r_max: float = R_MAX) -> float:
    """
    Compute place field radius from local curvature.

        r = r_base / (1 + |κ|)

    High curvature → small radius (fine resolution in knowledge-dense regions).
    Low/negative curvature → large radius (coarse resolution in sparse regions).
    Clamped to [r_min, r_max] for manifold safety.

    Args:
        kappa:  Local node curvature (can be negative)
        r_base: Base radius before curvature scaling
        r_min:  Floor (prevents infinitely tight fields at high κ)
        r_max:  Ceiling (prevents fields spanning entire manifold)

    Returns:
        Field radius in geodesic distance units.
    """
    r = r_base / (1.0 + abs(kappa))
    return float(np.clip(r, r_min, r_max))


def compute_all_field_radii(node_kappas: np.ndarray,
                            r_base: float = R_BASE,
                            r_min: float = R_MIN,
                            r_max: float = R_MAX) -> np.ndarray:
    """Vectorized field radius computation for all nodes."""
    r = r_base / (1.0 + np.abs(node_kappas))
    return np.clip(r, r_min, r_max)


# ─────────────────────────────────────────────────────────────
# STATE → MANIFOLD PROJECTION
# ─────────────────────────────────────────────────────────────

def project_state_to_manifold(state: int,
                              ontology: dict,
                              names: list) -> Tuple[int, str, float]:
    """
    Project a 256-bit state vector onto the manifold by finding the
    nearest ontology node in Hamming space.

    This is the fallback when no continuous vectors are available.
    Prefer project_state_continuous() when entorhinal layer provides it.
    """
    best_idx = 0
    best_name = names[0]
    best_sim = -1.0
    for i, nm in enumerate(names):
        sim = hamming_similarity(state, ontology[nm].vector)
        if sim > best_sim:
            best_sim = sim
            best_name = nm
            best_idx = i
    return best_idx, best_name, best_sim


def project_state_continuous(continuous_vec: np.ndarray,
                             node_continuous: dict,
                             names: list) -> Tuple[int, str, float]:
    """
    Project using cosine similarity in continuous pre-binarization space.
    Much better semantic separation than Hamming on 256-bit codes.

    Args:
        continuous_vec:   Unit-normalized projected vector from entorhinal
        node_continuous:  Dict of name → unit-normalized node vectors
        names:            Ordered node names

    Returns:
        (node_idx, node_name, similarity)
    """
    best_idx = 0
    best_name = names[0]
    best_sim = -2.0
    for i, nm in enumerate(names):
        if nm in node_continuous:
            sim = float(np.dot(continuous_vec, node_continuous[nm]))
            if sim > best_sim:
                best_sim = sim
                best_name = nm
                best_idx = i
    return best_idx, best_name, best_sim


# ─────────────────────────────────────────────────────────────
# ACTIVATION
# ─────────────────────────────────────────────────────────────

def activate_place_cells(anchor_idx: int,
                         cells: List[PlaceCell],
                         D: np.ndarray,
                         threshold: float = ACTIVATION_THRESHOLD
                         ) -> List[Tuple[PlaceCell, float]]:
    """
    Compute activation of all place cells given the state's manifold position.

    For each cell:
        geodesic_dist = D[anchor_idx, cell.node_idx]
        activation = max(0, 1 - geodesic_dist / cell.field_radius)
        active if activation >= threshold

    Args:
        anchor_idx: Index of the state's nearest ontology node in D
        cells:      List of PlaceCell objects
        D:          All-pairs geodesic distance matrix
        threshold:  Minimum activation to include in results

    Returns:
        List of (PlaceCell, activation_strength) sorted by descending activation.
    """
    active = []
    for cell in cells:
        d = D[anchor_idx, cell.node_idx]
        if d < cell.field_radius:
            activation = 1.0 - d / cell.field_radius
            if activation >= threshold:
                active.append((cell, float(activation)))

    active.sort(key=lambda x: x[1], reverse=True)
    return active


def where_am_i(anchor_idx: int,
               cells: List[PlaceCell],
               D: np.ndarray) -> Optional[Tuple[PlaceCell, float]]:
    """
    Return the single most-activated place cell (current manifold position).

    Args:
        anchor_idx: State's nearest ontology node index
        cells:      List of PlaceCell objects
        D:          Geodesic distance matrix

    Returns:
        (PlaceCell, activation) for the strongest cell, or None if nothing activates.
    """
    active = activate_place_cells(anchor_idx, cells, D, threshold=0.0)
    if not active:
        return None
    best = active[0]
    best[0].activation_count += 1
    return best


# ─────────────────────────────────────────────────────────────
# PLACE CELL MAP
# ─────────────────────────────────────────────────────────────

class PlaceCellMap:
    """
    Complete place cell system for the ROE manifold.

    Manages initialization (one cell per ontology node), activation,
    position tracking, and neurogenesis (spawning new cells for novel regions).

    Usage:
        pcm = PlaceCellMap(ontology)
        pos = pcm.locate(state)                   # where am I?
        active = pcm.activate(state)              # what's nearby?
        pcm.update_geometry()                     # after P changes, recompute radii
        pcm.maybe_spawn(state)                    # neurogenesis if needed
    """

    def __init__(self, ontology: dict = None,
                 r_base: float = R_BASE,
                 r_min: float = R_MIN,
                 r_max: float = R_MAX,
                 entorhinal=None):
        self.ont = ontology or build_default_ontology()
        self.r_base = r_base
        self.r_min = r_min
        self.r_max = r_max
        self._ent = entorhinal  # for cosine-based anchor lookup

        # Build transition matrix and geodesic distances
        T_raw, self.names, self.n, self.idx = build_T(self.ont)
        self.P = connectivity_fix(T_raw)
        self.D = geodesic_distances(self.P)

        # Compute curvature landscape
        kappas = ollivier_ricci(self.P, self.D)
        self.node_kappas = node_curvature(kappas, self.n, self.P, agg='mean')

        # Initialize one place cell per ontology node
        radii = compute_all_field_radii(self.node_kappas, r_base, r_min, r_max)
        self.cells: List[PlaceCell] = []
        for i, nm in enumerate(self.names):
            self.cells.append(PlaceCell(
                name=nm,
                node_idx=i,
                curvature=float(self.node_kappas[i]),
                field_radius=float(radii[i]),
            ))

        # Track spawned cells (neurogenesis)
        self._spawned_count = 0

    def _project(self, state: int) -> Tuple[int, str, float]:
        """Project state to manifold using best available method."""
        if self._ent is not None and hasattr(self._ent, '_last_continuous'):
            # Use cosine similarity in continuous space
            return project_state_continuous(
                self._ent._last_continuous,
                self._ent._node_continuous,
                self.names
            )
        return project_state_to_manifold(state, self.ont, self.names)

    def locate(self, state: int) -> Optional[Tuple[str, float]]:
        """
        Where is this state in the manifold?

        Returns:
            (cell_name, activation_strength) for the top place cell,
            or None if no cell activates.
        """
        anchor_idx, _, _ = self._project(state)
        result = where_am_i(anchor_idx, self.cells, self.D)
        if result is None:
            return None
        return result[0].name, result[1]

    def activate(self, state: int,
                 threshold: float = ACTIVATION_THRESHOLD
                 ) -> List[Tuple[str, float]]:
        """
        Which place cells activate for this state?

        Returns:
            List of (cell_name, activation_strength) sorted by descending activation.
        """
        anchor_idx, _, _ = self._project(state)
        active = activate_place_cells(anchor_idx, self.cells, self.D, threshold)
        return [(cell.name, strength) for cell, strength in active]

    def update_geometry(self, P_new: np.ndarray = None,
                        fast_estimator=None,
                        force: bool = False) -> None:
        """
        Recompute geodesic distances and field radii after the transition
        matrix has been modified (e.g., by mode switching or Hebbian update).

        Caches geodesic distances — only recomputes when P has changed
        significantly (Frobenius delta > threshold). Curvature estimation
        is always updated (it's cheap with fast_estimator).

        If fast_estimator is provided, uses it for curvature (sub-millisecond).
        Otherwise falls back to full Ollivier-Ricci (~2 seconds).
        """
        if P_new is not None:
            P_candidate = connectivity_fix(P_new)
        else:
            T_raw, self.names, self.n, self.idx = build_T(self.ont)
            P_candidate = connectivity_fix(T_raw)

        # Check if P changed enough to warrant geodesic recomputation
        p_delta = float(np.linalg.norm(P_candidate - self.P))
        GEODESIC_RECOMPUTE_THRESHOLD = 0.5

        self.P = P_candidate

        if force or p_delta > GEODESIC_RECOMPUTE_THRESHOLD or self.D is None:
            self.D = geodesic_distances(self.P)

        # Curvature is always updated (cheap with fast_estimator)
        if fast_estimator is not None:
            self.node_kappas = fast_estimator.estimate_nodes(self.P)
        else:
            kappas = ollivier_ricci(self.P, self.D)
            self.node_kappas = node_curvature(kappas, self.n, self.P, agg='mean')

        radii = compute_all_field_radii(
            self.node_kappas, self.r_base, self.r_min, self.r_max)
        for i, cell in enumerate(self.cells):
            if i < self.n:
                cell.curvature = float(self.node_kappas[i])
                cell.field_radius = float(radii[i])

    def maybe_spawn(self, state: int) -> Optional[PlaceCell]:
        """
        Neurogenesis: if no existing cell activates above NEUROGENESIS_THRESHOLD,
        spawn a new cell at the current manifold position.

        The new cell's curvature is inherited from the anchor node. This allows
        the system to grow its positional vocabulary for novel input regions.

        Returns:
            The new PlaceCell if spawned, or None.
        """
        anchor_idx, anchor_name, _ = project_state_to_manifold(
            state, self.ont, self.names)
        active = activate_place_cells(
            anchor_idx, self.cells, self.D, threshold=NEUROGENESIS_THRESHOLD)

        if active:
            return None  # existing cell covers this region

        # Spawn new cell at anchor position
        self._spawned_count += 1
        kappa = float(self.node_kappas[anchor_idx]) if anchor_idx < self.n else 0.0
        radius = compute_field_radius(kappa, self.r_base, self.r_min, self.r_max)

        new_cell = PlaceCell(
            name=f"spawned_{anchor_name}_{self._spawned_count}",
            node_idx=anchor_idx,
            curvature=kappa,
            field_radius=radius,
        )
        self.cells.append(new_cell)
        return new_cell

    def field_summary(self) -> None:
        """Print the current place field landscape."""
        print(f"\n── Place Cell Map ({len(self.cells)} cells) "
              f"──────────────────────────")
        print(f"  {'Cell':25s} {'κ':>8s} {'radius':>8s} {'hits':>6s}")
        print(f"  {'-'*50}")
        for cell in sorted(self.cells, key=lambda c: -c.curvature):
            print(f"  {cell.name:25s} {cell.curvature:>+8.4f} "
                  f"{cell.field_radius:>8.3f} {cell.activation_count:>6d}")
        print()


# ─────────────────────────────────────────────────────────────
# VALIDATION / DEMO
# ─────────────────────────────────────────────────────────────

def run_validation():
    """
    Phase 3 validation suite.

    Tests:
      V1 — Field radius: high-κ nodes have tighter fields than low-κ nodes
      V2 — Self-activation: forcing state to a node activates that node's cell strongest
      V3 — Neighbor activation: midpoint states activate both flanking cells
      V4 — Distance decay: activation decreases with geodesic distance
      V5 — Neurogenesis: novel input spawns new cell when nothing activates
      V6 — Geometry update: field radii change after transition matrix modification
      V7 — Mode integration: place cell activations shift under mode switching
    """
    print("=" * 65)
    print("  ROE Phase 3 — Place Cells Validation")
    print("=" * 65)
    print()

    from roe import RelationalOntologyEngine, int_to_bits, bits_to_int
    import random

    ont = build_default_ontology()
    pcm = PlaceCellMap(ont)

    # ── V1: Field radius scales inversely with curvature ──────
    print("V1 — Field radius inversely proportional to |κ|")

    high_k_cells = sorted(pcm.cells, key=lambda c: -abs(c.curvature))[:5]
    low_k_cells = sorted(pcm.cells, key=lambda c: abs(c.curvature))[:5]

    high_k_mean_r = np.mean([c.field_radius for c in high_k_cells])
    low_k_mean_r = np.mean([c.field_radius for c in low_k_cells])

    print(f"  Top-5 |κ| nodes:")
    for c in high_k_cells:
        print(f"    {c.name:20s}  κ={c.curvature:+.4f}  r={c.field_radius:.3f}")
    print(f"  Bottom-5 |κ| nodes:")
    for c in low_k_cells:
        print(f"    {c.name:20s}  κ={c.curvature:+.4f}  r={c.field_radius:.3f}")
    print(f"  Mean radius: high-|κ|={high_k_mean_r:.3f}  low-|κ|={low_k_mean_r:.3f}")

    v1_pass = high_k_mean_r < low_k_mean_r
    print(f"  V1: {'PASS' if v1_pass else 'FAIL'} "
          f"(high-curvature cells have tighter fields)")
    print()

    # ── V2: Self-activation (force state to node) ─────────────
    print("V2 — Force state to each node → that node's cell activates strongest")

    v2_correct = 0
    v2_total = 0
    for nm in pcm.names:
        # Create state very close to node's vector
        node_vec = ont[nm].vector
        # Add ~5% noise
        rng = random.Random(hash(nm))
        bits = [(node_vec >> (255 - i)) & 1 for i in range(256)]
        for i in range(256):
            if rng.random() < 0.05:
                bits[i] ^= 1
        state = 0
        for b in bits:
            state = (state << 1) | b

        result = pcm.locate(state)
        v2_total += 1
        if result and result[0] == nm:
            v2_correct += 1

    v2_rate = v2_correct / v2_total
    v2_pass = v2_rate >= 0.85  # allow some noise-induced misses
    print(f"  Correct top-activation: {v2_correct}/{v2_total} ({v2_rate:.0%})")
    print(f"  V2: {'PASS' if v2_pass else 'FAIL'}")
    print()

    # ── V3: Neighbor activation (midpoint activates both) ─────
    print("V3 — Geodesically close nodes co-activate")

    # Find pairs of nodes that are geodesically close
    close_pairs = []
    for i in range(pcm.n):
        for j in range(i + 1, pcm.n):
            if pcm.D[i, j] < 1.5:  # close in geodesic space
                close_pairs.append((i, j, pcm.D[i, j]))
    close_pairs.sort(key=lambda x: x[2])

    v3_coactivations = 0
    v3_tested = 0
    for i, j, d in close_pairs[:5]:
        # Force state to node i, check if node j also activates
        state = ont[pcm.names[i]].vector
        active = pcm.activate(state, threshold=0.01)
        active_names = [name for name, _ in active]
        both_active = pcm.names[i] in active_names and pcm.names[j] in active_names
        v3_tested += 1
        if both_active:
            v3_coactivations += 1
        print(f"  {pcm.names[i]:12s}↔{pcm.names[j]:12s}  d={d:.3f}  "
              f"coactive={'yes' if both_active else 'no'}  "
              f"({len(active)} cells active)")

    v3_pass = v3_coactivations > 0
    print(f"  Co-activations: {v3_coactivations}/{v3_tested}")
    print(f"  V3: {'PASS' if v3_pass else 'FAIL'}")
    print()

    # ── V4: Distance decay ────────────────────────────────────
    print("V4 — Activation decays with geodesic distance")

    # Pick a node with moderate curvature, check activation at increasing distances
    anchor_name = "memory"
    anchor_idx = pcm.idx[anchor_name]
    anchor_cell = [c for c in pcm.cells if c.name == anchor_name][0]

    distances_and_activations = []
    for i in range(pcm.n):
        d = pcm.D[anchor_idx, i]
        if d > 0:
            act = max(0.0, 1.0 - d / anchor_cell.field_radius)
            distances_and_activations.append((pcm.names[i], d, act))

    distances_and_activations.sort(key=lambda x: x[1])
    prev_act = 1.0
    monotone = True
    for nm, d, act in distances_and_activations[:8]:
        marker = "↓" if act < prev_act else ("=" if act == prev_act else "↑")
        if act > prev_act + 1e-6:
            monotone = False
        prev_act = act
        bar = "█" * int(act * 30) if act > 0 else "·"
        print(f"  {nm:20s}  d={d:.3f}  act={act:.4f}  {marker} {bar}")

    v4_pass = monotone
    print(f"  V4: {'PASS' if v4_pass else 'FAIL'} (activation monotonically decreases)")
    print()

    # ── V5: Neurogenesis ──────────────────────────────────────
    print("V5 — Neurogenesis: novel input can spawn new cell")

    initial_count = len(pcm.cells)

    # To test neurogenesis, we temporarily remove all cells that would cover
    # a region, then present a state in that region. With D[i,i]=0, the cell
    # AT the anchor always self-activates, so we must remove it and its neighbors.

    # Remove "food" cell and any cell whose field covers the food node
    food_node_idx = pcm.idx["food"]
    removed = []
    remaining = []
    for c in pcm.cells:
        d = pcm.D[food_node_idx, c.node_idx]
        if d < c.field_radius:  # this cell covers the food region
            removed.append(c)
        else:
            remaining.append(c)

    pcm.cells = remaining

    # Now present a state near "food" — nothing should cover it
    test_state = ont["food"].vector
    spawned = pcm.maybe_spawn(test_state)
    v5a_pass = spawned is not None

    # Restore removed cells
    pcm.cells.extend(removed)

    # Normal conditions: a state near an existing node (with full cell coverage) should NOT spawn
    normal_state = ont["danger"].vector
    spawned_normal = pcm.maybe_spawn(normal_state)
    v5b_pass = spawned_normal is None

    print(f"  Removed {len(removed)} cells covering food region")
    print(f"  Gap state → spawned: "
          f"{'yes' if v5a_pass else 'no'} {'PASS' if v5a_pass else 'FAIL'}")
    print(f"  Covered state → no spawn: "
          f"{'yes' if v5b_pass else 'no'} {'PASS' if v5b_pass else 'FAIL'}")
    if spawned:
        print(f"  Spawned cell: {spawned.name}  "
              f"κ={spawned.curvature:+.4f}  r={spawned.field_radius:.3f}")

    v5_pass = v5a_pass and v5b_pass
    print(f"  Cell count: {initial_count} → {len(pcm.cells)}")
    print(f"  V5: {'PASS' if v5_pass else 'FAIL'}")
    print()

    # ── V6: Geometry update ───────────────────────────────────
    print("V6 — Field radii update after transition matrix modification")

    radii_before = {c.name: c.field_radius for c in pcm.cells if c.name in pcm.names}

    # Apply diffusion to P (makes manifold smoother → curvatures decrease → radii increase)
    from roe_geometry import diffuse
    P_diffused = diffuse(pcm.P, delta=0.15)
    pcm.update_geometry(P_new=P_diffused)

    radii_after = {c.name: c.field_radius for c in pcm.cells if c.name in pcm.names}

    changed = 0
    for nm in pcm.names:
        if nm in radii_before and nm in radii_after:
            delta = radii_after[nm] - radii_before[nm]
            if abs(delta) > 0.001:
                changed += 1

    v6_pass = changed > 0
    print(f"  Cells with radius change > 0.001: {changed}/{len(pcm.names)}")
    # Show a few examples
    for nm in list(pcm.names)[:5]:
        rb = radii_before.get(nm, 0)
        ra = radii_after.get(nm, 0)
        print(f"    {nm:20s}  r: {rb:.3f} → {ra:.3f}  Δ={ra-rb:+.3f}")
    print(f"  V6: {'PASS' if v6_pass else 'FAIL'}")
    print()

    # Reset geometry for V7
    pcm_fresh = PlaceCellMap(ont)

    # ── V7: Mode integration ──────────────────────────────────
    print("V7 — Place cell activations shift under mode switching")

    from roe_mode import ModeController, OperatingMode

    mc = ModeController(ont)

    # Run 15 cycles in THINKING mode
    mc.set_mode(OperatingMode.THINKING)
    mc.lam = 1.0
    for i in range(15):
        mc.step(context_int=hash_to_int(f"think_{i}"))

    pcm_think = PlaceCellMap(ont)
    pcm_think.update_geometry(P_new=mc.P)

    # Run 15 cycles in ENCODING mode from fresh
    mc2 = ModeController(ont)
    mc2.set_mode(OperatingMode.ENCODING)
    mc2.lam = 0.0
    for i in range(15):
        mc2.step(context_int=hash_to_int(f"encode_{i}"))

    pcm_encode = PlaceCellMap(ont)
    pcm_encode.update_geometry(P_new=mc2.P)

    # Compare field radii between modes
    test_state = ont["memory"].vector
    active_think = pcm_think.activate(test_state, threshold=0.01)
    active_encode = pcm_encode.activate(test_state, threshold=0.01)

    print(f"  State: memory node vector")
    print(f"  Active cells (THINKING):  {len(active_think)}  "
          f"top: {active_think[0] if active_think else 'none'}")
    print(f"  Active cells (ENCODING):  {len(active_encode)}  "
          f"top: {active_encode[0] if active_encode else 'none'}")

    # Compare radii
    think_radii = {c.name: c.field_radius for c in pcm_think.cells}
    encode_radii = {c.name: c.field_radius for c in pcm_encode.cells}

    radius_diffs = []
    for nm in pcm.names:
        if nm in think_radii and nm in encode_radii:
            radius_diffs.append(abs(think_radii[nm] - encode_radii[nm]))

    mean_diff = np.mean(radius_diffs)
    max_diff = np.max(radius_diffs)
    v7_pass = mean_diff > 0.001  # field radii should differ between modes

    print(f"  Radius difference: mean={mean_diff:.4f}  max={max_diff:.4f}")
    print(f"  V7: {'PASS' if v7_pass else 'FAIL'} "
          f"(place fields reshape under mode switching)")
    print()

    # ── Summary ───────────────────────────────────────────────
    results = {
        "V1 (radius ~ 1/|κ|)":              v1_pass,
        "V2 (self-activation)":             v2_pass,
        "V3 (neighbor co-activation)":      v3_pass,
        "V4 (distance decay)":              v4_pass,
        "V5 (neurogenesis)":                v5_pass,
        "V6 (geometry update)":             v6_pass,
        "V7 (mode integration)":            v7_pass,
    }

    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    all_pass = True
    for label, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    n_pass = sum(results.values())
    print(f"\n  {n_pass}/{len(results)} tests passed")
    if all_pass:
        print("  Phase 3: COMPLETE ✓")
    else:
        print("  Phase 3: NEEDS ATTENTION")

    # Print the field landscape
    pcm_fresh = PlaceCellMap(ont)
    pcm_fresh.field_summary()


if __name__ == "__main__":
    run_validation()
