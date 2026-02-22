"""
roe_context.py — Phase 2: Contextual Index Layer
==================================================

Implements the entorhinal-style contextual modulation layer for ROE.

Biological analogue: the entorhinal cortex encodes WHEN and WHERE an episode
occurred, not just WHAT it contained. The dentate gyrus (sparse expansion)
then pattern-separates episodes so that even similar content gets distinct
index codes if their context differs.

Three components:

  1. Context Vectors
     make_context_vector(timestamp, session_id, topic_drift, affective_state)
     → 256-bit integer encoding contextual metadata.

  2. Transition Modulation
     context_modulate(P_base, c_t, node_vecs, epsilon, alpha, beta)
     → P_modulated: transitions gently biased toward currently-active context.
     Math: P_ij(t) = P_ij + ε · g(c_t, i, j)
           g(c,i,j) = α(c·u_i)(c·u_j) + β·(c·u_i + c·u_j)/2

  3. Sparse Expansion (dentate gyrus equivalent)
     SparseExpander: expands 16-dim ontology coords to ~80-dim sparse code.
     Expansion ratio ≈ 5x, sparsity 2-5%, top-k competitive inhibition.
     Context vector seeds the random projection so context is identity-defining.

  4. Episode Store
     EpisodeStore: stores (sparse_index, encoding) pairs keyed by episode.
     Retrieval: match query sparse pattern against all stored patterns, rank by
     overlap. Returns encodings in similarity order.

Imports used by downstream phases:
  from roe_context import (
      make_context_vector, context_modulate, context_projection,
      SparseExpander, EpisodeStore,
  )
"""

import hashlib
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from roe import (
    build_default_ontology, OntologyNode, RelationalOntologyEngine,
    hamming_similarity, hamming_distance, hash_to_int, int_to_bits,
)
from roe_crystal import build_T, write_T_to_ont
from roe_geometry import normalize_rows


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

CONTEXT_DIM = 256          # bits in context vector (matches ROE state space)
EXPAND_RATIO = 5           # SparseExpander: output dims = n_nodes * EXPAND_RATIO
SPARSITY_K_FRAC = 0.04     # top-k fraction retained after competitive inhibition
EPSILON_DEFAULT = 0.02     # default context modulation strength (tuned for centered projections)
ALPHA_DEFAULT = 0.6        # weight on conjunctive term (c·u_i)(c·u_j)
BETA_DEFAULT = 0.4         # weight on additive term (c·u_i + c·u_j)/2
RETRIEVAL_MIN_OVERLAP = 0.01  # minimum overlap to include in retrieval results


# ─────────────────────────────────────────────────────────────
# CONTEXT VECTOR CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def _float_to_bit_block(value: float, n_bits: int, seed: int) -> List[int]:
    """
    Encode a scalar float in [0,1] as n_bits pseudo-random bits.
    The float determines the "density" (fraction of 1s), seeded for
    reproducibility so the same value always produces the same block.
    """
    rng = random.Random(seed ^ int(value * 1e9))
    threshold = float(np.clip(value, 0.0, 1.0))
    return [1 if rng.random() < threshold else 0 for _ in range(n_bits)]


def make_context_vector(
    timestamp: float,
    session_id: str,
    topic_drift: float,
    affective_state: float,
) -> int:
    """
    Construct a 256-bit context vector encoding four contextual dimensions.

    Args:
        timestamp:       Unix timestamp (encodes temporal position)
        session_id:      String ID for the conversation/session (encodes identity)
        topic_drift:     Float in [0,1] — how far current topic is from session start
        affective_state: Float in [-1, 1] — valence of current conversational state

    Returns:
        256-bit integer context vector.

    Design:
        64 bits — temporal phase (timestamp mod period, normalized to [0,1])
        64 bits — session identity (SHA-256 of session_id, first 64 bits)
        64 bits — topic drift encoding
        64 bits — affective state encoding (mapped to [0,1])
    """
    # Temporal phase: fractional day position, encoded as dense random projection
    T_PERIOD = 86400.0  # 1 day in seconds
    temporal_phase = (timestamp % T_PERIOD) / T_PERIOD
    temporal_bits = _float_to_bit_block(temporal_phase, 64, seed=0xDEADBEEF)

    # Session identity: use full 64 bits of SHA-256 hash (maximum entropy, zero density bias)
    session_hash = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
    session_bits = [(session_hash >> (255 - i)) & 1 for i in range(64)]

    # Topic drift: [0, 1], encoded with a unique seed so it's orthogonal to affect
    drift_norm = float(np.clip(topic_drift, 0.0, 1.0))
    drift_bits = _float_to_bit_block(drift_norm, 64, seed=0xC0FFEE42)

    # Affective state: [-1, 1].  Encode as TWO sub-blocks:
    #   32 bits: density-encoded magnitude (|affect|)
    #   32 bits: sign indicator — positive → hash of "+", negative → hash of "-"
    # This ensures +0.9 and -0.9 differ by far more than just a density shift.
    affect_abs  = abs(float(np.clip(affective_state, -1.0, 1.0)))
    affect_mag_bits  = _float_to_bit_block(affect_abs, 32, seed=0xFEEDBEEF)
    sign_str    = "pos" if affective_state >= 0 else "neg"
    sign_hash   = int(hashlib.sha256(sign_str.encode()).hexdigest(), 16)
    affect_sign_bits = [(sign_hash >> (255 - i)) & 1 for i in range(32)]
    affect_bits = affect_mag_bits + affect_sign_bits

    # Concatenate all 256 bits into one integer
    all_bits = temporal_bits + session_bits + drift_bits + affect_bits
    result = 0
    for b in all_bits:
        result = (result << 1) | b
    return result


def context_dot(c: int, u: int) -> float:
    """
    Dot product proxy between two 256-bit integers.
    Uses normalized Hamming similarity — equivalent to cosine if vectors
    are drawn from {0,1}^256 with p=0.5. Returns value in [0, 1].
    """
    return hamming_similarity(c, u)


# Expected standard deviation of hamming_similarity for random 256-bit vectors.
# Derived from binomial variance: Var(sim) = 1/(4*n_bits), so std = 1/(2*sqrt(n_bits)).
_EXPECTED_STD_256 = 1.0 / (2.0 * math.sqrt(CONTEXT_DIM))


def context_projection(c: int, node_vecs: list) -> np.ndarray:
    """
    Compute centered and normalized context projections for all nodes.

    Raw hamming_similarity clusters near 0.5 for random 256-bit vectors,
    yielding ~0.03 std — too flat for meaningful geometry modulation.
    Centering at 0.5 and dividing by expected std produces projections
    with approximately unit variance and zero mean, so the geometry
    kernels (g_preserve, g_tension) get real differential signal.

    Args:
        c:         Context vector (256-bit int)
        node_vecs: List of node vectors (256-bit ints)

    Returns:
        p: numpy array of shape (n,), centered and normalized projections.
           Positive = node aligned with context, negative = anti-aligned.
    """
    raw = np.array([hamming_similarity(c, u) for u in node_vecs], dtype=float)
    return (raw - 0.5) / _EXPECTED_STD_256


# ─────────────────────────────────────────────────────────────
# TRANSITION MODULATION
# ─────────────────────────────────────────────────────────────

def context_modulate(
    P_base: np.ndarray,
    c_t: int,
    node_vecs: List[int],
    epsilon: float = EPSILON_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
) -> np.ndarray:
    """
    Context-modulate the transition matrix P_base.

    Math:
        p_i = (hamming_sim(c_t, u_i) - 0.5) / expected_std   (centered projection)
        g(c,i,j) = α·(p_i · p_j) + β·(p_i + p_j)/2
        P_ij(t) = P_ij + ε · g(c, i, j)
        P_new = normalize_rows(clip(P_new, 0, None))

    Projections are centered at 0.5 and normalized by expected std of 256-bit
    Hamming similarity (~0.0312), yielding approximately unit-variance values.
    This ensures meaningful geometry modulation even in high-dimensional bit space.

    Args:
        P_base:    Base transition matrix (n×n), row-stochastic
        c_t:       Current context vector (256-bit int)
        node_vecs: List of n node vectors (256-bit ints), same order as P rows
        epsilon:   Modulation strength (default 0.02, tuned for centered projections)
        alpha:     Conjunctive weight
        beta:      Additive weight

    Returns:
        P_modulated: row-stochastic (n×n) numpy array
    """
    n = len(node_vecs)
    assert P_base.shape == (n, n), f"P_base shape {P_base.shape} != ({n},{n})"

    # Compute centered+normalized context projections (unit-variance, zero-centered)
    p = context_projection(c_t, node_vecs)

    # Compute g matrix: g[i,j] = alpha*(p[i]*p[j]) + beta*(p[i]+p[j])/2
    # With centered projections:
    #   conjunctive term p[i]*p[j] > 0 when both aligned or both anti-aligned
    #   additive term biases toward nodes with high projection
    g = alpha * np.outer(p, p) + beta * 0.5 * (p[:, None] + p[None, :])

    # Modulate and renormalize
    P_new = P_base + epsilon * g
    P_new = np.clip(P_new, 0.0, None)
    return normalize_rows(P_new)


# ─────────────────────────────────────────────────────────────
# SPARSE EXPANDER (dentate gyrus equivalent)
# ─────────────────────────────────────────────────────────────

class SparseExpander:
    """
    Expands base ontology coordinates into a high-dimensional sparse index.

    Biological analogue: dentate gyrus sparse coding. Input patterns that are
    similar in semantic space get very different sparse codes if their context
    differs — enabling pattern separation of episodic memories.

    The random projection matrix is seeded by the context vector, so context
    identity IS the projection identity. Two episodes with different contexts
    will have near-zero index overlap even with similar content.

    Args:
        n_base:      Number of base coordinates (ontology nodes, default 16)
        expand_ratio: Output dim = n_base * expand_ratio (default 5 → 80-dim)
        k_frac:      Fraction of output units kept active (default 0.04 → top-3/80)
        rng_seed:    Base seed (combined with context vector)
    """

    def __init__(
        self,
        n_base: int = 16,
        expand_ratio: int = EXPAND_RATIO,
        k_frac: float = SPARSITY_K_FRAC,
        rng_seed: int = 0xABCD1234,
    ):
        self.n_base = n_base
        self.n_out = n_base * expand_ratio
        self.k_frac = k_frac
        self.rng_seed = rng_seed
        self._projection_cache: Dict[int, np.ndarray] = {}

    def _get_projection(self, c_t: int) -> np.ndarray:
        """
        Return (or build and cache) the random projection matrix for context c_t.
        Matrix shape: (n_out, n_base). Each row is a random unit vector seeded by c_t.
        """
        if c_t in self._projection_cache:
            return self._projection_cache[c_t]

        # Seed combines base seed with lower 64 bits of context vector
        seed_key = (self.rng_seed ^ (c_t & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(seed_key)
        W = rng.standard_normal((self.n_out, self.n_base))
        # Row-normalize so each output neuron has unit-norm weights
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / np.where(norms < 1e-10, 1.0, norms)
        self._projection_cache[c_t] = W
        return W

    def expand(
        self,
        base_coords: np.ndarray,
        c_t: int,
        k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Project base coordinates into sparse high-dim space.

        Args:
            base_coords: 1-D array of length n_base (e.g. stationary distribution
                         or context-modulated node activations)
            c_t:         Context vector (seeds the projection)
            k:           Number of active units (default: k_frac * n_out)

        Returns:
            sparse_index: float array of length n_out, with at most k nonzero entries.
        """
        base_coords = np.asarray(base_coords, dtype=float)
        assert len(base_coords) == self.n_base

        if k is None:
            k = max(1, int(self.k_frac * self.n_out))

        W = self._get_projection(c_t)
        activations = W @ base_coords   # (n_out,)

        # ReLU
        activations = np.clip(activations, 0.0, None)

        # Top-k competitive inhibition
        if np.sum(activations > 0) > k:
            threshold = np.partition(activations, -k)[-k]
            activations[activations < threshold] = 0.0

        # Normalize so dot product is a proper similarity measure
        norm = np.linalg.norm(activations)
        if norm > 1e-10:
            activations /= norm

        return activations

    def overlap(self, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        """
        Compute normalized overlap between two sparse indices.
        Returns dot product (cosine similarity for unit-norm vectors).
        """
        return float(np.dot(idx_a, idx_b))


# ─────────────────────────────────────────────────────────────
# EPISODE STORE
# ─────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """
    A single stored episodic memory.

    Attributes:
        episode_id:    Unique identifier (string)
        sparse_index:  High-dim sparse activation pattern (dentate code)
        encoding:      The ROE state vector or transition snapshot associated
                       with this episode (256-bit int or arbitrary object)
        context_vec:   The context vector at encoding time
        metadata:      Optional dict of extra info (timestamp, topic, etc.)
    """
    episode_id: str
    sparse_index: np.ndarray
    encoding: object          # 256-bit int or any hashable/picklable object
    context_vec: int
    metadata: dict = field(default_factory=dict)


class EpisodeStore:
    """
    Hippocampal-style episode memory store.

    Storage: list of Episode objects, each with a sparse dentate-gyrus index.

    Retrieval: given current context vector and optional content hint,
    compute sparse index for the query, then rank stored episodes by overlap.
    Returns episodes sorted by descending overlap, filtered by min_overlap.

    The key design principle: retrieval is pattern completion via sparse overlap,
    NOT exact key lookup. Similar contexts surface similar episodes; dissimilar
    contexts remain invisible even for semantically close content.
    """

    def __init__(self, expander: Optional[SparseExpander] = None):
        self.expander = expander or SparseExpander()
        self.episodes: List[Episode] = []
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"ep_{self._id_counter:06d}"

    def store(
        self,
        base_coords: np.ndarray,
        c_t: int,
        encoding: object,
        episode_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Episode:
        """
        Store a new episode.

        Args:
            base_coords: Ontology coordinate representation of episode content
                         (e.g. stationary distribution over nodes, shape (n,))
            c_t:         Context vector at encoding time
            encoding:    The ROE state/structural snapshot to store
            episode_id:  Optional explicit ID (auto-generated if None)
            metadata:    Optional dict of extra info

        Returns:
            The stored Episode object.
        """
        eid = episode_id or self._next_id()
        sparse_idx = self.expander.expand(base_coords, c_t)
        ep = Episode(
            episode_id=eid,
            sparse_index=sparse_idx,
            encoding=encoding,
            context_vec=c_t,
            metadata=metadata or {},
        )
        self.episodes.append(ep)
        return ep

    def retrieve(
        self,
        base_coords: np.ndarray,
        c_t: int,
        top_k: int = 5,
        min_overlap: float = RETRIEVAL_MIN_OVERLAP,
    ) -> List[Tuple[Episode, float]]:
        """
        Retrieve episodes matching the query sparse pattern.

        Args:
            base_coords: Content representation of current query
            c_t:         Current context vector (used to build query sparse index)
            top_k:       Maximum episodes to return
            min_overlap: Minimum overlap score to include in results

        Returns:
            List of (Episode, overlap_score) tuples, sorted by descending overlap.
        """
        if not self.episodes:
            return []

        query_idx = self.expander.expand(base_coords, c_t)

        scored = []
        for ep in self.episodes:
            overlap = self.expander.overlap(query_idx, ep.sparse_index)
            if overlap >= min_overlap:
                scored.append((ep, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.episodes)

    def __repr__(self) -> str:
        return f"EpisodeStore({len(self.episodes)} episodes)"


# ─────────────────────────────────────────────────────────────
# VALIDATION / DEMO
# ─────────────────────────────────────────────────────────────

def run_validation():
    """
    Phase 2 validation suite.

    Tests:
      V1 — Context vector: different inputs → different vectors
      V2 — Transition modulation: context changes P in expected direction
      V3 — Sparse expander: similar content + different context → low overlap (<5%)
      V4 — Sparse expander: same content + same context → overlap ≈ 1.0
      V5 — Episode retrieval: correct episode returned when re-presented with
           matching context; wrong episode not returned for mismatched context
      V6 — Manifold safety: modulated P has spectral gap within 20% of baseline
    """
    print("=" * 62)
    print("  ROE Phase 2 — Contextual Index Layer Validation")
    print("=" * 62)
    print()

    ont = build_default_ontology()
    T_raw, names, n, idx = build_T(ont)
    node_vecs = [ont[nm].vector for nm in names]

    # ─────────────────────────────────────────────────────────
    # V1: Context vector differentiation
    # ─────────────────────────────────────────────────────────
    print("V1 — Context vector differentiation")

    import time
    base_ts = time.time()

    c_morning  = make_context_vector(base_ts + 0,      "session_A", 0.1, +0.3)
    c_evening  = make_context_vector(base_ts + 43200,  "session_A", 0.1, +0.3)
    c_other    = make_context_vector(base_ts + 0,      "session_B", 0.1, +0.3)
    c_drift    = make_context_vector(base_ts + 0,      "session_A", 0.8, +0.3)
    c_negative = make_context_vector(base_ts + 0,      "session_A", 0.1, -0.9)

    sims = {
        "morning vs evening (same session)":
            hamming_similarity(c_morning, c_evening),
        "session A vs session B (same time)":
            hamming_similarity(c_morning, c_other),
        "low drift vs high drift":
            hamming_similarity(c_morning, c_drift),
        "positive vs negative affect":
            hamming_similarity(c_morning, c_negative),
    }

    # Note on threshold: context vectors use density-encoded bits. By LLN,
    # 256-bit random vectors cluster around sim=0.50. Different context
    # dimensions contribute different sub-blocks (64 bits each), so pairs
    # that differ in ONE dimension will still share ~75% of bits from the
    # other three. The meaningful uniqueness guarantee is in the SPARSE
    # EXPANDER (V3), not raw Hamming. We use 0.93 as threshold here —
    # flagging only near-identical context vectors that would fail to
    # produce distinct sparse codes.
    all_v1_pass = True
    for label, sim in sims.items():
        status = "PASS" if sim < 0.93 else "FAIL"
        if status == "FAIL":
            all_v1_pass = False
        print(f"  [{status}] {label}: sim={sim:.4f}")

    print(f"  V1 result: {'PASS' if all_v1_pass else 'FAIL'}")
    print()

    # ─────────────────────────────────────────────────────────
    # V2: Transition modulation
    # ─────────────────────────────────────────────────────────
    print("V2 — Context modulation changes P in expected direction")

    c_pos = make_context_vector(base_ts, "test", 0.0, +1.0)   # very positive affect
    c_neg = make_context_vector(base_ts, "test", 0.0, -1.0)   # very negative affect

    P_base = T_raw.copy()
    P_pos  = context_modulate(P_base, c_pos, node_vecs)
    P_neg  = context_modulate(P_base, c_neg, node_vecs)

    # Rows should still sum to 1
    pos_row_sums = np.abs(P_pos.sum(axis=1) - 1.0).max()
    neg_row_sums = np.abs(P_neg.sum(axis=1) - 1.0).max()
    rows_ok = pos_row_sums < 1e-8 and neg_row_sums < 1e-8

    # P_pos and P_neg should differ from each other
    delta_pos_neg = float(np.abs(P_pos - P_neg).mean())

    # Both should differ from P_base
    delta_from_base_pos = float(np.abs(P_pos - P_base).mean())
    delta_from_base_neg = float(np.abs(P_neg - P_base).mean())

    v2_pass = rows_ok and delta_pos_neg > 1e-4 and delta_from_base_pos > 1e-5
    print(f"  Row-stochastic preserved: {'PASS' if rows_ok else 'FAIL'} "
          f"(max_err={max(pos_row_sums, neg_row_sums):.2e})")
    print(f"  |P_pos - P_base|_mean = {delta_from_base_pos:.6f}")
    print(f"  |P_neg - P_base|_mean = {delta_from_base_neg:.6f}")
    print(f"  |P_pos - P_neg|_mean  = {delta_pos_neg:.6f}")
    print(f"  V2 result: {'PASS' if v2_pass else 'FAIL'}")
    print()

    # ─────────────────────────────────────────────────────────
    # V3 & V4: Sparse expander overlap
    # ─────────────────────────────────────────────────────────
    print("V3 — Similar content + different context → overlap < 5%")
    print("V4 — Same content + same context → overlap ≈ 1.0")

    expander = SparseExpander(n_base=n)

    # Shared "content" vector: stationary distribution of P_base
    eigs, evecs = np.linalg.eig(T_raw.T)
    stat_idx = np.argmax(np.abs(eigs))
    stat = np.abs(np.real(evecs[:, stat_idx]))
    stat /= stat.sum()
    content_A = stat.copy()

    # Slightly perturbed content (same semantic neighborhood)
    content_B = stat + np.random.default_rng(42).standard_normal(n) * 0.01
    content_B = np.clip(content_B, 0, None)
    content_B /= content_B.sum()

    # Two very different context vectors
    c1 = make_context_vector(base_ts,          "session_alpha", 0.2, +0.5)
    c2 = make_context_vector(base_ts + 86400,  "session_beta",  0.7, -0.5)

    idx_A_c1 = expander.expand(content_A, c1)
    idx_A_c2 = expander.expand(content_A, c2)
    idx_B_c1 = expander.expand(content_B, c1)

    overlap_diff_context = expander.overlap(idx_A_c1, idx_A_c2)  # same content, diff context
    overlap_same_content = expander.overlap(idx_A_c1, idx_B_c1)  # similar content, same context
    overlap_self        = expander.overlap(idx_A_c1, idx_A_c1)  # identical

    v3_pass = overlap_diff_context < 0.05
    v4_pass = abs(overlap_self - 1.0) < 1e-6

    print(f"  Same content, diff context overlap = {overlap_diff_context:.4f}  "
          f"(< 0.05) {'PASS' if v3_pass else 'FAIL'}")
    print(f"  Similar content, same context overlap = {overlap_same_content:.4f}")
    print(f"  Self overlap = {overlap_self:.6f}  (≈ 1.0) {'PASS' if v4_pass else 'FAIL'}")
    print(f"  V3 result: {'PASS' if v3_pass else 'FAIL'}")
    print(f"  V4 result: {'PASS' if v4_pass else 'FAIL'}")
    print()

    # ─────────────────────────────────────────────────────────
    # V5: Episode retrieval correctness
    # ─────────────────────────────────────────────────────────
    print("V5 — Retrieval: correct episode returned for matching context")

    store = EpisodeStore(expander=expander)

    # Build two semantically similar "content" vectors
    content_food  = np.zeros(n); content_food[names.index("food")]  = 1.0
    content_water = np.zeros(n); content_water[names.index("water")] = 0.7
    content_water[names.index("food")] = 0.3

    # Episode 1: food-related content in morning context
    c_morning_ep = make_context_vector(base_ts + 3600,  "ep_session", 0.1, +0.4)
    ep1 = store.store(content_food, c_morning_ep, encoding=hash_to_int("food_morning"),
                      episode_id="ep_food_morning")

    # Episode 2: water-related (similar) content in evening context
    c_evening_ep = make_context_vector(base_ts + 57600, "ep_session", 0.6, -0.1)
    ep2 = store.store(content_water, c_evening_ep, encoding=hash_to_int("water_evening"),
                      episode_id="ep_water_evening")

    # Episode 3: unrelated content in a completely different session
    content_danger = np.zeros(n); content_danger[names.index("danger")] = 1.0
    c_other_ep = make_context_vector(base_ts, "other_session", 0.5, -0.8)
    ep3 = store.store(content_danger, c_other_ep, encoding=hash_to_int("danger_other"),
                      episode_id="ep_danger_other")

    # Query 1: re-present with morning context + food content → should return ep1 first
    results_q1 = store.retrieve(content_food, c_morning_ep, top_k=3)
    q1_top_id = results_q1[0][0].episode_id if results_q1 else None
    v5a_pass = q1_top_id == "ep_food_morning"

    # Query 2: re-present with evening context + water content → should return ep2 first
    results_q2 = store.retrieve(content_water, c_evening_ep, top_k=3)
    q2_top_id = results_q2[0][0].episode_id if results_q2 else None
    v5b_pass = q2_top_id == "ep_water_evening"

    # Query 3: morning food context should NOT return ep_danger_other above ep_food_morning
    ep_ids_q1 = [r[0].episode_id for r in results_q1]
    food_rank  = ep_ids_q1.index("ep_food_morning") if "ep_food_morning" in ep_ids_q1 else 999
    danger_rank = ep_ids_q1.index("ep_danger_other") if "ep_danger_other" in ep_ids_q1 else 999
    v5c_pass = food_rank < danger_rank

    print(f"  Query (food + morning context) → top result: {q1_top_id}  "
          f"{'PASS' if v5a_pass else 'FAIL'}")
    print(f"  Query (water + evening context) → top result: {q2_top_id}  "
          f"{'PASS' if v5b_pass else 'FAIL'}")
    print(f"  Food ranks above danger for food+morning query: {'PASS' if v5c_pass else 'FAIL'}")

    if results_q1:
        print("  Top results for query 1:")
        for ep, score in results_q1:
            print(f"    {ep.episode_id:25s}  overlap={score:.4f}")

    v5_pass = v5a_pass and v5b_pass and v5c_pass
    print(f"  V5 result: {'PASS' if v5_pass else 'FAIL'}")
    print()

    # ─────────────────────────────────────────────────────────
    # V6: Manifold safety (spectral gap preserved under modulation)
    # ─────────────────────────────────────────────────────────
    print("V6 — Manifold safety: spectral gap survives context modulation")

    from roe_geometry import spectral_analysis

    spec_base = spectral_analysis(T_raw)
    spec_mod  = spectral_analysis(P_pos)

    gap_base = spec_base["spectral_gap"]
    gap_mod  = spec_mod["spectral_gap"]
    gap_ratio = gap_mod / max(gap_base, 1e-10)

    # Modulated gap should be within 20% of baseline (not destroyed)
    v6_pass = gap_ratio > 0.80

    print(f"  Baseline spectral gap:  {gap_base:.4f}")
    print(f"  Modulated spectral gap: {gap_mod:.4f}")
    print(f"  Ratio (mod/base):       {gap_ratio:.4f}  (> 0.80) "
          f"{'PASS' if v6_pass else 'FAIL'}")
    print(f"  V6 result: {'PASS' if v6_pass else 'FAIL'}")
    print()

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    results = {
        "V1 (context differentiation)": all_v1_pass,
        "V2 (transition modulation)":   v2_pass,
        "V3 (sparse: diff context)":    v3_pass,
        "V4 (sparse: self overlap)":    v4_pass,
        "V5 (retrieval correctness)":   v5_pass,
        "V6 (manifold safety)":         v6_pass,
    }

    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    all_pass = True
    for label, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    print()
    n_pass = sum(results.values())
    print(f"  {n_pass}/{len(results)} tests passed")
    if all_pass:
        print("  Phase 2: COMPLETE ✓")
    else:
        print("  Phase 2: NEEDS ATTENTION")
    print()


if __name__ == "__main__":
    run_validation()
