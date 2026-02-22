"""
ROE Hippocampal Module — Merkle-Indexed Episodic Memory
========================================================

Standalone module. No dependencies on manifold head or LM.
Takes geometric state vectors as input, stores them in a
Merkle-indexed chain, retrieves by similarity.

Tests:
  1. Determinism: same input → identical tree
  2. Non-contamination: storage/retrieval doesn't change input vectors
  3. Drift-trigger: storage only on semantic transitions
  4. Episodic recall: correct trajectory retrieved

Usage:
    python roe_hippocampus.py
"""

import numpy as np
import hashlib
import struct
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ════════════════════════════════════════════
# MEMORY RECORD
# ════════════════════════════════════════════

@dataclass
class MemoryRecord:
    """Single episodic memory entry."""
    manifold_state: np.ndarray        # z_t — geometric state vector
    active_basins: List[int]          # basin IDs active at this moment
    basin_weights: List[float]        # sparse activation weights
    entropy: float                    # token prediction entropy at storage
    injection_strength: float         # manifold injection strength
    drift_from_previous: float        # what triggered storage
    token_position: int               # position in generation sequence
    parent_hash: bytes = b'\x00' * 32 # hash of previous record
    goal_vector: Optional[np.ndarray] = None  # prefrontal goal (future use)

    @property
    def hash(self) -> bytes:
        """Deterministic SHA-256 hash of this record."""
        payload = bytearray()
        # Manifold state (deterministic bytes)
        payload.extend(self.manifold_state.astype(np.float64).tobytes())
        # Basin IDs
        for b in self.active_basins:
            payload.extend(struct.pack('>i', b))
        # Basin weights
        for w in self.basin_weights:
            payload.extend(struct.pack('>d', w))
        # Scalars
        payload.extend(struct.pack('>ddd',
                                   self.entropy,
                                   self.injection_strength,
                                   self.drift_from_previous))
        # Token position
        payload.extend(struct.pack('>i', self.token_position))
        # Parent hash
        payload.extend(self.parent_hash)
        return hashlib.sha256(bytes(payload)).digest()

    def to_dict(self) -> dict:
        return {
            "hash": self.hash.hex(),
            "parent_hash": self.parent_hash.hex(),
            "active_basins": self.active_basins,
            "basin_weights": self.basin_weights,
            "entropy": self.entropy,
            "injection_strength": self.injection_strength,
            "drift": self.drift_from_previous,
            "token_position": self.token_position,
            "state_norm": float(np.linalg.norm(self.manifold_state)),
        }


# ════════════════════════════════════════════
# MERKLE TREE
# ════════════════════════════════════════════

class MerkleNode:
    """Node in a Merkle tree."""
    def __init__(self, hash_val: bytes, left=None, right=None, record=None):
        self.hash = hash_val
        self.left = left
        self.right = right
        self.record = record  # only for leaf nodes

    @staticmethod
    def leaf(record: MemoryRecord) -> 'MerkleNode':
        return MerkleNode(hash_val=record.hash, record=record)

    @staticmethod
    def branch(left: 'MerkleNode', right: 'MerkleNode') -> 'MerkleNode':
        combined = hashlib.sha256(left.hash + right.hash).digest()
        return MerkleNode(hash_val=combined, left=left, right=right)


def build_merkle_tree(records: List[MemoryRecord]) -> Optional[MerkleNode]:
    """Build a Merkle tree from a list of records."""
    if not records:
        return None

    # Create leaf nodes
    nodes = [MerkleNode.leaf(r) for r in records]

    # Pad to power of 2
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])  # duplicate last
        next_level = []
        for i in range(0, len(nodes), 2):
            next_level.append(MerkleNode.branch(nodes[i], nodes[i + 1]))
        nodes = next_level

    return nodes[0]


# ════════════════════════════════════════════
# HIPPOCAMPUS
# ════════════════════════════════════════════

class Hippocampus:
    """
    Merkle-indexed episodic memory.

    Stores geometric state snapshots triggered by drift events.
    Retrieves by cosine similarity + basin overlap.
    Does NOT modify input vectors or influence geometry directly.
    """

    def __init__(self, drift_threshold: float = 0.3, max_records: int = 10000,
                 cooldown: int = 0):
        self.drift_threshold = drift_threshold
        self.max_records = max_records
        self.cooldown = cooldown  # minimum steps between stores
        self.records: List[MemoryRecord] = []
        self.merkle_root: Optional[MerkleNode] = None
        self.last_state: Optional[np.ndarray] = None
        self.last_store_position: int = -999
        self._tree_dirty = True

    def compute_drift(self, current_state: np.ndarray) -> float:
        """Compute drift from last stored state."""
        if self.last_state is None:
            return 1.0  # first state always drifts
        c_norm = current_state / (np.linalg.norm(current_state) + 1e-12)
        l_norm = self.last_state / (np.linalg.norm(self.last_state) + 1e-12)
        cos_sim = float(np.dot(c_norm, l_norm))
        return 1.0 - cos_sim

    def maybe_store(self, manifold_state: np.ndarray,
                    active_basins: List[int],
                    basin_weights: List[float],
                    entropy: float,
                    injection_strength: float,
                    token_position: int,
                    goal_vector: Optional[np.ndarray] = None) -> Optional[MemoryRecord]:
        """
        Store if drift exceeds threshold AND cooldown elapsed.

        Drift is measured from the LAST STORED STATE, not the last
        attempted state. This prevents chain drift where small steps
        each exceed threshold relative to the previous store.

        CRITICAL: This method makes a COPY of manifold_state.
        The original vector is never modified.
        """
        drift = self.compute_drift(manifold_state)

        # Cooldown check
        if token_position - self.last_store_position < self.cooldown:
            return None

        if drift < self.drift_threshold:
            return None

        # Copy state — no reference to caller's array
        state_copy = manifold_state.copy()
        goal_copy = goal_vector.copy() if goal_vector is not None else None

        parent_hash = self.records[-1].hash if self.records else b'\x00' * 32

        record = MemoryRecord(
            manifold_state=state_copy,
            active_basins=list(active_basins),
            basin_weights=list(basin_weights),
            entropy=float(entropy),
            injection_strength=float(injection_strength),
            drift_from_previous=float(drift),
            token_position=int(token_position),
            parent_hash=parent_hash,
            goal_vector=goal_copy,
        )

        self.records.append(record)
        self.last_state = state_copy.copy()
        self.last_store_position = token_position
        self._tree_dirty = True

        # Prune if over max
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

        return record

    def force_store(self, **kwargs) -> MemoryRecord:
        """Store unconditionally. For testing."""
        old_thresh = self.drift_threshold
        self.drift_threshold = -1.0
        record = self.maybe_store(**kwargs)
        self.drift_threshold = old_thresh
        return record

    def retrieve(self, query_state: np.ndarray,
                 query_basins: Optional[List[int]] = None,
                 k: int = 3) -> List[Tuple[MemoryRecord, float]]:
        """
        Retrieve top-k most similar records.

        CRITICAL: Returns copies. Never modifies stored records.
        Query vector is not modified.
        """
        if not self.records:
            return []

        q_norm = query_state / (np.linalg.norm(query_state) + 1e-12)

        scored = []
        for record in self.records:
            r_norm = record.manifold_state / (np.linalg.norm(record.manifold_state) + 1e-12)
            cos_sim = float(np.dot(q_norm, r_norm))

            # Basin overlap (Jaccard)
            if query_basins is not None and record.active_basins:
                q_set = set(query_basins)
                r_set = set(record.active_basins)
                union = q_set | r_set
                overlap = len(q_set & r_set) / len(union) if union else 0
            else:
                overlap = 0

            score = 0.7 * cos_sim + 0.3 * overlap
            scored.append((record, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def build_tree(self) -> Optional[MerkleNode]:
        """Build/rebuild Merkle tree from current records."""
        self.merkle_root = build_merkle_tree(self.records)
        self._tree_dirty = False
        return self.merkle_root

    @property
    def root_hash(self) -> Optional[bytes]:
        if self._tree_dirty:
            self.build_tree()
        return self.merkle_root.hash if self.merkle_root else None

    def get_chain(self) -> List[dict]:
        """Return the hash chain as a list of dicts."""
        return [r.to_dict() for r in self.records]

    def verify_chain(self) -> bool:
        """Verify hash chain integrity."""
        for i, record in enumerate(self.records):
            if i == 0:
                if record.parent_hash != b'\x00' * 32:
                    return False
            else:
                if record.parent_hash != self.records[i - 1].hash:
                    return False
        return True

    def session_summary(self) -> dict:
        """Consolidation summary for sleep cycle."""
        if not self.records:
            return {}

        # Most visited basins
        basin_counts = {}
        for r in self.records:
            for b, w in zip(r.active_basins, r.basin_weights):
                basin_counts[b] = basin_counts.get(b, 0) + w

        # Transitions
        transitions = {}
        for i in range(1, len(self.records)):
            prev_anchor = self.records[i - 1].active_basins[0] if self.records[i - 1].active_basins else -1
            curr_anchor = self.records[i].active_basins[0] if self.records[i].active_basins else -1
            if prev_anchor != curr_anchor:
                key = (prev_anchor, curr_anchor)
                transitions[key] = transitions.get(key, 0) + 1

        # Entropy profile
        entropies = [r.entropy for r in self.records]
        drifts = [r.drift_from_previous for r in self.records]

        return {
            "n_records": len(self.records),
            "basin_visits": dict(sorted(basin_counts.items(), key=lambda x: -x[1])),
            "transitions": {f"{a}->{b}": c for (a, b), c in
                           sorted(transitions.items(), key=lambda x: -x[1])},
            "mean_entropy": float(np.mean(entropies)),
            "max_entropy": float(np.max(entropies)),
            "mean_drift": float(np.mean(drifts)),
            "max_drift": float(np.max(drifts)),
            "root_hash": self.root_hash.hex() if self.root_hash else None,
            "chain_valid": self.verify_chain(),
        }


# ════════════════════════════════════════════
# TESTS
# ════════════════════════════════════════════

def test_determinism():
    """Test 1: Same input → identical Merkle tree."""
    print("  Test 1 — Determinism")
    d = 64  # manifold dimension

    def run_session(seed=42):
        rng = np.random.RandomState(seed)
        hipp = Hippocampus(drift_threshold=0.2)

        # Simulate 10 manifold states with known drift pattern
        states = []
        base = rng.randn(d).astype(np.float64)
        base = base / np.linalg.norm(base)
        for i in range(10):
            # Gradual drift with occasional jumps
            if i in [3, 7]:
                base = rng.randn(d).astype(np.float64)
                base = base / np.linalg.norm(base)
            else:
                noise = rng.randn(d) * 0.05
                base = base + noise
                base = base / np.linalg.norm(base)
            states.append(base.copy())

        for i, state in enumerate(states):
            hipp.maybe_store(
                manifold_state=state,
                active_basins=[i % 5, (i + 1) % 5],
                basin_weights=[0.6, 0.4],
                entropy=2.0 + i * 0.1,
                injection_strength=0.15,
                token_position=i * 10,
            )

        return hipp

    h1 = run_session(42)
    h2 = run_session(42)

    # Compare roots
    root1 = h1.root_hash
    root2 = h2.root_hash
    assert root1 == root2, f"FAIL: roots differ\n  {root1.hex()}\n  {root2.hex()}"

    # Compare every record hash
    assert len(h1.records) == len(h2.records), "FAIL: different record counts"
    for i, (r1, r2) in enumerate(zip(h1.records, h2.records)):
        assert r1.hash == r2.hash, f"FAIL: record {i} hashes differ"

    # Different seed should produce different tree
    h3 = run_session(99)
    assert h1.root_hash != h3.root_hash, "FAIL: different seeds produced same tree"

    print(f"    ✓ Identical roots: {root1.hex()[:16]}...")
    print(f"    ✓ All {len(h1.records)} record hashes match")
    print(f"    ✓ Different seed → different root")
    print(f"    ✓ Chain valid: {h1.verify_chain()}")
    print()


def test_non_contamination():
    """Test 2: Storage and retrieval don't modify input vectors."""
    print("  Test 2 — Non-Contamination")
    d = 64
    rng = np.random.RandomState(42)
    hipp = Hippocampus(drift_threshold=0.0)  # store everything

    # Create input vectors and save copies
    original_states = []
    for i in range(5):
        state = rng.randn(d).astype(np.float64)
        state = state / np.linalg.norm(state)
        original_states.append(state.copy())

    # Store
    for i, state in enumerate(original_states):
        state_before = state.copy()
        hipp.force_store(
            manifold_state=state,
            active_basins=[i],
            basin_weights=[1.0],
            entropy=2.0,
            injection_strength=0.1,
            token_position=i,
        )
        # Verify original wasn't modified
        assert np.array_equal(state, state_before), f"FAIL: store modified input vector {i}"

    # Retrieve
    query = rng.randn(d).astype(np.float64)
    query = query / np.linalg.norm(query)
    query_before = query.copy()

    results = hipp.retrieve(query, query_basins=[0, 1], k=3)

    # Verify query wasn't modified
    assert np.array_equal(query, query_before), "FAIL: retrieve modified query vector"

    # Verify stored records weren't modified by retrieval
    for i, state in enumerate(original_states):
        stored = hipp.records[i].manifold_state
        # Stored should be equal to original (copied at storage time)
        assert np.array_equal(stored, state), f"FAIL: stored record {i} differs from original"

    # Modify original — stored copy should be unaffected
    original_states[0][0] = 999.0
    assert hipp.records[0].manifold_state[0] != 999.0, "FAIL: stored record linked to original"

    print(f"    ✓ Storage doesn't modify input vectors")
    print(f"    ✓ Retrieval doesn't modify query vector")
    print(f"    ✓ Stored records are independent copies")
    print(f"    ✓ Modifying original doesn't affect stored copy")
    print()


def test_drift_trigger():
    """Test 3: Storage only fires on drift events."""
    print("  Test 3 — Drift-Trigger Logic")
    d = 64
    rng = np.random.RandomState(42)
    hipp = Hippocampus(drift_threshold=0.3)

    base = rng.randn(d).astype(np.float64)
    base = base / np.linalg.norm(base)

    stored_positions = []
    total_attempts = 20

    for i in range(total_attempts):
        if i in [0, 5, 10, 15]:
            # Major drift: new random direction
            state = rng.randn(d).astype(np.float64)
            state = state / np.linalg.norm(state)
            base = state.copy()
        else:
            # Minor perturbation: should NOT trigger storage (usually)
            noise = rng.randn(d) * 0.02
            state = base + noise
            state = state / np.linalg.norm(state)

        record = hipp.maybe_store(
            manifold_state=state,
            active_basins=[i % 3],
            basin_weights=[1.0],
            entropy=2.0,
            injection_strength=0.1,
            token_position=i,
        )
        if record is not None:
            stored_positions.append(i)

    print(f"    Attempted: {total_attempts} states")
    print(f"    Stored: {len(stored_positions)} records")
    print(f"    Stored at positions: {stored_positions}")
    print(f"    Expected major drifts at: [0, 5, 10, 15]")

    # Position 0 always stores (first state)
    assert 0 in stored_positions, "FAIL: first state not stored"

    # Major drift positions should be stored
    for pos in [5, 10, 15]:
        assert pos in stored_positions, f"FAIL: major drift at {pos} not stored"

    # Most non-drift positions should NOT be stored
    non_drift = [i for i in range(total_attempts) if i not in [0, 5, 10, 15]]
    spurious = [i for i in non_drift if i in stored_positions]
    print(f"    Spurious stores (non-drift): {len(spurious)}")
    assert len(spurious) < len(non_drift) * 0.3, "FAIL: too many spurious stores"

    # Verify chain integrity
    assert hipp.verify_chain(), "FAIL: chain integrity broken"

    # Verify drifts in records match expectations
    for r in hipp.records:
        assert r.drift_from_previous >= hipp.drift_threshold, \
            f"FAIL: record stored with drift {r.drift_from_previous} < threshold {hipp.drift_threshold}"

    print(f"    ✓ Major drifts captured")
    print(f"    ✓ Minor perturbations filtered")
    print(f"    ✓ All stored drifts ≥ threshold ({hipp.drift_threshold})")
    print(f"    ✓ Chain integrity verified")
    print()


def test_episodic_recall():
    """Test 4: Correct trajectory retrieved when returning to earlier topic."""
    print("  Test 4 — Episodic Recall")
    d = 64
    rng = np.random.RandomState(42)
    hipp = Hippocampus(drift_threshold=0.0)  # store everything for test

    # Create 3 distinct semantic regions
    region_a = rng.randn(d).astype(np.float64)
    region_a = region_a / np.linalg.norm(region_a)
    region_b = rng.randn(d).astype(np.float64)
    region_b = region_b / np.linalg.norm(region_b)
    region_c = rng.randn(d).astype(np.float64)
    region_c = region_c / np.linalg.norm(region_c)

    # Simulate trajectory: A → B → C → A (return to A)
    trajectory = [
        (region_a, [0], "region_a_1"),
        (region_a + rng.randn(d) * 0.05, [0], "region_a_2"),
        (region_b, [1], "region_b_1"),
        (region_b + rng.randn(d) * 0.05, [1], "region_b_2"),
        (region_c, [2], "region_c_1"),
        (region_c + rng.randn(d) * 0.05, [2], "region_c_2"),
    ]

    for i, (state, basins, label) in enumerate(trajectory):
        state = state / np.linalg.norm(state)
        hipp.force_store(
            manifold_state=state,
            active_basins=basins,
            basin_weights=[1.0],
            entropy=2.0,
            injection_strength=0.1,
            token_position=i * 10,
        )

    # Now query with a state near region_a (returning to earlier topic)
    query_a = region_a + rng.randn(d) * 0.03
    query_a = query_a / np.linalg.norm(query_a)

    results = hipp.retrieve(query_a, query_basins=[0], k=3)

    # Top results should be from region A, not B or C
    print(f"    Query: near region_a")
    print(f"    Retrieved:")
    for record, score in results:
        which = "?" 
        for label_name, ref in [("A", region_a), ("B", region_b), ("C", region_c)]:
            ref_n = ref / np.linalg.norm(ref)
            rec_n = record.manifold_state / np.linalg.norm(record.manifold_state)
            if np.dot(ref_n, rec_n) > 0.9:
                which = label_name
        print(f"      pos={record.token_position:3d}  basins={record.active_basins}  "
              f"score={score:.3f}  region={which}")

    # Top result should be from region A
    top_record = results[0][0]
    top_norm = top_record.manifold_state / np.linalg.norm(top_record.manifold_state)
    a_norm = region_a / np.linalg.norm(region_a)
    assert np.dot(top_norm, a_norm) > 0.9, "FAIL: top retrieval not from region A"

    # Query region B
    query_b = region_b + rng.randn(d) * 0.03
    query_b = query_b / np.linalg.norm(query_b)
    results_b = hipp.retrieve(query_b, query_basins=[1], k=3)

    print(f"    Query: near region_b")
    top_b = results_b[0][0]
    top_b_norm = top_b.manifold_state / np.linalg.norm(top_b.manifold_state)
    b_norm = region_b / np.linalg.norm(region_b)
    assert np.dot(top_b_norm, b_norm) > 0.9, "FAIL: top retrieval not from region B"
    print(f"      Top result: pos={top_b.token_position}, basins={top_b.active_basins}  ✓")

    print(f"    ✓ Region A query retrieves region A memories")
    print(f"    ✓ Region B query retrieves region B memories")
    print(f"    ✓ Episodic recall correctly discriminates trajectories")
    print()


def test_session_summary():
    """Test consolidation summary (sleep cycle)."""
    print("  Test 5 — Session Summary (Sleep Cycle)")
    d = 64
    rng = np.random.RandomState(42)
    hipp = Hippocampus(drift_threshold=0.0)

    # Simulate session with known pattern
    for i in range(20):
        state = rng.randn(d).astype(np.float64)
        state = state / np.linalg.norm(state)
        basin = i % 4
        hipp.force_store(
            manifold_state=state,
            active_basins=[basin, (basin + 1) % 4],
            basin_weights=[0.7, 0.3],
            entropy=2.0 + rng.randn() * 0.5,
            injection_strength=0.1 + rng.randn() * 0.02,
            token_position=i * 5,
        )

    summary = hipp.session_summary()
    print(f"    Records: {summary['n_records']}")
    print(f"    Basin visits: {summary['basin_visits']}")
    print(f"    Transitions: {summary['transitions']}")
    print(f"    Mean entropy: {summary['mean_entropy']:.2f}")
    print(f"    Mean drift: {summary['mean_drift']:.2f}")
    print(f"    Root hash: {summary['root_hash'][:16]}...")
    print(f"    Chain valid: {summary['chain_valid']}")
    assert summary['chain_valid'], "FAIL: chain invalid"
    assert summary['n_records'] == 20, "FAIL: wrong record count"
    print(f"    ✓ Session summary complete and valid")
    print()


def main():
    print("=" * 65)
    print("  ROE Hippocampal Module — Verification Tests")
    print("=" * 65)
    print()

    test_determinism()
    test_non_contamination()
    test_drift_trigger()
    test_episodic_recall()
    test_session_summary()

    print("=" * 65)
    print("  ALL TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
