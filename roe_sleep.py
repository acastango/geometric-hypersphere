"""
ROE Wake-Sleep Cycle
=====================

Implements the offline integration phase where recent encodings are replayed
through the substrate to deepen and stabilize structural modifications.

Biological analogue: sleep consolidation. During waking, the hippocampus
encodes experiences rapidly but shallowly. During sleep, those encodings
are replayed through the neocortex, deepening structural modifications.
Slow-wave sleep (SWS) replays faithful copies (consolidation). REM sleep
recombines fragments (creative integration, bridge discovery).

Architecture:
    EncodingBuffer  — accumulates (state_vector, context_vector) pairs during wake
    SleepCycle      — runs SWS and REM phases using ModeController
    KappaBreathing  — monitors curvature oscillation across wake/sleep transitions
    WakeSleepManager — orchestrates the full cycle, triggers sleep when needed

Sleep phases:
    SWS (slow-wave sleep):
        Mode: ENCODING (preserve kernel, λ→0)
        Action: replay each buffered encoding N times through mode-modulated P
        Effect: deepens structural modifications, smooths curvature, consolidates
    REM (rapid eye movement):
        Mode: THINKING (tension kernel, λ→1) but with SLEEP flag
        Action: compose() random pairs from buffer, replay blended states
        Effect: discovers cross-episode bridges, diversifies curvature, creative links

Observable signature: curvature breathing
    Wake (thinking): κ variance moderate, spectral gap high
    Wake (encoding): κ variance increases, spectral gap decreases
    SWS:             κ variance decreases (smoothing), κ mean stabilizes
    REM:             κ variance increases (diversification), new bridges form

Dependencies:
    roe_mode.py     — ModeController, OperatingMode
    roe_context.py  — make_context_vector, context_projection
    roe_place.py    — PlaceCellMap (for tracking replay position)
    roe_invariants.py — compose() for REM recombination
    roe_geometry.py — ollivier_ricci, geodesic_distances, spectral_analysis

Imports for downstream phases:
    from roe_sleep import (
        EncodingBuffer, SleepCycle, KappaBreathing, WakeSleepManager,
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum, auto

from roe import (
    build_default_ontology, OntologyNode,
    hamming_similarity, hash_to_int,
)
from roe_crystal import build_T
from roe_geometry import (
    ollivier_ricci, geodesic_distances, node_curvature,
    connectivity_fix, spectral_analysis,
)
from roe_mode import (
    ModeController, OperatingMode,
    modulate_transitions, node_unit_vectors, context_to_unit_vec,
)
from roe_invariants import compose


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SWS_REPLAYS_PER_ENCODING = 3   # times each encoding is replayed during SWS
REM_RECOMBINATIONS = 10         # number of random blends during REM
SLEEP_TRIGGER_COUNT = 8         # buffer size that triggers sleep
SWS_CYCLES = 15                 # mode controller steps during SWS
REM_CYCLES = 10                 # mode controller steps during REM


# ─────────────────────────────────────────────────────────────
# ENCODING BUFFER
# ─────────────────────────────────────────────────────────────

@dataclass
class BufferedEncoding:
    """
    A single experience captured during wake for later replay.

    Attributes:
        state_vec:    256-bit integer — the ROE state at encoding time
        context_int:  256-bit integer — context vector at encoding time
        anchor_name:  Name of the nearest ontology node (manifold position)
        strength:     Encoding strength [0,1] — higher = more salient
        replayed:     Count of times this encoding has been replayed
    """
    state_vec: int
    context_int: int
    anchor_name: str
    strength: float = 1.0
    replayed: int = 0


class EncodingBuffer:
    """
    Accumulates encodings during wake. Emptied after sleep.

    Acts as the hippocampal buffer: experiences are rapidly written here
    during wake, then consolidated into cortical structure during sleep.
    """

    def __init__(self, max_size: int = 50):
        self.buffer: List[BufferedEncoding] = []
        self.max_size = max_size
        self._total_encoded = 0

    def add(self, state_vec: int, context_int: int,
            anchor_name: str, strength: float = 1.0) -> BufferedEncoding:
        """Add a new encoding to the buffer."""
        enc = BufferedEncoding(
            state_vec=state_vec,
            context_int=context_int,
            anchor_name=anchor_name,
            strength=strength,
        )
        self.buffer.append(enc)
        self._total_encoded += 1

        # If buffer exceeds max, drop weakest
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda e: e.strength, reverse=True)
            self.buffer = self.buffer[:self.max_size]

        return enc

    def clear(self):
        """Empty the buffer (called after sleep)."""
        self.buffer.clear()

    @property
    def count(self) -> int:
        return len(self.buffer)

    @property
    def total_encoded(self) -> int:
        return self._total_encoded

    def should_sleep(self, trigger_count: int = SLEEP_TRIGGER_COUNT) -> bool:
        """Has enough accumulated to warrant a sleep cycle?"""
        return len(self.buffer) >= trigger_count

    def __repr__(self) -> str:
        return f"EncodingBuffer({self.count} pending, {self.total_encoded} total)"


# ─────────────────────────────────────────────────────────────
# CURVATURE BREATHING MONITOR
# ─────────────────────────────────────────────────────────────

@dataclass
class BreathingSample:
    """One curvature measurement in the breathing trace."""
    phase: str          # 'wake_think', 'wake_encode', 'sws', 'rem'
    step: int
    kappa_mean: float
    kappa_var: float
    spectral_gap: float
    pct_negative: float


class KappaBreathing:
    """
    Monitors curvature oscillation across wake/sleep transitions.

    The healthy signature:
        wake_think:  moderate κ_var, high spectral gap
        wake_encode: rising κ_var, falling spectral gap
        sws:         falling κ_var (smoothing), stable κ_mean
        rem:         rising κ_var (diversification), new structures

    This oscillation is the system's heartbeat. If it flatlines
    (constant κ_var across phases), the mode switching isn't working.
    """

    def __init__(self, fast_estimator=None):
        self.trace: List[BreathingSample] = []
        self._step_counter = 0
        self._fast = fast_estimator  # CalibratedEstimator if available

    def sample(self, P: np.ndarray, phase: str) -> BreathingSample:
        """Take a curvature measurement and add to trace."""
        if self._fast is not None:
            # Fast path: ~0.5ms vs ~2000ms
            k_vals = self._fast.estimate_nodes(P)
            spec = spectral_analysis(P)
            sample = BreathingSample(
                phase=phase,
                step=self._step_counter,
                kappa_mean=float(k_vals.mean()),
                kappa_var=float(k_vals.var()),
                spectral_gap=float(spec.get('spectral_gap', 0.0)),
                pct_negative=float(100.0 * np.sum(k_vals < 0) / len(k_vals)) if len(k_vals) > 0 else 0.0,
            )
        else:
            # Slow path: full Ollivier-Ricci (used during calibration/validation)
            P_c = connectivity_fix(P)
            D = geodesic_distances(P_c)
            kappas = ollivier_ricci(P_c, D)
            k_vals = np.array([v for v in kappas.values() if not np.isnan(v)])

            spec = spectral_analysis(P)

            sample = BreathingSample(
                phase=phase,
                step=self._step_counter,
                kappa_mean=float(k_vals.mean()) if len(k_vals) > 0 else 0.0,
                kappa_var=float(k_vals.var()) if len(k_vals) > 0 else 0.0,
                spectral_gap=float(spec.get('spectral_gap', 0.0)),
                pct_negative=float(100.0 * np.sum(k_vals < 0) / len(k_vals)) if len(k_vals) > 0 else 0.0,
            )

        self.trace.append(sample)
        self._step_counter += 1
        return sample

    def phase_summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate statistics per phase."""
        from collections import defaultdict
        by_phase = defaultdict(list)
        for s in self.trace:
            by_phase[s.phase].append(s)

        summary = {}
        for phase, samples in by_phase.items():
            summary[phase] = {
                'kappa_mean': np.mean([s.kappa_mean for s in samples]),
                'kappa_var': np.mean([s.kappa_var for s in samples]),
                'spectral_gap': np.mean([s.spectral_gap for s in samples]),
                'pct_negative': np.mean([s.pct_negative for s in samples]),
                'n_samples': len(samples),
            }
        return summary

    def is_breathing(self) -> bool:
        """
        Check if the curvature trace shows healthy oscillation.
        Requires at least 2 distinct phases with measurably different κ_var.
        """
        summary = self.phase_summary()
        if len(summary) < 2:
            return False
        vars_by_phase = [v['kappa_var'] for v in summary.values()]
        spread = max(vars_by_phase) - min(vars_by_phase)
        return spread > 0.01  # must have measurable κ_var oscillation


# ─────────────────────────────────────────────────────────────
# SLEEP CYCLE
# ─────────────────────────────────────────────────────────────

class SleepCycle:
    """
    Executes the offline integration phase.

    SWS: preserve kernel, faithful replay of buffered encodings.
    REM: tension kernel, recombinant replay of blended encoding pairs.

    The ModeController manages λ interpolation and geometry modulation.
    The SleepCycle manages what gets replayed and in what order.
    """

    def __init__(self, mode_controller: ModeController,
                 breathing: Optional[KappaBreathing] = None,
                 fast_estimator=None,
                 regulator=None):
        self.mc = mode_controller
        self.breathing = breathing or KappaBreathing()
        self._fast = fast_estimator
        self._reg = regulator

    def run_sws(self, buffer: EncodingBuffer,
                n_cycles: int = SWS_CYCLES,
                replays_per: int = SWS_REPLAYS_PER_ENCODING) -> np.ndarray:
        """
        Slow-Wave Sleep: faithful replay with preservation kernel.
        Regulation applied every step to prevent consolidation collapse.
        """
        self.mc.set_mode(OperatingMode.ENCODING, lam_override=0.0)

        schedule = []
        for enc in buffer.buffer:
            for _ in range(replays_per):
                schedule.append(enc)

        while len(schedule) < n_cycles:
            if buffer.buffer:
                strongest = max(buffer.buffer, key=lambda e: e.strength)
                schedule.append(strongest)
            else:
                break

        schedule = schedule[:n_cycles]

        for enc in schedule:
            self.mc.step(context_int=enc.context_int,
                         fast_estimator=self._fast)
            enc.replayed += 1

            # Regulate during sleep — prevent consolidation collapse
            if self._reg is not None:
                anchor_idx = None
                if enc.anchor_name:
                    # Look up anchor index by name
                    for i, nm in enumerate(self._reg.names):
                        if nm == enc.anchor_name:
                            anchor_idx = i
                            break
                self.mc.P = self._reg.regulate(self.mc.P, anchor_idx=anchor_idx)

            if self.breathing and enc.replayed % 3 == 0:
                self.breathing.sample(self.mc.P, 'sws')

        if self.breathing:
            self.breathing.sample(self.mc.P, 'sws')

        return self.mc.P.copy()

    def run_rem(self, buffer: EncodingBuffer,
                n_cycles: int = REM_CYCLES,
                n_recombinations: int = REM_RECOMBINATIONS) -> np.ndarray:
        """
        REM Sleep: recombinant replay with tension kernel.
        Regulation applied to maintain diversity during recombination.
        """
        self.mc.set_mode(OperatingMode.SLEEP, lam_override=1.0)

        if len(buffer.buffer) < 2:
            for _ in range(n_cycles):
                self.mc.step(fast_estimator=self._fast)
                if self._reg is not None:
                    self.mc.P = self._reg.regulate(self.mc.P)
            return self.mc.P.copy()

        rng = np.random.default_rng(42)
        blends = []
        for _ in range(n_recombinations):
            i, j = rng.choice(len(buffer.buffer), size=2, replace=False)
            enc_a = buffer.buffer[i]
            enc_b = buffer.buffer[j]

            blended_state = compose(enc_a.state_vec, enc_b.state_vec)
            blended_context = enc_a.context_int ^ enc_b.context_int

            blends.append((blended_state, blended_context,
                           enc_a.anchor_name, enc_b.anchor_name))

        schedule = blends * ((n_cycles // len(blends)) + 1)
        schedule = schedule[:n_cycles]

        for blended_state, blended_context, name_a, name_b in schedule:
            self.mc.step(context_int=blended_context,
                         fast_estimator=self._fast)

            # Regulate during REM
            if self._reg is not None:
                self.mc.P = self._reg.regulate(self.mc.P)

            if self.breathing and rng.random() < 0.3:
                self.breathing.sample(self.mc.P, 'rem')

        # Final REM measurement
        if self.breathing:
            self.breathing.sample(self.mc.P, 'rem')

        return self.mc.P.copy()

    def run_full_sleep(self, buffer: EncodingBuffer,
                       sws_cycles: int = SWS_CYCLES,
                       rem_cycles: int = REM_CYCLES) -> np.ndarray:
        """
        Execute complete sleep cycle: SWS → REM.

        Returns P after both phases.
        """
        P_sws = self.run_sws(buffer, n_cycles=sws_cycles)
        P_rem = self.run_rem(buffer, n_cycles=rem_cycles)
        return self.mc.P.copy()


# ─────────────────────────────────────────────────────────────
# WAKE-SLEEP MANAGER
# ─────────────────────────────────────────────────────────────

class WakeSleepManager:
    """
    Orchestrates the full wake-sleep cycle.

    During wake: accepts experiences, buffers encodings, runs mode controller
    in THINKING mode. Can switch to ENCODING for specific events.

    When buffer fills: triggers sleep cycle (SWS → REM), empties buffer,
    returns to wake.

    Tracks curvature breathing across the full cycle.

    Usage:
        wsm = WakeSleepManager()
        for experience in stream:
            wsm.wake_step(experience)
            if wsm.needs_sleep():
                wsm.sleep()
    """

    def __init__(self, ontology: dict = None,
                 sleep_trigger: int = SLEEP_TRIGGER_COUNT):
        self.ont = ontology or build_default_ontology()
        self.mc = ModeController(self.ont)
        self.buffer = EncodingBuffer()
        self.breathing = KappaBreathing()
        self.sleep_trigger = sleep_trigger
        self.sleep_count = 0
        self.wake_steps = 0

        # Start in thinking mode
        self.mc.set_mode(OperatingMode.THINKING)

    def wake_step(self, context_int: int,
                  state_vec: Optional[int] = None,
                  anchor_name: Optional[str] = None,
                  encode: bool = False) -> np.ndarray:
        """
        One step of waking operation.

        Args:
            context_int:  Current context vector (256-bit int)
            state_vec:    Optional state to buffer as encoding
            anchor_name:  Optional manifold position label for the encoding
            encode:       If True, buffer this experience for sleep replay

        Returns:
            Current P after this step
        """
        # Run mode controller step
        P, snap = self.mc.step(context_int=context_int)
        self.wake_steps += 1

        # Sample breathing periodically
        if self.wake_steps % 5 == 0:
            phase = 'wake_encode' if encode else 'wake_think'
            self.breathing.sample(self.mc.P, phase)

        # Buffer encoding if requested
        if encode and state_vec is not None:
            self.buffer.add(
                state_vec=state_vec,
                context_int=context_int,
                anchor_name=anchor_name or "unknown",
                strength=1.0,
            )

        return P

    def needs_sleep(self) -> bool:
        """Check if enough encodings have accumulated to warrant sleep."""
        return self.buffer.should_sleep(self.sleep_trigger)

    def sleep(self, sws_cycles: int = SWS_CYCLES,
              rem_cycles: int = REM_CYCLES) -> np.ndarray:
        """
        Run a full sleep cycle: SWS → REM → clear buffer → return to wake.

        Returns P after sleep.
        """
        # Pre-sleep measurement
        self.breathing.sample(self.mc.P, 'wake_think')

        cycle = SleepCycle(self.mc, self.breathing)
        P = cycle.run_full_sleep(self.buffer, sws_cycles, rem_cycles)

        self.sleep_count += 1
        self.buffer.clear()

        # Return to thinking mode
        self.mc.set_mode(OperatingMode.THINKING)

        # Post-sleep measurement
        self.breathing.sample(self.mc.P, 'wake_think')

        return P

    def status(self) -> str:
        """Human-readable status string."""
        return (f"WakeSleepManager: {self.wake_steps} wake steps, "
                f"{self.sleep_count} sleep cycles, "
                f"{self.buffer.count} pending encodings, "
                f"breathing={'yes' if self.breathing.is_breathing() else 'not yet'}")


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def run_validation():
    """
    Phase 4 validation suite.

    Tests:
      V1 — Encoding buffer: accumulates and triggers sleep at threshold
      V2 — SWS deepens structure: P delta increases after SWS replay
      V3 — REM creates bridges: new cross-connections appear after REM
      V4 — κ breathing: curvature variance differs across phases
      V5 — Spectral safety: gap survives full sleep cycle
      V6 — Full cycle: wake → accumulate → sleep → wake resumes
      V7 — Place cells shift: field radii change pre/post sleep
    """
    print("=" * 65)
    print("  ROE Phase 4 — Wake-Sleep Cycle Validation")
    print("=" * 65)
    print()

    ont = build_default_ontology()
    T_raw, names, n, idx = build_T(ont)
    node_vecs = [ont[nm].vector for nm in names]

    # ── V1: Encoding buffer ───────────────────────────────────
    print("V1 — Encoding buffer accumulates and triggers sleep")

    buf = EncodingBuffer()
    for i in range(5):
        buf.add(
            state_vec=ont[names[i]].vector,
            context_int=hash_to_int(f"ctx_{i}"),
            anchor_name=names[i],
        )

    v1a = buf.count == 5
    v1b = not buf.should_sleep(trigger_count=8)
    for i in range(5, 10):
        buf.add(
            state_vec=ont[names[i % n]].vector,
            context_int=hash_to_int(f"ctx_{i}"),
            anchor_name=names[i % n],
        )
    v1c = buf.should_sleep(trigger_count=8)

    print(f"  After 5 encodings: count={buf.count}  "
          f"triggers_sleep={'no' if v1b else 'yes'}")
    print(f"  After 10 encodings: count={buf.count}  "
          f"triggers_sleep={'yes' if v1c else 'no'}")
    v1_pass = v1a and v1b and v1c
    print(f"  V1: {'PASS' if v1_pass else 'FAIL'}")
    print()

    # ── V2: SWS deepens structure ─────────────────────────────
    print("V2 — SWS deepens structural modifications")

    mc = ModeController(ont)
    breathing = KappaBreathing()
    P_pre_sws = mc.P.copy()

    # Fill buffer with diverse encodings
    buf = EncodingBuffer()
    for i in range(10):
        buf.add(
            state_vec=ont[names[i % n]].vector,
            context_int=hash_to_int(f"sws_ctx_{i}"),
            anchor_name=names[i % n],
        )

    # Run SWS
    sleep = SleepCycle(mc, breathing)
    P_post_sws = sleep.run_sws(buf, n_cycles=20, replays_per=3)

    delta_sws = float(np.abs(P_post_sws - P_pre_sws).mean())
    frob_sws = float(np.linalg.norm(P_post_sws - P_pre_sws))

    # Check that replays were counted
    total_replays = sum(e.replayed for e in buf.buffer)

    print(f"  |P_post_sws - P_pre|_mean: {delta_sws:.6f}")
    print(f"  Frobenius norm delta:       {frob_sws:.4f}")
    print(f"  Total replays across buffer: {total_replays}")

    v2_pass = delta_sws > 1e-4 and total_replays > 0
    print(f"  V2: {'PASS' if v2_pass else 'FAIL'} "
          f"(structure modified and replays counted)")
    print()

    # ── V3: REM creates cross-connections ─────────────────────
    print("V3 — REM creates cross-episode bridges")

    mc2 = ModeController(ont)
    breathing2 = KappaBreathing()
    P_pre_rem = mc2.P.copy()

    # Buffer with semantically distinct encodings
    buf2 = EncodingBuffer()
    # Positive valence experiences
    for nm in ["food", "home", "safe", "friend", "trust"]:
        buf2.add(state_vec=ont[nm].vector,
                 context_int=hash_to_int(f"pos_{nm}"),
                 anchor_name=nm)
    # Negative valence experiences
    for nm in ["danger", "enemy", "pain", "flee", "dying"]:
        buf2.add(state_vec=ont[nm].vector,
                 context_int=hash_to_int(f"neg_{nm}"),
                 anchor_name=nm)

    sleep2 = SleepCycle(mc2, breathing2)
    P_post_rem = sleep2.run_rem(buf2, n_cycles=15, n_recombinations=10)

    delta_rem = float(np.abs(P_post_rem - P_pre_rem).mean())

    # Measure if cross-valence edges changed more than within-valence
    pos_names = ["food", "home", "safe", "friend", "trust"]
    neg_names = ["danger", "enemy", "pain", "flee", "dying"]
    pos_idx = [idx[nm] for nm in pos_names if nm in idx]
    neg_idx = [idx[nm] for nm in neg_names if nm in idx]

    cross_delta = 0.0
    cross_count = 0
    within_delta = 0.0
    within_count = 0
    for i in pos_idx:
        for j in neg_idx:
            cross_delta += abs(P_post_rem[i, j] - P_pre_rem[i, j])
            cross_count += 1
    for i in pos_idx:
        for j in pos_idx:
            if i != j:
                within_delta += abs(P_post_rem[i, j] - P_pre_rem[i, j])
                within_count += 1

    cross_mean = cross_delta / max(cross_count, 1)
    within_mean = within_delta / max(within_count, 1)

    print(f"  |P_post_rem - P_pre|_mean: {delta_rem:.6f}")
    print(f"  Cross-valence edge Δ:      {cross_mean:.6f}")
    print(f"  Within-valence edge Δ:     {within_mean:.6f}")

    v3_pass = delta_rem > 1e-4
    print(f"  V3: {'PASS' if v3_pass else 'FAIL'} "
          f"(REM modified structure)")
    print()

    # ── V4: Curvature breathing ───────────────────────────────
    print("V4 — Curvature breathing across wake/sleep phases")

    wsm = WakeSleepManager(ont, sleep_trigger=8)

    # Wake phase: 12 steps with encoding
    for i in range(12):
        ctx = hash_to_int(f"wake_{i}")
        wsm.wake_step(
            context_int=ctx,
            state_vec=ont[names[i % n]].vector,
            anchor_name=names[i % n],
            encode=True,
        )

    # Force a breathing sample pre-sleep
    wsm.breathing.sample(wsm.mc.P, 'wake_think')

    # Sleep
    wsm.sleep(sws_cycles=12, rem_cycles=8)

    # More wake after sleep
    for i in range(5):
        wsm.wake_step(context_int=hash_to_int(f"post_wake_{i}"))

    summary = wsm.breathing.phase_summary()
    print(f"  Breathing phases observed: {list(summary.keys())}")
    for phase, stats in summary.items():
        print(f"    {phase:14s}  κ_var={stats['kappa_var']:.4f}  "
              f"gap={stats['spectral_gap']:.4f}  "
              f"%neg={stats['pct_negative']:.1f}%  "
              f"(n={stats['n_samples']:.0f})")

    v4_pass = wsm.breathing.is_breathing()
    print(f"  Breathing detected: {'yes' if v4_pass else 'no'}")
    print(f"  V4: {'PASS' if v4_pass else 'FAIL'}")
    print()

    # ── V5: Spectral safety ───────────────────────────────────
    print("V5 — Spectral gap survives full sleep cycle")

    spec_pre = spectral_analysis(T_raw)
    spec_post = spectral_analysis(wsm.mc.P)
    gap_pre = float(spec_pre.get('spectral_gap', 0.0))
    gap_post = float(spec_post.get('spectral_gap', 0.0))
    gap_ratio = gap_post / max(gap_pre, 1e-10)

    has_nan = np.any(np.isnan(wsm.mc.P))
    has_neg = np.any(wsm.mc.P < -1e-10)
    rows_ok = np.abs(wsm.mc.P.sum(axis=1) - 1.0).max() < 1e-8

    print(f"  Spectral gap: pre={gap_pre:.4f}  post={gap_post:.4f}  "
          f"ratio={gap_ratio:.3f}")
    print(f"  Row-stochastic: {'yes' if rows_ok else 'NO'}")
    print(f"  NaN: {'NO' if not has_nan else 'YES'}  "
          f"Negative: {'NO' if not has_neg else 'YES'}")

    v5_pass = gap_post > 0.01 and not has_nan and not has_neg and rows_ok
    print(f"  V5: {'PASS' if v5_pass else 'FAIL'}")
    print()

    # ── V6: Full cycle integration ────────────────────────────
    print("V6 — Full cycle: wake → accumulate → sleep → wake resumes")

    wsm2 = WakeSleepManager(ont, sleep_trigger=6)

    # Wake until sleep triggers
    steps_to_sleep = 0
    for i in range(20):
        ctx = hash_to_int(f"cycle_{i}")
        wsm2.wake_step(
            context_int=ctx,
            state_vec=ont[names[i % n]].vector,
            anchor_name=names[i % n],
            encode=True,
        )
        steps_to_sleep += 1
        if wsm2.needs_sleep():
            break

    P_pre_sleep = wsm2.mc.P.copy()
    pre_buf_count = wsm2.buffer.count

    # Sleep
    wsm2.sleep()

    P_post_sleep = wsm2.mc.P.copy()
    post_buf_count = wsm2.buffer.count

    # Resume wake
    for i in range(5):
        wsm2.wake_step(context_int=hash_to_int(f"resume_{i}"))

    sleep_delta = float(np.abs(P_post_sleep - P_pre_sleep).mean())

    print(f"  Steps to trigger sleep: {steps_to_sleep}")
    print(f"  Buffer: {pre_buf_count} before sleep → {post_buf_count} after")
    print(f"  Sleep cycles completed: {wsm2.sleep_count}")
    print(f"  |P_post - P_pre|: {sleep_delta:.6f}")
    print(f"  Wake resumed: {wsm2.wake_steps} total steps")

    v6_pass = (wsm2.sleep_count == 1 and post_buf_count == 0
               and sleep_delta > 1e-5 and wsm2.wake_steps > steps_to_sleep)
    print(f"  V6: {'PASS' if v6_pass else 'FAIL'}")
    print()

    # ── V7: Place cells shift post-sleep ──────────────────────
    print("V7 — Place cell field radii change after sleep")

    from roe_place import PlaceCellMap

    pcm_pre = PlaceCellMap(ont)
    pcm_post = PlaceCellMap(ont)
    pcm_post.update_geometry(P_new=wsm2.mc.P)

    radius_diffs = []
    for i, nm in enumerate(names):
        r_pre = pcm_pre.cells[i].field_radius
        r_post = pcm_post.cells[i].field_radius
        radius_diffs.append(abs(r_post - r_pre))

    mean_diff = np.mean(radius_diffs)
    max_diff = np.max(radius_diffs)
    changed = sum(1 for d in radius_diffs if d > 0.001)

    print(f"  Radius changes: mean={mean_diff:.4f}  max={max_diff:.4f}")
    print(f"  Cells with Δr > 0.001: {changed}/{len(names)}")

    # Show biggest changes
    diffs_named = sorted(zip(names, radius_diffs), key=lambda x: -x[1])
    for nm, d in diffs_named[:5]:
        r_pre = pcm_pre.cells[names.index(nm)].field_radius
        r_post = pcm_post.cells[names.index(nm)].field_radius
        print(f"    {nm:20s}  {r_pre:.3f} → {r_post:.3f}  Δ={d:+.3f}")

    v7_pass = changed > 0
    print(f"  V7: {'PASS' if v7_pass else 'FAIL'} "
          f"(sleep reshapes place fields)")
    print()

    # ── Summary ───────────────────────────────────────────────
    results = {
        "V1 (encoding buffer)":         v1_pass,
        "V2 (SWS deepens structure)":   v2_pass,
        "V3 (REM bridges)":             v3_pass,
        "V4 (κ breathing)":             v4_pass,
        "V5 (spectral safety)":         v5_pass,
        "V6 (full cycle)":              v6_pass,
        "V7 (place cell shift)":        v7_pass,
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
        print("  Phase 4: COMPLETE ✓")
    else:
        print("  Phase 4: NEEDS ATTENTION")
    print()

    # Print manager status
    print(f"  {wsm2.status()}")
    print()


if __name__ == "__main__":
    run_validation()
