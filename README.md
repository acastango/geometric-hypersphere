# ROE — Relational Ontology Engine

A biologically grounded cognitive memory architecture that gives AI systems selective long-term memory, context-sensitive recall, and continuous identity. Models the hippocampal encoding system as a spectral band compressor operating on a transition manifold.

**Language:** Python 3.10+ | **Dependencies:** numpy, scipy (core), torch (vision/training)

## Architecture

The LLM is the neocortex (frozen substrate). ROE is the hippocampus (spectral band encoder that watches, compresses, and writes structural modifications). The entorhinal cortex translates between them. Memory is not stored — it IS the structural modification of the transition matrix P.

The system breathes between three modes:
- **Thinking** — tension kernel, explores cross-context bridges
- **Encoding** — preserve kernel, stabilizes and crystallizes structure
- **Sleep** — offline integration (SWS consolidation + REM recombination)

## Project Structure

```
├── roe.py                      # Core ontology (16-node, 256-bit vectors)
├── roe_crystal.py              # Transition matrix P operations
├── roe_geometry.py             # Manifold geometry (curvature, geodesics, spectral)
├── roe_spectral.py             # Spectral snapshot monitoring
├── roe_engine.py               # Phase 1 visual geometric core (64D hypersphere)
├── roe_scalar_field.py         # Token as scalar field on sphere
├── roe_token_gradient.py       # Token as gradient/force on sphere
├── roe_continuous.py           # Continuous training engine
├── roe_hippocampus.py          # Merkle-indexed episodic memory
├── roe_hippocampus_continuous.py
├── locus_core.py               # Mathematical substrate (observer, triangulation)
│
├── roe_mode.py                 # Mode-dependent geometry modulation
├── roe_context.py              # Contextual index layer (dentate gyrus)
├── roe_place.py                # Place cells (manifold position awareness)
├── roe_sleep.py                # Wake-sleep cycle (SWS + REM)
├── roe_entorhinal.py           # Entorhinal translation (LLM <-> ROE codec)
├── roe_loop.py                 # Strange loop (complete architecture)
│
├── roe_invariants.py           # Composition operations
├── roe_spectral_fast.py        # Fast spectral estimation
├── roe_regulate.py             # Attractor regulation
│
├── experiments/                # Standalone experiment and analysis scripts
├── tests/                      # Adversarial tests and validation
├── visualization/              # 3D sphere visualizer and HTML exports
├── data/                       # Ontology snapshots, corpora, vocab
├── docs/                       # Architecture spec, agent guide, validation plan
└── archive/                    # Historical/unused files
```

## Key Concepts

- **Transition matrix P**: `P[i,j]` = probability from node i to j. Rows sum to 1. Memory = modifying P.
- **State vectors**: 256-bit integers. Similarity via Hamming distance. Input via `hash_to_int()`.
- **Curvature**: Ollivier-Ricci. Positive = well-connected, negative = bottleneck.
- **Spectral gap**: Health metric for manifold connectivity.

## Documentation

- `docs/ROE_AGENT_GUIDE.md` — Full implementation guide and architecture reference
- `docs/ROE_VALIDATION_EXPERIMENTS.md` — Experimental roadmap and scaling results
- `docs/ROE_Architecture_Specification.docx` — Formal specification
