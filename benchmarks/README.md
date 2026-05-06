# Benchmarks

This folder contains benchmark utilities and benchmark scripts.

## Contents

- `benchmark_utils.py`: shared helper functions used by benchmark scripts.
- `benes_sde/`: Benes SDE benchmark suite:
  - `benes_gsf_vs_em.py`
  - `benes_mgsf_gsf_em.py`
  - `benes_marginalised_gsf_em.py`
  - `minimal_profiling.py`

## Running benchmarks

From the project root:

```bash
python benchmarks/benes_sde/benes_gsf_vs_em.py
python benchmarks/benes_sde/benes_mgsf_gsf_em.py
python benchmarks/benes_sde/benes_marginalised_gsf_em.py
python benchmarks/benes_sde/minimal_profiling.py
```

## Notes:
- Some scripts require optional dependencies such as `matplotlib` and `tqdm`.
- Outputs (plots/tables) are produced by the scripts themselves.

## Benchmark roadmap

Planned improvements:

- Add additional benchmark problems beyond Benes SDE (e.g., linear SDEs, FitzHugh-Nagumo-type systems, and other nonlinear examples).
- Improve runtime performance of benchmark scripts (especially Monte Carlo loops and repeated path construction).
- Increase JAX-native vectorization to reduce Python-loop overhead and improve `jit`/`vmap` efficiency.
- Standardize benchmark outputs (tables, plots, summary metrics) for easier method-to-method comparison.
- Add reproducible benchmark presets (fixed seeds and documented configurations).