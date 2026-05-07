# Benchmarks

This folder contains benchmark utilities and benchmark scripts.

## Contents

- `benes_sde/`: Benes SDE benchmark suite:
  - `benes_gsf_vs_em.py`
  - `benes_mgsf_gsf_em.py`
  - `benes_marginalised_gsf_em.py`

  Profiling / timing harnesses (performance baseline):
  - `end_to_end_matrix.py`
  - `section_timing_matrix.py`
  - `marginalised_scaling_bench.py`
  - `profile_marginalised_inner_solve.py`
  - `time_marginalised_ensemble_paths.py`

  Minimal timing:
  - `minimal_profiling.py`

## Running benchmarks

From the project root:

```bash
python benchmarks/benes_sde/benes_gsf_vs_em.py
python benchmarks/benes_sde/benes_mgsf_gsf_em.py
python benchmarks/benes_sde/benes_marginalised_gsf_em.py
```

## Profiling / timing harnesses

These scripts measure end-to-end and section-level runtime, plus isolated solver costs. They are intended to support the roadmap item “Performance baseline + profiling”.

From the project root:

```bash
python benchmarks/benes_sde/end_to_end_matrix.py
python benchmarks/benes_sde/section_timing_matrix.py
python benchmarks/benes_sde/marginalised_scaling_bench.py
python benchmarks/benes_sde/profile_marginalised_inner_solve.py
python benchmarks/benes_sde/time_marginalised_ensemble_paths.py
python benchmarks/benes_sde/minimal_profiling.py
```

### Notes:
- The first run includes JAX compilation; prefer comparing warm-run timings.
- Default Monte Carlo settings can take many minutes on CPU.


## Notes:
- Some scripts require optional dependencies such as `matplotlib` and `tqdm`  (install via your configured extras if applicable).
- Outputs (plots/tables) are produced by the scripts themselves.

## Benchmark roadmap

Planned improvements:

- Add additional benchmark problems beyond Benes SDE (e.g., linear SDEs, FitzHugh-Nagumo-type systems, and other nonlinear examples).
- Improve runtime performance of benchmark scripts (especially Monte Carlo loops and repeated path construction).
- Increase JAX-native vectorization to reduce Python-loop overhead and improve `jit`/`vmap` efficiency.
- Standardize benchmark outputs (tables, plots, summary metrics) for easier method-to-method comparison.
- Add reproducible benchmark presets (fixed seeds and documented configurations).