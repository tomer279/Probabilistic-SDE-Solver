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
- Increase JAX-native vectorization to optimize remaining coupled per-seed-work and improve `jit`/`vmap` efficiency.
- Standardize benchmark outputs (tables, plots, summary metrics) for easier method-to-method comparison.
- Add reproducible benchmark presets (fixed seeds and documented configurations).

## Marginalised batched sampling results

The marginalised solver now evaluates independent Algorithm-4 trajectories with
`solve_sde_marginalised_batch` (`jax.vmap` over PRNG keys). This replaces the
previous per-sample Python dispatch pattern in the default marginalised facade
path.

### 1) Marginalised scaling benchmark (`benes_sde/marginalised_scaling_bench.py`)

Configuration: `N=100` marginalised samples on the default Benes grid.

Representative warm-run timings:
- single `solve_sde_marginalised`: ~0.29s
- direct batched mean/var: ~0.43s
- facade `solve_sde(..., method="marginalised")`: ~0.43s
- old explicit Python loop baseline: ~49.8s

Observed speedup (facade vs old loop baseline): about `117x`.

### 2) Ensemble path timing (`benes_sde/time_marginalised_ensemble_paths.py`)

Configuration: `num_sample_paths=500`.

Representative timing breakdown:
- total instrumented loop: ~26.17s
- `prepare_coupled_discretization`: ~14.61s (~55.8%)
- coarse Euler-Maruyama paths: ~9.89s (~37.8%)
- batched marginalised solve: ~1.65s (~6.3%)

Interpretation: after batching, marginalised path construction is a small part
of total ensemble-path runtime; the dominant costs are now coupled
discretization and coarse EM path construction.

### Reproducibility notes

- First execution includes JAX tracing/compilation; compare warm-run timings.
- Runtime depends on hardware/backend; report seed and sample count.
- Suggested for published numbers: fixed seed (for example `--seed 0`) and
  explicit `--num-sample-paths`.