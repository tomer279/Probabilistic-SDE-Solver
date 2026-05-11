# Probabilistic SDE Solver

Probabilistic numerical solvers for stochastic differential equations (SDEs) with pathwise uncertainty estimates. This package implements the methodology of **Le Fay, Särkkä & Corenflos (2025)**: [*Modelling pathwise uncertainty of Stochastic Differential Equations samplers via Probabilistic Numerics*](https://arxiv.org/abs/2401.03338), Bayesian Analysis (2025).

## Status
Alpha / research code. The public API may change.

## Summary

The method turns an SDE into a **sequence of random ODEs** by replacing Brownian motion with a **piecewise differentiable approximation** on each time step. Each random ODE is solved with a **Gaussian ODE filter** (EKF0/1), yielding a pathwise solution with **uncertainty bands** (credible intervals). The paper establishes strong convergence orders (e.g. 1.5 local, 1.0 global for a specific instance) and shows how to **marginalise** over the Brownian approximation to compute exact transition densities.

## Features

- **Pathwise solver**: piecewise differentiable Brownian approximation + Gaussian ODE filter per step.
- **Uncertainty bands**: Posterior mean and covariance from the filter at each step.
- **Marginalised version**: Monte Carlo sampling is evaluated with batched Algorithm-4 trajectories (`solve_sde_marginalised_batch`), and the facade returns pointwise mean/variance.
- **Benchmarks**: strong/weak convergence and runtime comparisons (see `benchmarks/README.md`)

## Methods

This package provides three related probabilistic solvers for SDEs:

- **Gaussian SDE Filter (GSF, Algorithm 2)**: propagates a single Gaussian latent state over the time grid. Produces one pathwise trajectory and can optionally return per-step latent posterior means/covariances (uncertainty).

- **Mixture Gaussian SDE Filter (MGSF, Algorithm 3)**: a pathwise solver that samples Brownian-approximation coefficients per interval, builds a local random ODE, and performs a Gaussian ODE-filter update each step. Produces one trajectory (mean or sampled posterior position) and can optionally return uncertainty.

- **Marginalised Gaussian SDE Filter (Algorithm 4)**: marginalises over Brownian-approximation randomness using an augmented Gaussian state and Monte Carlo aggregation. In this codebase it is currently **scalar-state** oriented.
In the unified facade (`method="marginalised"`), independent trajectories are generated via batched execution and aggregated into `mean_trajectory` / `var_trajectory`.

You can select the backend via `SDESolverConfig.method` in the unified `solve_sde(...)` facade (`"gsf"`, `"mgsf"`, or `"marginalised"`).

## Installation

From the project root:

```bash
py -m pip install -e .
# with dev dependencies:
py -m pip install -e ".[dev]"
```

Requirements: **Python ≥ 3.9**, **JAX** (and **jaxlib**), **NumPy**, **matplotlib**, **tqdm**.

## Quick start

See examples/ or benchmarks/ for runnable scripts.

```python
import jax
import jax.numpy as jnp

from prob_sde import (
    SDESpec,
    piecewise_parabolic_brownian,
    SDESolverConfig,
    TimeGridConfig,
    solve_sde,
)

key = jax.random.PRNGKey(0)

drift = lambda x, t: 0.1 * x
diffusion = lambda x, t: 0.2
x0 = jnp.array(1.0)

sde = SDESpec(drift=drift, diffusion=diffusion, x0=x0, bm_factory=piecewise_parabolic_brownian)

cfg = SDESolverConfig(
    method="mgsf",
    grid=TimeGridConfig(delta=0.01, num_steps=100),
)

result = solve_sde(key=key, sde=sde, cfg=cfg)

ts = result.ts
traj = result.trajectory
```

## Project layout


- `src/prob_sde/`: core package
  - `__init__.py`: public API (top-level exports)
  - `core/`
    - `sde.py`: `SDESpec` (SDE specification container)
    - `prior_models.py`: `IWPPrior`, `IWP2Prior`, `IWP3Prior`
  - `brownian/`
    - `brownian.py`: Brownian approximation factories (`piecewise_linear_brownian`, `piecewise_parabolic_brownian`)
    - `pathwise_rhs.py`: per-interval random ODE RHS construction
  - `filtering/`
    - `ode/`: Gaussian ODE filter routines (e.g. `ode_filter_step`, `ode_integrator_factory`)
    - `sde/`: SDE filtering algorithms (Gaussian / mixture / marginalised implementations)
  - `solvers/`
    - `sde_solver.py`: unified facade `solve_sde` + config/result dataclasses
  - `mixture_utils/`: scan/rollout utilities used by mixture methods
  - `utils/`: small helpers (`insert`, `time_grid`, `split_key`)
- **examples/**: Scripts for pathwise and marginalised runs and uncertainty bands.
- **benchmarks/**: Strong/weak convergence and runtime (see `benchmarks/README.md`).
  - `benchmarks/benes_sde/` : Benes SDE benchmark scripts (EM vs GSF/MGSF/Marginalised)
- **tests/**: Pytest suite.

## Tests

From the project root:

```bash
python -m pytest -q
```

Or on Windows (Python launcher):
```bash
py -m pytest -q
```

Run only test collection (useful for import/discovery checks):
```bash
python -m pytest --collect-only -q
```

## Examples

From the project root:

```bash
python examples/kalman_filter_demo.py
python examples/kalman_car_demo.py
python examples/extended_kalman_pendulum_demo.py
python examples/extended_kalman_filter_turn_model_demo.py
python examples/ode_filter_ekf0_demo.py
python examples/brownian_algorithm1_accuracy.py
```

(Requires `matplotlib` for plotting.)

## Benchmarks

See `benchmarks/benes_sde/` for Benes SDE experiments comparing Euler–Maruyama (EM) against the probabilistic solvers.

For profiling/timing harness entry points (performance baseline), see `benchmarks/README.md`.

```bash
python benchmarks/benes_sde/benes_gsf_vs_em.py
python benchmarks/benes_sde/benes_mgsf_gsf_em.py
python benchmarks/benes_sde/benes_marginalised_gsf_em.py

# Profiling / timing harnesses (performance baseline)
python benchmarks/benes_sde/end_to_end_matrix.py
python benchmarks/benes_sde/section_timing_matrix.py
python benchmarks/benes_sde/marginalised_scaling_bench.py
python benchmarks/benes_sde/profile_marginalised_inner_solve.py
python benchmarks/benes_sde/time_marginalised_ensemble_paths.py

# Minimal marginalised-only timing
python benchmarks/benes_sde/minimal_profiling.py
```

### Notes
- The first run includes JAX compilation; prefer comparing warm-run timings.
- The profiling scripts fix mc.seed for reproducibility by default.
- Default Monte Carlo settings can take many minutes on CPU.

### Performance note (marginalised solver)

The marginalised solver now evaluates Monte Carlo trajectories with batched
Algorithm-4 execution (`solve_sde_marginalised_batch`) instead of Python-level
per-sample dispatch in the default facade path.

In local benchmark runs (`N=100` in `marginalised_scaling_bench.py`), the
marginalised facade runtime dropped from about `49.8s` (old explicit loop
baseline) to about `0.43s` (batched facade), a speedup of roughly `117x`.

In the Benes ensemble timing harness (`time_marginalised_ensemble_paths.py`,
`num_sample_paths=500`), the batched marginalised component is now about `1.65s`
(~`6.3%` of `~26.17s` total), so remaining runtime is dominated by coupled
discretization and coarse EM path construction.

## Roadmap

### Near-term (next 1-2 releases)

- **Performance baseline + profiling** (partially done)
  - Profile end-to-end runtime and per-method hotspots (GSF, MGSF, Marginalised) (done).
  - Continue optimizing coupled EM, GSF, and reference Monte Carlo costs (in progress; 0.1.4 improves Benes GSF-vs-EM MC vectorization and coeff batching; full pipeline still bottlenecked elsewhere per benchmarks).

- **JAX-native execution path** (in progress)
  - Migrate remaining non-JAX, NumPy, or Python-loop code paths in core solvers and coupled paths to JAX-native operations.
  - Broaden jit / vmap compatibility for batched experiments (0.1.4: stronger coverage in Benes benchmarks and stacked Brownian coeffs; library-wide coverage still incomplete).

- **Documentation and examples**
  - Expand method-specific examples for `"gsf"`, `"mgsf"`, and `"marginalised"` via `solve_sde`.
  - Add clear guidance on method selection and expected outputs (`trajectory`, `means/covs`, `mean_trajectory/var_trajectory`).

- **Repository quality gates**
  - Add CI (tests + lint) on pull requests and main branch updates.
  - Add reproducible benchmark/test entry points for easier contribution.

### Mid-term (next 3-6 releases)

- **Benchmark expansion**
  - Add benchmark suites beyond Benes SDE, including:
    - linear SDEs
    - FitzHugh-Nagumo-type systems
    - additional nonlinear testbeds
  - Compare methods across strong/weak error, runtime, and memory usage.

- **Uncertainty calibration**
  - Implement uncertainty calibration for affine SDEs (Section 3.4 of the paper).
  - Add quantitative calibration diagnostics (e.g., empirical coverage vs nominal confidence).

- **Reproducibility improvements**
  - Add fixed-seed benchmark presets and standard output artifacts (figures/tables).
  - Provide optional pinned environments for reproducible runs.

- **Numerical diagnostics**
  - Add diagnostics for covariance behavior, stability checks, and failure-mode tracing.
  - Add regression-style tests for expected convergence trends.

### Long-term (project maturity)

- **User-facing interface**
  - Build a Streamlit app for interactive solver configuration, trajectory visualization, and uncertainty inspection.

- **API stabilization**
  - Stabilize and version the high-level API (`SDESpec`, `SDESolverConfig`, `solve_sde`).
  - Improve backward-compatibility policy and migration notes for users.

- **Scalability and vectorization**
  - Support larger batched experiments and higher-dimensional settings with better vectorized execution.
  - Reduce host/device transfer overhead and improve throughput for research workloads.

- **Documentation and adoption**
  - Publish richer tutorials and “cookbook” style notebooks.
  - Add contributor guides for extending methods and benchmarks.

See CHANGELOG.md for release history.

## Contributing

Contributions are welcome.

If you would like to contribute:

1. Fork the repository and create a feature branch.
2. Make your changes with clear commit messages.
3. Run tests locally before opening a pull request:
   ```bash
   pip install -e ".[dev]"
   pytest -v
4. Open a pull request describing:
 - what changed,
 - why it changed,
 - and how it was tested,

Please keep changes focused, add/update tests when relevant, and update documentation for user-facing changes.

## Reference

- Paper: [arXiv:2401.03338](https://arxiv.org/abs/2401.03338)  
- Companion code (JAX): [ylefay/bayesianSDEsolver](https://github.com/ylefay/bayesianSDEsolver)

## License

MIT (see LICENSE).
