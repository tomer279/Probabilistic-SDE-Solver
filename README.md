# Probabilistic SDE Solver

Probabilistic numerical solvers for stochastic differential equations (SDEs) with pathwise uncertainty estimates. This package implements the methodology of **Le Fay, Särkkä & Corenflos (2025)**: [*Modelling pathwise uncertainty of Stochastic Differential Equations samplers via Probabilistic Numerics*](https://arxiv.org/abs/2401.03338), Bayesian Analysis (2025).

## Summary

The method turns an SDE into a **sequence of random ODEs** by replacing Brownian motion with a **piecewise differentiable approximation** on each time step. Each random ODE is solved with a **Gaussian ODE filter** (EKF0), yielding a pathwise solution with **uncertainty bands** (credible intervals). The paper establishes strong convergence orders (e.g. 1.5 local, 1.0 global for a specific instance) and shows how to **marginalise** over the Brownian approximation to compute exact transition densities.

## Features

- **Pathwise solver**: Piecewise linear (or polynomial) Brownian approximation + Gaussian ODE filter per step.
- **Uncertainty bands**: Posterior mean and covariance from the filter at each step.
- **Marginalised version**: Mean and variance by averaging over pathwise samples (optional exact transition densities via extended state).
- **Benchmarks**: Strong/weak convergence and runtime vs Euler–Maruyama (and Milstein).

## Installation

From the project root:

```bash
pip install -e .
# with dev dependencies (pytest, matplotlib, ruff):
pip install -e ".[dev]"
```

Requirements: **Python ≥ 3.9**, **JAX** (and **jaxlib**), **NumPy**.

## Quick start

```python
import jax
import jax.numpy as jnp
from prob_sde import (
    piecewise_linear_brownian,
    IWP2Prior,
    ode_integrator_factory,
    solve_sde_pathwise,
)

key = jax.random.PRNGKey(0)
drift = lambda x, t: 0.1 * x
diffusion = lambda x, t: 0.2
x0 = jnp.array(1.0)
prior = IWP2Prior(1.0)
ode_int = ode_integrator_factory(prior)

ts, trajectory = solve_sde_pathwise(
    key, drift, diffusion, x0, piecewise_linear_brownian,
    delta=0.01, num_steps=100, ode_integrator=ode_int
)
# With uncertainty bands:
ts, trajectory, (means, covs) = solve_sde_pathwise(
    key, drift, diffusion, x0, piecewise_linear_brownian,
    delta=0.01, num_steps=100, ode_integrator=ode_int, return_uncertainty=True
)
```

## Project layout

- **src/prob_sde/**: Core package  
  - `brownian`: Piecewise differentiable BM approximation (`piecewise_linear_brownian`)  
  - `prior_models`: State-space prior (`IWP2Prior`)  
  - `ode_filter`: Gaussian ODE filter (`ode_filter_step`, `ode_integrator_factory`)  
  - `sde_solver`: Pathwise SDE solver (`solve_sde_pathwise`)  
  - `marginalised`: Marginalised solver (`solve_sde_marginalised`)  
  - `utils`: Helpers (insert, time_grid, split_key)
- **examples/**: Scripts for pathwise and marginalised runs and uncertainty bands.
- **benchmarks/**: Strong/weak convergence and runtime (see `benchmarks/README.md`).
- **tests/**: Pytest suite.

## Tests

```bash
# From project root, with src on PYTHONPATH:
pytest tests/ -v
# Or after pip install -e .:
pytest tests/ -v
```

## Examples

From the project root:

```bash
python examples/scalar_sde_pathwise.py   # Path + uncertainty bands
python examples/scalar_sde_marginalised.py
python examples/uncertainty_bands_demo.py
```

(Requires `matplotlib` for plotting.)

## Reference

- Paper: [arXiv:2401.03338](https://arxiv.org/abs/2401.03338)  
- Companion code (JAX): [ylefay/bayesianSDEsolver](https://github.com/ylefay/bayesianSDEsolver)

## License

MIT (see LICENSE).
