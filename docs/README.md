# Documentation

This folder contains project documentation for `prob-sde`.

Current primary references are:

- Project overview and quickstart: `README.md`
- Runnable demos: `examples/`
- Benchmark scripts: `benchmarks/`

## Planned docs

- `methods.md`  
  Overview of implemented methods:
  - Gaussian SDE Filter (GSF, Algorithm 2)
  - Mixture Gaussian SDE Filter (MGSF, Algorithm 3)
  - Marginalised Gaussian SDE Filter (Algorithm 4)

- `api.md`  
  Public API reference for top-level imports and solver facade:
  - `SDESpec`
  - `SDESolverConfig`, `TimeGridConfig`
  - `solve_sde`
  - result containers and uncertainty outputs

- `benchmarks.md`  
  Benchmark structure, Benes benchmark suite usage, and roadmap
  (additional benchmark systems, optimization, and reproducibility).

- `development.md`  
  Development notes:
  - environment setup
  - testing and linting
  - contribution workflow and release/versioning process

- Hebrew background document

## Status

Documentation is under active development.  
For now, the most up-to-date usage patterns are in `README.md`,
`examples/`, and `benchmarks/benes_sde/`.