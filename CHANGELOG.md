# Changelog

All notable changes to this project will be documented in this file.

## [0.1.3] - 2026-05-08

### Added
- Added weak-only terminal-time comparsion outputs in Benes marginalised benchamrk (`weak_em_g`, `weak_gsf_g`, `weak_marg_g`)

### Changed
- Updated benchmark presets to support faster local iteration (smoke) while preserving full-report defaults (publish), in `benchmarks\benes_sde\benes_marginalised_gsf_em.py`
- Updated weak-error reporting to align with article-style terminal weak metric at time T, in `benchmarks\benes_sde\benes_marginalised_gsf_em.py`.
- Updated benchmark and solver docstrings/comments to reflect terminal weak-g metrics.

### Fixed
- 

### Notes
- Profiling baseline established.
- Remaining hotspot optimization deferred to 0.2.0 (JAX-native/vectorized coupled path work).

## [0.1.2] - 2026-05-07

### Added
- `solve_sde_marginalised_batch` in `prob_sde\src\prob_sde\filtering\sde\marginalised.py`.

### Changed
- `solve_marginalised` in `prob_sde\src\prob_sde\solvers\sde_solver.py` now uses batched marginalised trajectories.
- Benes benchmark path/MC marginalised sections now use batched path generation.
- benchmark docs/comments updated to reflect batch behavior

### Fixed
- 


## [0.1.1] - 2026-05-07

### Added
- Add benchmark profiling harness scripts for the Benes SDE suite (end-to-end timing matrix, section timing matrix, marginalised scaling bench, and marginalised inner-solve/ensemble timing scripts).

### Changed
- 

### Fixed
- 

## [0.1.0] - 2026-05-06

### Added
- Initial public release.
- Core probabilistic SDE solvers (GSF, MGSF, Marginalised).
- Benchmarks and examples.