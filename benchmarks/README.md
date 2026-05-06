# Benchmarks

## Convergence

- **convergence_strong_weak.py**: Compares strong error at a fixed time for the pathwise probabilistic solver and Euler–Maruyama. Run with a fine reference path; errors are reported for several step counts. Expect strong convergence order around 0.5–1.0 for the probabilistic solver (paper: global order 1.0).

  ```bash
  cd benchmarks && python convergence_strong_weak.py
  ```

## Runtime

- **runtime_benchmark.py**: Measures wall-clock time for 500 steps (pathwise solver vs Euler–Maruyama) after JIT compilation. Run from the project root or with `PYTHONPATH` including `src`.

  ```bash
  cd benchmarks && python runtime_benchmark.py
  ```

## Interpreting results

- **Strong error**: Single-path difference vs a fine reference. Smaller step size should give smaller error; slope in log-log plot indicates convergence order.
- **Runtime**: Pathwise solver does more work per step (ODE filter + Brownian sampling); use when uncertainty estimates are needed.
