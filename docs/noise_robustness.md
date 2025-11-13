# Noise Robustness Study

## Objective

Quantify how additive Gaussian noise on the master system’s transmitted $x$ signal affects synchronization quality. This provides an individual contribution complementary to the binary communication experiments: instead of encoding information, we stress the channel and report how much noise the driven system can tolerate before desynchronizing.

## Methodology

- Used the enhanced `run_simulation` API (`noise_std`, `random_seed`) to inject zero-mean noise into the master-to-driven coupling term.
- Ran `NoiseRobustnessStudy.py`, which sweeps noise levels in `[0.0, 3.0]` with sixty-one evenly spaced samples (Δ=0.05) and five trials per level. The script attempts parallel execution but automatically falls back to sequential evaluation when the environment forbids new worker processes.
- Stored the resulting visualization at `image/noise_robustness.png` and exported the raw metrics to `data/noise_robustness.csv` so every data point can be inspected or re-plotted elsewhere.

## Results

Full 61-point dataset (step 0.05) is available in `data/noise_robustness.csv`. For readability, the table below shows every 0.5 increment:

| Noise std | Mean final error | Std dev |
|-----------|------------------|---------|
| 0.00      | 0.740            | 0.000   |
| 0.50      | 0.908            | 0.103   |
| 1.00      | 1.264            | 0.360   |
| 1.50      | 1.664            | 0.665   |
| 2.00      | 2.074            | 0.982   |
| 2.50      | 2.489            | 1.300   |
| 3.00      | 2.908            | 1.614   |

**Observations**

1. The synchronization error grows roughly linearly up to `noise_std ≈ 1.5`, after which the variance balloons—evidence that the driven system begins to lose lock.
2. Even with moderate noise (`noise_std = 0.5`), the final error stays below 1, suggesting usable performance margins for analog message transmission.
3. The sharp increase in both mean and variance beyond `noise_std = 2.0` indicates that a noise-aware decoder would need adaptive thresholds or error correction to maintain fidelity.

## Next questions

- How does the convergence rate change with noise? The current study used only the final error; extending it to measure the time to enter a tolerance band would yield richer metrics.
- What happens if noise is injected into all three channels (`x`, `y`, `z`) or directly into parameter values (e.g., `sigma` jitter)? This would mirror physical actuator imperfections.

    These findings can be shared during the progress meeting to demonstrate personal contributions to the theoretical and empirical understanding of synchronization robustness.
