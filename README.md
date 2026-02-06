# MTHE493-Capstone

## Local setup

Create an isolated virtual environment so that `numpy` and `matplotlib` can be installed with compatible binary wheels instead of mixing the system packages (which triggered the `numpy.core.multiarray failed to import` error when `/bin/python` was used).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the simulation

After the environment is activated, invoke the driver with the interpreter from that environment:

```bash
python TrialSimulation.py
```

Avoid calling `/bin/python` because it bypasses the virtual environment and re-introduces the incompatible system packages that are compiled against NumPy 1.x.

## Running tests

Unit tests validate the precision and synchronization guarantees (e.g., double-precision data and burn-in convergence). Execute them with:

```bash
python -m unittest discover -s tests
```

## Noise robustness experiment

Run the dedicated sweep to reproduce the Gaussian-noise study and regenerate `image/noise_robustness.png`:

```bash
python NoiseRobustnessStudy.py
```

Results and interpretation are summarized in `docs/noise_robustness.md`, and the raw dataset is exported to `data/noise_robustness.csv` for further analysis.

## Event-triggered threshold optimization

An event-triggered symbol-termination simulation and threshold optimizer are included to support:

- threshold hysteresis (`epsilon0`, `epsilon1`)
- confidence window (`tconf`)
- mismatch amplitude (`delta_sigma`)
- filtered threshold metrics (`metric_filter_alpha`, `abs_dx_floor_ratio`)

Run the built-in CEM (Cross-Entropy Method) parameter search:

```bash
python -m event_triggered.threshold_optimization
```

Detailed usage is documented in `docs/event_triggered_threshold_optimization.md`.
