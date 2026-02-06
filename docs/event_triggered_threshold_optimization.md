# Event-Triggered Threshold Optimization

This contribution implements the event-triggered symbol termination concept from `Event_Trigger_Revised.pdf`.

## What was added

- `event_triggered/event_triggered_simulation.py`
  - Implements bit-driven mismatch (`delta_sigma`) and state-dependent symbol termination.
  - Uses two thresholds (`epsilon0`, `epsilon1`) and confidence time (`tconf`) as acceptance logic.
  - Includes filtered threshold metric options (`use_metric_filter`, `metric_filter_alpha`, `abs_dx_floor_ratio`) to avoid false low-error decisions when signal components cross.
  - Tracks BER, average symbol duration, trigger failure rate, timeout rate, and max observed error.

- `event_triggered/threshold_optimization.py`
  - Adds CEM optimization and grid search for `(epsilon0, epsilon1, tconf, delta_sigma)`.
  - Produces ranked candidates using a weighted objective on BER, trigger failures, duration, and max error.

- `tests/test_event_trigger_optimization.py`
  - Validates confidence-window behavior and optimizer sorting/output contracts.

## Quick start

```bash
python -m event_triggered.threshold_optimization
```

This runs an example search and prints the best candidate plus top alternatives.

## Custom optimization run

```python
from event_triggered.threshold_optimization import build_random_bits, optimize_thresholds

bits = build_random_bits(64, seed=42)
result = optimize_thresholds(
    epsilon0_values=[1.0, 1.5, 2.0],
    epsilon1_values=[10.0, 12.0, 15.0],
    tconf_values=[0.05, 0.1, 0.2],
    delta_sigma_values=[2.5, 3.5, 5.0],
    message_bits=bits,
    use_metric_filter=True,
    metric_filter_alpha=0.3,
    abs_dx_floor_ratio=0.2,
    noise_std=0.75,
    random_seed=42,
    top_k=5,
)
print(result["best"])
```
