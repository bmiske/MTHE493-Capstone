from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np

from event_triggered.event_triggered_simulation import (
    EventTriggerParams,
    run_event_triggered_message,
)


@dataclass(frozen=True)
class OptimizationWeights:
    ber: float = 100.0
    tx_fail: float = 20.0
    avg_duration: float = 1.0
    max_error: float = 0.02


@dataclass(frozen=True)
class CEMConfig:
    iterations: int = 8
    population_size: int = 24
    elite_fraction: float = 0.25
    smoothing: float = 0.7
    min_gap: float = 0.1


def _objective(metrics: dict, weights: OptimizationWeights) -> float:
    return (
        weights.ber*metrics["ber"]
        + weights.tx_fail*metrics["tx_trigger_failure_rate"]
        + weights.avg_duration*metrics["avg_symbol_duration"]
        + weights.max_error*metrics["max_error"]
    )


def evaluate_parameter_set(
    params: EventTriggerParams,
    *,
    message_bits,
    timestep: float = 0.01,
    metric: str = "norm",
    use_metric_filter: bool = True,
    metric_filter_alpha: float = 0.3,
    abs_dx_floor_ratio: float = 0.2,
    noise_std: float = 0.5,
    random_seed: int = 0,
    max_symbol_steps: int = 12_000,
    weights: OptimizationWeights | None = None,
) -> dict:
    weights = weights or OptimizationWeights()
    metrics = run_event_triggered_message(
        message_bits,
        params,
        timestep=timestep,
        metric=metric,
        use_metric_filter=use_metric_filter,
        metric_filter_alpha=metric_filter_alpha,
        abs_dx_floor_ratio=abs_dx_floor_ratio,
        noise_std=noise_std,
        random_seed=random_seed,
        max_symbol_steps=max_symbol_steps,
    )
    score = _objective(metrics, weights)

    return {
        "score": float(score),
        "epsilon0": float(params.epsilon0),
        "epsilon1": float(params.epsilon1),
        "tconf": float(params.tconf),
        "delta_sigma": float(params.delta_sigma),
        "ber": float(metrics["ber"]),
        "avg_symbol_duration": float(metrics["avg_symbol_duration"]),
        "max_error": float(metrics["max_error"]),
        "tx_trigger_failure_rate": float(metrics["tx_trigger_failure_rate"]),
        "rx_timeout_rate": float(metrics["rx_timeout_rate"]),
    }


def optimize_thresholds(
    *,
    epsilon0_values,
    epsilon1_values,
    tconf_values,
    delta_sigma_values,
    message_bits,
    timestep: float = 0.01,
    metric: str = "norm",
    use_metric_filter: bool = True,
    metric_filter_alpha: float = 0.3,
    abs_dx_floor_ratio: float = 0.2,
    noise_std: float = 0.5,
    random_seed: int = 0,
    max_symbol_steps: int = 12_000,
    top_k: int = 10,
    weights: OptimizationWeights | None = None,
    verbose: bool = False,
    progress_interval: int = 25,
):
    """
    Grid-search event-triggered threshold parameters.

    Returns all valid runs sorted by score, plus `best` and `top_k` convenience entries.
    """
    epsilon0_values = list(epsilon0_values)
    epsilon1_values = list(epsilon1_values)
    tconf_values = list(tconf_values)
    delta_sigma_values = list(delta_sigma_values)

    total_candidates = (
        len(epsilon0_values)
        * len(epsilon1_values)
        * len(tconf_values)
        * len(delta_sigma_values)
    )

    if verbose:
        print("Threshold optimization configuration")
        print(f"  epsilon0 points: {len(epsilon0_values)}")
        print(f"  epsilon1 points: {len(epsilon1_values)}")
        print(f"  tconf points: {len(tconf_values)}")
        print(f"  delta_sigma points: {len(delta_sigma_values)}")
        print(f"  total candidates: {total_candidates}")
        print(f"  metric filter enabled: {use_metric_filter}")
        print(f"  metric filter alpha: {metric_filter_alpha:.2f}")
        print(f"  abs_dx floor ratio: {abs_dx_floor_ratio:.2f}")

    if total_candidates == 0:
        return {
            "best": None,
            "top_k": [],
            "all_results": [],
        }

    valid_results = []
    start_time = perf_counter()
    evaluated = 0

    for eps0, eps1, tconf, delta_sigma in product(
        epsilon0_values,
        epsilon1_values,
        tconf_values,
        delta_sigma_values,
    ):
        evaluated += 1
        params = EventTriggerParams(
            epsilon0=float(eps0),
            epsilon1=float(eps1),
            tconf=float(tconf),
            delta_sigma=float(delta_sigma),
        )
        try:
            params.validate()
        except ValueError:
            continue

        summary = evaluate_parameter_set(
            params,
            message_bits=message_bits,
            timestep=timestep,
            metric=metric,
            use_metric_filter=use_metric_filter,
            metric_filter_alpha=metric_filter_alpha,
            abs_dx_floor_ratio=abs_dx_floor_ratio,
            noise_std=noise_std,
            random_seed=random_seed,
            max_symbol_steps=max_symbol_steps,
            weights=weights,
        )        
        valid_results.append(summary)

        if verbose and (
            evaluated == total_candidates
            or evaluated == 1
            or evaluated % max(1, progress_interval) == 0
        ):
            elapsed = perf_counter() - start_time
            rate = evaluated / elapsed if elapsed > 0 else 0.0
            remaining = total_candidates - evaluated
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_minutes = eta_seconds / 60.0 if np.isfinite(eta_seconds) else float("inf")
            eta_display = f"{eta_minutes:.2f}m" if np.isfinite(eta_minutes) else "n/a"
            print(
                f"  progress: {evaluated}/{total_candidates} "
                f"({100.0*evaluated/total_candidates:.1f}%), "
                f"elapsed: {elapsed:.1f}s, eta: {eta_display}"
            )

    valid_results.sort(key=lambda row: row["score"])

    return {
        "best": valid_results[0] if valid_results else None,
        "top_k": valid_results[:top_k],
        "all_results": valid_results,
    }


def build_random_bits(length: int, *, seed: int = 1234):
    if length <= 0:
        raise ValueError("length must be > 0")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=length, endpoint=False).tolist()


def _validate_bounds(bounds: dict) -> dict:
    required = ("epsilon0", "epsilon1", "tconf", "delta_sigma")
    normalized = {}
    for key in required:
        if key not in bounds:
            raise ValueError(f"Missing bounds for '{key}'")
        lo, hi = bounds[key]
        lo = float(lo)
        hi = float(hi)
        if not hi > lo:
            raise ValueError(f"Invalid bounds for '{key}': upper must be > lower")
        normalized[key] = (lo, hi)
    return normalized


def optimize_thresholds_cem(
    *,
    bounds,
    message_bits,
    timestep: float = 0.01,
    metric: str = "norm",
    use_metric_filter: bool = True,
    metric_filter_alpha: float = 0.3,
    abs_dx_floor_ratio: float = 0.2,
    noise_std: float = 0.5,
    random_seed: int = 0,
    max_symbol_steps: int = 12_000,
    top_k: int = 10,
    weights: OptimizationWeights | None = None,
    config: CEMConfig | None = None,
    verbose: bool = False,
):
    """
    Cross-Entropy Method (CEM) optimization over continuous parameter ranges.
    """
    config = config or CEMConfig()
    if config.iterations <= 0:
        raise ValueError("iterations must be > 0")
    if config.population_size <= 1:
        raise ValueError("population_size must be > 1")
    if not (0.0 < config.elite_fraction <= 1.0):
        raise ValueError("elite_fraction must be in (0, 1]")
    if not (0.0 < config.smoothing <= 1.0):
        raise ValueError("smoothing must be in (0, 1]")

    bounds = _validate_bounds(bounds)
    rng = np.random.default_rng(random_seed)
    elite_count = max(1, int(np.ceil(config.population_size*config.elite_fraction)))
    total_evals = config.iterations*config.population_size

    keys = ("epsilon0", "epsilon1", "tconf", "delta_sigma")
    lower = np.array([bounds[k][0] for k in keys], dtype=float)
    upper = np.array([bounds[k][1] for k in keys], dtype=float)
    mean = 0.5*(lower + upper)
    std = 0.5*(upper - lower)

    all_results = []
    history = []
    evaluations = 0
    start_time = perf_counter()

    if verbose:
        print("CEM optimization configuration")
        print(f"  iterations: {config.iterations}")
        print(f"  population_size: {config.population_size}")
        print(f"  elite_fraction: {config.elite_fraction:.2f}")
        print(f"  total evaluations: {total_evals}")
        print(f"  metric filter enabled: {use_metric_filter}")
        print(f"  metric filter alpha: {metric_filter_alpha:.2f}")
        print(f"  abs_dx floor ratio: {abs_dx_floor_ratio:.2f}")
        print("  bounds:")
        for key in keys:
            lo, hi = bounds[key]
            print(f"    {key}: [{lo:.3f}, {hi:.3f}]")

    for iteration in range(config.iterations):
        iteration_results = []
        for _ in range(config.population_size):
            sampled = rng.normal(loc=mean, scale=np.maximum(std, 1e-6))
            sampled = np.clip(sampled, lower, upper)
            eps0, eps1, tconf, delta_sigma = sampled.tolist()

            if eps1 <= eps0 + config.min_gap:
                eps1 = min(upper[1], eps0 + config.min_gap)
                if eps1 <= eps0:
                    eps0 = max(lower[0], eps1 - config.min_gap)

            params = EventTriggerParams(
                epsilon0=float(eps0),
                epsilon1=float(eps1),
                tconf=float(tconf),
                delta_sigma=float(delta_sigma),
            )
            try:
                params.validate()
            except ValueError:
                continue

            summary = evaluate_parameter_set(
                params,
                message_bits=message_bits,
                timestep=timestep,
                metric=metric,
                use_metric_filter=use_metric_filter,
                metric_filter_alpha=metric_filter_alpha,
                abs_dx_floor_ratio=abs_dx_floor_ratio,
                noise_std=noise_std,
                random_seed=int(rng.integers(0, 1_000_000_000)),
                max_symbol_steps=max_symbol_steps,
                weights=weights,
            )
            all_results.append(summary)
            iteration_results.append(summary)
            evaluations += 1

        if not iteration_results:
            continue

        iteration_results.sort(key=lambda row: row["score"])
        elites = iteration_results[:elite_count]
        elite_vectors = np.array(
            [
                [row["epsilon0"], row["epsilon1"], row["tconf"], row["delta_sigma"]]
                for row in elites
            ],
            dtype=float,
        )

        new_mean = elite_vectors.mean(axis=0)
        new_std = elite_vectors.std(axis=0, ddof=0)
        mean = config.smoothing*mean + (1.0 - config.smoothing)*new_mean
        std = config.smoothing*std + (1.0 - config.smoothing)*np.maximum(new_std, 1e-4)

        best_row = iteration_results[0]
        history.append(
            {
                "iteration": iteration + 1,
                "best_score": float(best_row["score"]),
                "best_ber": float(best_row["ber"]),
            }
        )

        if verbose:
            elapsed = perf_counter() - start_time
            rate = evaluations / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total_evals - evaluations)
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_minutes = eta_seconds / 60.0 if np.isfinite(eta_seconds) else float("inf")
            eta_display = f"{eta_minutes:.2f}m" if np.isfinite(eta_minutes) else "n/a"
            print(
                f"  iter {iteration + 1}/{config.iterations}: "
                f"best_score={best_row['score']:.4f}, "
                f"evals={evaluations}/{total_evals}, eta={eta_display}"
            )

    all_results.sort(key=lambda row: row["score"])
    return {
        "best": all_results[0] if all_results else None,
        "top_k": all_results[:top_k],
        "all_results": all_results,
        "history": history,
    }


def _print_candidate(label: str, row: dict) -> None:
    print(label)
    print("  params:")
    print(f"    epsilon0={row['epsilon0']:.3f}")
    print(f"    epsilon1={row['epsilon1']:.3f}")
    print(f"    tconf={row['tconf']:.3f}")
    print(f"    delta_sigma={row['delta_sigma']:.3f}")
    print("  metrics:")
    print(f"    score={row['score']:.4f}")
    print(f"    ber={row['ber']:.4f}")
    print(f"    avg_symbol_duration={row['avg_symbol_duration']:.4f}")
    print(f"    max_error={row['max_error']:.4f}")
    print("  reliability:")
    print(f"    tx_trigger_failure_rate={row['tx_trigger_failure_rate']:.4f}")
    print(f"    rx_timeout_rate={row['rx_timeout_rate']:.4f}")


def run_default_cem_demo() -> None:
    bitstream = build_random_bits(16, seed=42)
    search = optimize_thresholds_cem(
        bounds={
            "epsilon0": (0.5, 2.0),
            "epsilon1": (6.0, 14.0),
            "tconf": (0.02, 0.12),
            "delta_sigma": (1.5, 6.0),
        },
        message_bits=bitstream,
        noise_std=0.5,
        random_seed=42,
        top_k=5,
        max_symbol_steps=2_000,
        config=CEMConfig(iterations=6, population_size=20, elite_fraction=0.25, smoothing=0.7),
        verbose=True,
    )

    print("Best parameter set (CEM):")
    _print_candidate("  best", search["best"])
    print("\nTop candidates:")
    for idx, row in enumerate(search["top_k"], start=1):
        _print_candidate(f"  #{idx}", row)


if __name__ == "__main__":
    run_default_cem_demo()
