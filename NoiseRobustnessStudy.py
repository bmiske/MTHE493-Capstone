import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from TrialSimulation import run_simulation, NUM_STEPS, BURN_IN_STEPS, SAMPLE_STRIDE, TIME_STEP


def _noise_level_metrics(args):
    noise_std, num_trials, num_steps, burn_in_steps, sample_stride, timestep = args
    trial_errors = []
    for trial in range(num_trials):
        results = run_simulation(
            num_steps=num_steps,
            burn_in_steps=burn_in_steps,
            sample_stride=sample_stride,
            timestep=timestep,
            noise_std=noise_std,
            random_seed=trial,
        )
        trial_errors.append(results["errors"][-1])

    errors = np.array(trial_errors, dtype=float)
    return {
        "noise_std": noise_std,
        "mean_final_error": errors.mean(),
        "std_final_error": errors.std(ddof=1) if errors.size > 1 else 0.0,
    }


def sweep_noise_levels(
    noise_levels,
    *,
    num_trials=5,
    num_steps=NUM_STEPS,
    burn_in_steps=BURN_IN_STEPS,
    sample_stride=SAMPLE_STRIDE,
    timestep=None,
    max_workers=1,
):
    """
    Run multiple simulations per-noise level and summarize the final synchronization error.

    Args:
        noise_levels: Iterable of noise standard deviations to evaluate.
        num_trials: Number of simulations per noise level.
        num_steps/burn_in_steps/sample_stride/timestep: forwarded to `run_simulation`.
        max_workers: Number of worker processes (defaults to CPU count). Use 1 for sequential execution.

    Returns:
        List of dicts with keys: noise_std, mean_final_error, std_final_error.
    """
    timestep = timestep or TIME_STEP
    noise_levels = [float(level) for level in noise_levels]
    worker_args = [
        (level, num_trials, num_steps, burn_in_steps, sample_stride, timestep)
        for level in noise_levels
    ]

    def run_sequential():
        return [_noise_level_metrics(args) for args in worker_args]

    if max_workers == 1:
        metrics = run_sequential()
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                metrics = list(executor.map(_noise_level_metrics, worker_args))
        except (OSError, PermissionError) as exc:
            print(
                f"ProcessPoolExecutor unavailable ({exc!s}). "
                "Falling back to sequential computation."
            )
            metrics = run_sequential()

    metrics.sort(key=lambda entry: entry["noise_std"])
    return metrics


def plot_noise_sweep(metrics, *, output_path=None, show_plot=False):
    """Plot final error as a function of injected noise standard deviation."""
    noise = [m["noise_std"] for m in metrics]
    mean_error = [m["mean_final_error"] for m in metrics]
    std_error = [m["std_final_error"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(noise, mean_error, yerr=std_error, fmt="-o", capsize=4, color="tab:red")
    ax.set_xlabel("Injected noise std (on master x)")
    ax.set_ylabel("Final post burn-in error (state units)")
    ax.set_title("Noise robustness of synchronized Lorenz system")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200)

    if show_plot:
        try:
            plt.show()
        except Exception as exc:
            print(
                "Could not display plot interactively "
                f"({exc!s}). Figure saved to {output_path or 'figure object'}."
            )

    plt.close(fig)


def export_metrics_to_csv(metrics, csv_path):
    """Persist the sweep results so they can be shared alongside the plot."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["noise_std", "mean_final_error", "std_final_error"])
        for entry in metrics:
            writer.writerow(
                [
                    f"{entry['noise_std']:.6f}",
                    f"{entry['mean_final_error']:.6f}",
                    f"{entry['std_final_error']:.6f}",
                ]
            )


if __name__ == "__main__":
    sweep = sweep_noise_levels(
        noise_levels=np.linspace(0.0, 40.0, num=200),
        num_trials=10,
        num_steps=4_000,
        burn_in_steps=2_000,
        sample_stride=40,
        max_workers=None,
    )
    plot_noise_sweep(
        sweep,
        output_path="image/noise_robustness.png",
        show_plot=True,
    )
    export_metrics_to_csv(sweep, "data/noise_robustness.csv")
    for entry in sweep:
        print(
            f"noise_std={entry['noise_std']:.2f} "
            f"mean_error={entry['mean_final_error']:.3f} "
            f"std={entry['std_final_error']:.3f}"
        )
