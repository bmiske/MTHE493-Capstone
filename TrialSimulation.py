import matplotlib.pyplot as plt
import numpy as np

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem

NUM_STEPS = 4*10**4
TIME_STEP = 0.01
MASTER_INITIAL_STATE = (10.0, -20.0, 30.0)
DRIVEN_INITIAL_STATE = (-10.1, 10.2, 9.9)
BURN_IN_STEPS = 2_000
SAMPLE_STRIDE = 40


def run_simulation(
    num_steps=NUM_STEPS,
    *,
    timestep=TIME_STEP,
    master_initial_state=MASTER_INITIAL_STATE,
    driven_initial_state=DRIVEN_INITIAL_STATE,
    burn_in_steps=BURN_IN_STEPS,
    sample_stride=SAMPLE_STRIDE,
    noise_std=0.0,
    random_seed=None,
):
    """
    Integrate coupled Lorenz systems and return sampled trajectories plus errors.

    Args:
        num_steps: Total integration steps.
        timestep: RK4 timestep size.
        master_initial_state: Starting state for the master system.
        driven_initial_state: Starting state for the driven system.
        burn_in_steps: Steps ignored when computing the error metric.
        sample_stride: Every `sample_stride`-th state is stored.
        noise_std: Standard deviation of zero-mean Gaussian noise added to the driving signal.
        random_seed: Seed forwarded to NumPy's default RNG for reproducibility when injecting noise.
    """
    if burn_in_steps >= num_steps:
        raise ValueError("burn_in_steps must be smaller than num_steps")

    master_system = MasterLorenzSystem(master_initial_state, timestep)
    driven_system = DrivenLorenzSystem(driven_initial_state, timestep)
    rng = np.random.default_rng(random_seed)

    sample_times = [0.0]
    master_samples = [np.array(master_initial_state, dtype=float)]
    driven_samples = [np.array(driven_initial_state, dtype=float)]

    error_times = []
    errors = []

    for step in range(num_steps):
        master_state = master_system.nextState()
        driving_signal = master_state[0]

        if noise_std > 0.0:
            driving_signal = driving_signal + rng.normal(0.0, noise_std)

        driven_state = driven_system.nextState(driving_signal)

        step_time = (step + 1)*timestep

        if (step + 1) % sample_stride == 0:
            sample_times.append(step_time)
            master_samples.append(master_state)
            driven_samples.append(driven_state)

        if step + 1 >= burn_in_steps:
            error_times.append(step_time)
            errors.append(np.linalg.norm(master_state - driven_state))

    return {
        "times": np.array(sample_times),
        "master_states": np.vstack(master_samples),
        "driven_states": np.vstack(driven_samples),
        "error_times": np.array(error_times),
        "errors": np.array(errors),
    }


def plot_results(results):
    times = results["times"]
    master_states = results["master_states"]
    driven_states = results["driven_states"]
    error_times = results["error_times"]
    errors = results["errors"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Lorenz Systems Comparison", fontsize=16)

    axes[0, 0].plot(times, master_states[:, 0], label="Master System", color="blue", linewidth=1.5)
    axes[0, 0].plot(times, driven_states[:, 0], label="Driven System", color="red", linewidth=1.5, alpha=0.8)
    axes[0, 0].set_xlabel("Time", fontsize=12)
    axes[0, 0].set_ylabel("X State", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(times, master_states[:, 1], label="Master System", color="blue", linewidth=1.5)
    axes[0, 1].plot(times, driven_states[:, 1], label="Driven System", color="red", linewidth=1.5, alpha=0.8)
    axes[0, 1].set_xlabel("Time", fontsize=12)
    axes[0, 1].set_ylabel("Y State", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times, master_states[:, 2], label="Master System", color="blue", linewidth=1.5)
    axes[1, 0].plot(times, driven_states[:, 2], label="Driven System", color="red", linewidth=1.5, alpha=0.8)
    axes[1, 0].set_xlabel("Time", fontsize=12)
    axes[1, 0].set_ylabel("Z State", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(error_times, errors, color="green", linewidth=2)
    axes[1, 1].set_xlabel("Time", fontsize=12)
    axes[1, 1].set_ylabel("Post Burn-in Error", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sim_results = run_simulation()
    plot_results(sim_results)
