from skopt import gp_minimize
from skopt.space import Real, Integer
from EventTriggered import EventTriggeredParameters, EventTriggeredCommunication

SYMBOLS_PER_TRIAL = 64

# C = r(e/e0) + (1-r)(t/t0)
#   r  = error rate weighting constant between 0 and 1
#   e  = error rate with current parameters
#   e0 = baseline error rate from other optimal algorithm
#   t  = average time steps per symbol with current parameters
#   t0 = baseline time steps per symbol from other optimal algorithm

def findMinimumCost(params):
    NUMBER_OF_INITIAL_TRIALS = 10
    INITIAL_COST_NEEDED_TO_CONTINUE = 1.0
    NUMBER_OF_ADDITIONAL_TRIALS = 40

    WEIGHT_CONSTANT = 0.75
    BASELINE_ERROR_RATE = 0.0042
    BASELINE_TIME_STEPS = 150

    # Unpack parameters and create test configuration
    epsilon_zero, epsilon_one, sigma_offset, conf_time_steps, moving_average_window_size = params
    test_params = EventTriggeredParameters(
        epsilon_zero=epsilon_zero,
        epsilon_one=epsilon_one,
        sigma_offset=sigma_offset,
        confidence_time_steps=conf_time_steps,
        average_window_size=moving_average_window_size,
    )

    errors = 0
    time_steps = 0

    # Initial trials
    for _ in range(NUMBER_OF_INITIAL_TRIALS):
        result = EventTriggeredCommunication.EventTriggeredTest(test_params)
        errors += result.errors
        time_steps += result.time_steps

    avg_error = errors / NUMBER_OF_INITIAL_TRIALS
    avg_time_per_symbol = time_steps / (NUMBER_OF_INITIAL_TRIALS * SYMBOLS_PER_TRIAL)

    initial_cost = (
        WEIGHT_CONSTANT * (avg_error / BASELINE_ERROR_RATE)
        + (1 - WEIGHT_CONSTANT) * (avg_time_per_symbol / BASELINE_TIME_STEPS)
    )

    # Continue only if the preliminary cost is promising
    if initial_cost <= INITIAL_COST_NEEDED_TO_CONTINUE:
        for _ in range(NUMBER_OF_ADDITIONAL_TRIALS):
            result = EventTriggeredCommunication.EventTriggeredTest(test_params)
            errors += result.errors
            time_steps += result.time_steps

        total_trials = NUMBER_OF_INITIAL_TRIALS + NUMBER_OF_ADDITIONAL_TRIALS
        avg_error = errors / total_trials
        avg_time_per_symbol = time_steps / (total_trials * SYMBOLS_PER_TRIAL)

        final_cost = (
            WEIGHT_CONSTANT * (avg_error / BASELINE_ERROR_RATE)
            + (1 - WEIGHT_CONSTANT) * (avg_time_per_symbol / BASELINE_TIME_STEPS)
        )

        print(
            f"Achieved cost {final_cost:.3f} with params: "
            f"epsilon_zero={epsilon_zero:.2f}, "
            f"epsilon_one={epsilon_one:.2f}, "
            f"sigma_offset={sigma_offset:.2f}, "
            f"conf_time={conf_time_steps}, "
            f"window_size={moving_average_window_size} - "
            f"Average Error Rate: {avg_error:.3f}, "
            f"Average Time Steps per Symbol: {avg_time_per_symbol:.3f}"
        )

        return final_cost

    print(
        f"Achieved cost {initial_cost:.3f} with params: "
        f"epsilon_zero={epsilon_zero:.2f}, "
        f"epsilon_one={epsilon_one:.2f}, "
        f"sigma_offset={sigma_offset:.2f}, "
        f"conf_time={conf_time_steps}, "
        f"window_size={moving_average_window_size} - "
        f"Average Error Rate: {avg_error:.3f}, "
        f"Average Time Steps per Symbol: {avg_time_per_symbol:.3f}. "
        f"Ending Early."
    )

    return initial_cost


# Parameters to optimize
dimensions = [
    Real(0.01, 0.5, name="epsilon_zero"),
    Real(0.5, 2.0, name="epsilon_one"),
    Real(5.0, 20.0, name="sigma_offset"),
    Integer(10, 50, name="confidence_time_steps"),
    Integer(5, 30, name="moving_average_window_size"),
]


# Bayesian optimization
result = gp_minimize(findMinimumCost, dimensions, n_calls=120, n_initial_points=20)
print(f"Optimal parameters: {result.x}")
print(f"Minimal cost achieved: {result.fun}")
