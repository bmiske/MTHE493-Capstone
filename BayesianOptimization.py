from skopt import gp_minimize
from skopt.space import Real, Integer
from BinaryCommunication import BinaryCommunicationParameters, BinaryCommunication

def findMinimumSymbolLength(params):

    INITIAL_TIME_STEPS = 200
    WORST_CASE_VALUE = 300
    TRIALS_PER_TIME_STEP_VALUE = 5

    error_threshold, sigma_offset, transition_time_steps = params

    test_params = BinaryCommunicationParameters (
        error_threshold = error_threshold,
        sigma_offset = sigma_offset,
        transition_time_steps = transition_time_steps,
    )

    test_params.set_time_steps_per_symbol(INITIAL_TIME_STEPS)

    adjustment_factor = 25 #We will start at 200 and reduce by 25 until failure. Then we will fine tune our search.
    last_error_free_value = None
    # We will stay in this loop reducing time_steps_per_symbol until we experience errors.
    # TODO: Make this loop more readable/add comments
    while True:
        errors = 0
        for _ in range(TRIALS_PER_TIME_STEP_VALUE):
            errors += BinaryCommunication.BinaryCommunicationTest(test_params)
        if errors == 0:
            last_error_free_value = test_params.time_steps_per_symbol
        else:
            if last_error_free_value == None: last_error_free_value = WORST_CASE_VALUE
            elif adjustment_factor > 1 and last_error_free_value < 200: adjustment_factor /= 5
            else: break
        test_params.set_time_steps_per_symbol(int(last_error_free_value - adjustment_factor))
    
    return last_error_free_value

dimensions = [
    Real(0.05, 1.0, name="error_threshold"),
    Real(0.5, 20.0, name="sigma_offset"),
    Integer(0, 150, name="transition_time"),
#   Commented these params out since changing them can cause Lorenz to not converge. Would like to experiment with some variations though
#    Real(-20.0, 20.0, name="sigma"),
#    Real(-20.0, 20.0, name="beta"),
#    Real(-20.0, 20.0, name="rho"),
]

result = gp_minimize(findMinimumSymbolLength, dimensions, n_calls=50)

print(f"Optimal parameters: {result.x}")
print(f"Shortest symbol length achieved: {result.fun}")



'''
I have only run this once so far, with 0.1 White Noise Std. Dev. and achieved the following param set:
    Optimal parameters: [0.2950381815949865, 10.39831407260162, np.int64(0)]
    Shortest symbol length achieved: 145

I would like to try running more tests, and varying the parameters of the gp_minimize function, and also try with 0.2 noise st.dev.
This seems really cool though.
-Ben
'''