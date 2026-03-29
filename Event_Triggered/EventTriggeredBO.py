from skopt import gp_minimize
from skopt.space import Real, Integer
from EventTriggered import EventTriggeredParameters, EventTriggeredCommunication

SYMBOLS_PER_TRIAL = 64

# This is the cost function currently used, which only minimizes error rate
def findMinimumErrorRate(params):
    
    NUMBER_OF_INITIAL_TRIALS = 10               #Trials initially completed to get rough idea of error rate
    INITIAL_ERROR_RATE_NEEDED_TO_CONTINUE = 1.0 #Maximum error rate to continue trials
    NUMBER_OF_ADDITIONAL_TRIALS = 40            #Used in cases where error rate is less than above rate

    # Unpackages parameters and then instantiates an EventTriggeredParameters varialble using the supplied values
    epsilon_zero, epsilon_one, sigma_offset, conf_time_steps, moving_average_window_size = params
    test_params = EventTriggeredParameters (
        epsilon_zero=epsilon_zero,
        epsilon_one=epsilon_one,
        sigma_offset = sigma_offset,
        confidence_time_steps = conf_time_steps,
        average_window_size = moving_average_window_size,
    )
    
    errors = 0
    time_steps = 0
    # First runs a preliminary number of trials. If preforms poorly, doesn't waste time on further computation
    for _ in range(NUMBER_OF_INITIAL_TRIALS):
        errors += (EventTriggeredCommunication.EventTriggeredTest(test_params)).errors
        time_steps += (EventTriggeredCommunication.EventTriggeredTest(test_params)).time_steps
    # If the error rate is belowed a designated threshold, we attempt more trials to get a better estimation of performance
    if (errors/NUMBER_OF_INITIAL_TRIALS) <= INITIAL_ERROR_RATE_NEEDED_TO_CONTINUE: 
        for _ in range (NUMBER_OF_ADDITIONAL_TRIALS):
            errors += (EventTriggeredCommunication.EventTriggeredTest(test_params)).errors
            time_steps += (EventTriggeredCommunication.EventTriggeredTest(test_params)).time_steps
        print(f"Achieved {errors/(NUMBER_OF_INITIAL_TRIALS+NUMBER_OF_ADDITIONAL_TRIALS):.3f} error rate with params:"+
            f"epsilon_zero={epsilon_zero:.2f}, epsilon_one={epsilon_one:.2f} sigma_offset={sigma_offset:.2f}, conf_time={conf_time_steps}, window_size={moving_average_window_size} - "+
            f"Average Time Steps per Symbol: {time_steps/((NUMBER_OF_INITIAL_TRIALS+NUMBER_OF_ADDITIONAL_TRIALS)*SYMBOLS_PER_TRIAL):.3f}")
        return (errors/(NUMBER_OF_INITIAL_TRIALS+NUMBER_OF_ADDITIONAL_TRIALS))
    print(f"Achieved {errors/(NUMBER_OF_INITIAL_TRIALS):.3f} error rate with params:"+
        f"epsilon_zero={epsilon_zero:.2f}, epsilon_one={epsilon_one:.2f} sigma_offset={sigma_offset:.2f}, conf_time={conf_time_steps}, window_size={moving_average_window_size} - "+
        f"Average Time Steps per Symbol: {time_steps/(NUMBER_OF_INITIAL_TRIALS*SYMBOLS_PER_TRIAL):.3f}. "+
        "Ending Early.")
    # We return the final error rate to the optimization function.
    return (errors/(NUMBER_OF_INITIAL_TRIALS))



# These are the parameters to be passed into the cost function. Formatted as:
#   [type (Real/Integer)]([min_value], [max_value], name="[parameter_name]") 
dimensions = [
    Real(0.01, 0.5, name="epsilon_zero"),
    Real(0.5, 2.0, name="epsilon_one"),
    Real(5.0, 20.0, name="sigma_offset"),
    Integer(10, 50, name="confidence_time_steps"),
    Integer(5, 30, name="moving_average_window_size"),
    # Integer(999, 1000, name="max_time_steps"), Depreciated
]

# This is the bayseian optimization function. 
# Increasing n_initial_points will make it more likely to find a global min, while n_calls willhelp optimize whichever minimum it locates
result = gp_minimize(findMinimumErrorRate, dimensions, n_calls = 120, n_initial_points=20)

print(f"Optimal parameters: {result.x}")
print(f"Minimal error rate achieved: {result.fun}")

'''

'''