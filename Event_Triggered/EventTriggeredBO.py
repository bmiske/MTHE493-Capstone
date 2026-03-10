from skopt import gp_minimize
from skopt.space import Real, Integer
from EventTriggered import EventTriggeredParameters, EventTriggeredCommunication

def findMinimumErrorRate(params):
    
    NUMBER_OF_INITIAL_TRIALS = 10               #Trials initially completed to get rough idea of error rate
    INITIAL_ERROR_RATE_NEEDED_TO_CONTINUE = 1.0 #Maximum error rate to continue trials
    NUMBER_OF_ADDITIONAL_TRIALS = 40            #Used in cases where error rate is less than above rate

    epsilon_zero, epsilon_one, sigma_offset, conf_time_steps = params
    test_params = EventTriggeredParameters (
        epsilon_zero=epsilon_zero,
        epsilon_one=epsilon_one,
        sigma_offset = sigma_offset,
        confidence_time_steps = conf_time_steps,
    )
    
    errors = 0
    for _ in range(NUMBER_OF_INITIAL_TRIALS):
        errors += EventTriggeredCommunication.EventTriggeredTest(test_params)
    if (errors/NUMBER_OF_INITIAL_TRIALS) <= INITIAL_ERROR_RATE_NEEDED_TO_CONTINUE: 
        for _ in range (NUMBER_OF_ADDITIONAL_TRIALS):
            errors += EventTriggeredCommunication.EventTriggeredTest(test_params)
        print(f"Achieved {errors/(NUMBER_OF_INITIAL_TRIALS+NUMBER_OF_ADDITIONAL_TRIALS):.3f} error rate with params: epsilon_zero={epsilon_zero:.2f}, epsilon_one={epsilon_one:.2f} sigma_offset={sigma_offset:.2f}, conf_time={conf_time_steps}")
        return (errors/(NUMBER_OF_INITIAL_TRIALS+NUMBER_OF_ADDITIONAL_TRIALS))
    print(f"Achieved {errors/(NUMBER_OF_INITIAL_TRIALS):.3f} error rate with params: epsilon_zero={epsilon_zero:.2f}, epsilon_one={epsilon_one:.2f} sigma_offset={sigma_offset:.2f}, conf_time={conf_time_steps}. Ending early.")
    return (errors/(NUMBER_OF_INITIAL_TRIALS))

dimensions = [
    Real(0.01, 0.5, name="epsilon_zero"),
    Real(0.5, 2.0, name="epsilon_one"),
    Real(5.0, 20.0, name="sigma_offset"),
    Integer(50, 150, name="confidence_time_steps"),
#   Commented these params out since changing them can cause Lorenz to not converge. Would like to experiment with some variations though
#    Real(-20.0, 20.0, name="sigma"),
#    Real(-20.0, 20.0, name="beta"),
#    Real(-20.0, 20.0, name="rho"),
]

result = gp_minimize(findMinimumErrorRate, dimensions, n_calls = 120, n_initial_points=20)

print(f"Optimal parameters: {result.x}")
print(f"Minimal error rate achieved: {result.fun}")

'''

'''