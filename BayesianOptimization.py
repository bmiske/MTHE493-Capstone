from skopt import gp_minimize
from skopt.space import Real, Integer
from BinaryCommunication import BinaryCommunicationParameters, BinaryCommunication

def findMinimumSymbolLength(params):

    INITIAL_TIME_STEPS = 200
    WORST_CASE_VALUE = 300
    TRIALS_PER_TIME_STEP_VALUE = 10

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
            elif adjustment_factor > 1 and last_error_free_value <= 200: adjustment_factor /= 5
            else: break
        test_params.set_time_steps_per_symbol(int(last_error_free_value - adjustment_factor))
    print(f"Achieved {last_error_free_value} with params: error_threshold={error_threshold:.2f}, sigma_offset={sigma_offset:.2f}, transition_time={transition_time_steps}")
    return last_error_free_value

def findMinimumErrorRate(params):
    
    SYMBOL_TIME_STEPS = 130
    NUMBER_OF_INITIAL_TRIALS = 30               #Trials initially completed to get rough idea of error rate
    INITIAL_ERROR_RATE_NEEDED_TO_CONTINUE = 1.0 #Maximum error rate to continue trials
    NUMBER_OF_ADDITIONAL_TRIALS = 70            #Used in cases where error rate is less than above rate
 
    error_threshold, sigma_offset, transition_time_steps = params
    test_params = BinaryCommunicationParameters (
        error_threshold = error_threshold,
        sigma_offset = sigma_offset,
        transition_time_steps = transition_time_steps,
        white_noise_std_dev=0.1
    )
    test_params.set_time_steps_per_symbol(SYMBOL_TIME_STEPS)
    
    errors = 0
    for _ in range(NUMBER_OF_INITIAL_TRIALS):
        errors += BinaryCommunication.BinaryCommunicationTest(test_params)
    if (errors/NUMBER_OF_INITIAL_TRIALS) <= INITIAL_ERROR_RATE_NEEDED_TO_CONTINUE: 
        for _ in range (NUMBER_OF_ADDITIONAL_TRIALS):
            errors += BinaryCommunication.BinaryCommunicationTest(test_params)
        print(f"Achieved {errors/(NUMBER_OF_INITIAL_TRIALS+NUMBER_OF_ADDITIONAL_TRIALS):.3f} error rate with params: error_threshold={error_threshold:.2f}, sigma_offset={sigma_offset:.2f}, transition_time={transition_time_steps}")
        return (errors/(NUMBER_OF_INITIAL_TRIALS+NUMBER_OF_ADDITIONAL_TRIALS))
    print(f"Achieved {errors/(NUMBER_OF_INITIAL_TRIALS):.3f} error rate with params: error_threshold={error_threshold:.2f}, sigma_offset={sigma_offset:.2f}, transition_time={transition_time_steps}. Ending early.")
    return (errors/(NUMBER_OF_INITIAL_TRIALS))

dimensions = [
    Real(0.01, 1.5, name="error_threshold"),
    Real(5.0, 40.0, name="sigma_offset"),
    Integer(30, 130, name="transition_time"),
#   Commented these params out since changing them can cause Lorenz to not converge. Would like to experiment with some variations though
#    Real(-20.0, 20.0, name="sigma"),
#    Real(-20.0, 20.0, name="beta"),
#    Real(-20.0, 20.0, name="rho"),
]

result = gp_minimize(findMinimumErrorRate, dimensions, n_calls = 120, n_initial_points=20)

print(f"Optimal parameters: {result.x}")
print(f"Minimal error rate achieved: {result.fun}")

'''
Since fixing the threshold_time bug, I have run this once (n_calls=100, n_initial_points=10) with the following result:
    Optimal parameters: [0.1809420537408949, 9.764819670083668, np.int64(110)]
    Shortest symbol length achieved: 133
Ran again increasing initial points (n_calls=100, n_initial_points=20):
    Optimal parameters: [0.2963555190346865, 20.0, np.int64(109)]
    Shortest symbol length achieved: 125
Seems like sigma_offset range should be increased given best result occured when it was at the maximum of 20.0 at the time
Increased sigma_offset max (20.0 -> 30.0) and min (0.5 -> 5.0). Left gp_minimize params the same:
  1.Optimal parameters: [0.26082288140275217, 18.44445671495927, np.int64(112)]
    Shortest symbol length achieved: 129
  2.Optimal parameters: [0.30928273894586844, 17.30289569830851, np.int64(103)]
    Shortest symbol length achieved: 124
Going to try reducing range of transition time since it always seens to be in (100, 115). Set to (85, 120) to allow some room.
    Optimal parameters: [0.22482418458932746, 10.072602351008907, np.int64(120)]
    Shortest symbol length achieved: 130
We ended up butting against new range (120), so increasing range (now is 50-150). Also, this is the first one I watched the process.
Decided to increase number of trials to hopefully increase consistency. Many are failing at 200, which is interesting to me.
Expect lower results on next one due to increased trials. Maybe worth implementing max. error rate.
    Optimal parameters: [0.3100783663687425, 14.597340884589109, np.int64(150)]
    Shortest symbol length achieved: 160
Well that was a lot worse than expected... 
Wondering if it makes sense to instead perscribe a symbol length, and find params that minimize error rate? Would be quicker, and likely less noisy

Okay, I've implemented the other one. Now you can set a Symbol Length, and a Number of Trials, and can see what parameters limit error rate

With symbol_len = 130:
    Optimal parameters: [0.6839746463811801, 27.308377230670587, np.int64(110)]
    Minimal error rate achieved: 0.24

Trying without any white noise, symbol_len=130:
    Optimal parameters: [0.2, 12.241486886058013, np.int64(112)]
    Minimal error rate achieved: 0.06
Predictably, error threshold should be lower if no WN

-Ben

March 29: Finding optimal accuracy at symbol lengths 100, 120, 140, 160, 180, and 200:
All tests conducted with 0.1 AWGN, and min 100 trials at optimal:

100:
Optimal parameters: [0.5053060121080802, 16.935974363753527, np.int64(76)]
Minimal error rate achieved: 1.09

120:
Optimal parameters: [0.3865732549397399, 16.9512783131441, np.int64(98)]
Minimal error rate achieved: 0.36

140:
Optimal parameters: [0.2623758292747531, 19.19955476428411, np.int64(117)]
Minimal error rate achieved: 0.05

160:
Optimal parameters: [0.36803638400630945, 21.866183469294334, np.int64(100)]
Minimal error rate achieved: 0.03

180:
Optimal parameters: [0.3473963162600759, 21.99607499797826, np.int64(151)]
Minimal error rate achieved: 0.0




'''