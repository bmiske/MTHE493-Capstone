import numpy as np

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem
from typing import Tuple

class EventTriggeredParameters:

    # Default values for test parameters
    DEFAULT_SIGMA_OFFSET = 5.0
    DEFAULT_TRANSITION_TIME_STEPS = 100
    DEFAULT_WHITE_NOISE_STD_DEV = 0.1
    DEFAULT_TIME_STEP = 0.01
    DEFAULT_SIGMA = 16.0
    DEFAULT_BETA = 4.0
    DEFAULT_RHO = 45.0
    DEFAULT_MASTER_INITIAL_STATE = (20.0, 10.0, 9.9)
    DEFAULT_DRIVEN_INITIAL_STATE = (20.1, 10.2, 9.9)
    DEFAULT_EPSILON_ZERO = 0.25
    DEFAULT_EPSILON_ONE = 1.25
    DEFAULT_CONFIDENCE_TIME_STEPS = 20
    DEFAULT_MAX_STEPS_PER_SYMBOL = 5000

    def __init__(
        self, 
        sigma_offset: float = DEFAULT_SIGMA_OFFSET,
        white_noise_std_dev: float = DEFAULT_WHITE_NOISE_STD_DEV,
        time_step: float = DEFAULT_TIME_STEP,
        sigma: float = DEFAULT_SIGMA,
        beta: float = DEFAULT_BETA,
        rho: float = DEFAULT_RHO,
        master_system_initial_state: Tuple[float, float, float] = DEFAULT_MASTER_INITIAL_STATE,
        driven_system_initial_state: Tuple[float, float, float] = DEFAULT_DRIVEN_INITIAL_STATE,
        epsilon_zero: float = DEFAULT_EPSILON_ZERO,
        epsilon_one: float = DEFAULT_EPSILON_ONE,
        confidence_time_steps: int = DEFAULT_CONFIDENCE_TIME_STEPS,
        max_steps_per_symbol: int = DEFAULT_MAX_STEPS_PER_SYMBOL,
    ):
        """
        Parameters
        -------------
        sigmaOffset: float
            The amount that the sigma parameter of the transmitting system is increased by when transmitting a 1 bit
        whiteNoiseStDev: float
            The standard deviation of the gaussian white noise added to the transmitted signal
        timeStep: float
            How much time passes inbetween each iteration of the chaotic systems
        sigma: float
            A parameter used for the Lorenz System. Sigma of the transmitting system is increased by sigmaOffset when transmitting a 1
        beta: float
            A parameter used for the Lorenz System.
        rho: float
            A parameter used for the Lorenz System.
        masterSystemInitialState: float[3]
            An ordered list contiaing the intial x, y, and z states of the master (transmitting) system
        drivenSystemInitialState: float[3]
            An ordered list contiaing the intial x, y, and z states of the driven (receiving) system
        epsilon_zero: float
            threshold the error must be below to indicate a zero bit
        epsilon_one: float
            threshold the error must be above to indicate a one bit
        confidence_time_steps: int
            ammount of time steps that the error must be below epsilon 0 or above epsilon 1 before indicating a bit
        max_steps_per_symbol: int
            the maximum amount of steps the receiver will attempt to decode a single bit for
        """
        self.time_step = time_step
        self.white_noise_std_dev = white_noise_std_dev
        self.sigma_offset = sigma_offset
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.master_system_initial_state = master_system_initial_state
        self.driven_system_initial_state = driven_system_initial_state
        self.epsilon_zero = epsilon_zero
        self.epsilon_one = epsilon_one
        self.max_steps_per_symbol = max_steps_per_symbol
        self.confidence_time_steps = confidence_time_steps
        
        self._validate_parameters()

    
    def _validate_parameters(self):
        """Validate parameter values are reasonable."""
        if self.max_steps_per_symbol <= 0:
            raise ValueError("max_steps_per_symbol must be positive")
        if self.epsilon_zero < 0 or self.epsilon_one < 0:
            raise ValueError("epsilon values must be non-negative")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.epsilon_one < self.epsilon_zero:
            raise ValueError("epsilon_zero must be less than epsilon_one") 

class EventTriggeredTestResults:

    def __init__(self, errors: int, time_steps: int):
        self.errors = errors
        self.time_steps = time_steps

class EventTriggeredCommunication:

    @staticmethod
    def compareMessages(m1, m2):
        """
        Comapares two messages of equal length, and returns the number of differences between them

        Params
        --------
        m1: string or array
        m2: string or array

        Returns
        --------
        int
            Number of mismatched charachters between m1 and m2
        """
        if len(m1) != len(m2):
            raise ValueError("m1 and m2 must have the same length")
        num_errors = 0
        for i in range(len(m1)):
            if m1[i] != m2[i]: num_errors += 1
        return num_errors 

    @staticmethod
    def EventTriggeredTest(test_parameters: EventTriggeredParameters, message=None):
        """
        Test whether a binary communication with the given parameters succeeds

        Params
        --------
        test_parameters: BinaryCommunicationParameters
            The parameters that the test should be run under
        message: string or Array
            A string or array containing bits to be encoded. Should only contain 0s or 1s.

        Returns
        ---------
        int
            The number of errors in the decoded message. 0 if the decoded message matches the input message exactly.
        """
        DEFAULT_MESSAGE_LENGTH = 64
        if not message:
            message = np.random.randint(0, 2, DEFAULT_MESSAGE_LENGTH)

        master_system = MasterLorenzSystem(test_parameters.master_system_initial_state, test_parameters.time_step, sigma=test_parameters.sigma, rho=test_parameters.rho, beta=test_parameters.beta)
        driven_system = DrivenLorenzSystem(test_parameters.driven_system_initial_state, test_parameters.time_step, sigma=test_parameters.sigma, rho=test_parameters.rho, beta=test_parameters.beta)

        times = [0.0]
        master_x = [master_system.getXState()]
        driven_x = [driven_system.getXState()]
        errors = []  # r(t) = |x_m - x_d| at each time step

        # For reconstructing square waves with variable symbol lengths
        symbol_steps = []     # number of time steps spent per symbol
        event_times = [0.0]   # absolute time when each symbol ends (t0=0)

        # Decoded bits (receiver output)
        received_bits = []

        # -----------------------------
        # Main event-triggered simulation
        # -----------------------------
        t = 0.0

        for bit in message:
            zero_count = 0
            one_count = 0
            steps_this_symbol = 0
            triggered_bit = None  # what receiver decides for this symbol

            while True:
                # Use current master x as the transmitted driving signal
                xm = master_system.getXState()

                # Measure sync error BEFORE stepping (consistent with your original test style)
                r = driven_system.getError(xm)
                errors.append(r)

                # Step driven system with noisy transmitted xm
                driven_system.nextState(xm + np.random.normal(0, test_parameters.white_noise_std_dev))

                # Step master system, modulating sigma when transmitting a 1
                if bit == 1:
                    master_system.nextState(test_parameters.sigma_offset)
                else:
                    master_system.nextState(0.0)

                # Advance time and log states AFTER stepping
                t += test_parameters.time_step
                times.append(t)
                master_x.append(master_system.getXState())
                driven_x.append(master_system.getXState())

                steps_this_symbol += 1

                if r <= test_parameters.epsilon_zero:
                    zero_count += 1
                    one_count = 0
                    if zero_count >= test_parameters.confidence_time_steps:
                        triggered_bit = 0  # receiver interprets this as a 0-region trigger
                        break
                elif r >= test_parameters.epsilon_one:
                    one_count += 1
                    zero_count = 0
                    if one_count >= test_parameters.confidence_time_steps:
                        triggered_bit = 1  # receiver interprets this as a 1-region trigger
                        break
                else:
                    zero_count = 0
                    one_count = 0

                # Safety escape if event never triggers
                if steps_this_symbol >= test_parameters.max_steps_per_symbol:
                    # Fallback: decide based on where we ended up relative to mid-threshold
                    mid = 0.5 * (test_parameters.epsilon_zero + test_parameters.epsilon_one)
                    triggered_bit = 1 if r > mid else 0
                    break

            symbol_steps.append(steps_this_symbol)
            received_bits.append(triggered_bit)
            event_times.append(t)

        received_bits = np.array(received_bits, dtype=int)

        num_errors = EventTriggeredCommunication.compareMessages(message, received_bits)
        num_time_steps = len(times)
        result = EventTriggeredTestResults(num_errors, num_time_steps)
        return(result)