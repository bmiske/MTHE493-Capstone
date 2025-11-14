import numpy as np

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem
from typing import Tuple

class BinaryCommunicationParameters:

    # Default values for test parameters
    DEFAULT_TIME_STEPS_PER_SYMBOL = 200
    DEFAULT_ERROR_THRESHOLD = 0.2
    DEFAULT_SIGMA_OFFSET = 5.0
    DEFAULT_TRANSITION_TIME_STEPS = 100
    DEFAULT_WHITE_NOISE_STD_DEV = 0.1
    DEFAULT_TIME_STEP = 0.01
    DEFAULT_SIGMA = 16.0
    DEFAULT_BETA = 4.0
    DEFAULT_RHO = 45.0
    DEFAULT_MASTER_INITIAL_STATE = (20.0, 10.0, 9.9)
    DEFAULT_DRIVEN_INITIAL_STATE = (20.1, 10.2, 9.9)

    def __init__(
        self, 
        time_steps_per_symbol: int = DEFAULT_TIME_STEPS_PER_SYMBOL,
        error_threshold: float = DEFAULT_ERROR_THRESHOLD,
        sigma_offset: float = DEFAULT_SIGMA_OFFSET,
        transition_time_steps: int = DEFAULT_TRANSITION_TIME_STEPS,
        white_noise_std_dev: float = DEFAULT_WHITE_NOISE_STD_DEV,
        time_step: float = DEFAULT_TIME_STEP,
        sigma: float = DEFAULT_SIGMA,
        beta: float = DEFAULT_BETA,
        rho: float = DEFAULT_RHO,
        master_system_initial_state: Tuple[float, float, float] = DEFAULT_MASTER_INITIAL_STATE,
        driven_system_initial_state: Tuple[float, float, float] = DEFAULT_DRIVEN_INITIAL_STATE
    ):
        """
        Parameters
        -------------
        message: Array or String of bits                                                          
            The message to be encoded and sent using the binary communication. Defaults to a random 64 bit message.
        timeStepsPerSymbol: int
            The amount of time used to send each symbol of the message
        errorThreshold: float
            The amount of desynchronization error required before the bit is decoded as a 1
        sigmaOffset: float
            The amount that the sigma parameter of the transmitting system is increased by when transmitting a 1 bit
        transitionTimeSteps: int
            The amount of time steps waited before averaging, to allow the system to synchronize/desynchronize
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
        """

        self.time_steps_per_symbol = time_steps_per_symbol
        self.error_threshold = error_threshold
        self.transition_time_steps = transition_time_steps
        self.time_step = time_step
        self.white_noise_std_dev = white_noise_std_dev
        self.sigma_offset = sigma_offset
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.master_system_initial_state = master_system_initial_state
        self.driven_system_initial_state = driven_system_initial_state
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate parameter values are reasonable."""
        if self.time_steps_per_symbol <= 0:
            raise ValueError("time_steps_per_symbol must be positive")
        if self.error_threshold < 0:
            raise ValueError("error_threshold must be non-negative")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.transition_time_steps >= self.time_steps_per_symbol:
            raise ValueError("time_steps_per_symbol must be greater than transition_time_steps")

class BinaryCommunication:

    @staticmethod
    def binaryToSquareWave(symbols, timePerSymbol): 
        squareWave = np.zeros(len(symbols)*timePerSymbol)
        for i in range(len(symbols)):
            if int(symbols[i]) == 1:
                squareWave[i*timePerSymbol:(i+1)*timePerSymbol] = np.ones(timePerSymbol)
            elif int(symbols[i]) != 0:
                raise ValueError("symbols must be an array of only 0s and 1s")
        return squareWave

    @staticmethod
    def errorDecoder(error_array, symbol_length, sensitivity):
        num_symbols = int(len(error_array)/symbol_length)
        decoded_message = np.zeros(num_symbols)
        for i in range(num_symbols):
            avg_window_error = np.average(error_array[int((i+0.5)*symbol_length):(i+1)*symbol_length])
            if avg_window_error > sensitivity: decoded_message[i] = 1
        return decoded_message

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
    def BinaryCommunicationTest(test_parameters: BinaryCommunicationParameters, message=None):
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
        
        message_length = len(message)
        total_number_of_time_steps = message_length * test_parameters.time_steps_per_symbol

        m = BinaryCommunication.binaryToSquareWave(message, test_parameters.time_steps_per_symbol)

        master_system = MasterLorenzSystem(test_parameters.master_system_initial_state, test_parameters.time_step, sigma=test_parameters.sigma, rho=test_parameters.rho, beta=test_parameters.beta)
        driven_system = DrivenLorenzSystem(test_parameters.driven_system_initial_state, test_parameters.time_step, sigma=test_parameters.sigma, rho=test_parameters.rho, beta=test_parameters.beta)

        errors = np.empty(total_number_of_time_steps)
        for i in range(total_number_of_time_steps):
            errors[i] = driven_system.getError(master_system.getXState())
            driven_system.nextState(master_system.getXState()+np.random.normal(0, test_parameters.white_noise_std_dev))
            if m[i] >= 0.5: master_system.nextState(test_parameters.sigma_offset)
            else: master_system.nextState()
        
        received_message = BinaryCommunication.errorDecoder(errors, test_parameters.time_steps_per_symbol, test_parameters.error_threshold)

        return(BinaryCommunication.compareMessages(message, received_message))