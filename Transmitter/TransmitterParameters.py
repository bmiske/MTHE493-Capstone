from typing import Tuple

class TransmitterParameters:

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
    DEFAULT_INITIAL_STATE = (20.0, 10.0, 9.9)

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
        initial_state: Tuple[float, float, float] = DEFAULT_INITIAL_STATE,
    ):
        """
        Parameters
        -------------
        message: Array or String of bits                                                          
            The message to be encoded and sent using the binary communication. Defaults to a random 64 bit message.
        time_steps_per_symbol: int
            The amount of time used to send each symbol of the message
        error_threshold: float
            The amount of desynchronization error required before the bit is decoded as a 1
        sigma_offset: float
            The amount that the sigma parameter of the transmitting system is increased by when transmitting a 1 bit
        transition_time_steps: int
            The amount of time steps waited before averaging, to allow the system to synchronize/desynchronize
        white_noise_std_dev: float
            The standard deviation of the gaussian white noise added to the transmitted signal
        time_step: float
            How much time passes inbetween each iteration of the chaotic systems
        sigma: float
            A parameter used for the Lorenz System. Sigma of the transmitting system is increased by sigmaOffset when transmitting a 1
        beta: float
            A parameter used for the Lorenz System.
        rho: float
            A parameter used for the Lorenz System.
        initial_state: float[3]
            An ordered list contiaing the intial x, y, and z states of the transmitting system
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
        self.initial_state = initial_state
        
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