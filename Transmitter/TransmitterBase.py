import numpy as np

from TransmitterParameters import TransmitterParameters
from MasterLorenzSystem import MasterLorenzSystem
from transmitter import transmitter

class TransmitterBase:

    def __init__(self, top_block: transmitter, parameters: TransmitterParameters):
        self.parameters = parameters
        self.top_block = top_block
        self.system = MasterLorenzSystem(parameters.initial_state, parameters.time_step, sigma=parameters.sigma, rho=parameters.rho, beta=parameters.beta)

    def transmitMessage(self, message):
        message_length = len(message)
        message_square_wave = self.binaryToSquareWave(message, self.parameters.time_steps_per_symbol)
        total_number_of_time_steps = message_length * self.parameters.time_steps_per_symbol

        for i in range(total_number_of_time_steps):
            current_x_state = self.system.getXState()
            freq_offset = self.determineFreqOffset(current_x_state)
            self.top_block.set_delta_freq(freq_offset)
            if message_square_wave[i] >= 0.5: self.system.nextState(self.parameters.sigma_offset)
            else: self.system.nextState()

    def determineFreqOffset(self, x_state: float):
        if x_state <= (-1)*self.parameters.x_max: return (-1)*self.parameters.freq_offset_max_hz
        elif x_state >= self.parameters.x_max: return self.parameters.freq_offset_max_hz
        else: return (x_state/self.parameters.x_max)*self.parameters.freq_offset_max_hz

        

    @staticmethod
    def binaryToSquareWave(symbols, time_steps_per_symbol): 
        squareWave = np.zeros(len(symbols)*time_steps_per_symbol)
        for i in range(len(symbols)):
            if int(symbols[i]) == 1:
                squareWave[i*time_steps_per_symbol:(i+1)*time_steps_per_symbol] = np.ones(time_steps_per_symbol)
            elif int(symbols[i]) != 0:
                raise ValueError("symbols must be an array of only 0s and 1s")
        return squareWave
