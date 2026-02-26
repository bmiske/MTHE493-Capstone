import numpy as np

class Transmitter:

    @staticmethod
    def binaryToSquareWave(symbols, timePerSymbol): 
        squareWave = np.zeros(len(symbols)*timePerSymbol)
        for i in range(len(symbols)):
            if int(symbols[i]) == 1:
                squareWave[i*timePerSymbol:(i+1)*timePerSymbol] = np.ones(timePerSymbol)
            elif int(symbols[i]) != 0:
                raise ValueError("symbols must be an array of only 0s and 1s")
        return squareWave
