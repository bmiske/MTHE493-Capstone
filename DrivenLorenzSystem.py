import numpy as np

class DrivenLorenzSystem:

    def __init__(self, initial_xyz, timestep, *, sigma=10, rho=28, beta=2.667):
        self.xyz = initial_xyz
        self.timestep = timestep
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def nextState(self, driving_signal):
        self.xyz += self.timestep*self.getDerivatives(self.xyz, driving_signal)
        return(self.xyz)

    def getDerivatives(self, xyz, driving_signal):
        x, y, z = xyz
        x_dot = self.sigma*(y - x)
        y_dot = self.rho*driving_signal - y - x*z
        z_dot = driving_signal*y - self.beta*z
        return np.array([x_dot, y_dot, z_dot])
    
    def getError(self, driving_signal):
        return np.abs(driving_signal-self.xyz[0])