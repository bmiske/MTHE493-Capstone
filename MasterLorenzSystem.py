import numpy as np

class MasterLorenzSystem:

    def __init__(self, initial_xyz, timestep, *, sigma=10, rho=28, beta=2.667):
        self.xyz = initial_xyz
        self.timestep = timestep
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def nextState(self, increaseToSigma = 0):
        self.xyz += self.timestep*self.getDerivatives(self.xyz, increaseToSigma)
        return(self.xyz)

    def getDerivatives(self, xyz, increaseToSigma = 0):
        x, y, z = xyz
        x_dot = (self.sigma+increaseToSigma)*(y - x)
        y_dot = self.rho*x - y - x*z
        z_dot = x*y - self.beta*z
        return np.array([x_dot, y_dot, z_dot])