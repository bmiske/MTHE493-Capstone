import numpy as np


class MasterLorenzSystem:
    def __init__(self, initial_xyz, timestep, *, sigma=10, rho=28, beta=2.667):
        self.xyz = np.array(initial_xyz, dtype=float, copy=True)
        self.timestep = timestep
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def nextState(self):
        self.xyz = self._rk4_step(self.xyz)
        return self.xyz.copy()

    def _rk4_step(self, xyz):
        dt = self.timestep
        k1 = self.getDerivatives(xyz)
        k2 = self.getDerivatives(xyz + 0.5*dt*k1)
        k3 = self.getDerivatives(xyz + 0.5*dt*k2)
        k4 = self.getDerivatives(xyz + dt*k3)
        return xyz + (dt / 6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def getDerivatives(self, xyz):
        x, y, z = xyz
        x_dot = self.sigma*(y - x)
        y_dot = self.rho*x - y - x*z
        z_dot = x*y - self.beta*z
        return np.array([x_dot, y_dot, z_dot])
