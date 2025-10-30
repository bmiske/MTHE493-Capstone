import matplotlib.pyplot as plt
import numpy as np
import math

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem
import MessageEncoder

NUM_STEPS = 10**4
TIME_STEP = 0.01
SIGMA = 16.0
RHO = 45.2
BETA = 4
SIGMA_OFFSET = 4
MASTER_INITIAL_STATE = (20.0, -20.0, 30.0)
DRIVEN_INITIAL_STATE = (-20.1, 10.2, 9.9)

def binaryToSquareWave(symbols, timePerSymbol):
    lengthOfWave = len(symbols)*timePerSymbol
    squareWave = np.zeros(lengthOfWave)
    for i in range(len(symbols)):
        if symbols[i] == '1':
            squareWave[i*timePerSymbol:(i+1)*timePerSymbol] = np.ones(timePerSymbol)
    return squareWave

master_system = MasterLorenzSystem(MASTER_INITIAL_STATE, TIME_STEP, sigma=SIGMA, rho=RHO, beta=BETA)
master_system_states = np.empty((NUM_STEPS+1, 3))
master_system_states[0] = (MASTER_INITIAL_STATE)
driven_system = DrivenLorenzSystem(DRIVEN_INITIAL_STATE, TIME_STEP, sigma=SIGMA, rho=RHO, beta=BETA)
driven_system_states = np.empty((NUM_STEPS+1, 3))
driven_system_states[0] = (DRIVEN_INITIAL_STATE)

message = '1011010011011100'
m = binaryToSquareWave(message, math.floor(NUM_STEPS/len(message)))

errors = np.empty(NUM_STEPS)
times = np.empty(NUM_STEPS)

for i in range(NUM_STEPS):
    if m[i] >= 0.5: master_system_states[i+1] = master_system.nextState(SIGMA_OFFSET)
    else: master_system_states[i+1] = master_system.nextState()
    errors[i] = driven_system.getError(master_system_states[i,0])
    driven_system_states[i+1] = driven_system.nextState(master_system_states[i,0])
    times[i] = i*TIME_STEP

times_extended = np.append(times, NUM_STEPS * TIME_STEP)

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
fig.suptitle('Lorenz Systems Comparison', fontsize=16)

axes[0].plot(times_extended, master_system_states[:, 0], label='Master System', color='blue', linewidth=1.5)
axes[0].plot(times_extended, driven_system_states[:, 0], label='Driven System', color='red', linewidth=1.5, alpha=0.8)
axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('X State', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(times, m, label='Original Message', color='blue', linewidth=1.5)
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Original Message', fontsize=12)
axes[1].grid(True, alpha=0.3)

axes[2].plot(times, errors, label='Message Decoding Error', color='blue', linewidth=1.5)
axes[2].set_xlabel('Time', fontsize=12)
axes[2].set_ylabel('Message Error', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

