import matplotlib.pyplot as plt
import numpy as np

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem

NUM_STEPS = 10**4
TIME_STEP = 0.01
MASTER_INITIAL_STATE = (10.0, 20.0, 30.0)
DRIVEN_INITIAL_STATE = (10.1, 20.2, 29.9)

master_system = MasterLorenzSystem(MASTER_INITIAL_STATE, TIME_STEP)
master_system_states = np.empty((NUM_STEPS+1, 3))
master_system_states[0] = (MASTER_INITIAL_STATE)
driven_system = DrivenLorenzSystem(DRIVEN_INITIAL_STATE, TIME_STEP)
driven_system_states = np.empty((NUM_STEPS+1, 3))
driven_system_states[0] = (DRIVEN_INITIAL_STATE)
errors = np.empty(NUM_STEPS)
times = np.empty(NUM_STEPS)

for i in range(NUM_STEPS):
    master_system_states[i+1] = master_system.nextState()
    driven_system_states[i+1] = driven_system.nextState(master_system_states[i,0])
    errors[i] = np.linalg.norm(master_system_states[i]-driven_system_states[i])
    times[i] = TIME_STEP*i

plt.figure(figsize=(10, 6))
plt.plot(times, errors)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.grid(True, alpha=0.3)

plt.show()