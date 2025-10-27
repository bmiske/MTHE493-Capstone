import matplotlib.pyplot as plt
import numpy as np

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem

NUM_STEPS = 10**4
TIME_STEP = 0.01
MASTER_INITIAL_STATE = (10.0, -20.0, 30.0)
DRIVEN_INITIAL_STATE = (-10.1, 10.2, 9.9)

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

times_extended = np.append(times, NUM_STEPS * TIME_STEP)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Lorenz Systems Comparison', fontsize=16)

axes[0, 0].plot(times_extended, master_system_states[:, 0], label='Master System', color='blue', linewidth=1.5)
axes[0, 0].plot(times_extended, driven_system_states[:, 0], label='Driven System', color='red', linewidth=1.5, alpha=0.8)
axes[0, 0].set_xlabel('Time', fontsize=12)
axes[0, 0].set_ylabel('X State', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(times_extended, master_system_states[:, 1], label='Master System', color='blue', linewidth=1.5)
axes[0, 1].plot(times_extended, driven_system_states[:, 1], label='Driven System', color='red', linewidth=1.5, alpha=0.8)
axes[0, 1].set_xlabel('Time', fontsize=12)
axes[0, 1].set_ylabel('Y State', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(times_extended, master_system_states[:, 2], label='Master System', color='blue', linewidth=1.5)
axes[1, 0].plot(times_extended, driven_system_states[:, 2], label='Driven System', color='red', linewidth=1.5, alpha=0.8)
axes[1, 0].set_xlabel('Time', fontsize=12)
axes[1, 0].set_ylabel('Z State', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(times, errors, color='green', linewidth=2)
axes[1, 1].set_xlabel('Time', fontsize=12)
axes[1, 1].set_ylabel('Error', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()