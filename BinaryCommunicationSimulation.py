import matplotlib.pyplot as plt
import numpy as np

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem
from BinaryCommunication import BinaryCommunication

TIME_STEP = 0.01
SIGMA = 16.0
RHO = 45.2
BETA = 4
SIGMA_OFFSET = 10.0
MASTER_INITIAL_STATE = (20.0, 10.0, 9.9)
DRIVEN_INITIAL_STATE = (20.1, 10.2, 9.9)
WHITE_NOISE_STDEV = 0.2
ERROR_SENSITIVITY = 0.35

MESSAGE_LEN = 64
TIME_PER_SYMBOL = 120
TRANSITION_TIME = TIME_PER_SYMBOL/2
message = np.random.randint(0, 2, MESSAGE_LEN)
COMMUNICATION_TIME = MESSAGE_LEN * TIME_PER_SYMBOL

master_system = MasterLorenzSystem(MASTER_INITIAL_STATE, TIME_STEP, sigma=SIGMA, rho=RHO, beta=BETA)
master_system_states = np.empty((COMMUNICATION_TIME+1, 3))
master_system_states[0] = (MASTER_INITIAL_STATE)
driven_system = DrivenLorenzSystem(DRIVEN_INITIAL_STATE, TIME_STEP, sigma=SIGMA, rho=RHO, beta=BETA)
driven_system_states = np.empty((COMMUNICATION_TIME+1, 3))
driven_system_states[0] = (DRIVEN_INITIAL_STATE)

m = BinaryCommunication.binaryToSquareWave(message, TIME_PER_SYMBOL)

errors = np.empty(COMMUNICATION_TIME)
times = np.empty(COMMUNICATION_TIME)

for i in range(COMMUNICATION_TIME):
    if m[i] >= 0.5: master_system_states[i+1] = master_system.nextState(SIGMA_OFFSET)
    else: master_system_states[i+1] = master_system.nextState()
    errors[i] = driven_system.getError(master_system_states[i,0])
    driven_system_states[i+1] = driven_system.nextState(master_system_states[i,0]+np.random.normal(0, WHITE_NOISE_STDEV))
    times[i] = i*TIME_STEP

receivedMessage = BinaryCommunication.errorDecoder(errors, TIME_PER_SYMBOL, TRANSITION_TIME, ERROR_SENSITIVITY)
times_extended = np.append(times, COMMUNICATION_TIME * TIME_STEP)
rm = BinaryCommunication.binaryToSquareWave(receivedMessage, TIME_PER_SYMBOL)

symbol_start_times = np.arange(0, COMMUNICATION_TIME*TIME_STEP, TIME_PER_SYMBOL*TIME_STEP)

fig, axes = plt.subplots(4, 1, figsize=(15, 10))
fig.suptitle('Lorenz Systems Comparison', fontsize=16)

print(BinaryCommunication.compareMessages(message, receivedMessage))

axes[0].plot(times_extended, master_system_states[:, 0], label='Master System', color='blue', linewidth=1.5)
axes[0].plot(times_extended, driven_system_states[:, 0], label='Driven System', color='red', linewidth=1.5, alpha=0.8)
axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('X State', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(times, rm, label='Received Message', color='red', linewidth=1.5)
axes[1].plot(times, m, label='Original Message', color='blue', linewidth=1.5)
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Original Message', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(times, errors, label='Message Decoding Error', color='blue', linewidth=1.5)
axes[2].axhline(y=ERROR_SENSITIVITY, color='g', linestyle='--', label='Threshold')
axes[2].vlines(x=symbol_start_times, ymin=0, ymax=5, color='r', linestyle='--')
axes[2].plot(times, m*8, label='Original Message', color='blue', linewidth=0.5)
axes[2].set_xlabel('Time', fontsize=12)
axes[2].set_ylabel('X-State Error', fontsize=12)
axes[2].grid(True, alpha=0.3)

axes[3].plot(times, rm, label='Received Message', color='red', linewidth=1.5)
axes[3].set_xlabel('Time', fontsize=12)
axes[3].set_ylabel('Received Message', fontsize=12)
axes[3].grid(True, alpha=0.3)

# display simulation in windows
#plt.tight_layout()
#plt.show()

# display simulation in linux
plt.tight_layout()
plt.savefig("binary_communication.png", dpi=200)
plt.close()