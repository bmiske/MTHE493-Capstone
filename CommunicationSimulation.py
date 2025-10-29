import matplotlib.pyplot as plt
import numpy as np
import math

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem
import MessageEncoder

NUM_STEPS = 10**4
TIME_STEP = 0.01
MASTER_INITIAL_STATE = (20.0, -20.0, 30.0)
DRIVEN_INITIAL_STATE = (-20.1, 10.2, 9.9)

MESSAGE_ENCODING_DELAY = 500
MESSAGE_AMPLITUDE = 0.1

master_system = MasterLorenzSystem(MASTER_INITIAL_STATE, TIME_STEP)
master_system_states = np.empty((NUM_STEPS+1, 3))
master_system_states[0] = (MASTER_INITIAL_STATE)
driven_system = DrivenLorenzSystem(DRIVEN_INITIAL_STATE, TIME_STEP)
driven_system_states = np.empty((NUM_STEPS+1, 3))
driven_system_states[0] = (DRIVEN_INITIAL_STATE)
messages = np.empty((NUM_STEPS, 3)) #Stores transmitted message in (:,0) and received in (:,1)
errors = np.empty(NUM_STEPS)
times = np.empty(NUM_STEPS)

for i in range(NUM_STEPS):
    master_system_states[i+1] = master_system.nextState()
    if i > MESSAGE_ENCODING_DELAY:
        transmittedSignal = MessageEncoder.encodeMessage(MESSAGE_AMPLITUDE*math.sin(TIME_STEP*i), master_system_states[i,0])
    else:
        transmittedSignal =  master_system_states[i,0]
    messages[i,0] = MESSAGE_AMPLITUDE*math.sin(TIME_STEP*i)
    driven_system_states[i+1] = driven_system.nextState(transmittedSignal)
    estimatedMessage = MessageEncoder.decodeMessage(transmittedSignal, driven_system_states[i,0])
    messages[i,1] = estimatedMessage
    times[i] = TIME_STEP*i

def SimpleMovingAverage(windowSize, data):
    weights = np.ones(windowSize) / windowSize
    averagedData = np.convolve(data, weights, mode='same')
    return averagedData

messages[:,2] = SimpleMovingAverage(17, messages[:,1])
times_extended = np.append(times, NUM_STEPS * TIME_STEP)
unprocessed_message_error = np.abs(messages[:,0]-messages[:,1])
averaged_message_error = np.abs(messages[:,0]-messages[:,2])
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
fig.suptitle('Lorenz Systems Comparison', fontsize=16)

axes[0].plot(times_extended, master_system_states[:, 0], label='Master System', color='blue', linewidth=1.5)
axes[0].plot(times_extended, driven_system_states[:, 0], label='Driven System', color='red', linewidth=1.5, alpha=0.8)
axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('X State', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(times[MESSAGE_ENCODING_DELAY:], messages[MESSAGE_ENCODING_DELAY:, 0], label='Original Message', color='blue', linewidth=1.5)
#axes[1].plot(times[MESSAGE_ENCODING_DELAY:], messages[MESSAGE_ENCODING_DELAY:, 1], label='Received Message', color='red', linewidth=1.5, alpha=0.8)
axes[1].plot(times[MESSAGE_ENCODING_DELAY:], messages[MESSAGE_ENCODING_DELAY:, 2], label='Received Message', color='green', linewidth=1.5, alpha=0.8)
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Message Recovery', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(times[MESSAGE_ENCODING_DELAY:], unprocessed_message_error[MESSAGE_ENCODING_DELAY:], label='Message Decoding Error', color='blue', linewidth=1.5)
axes[2].plot(times[MESSAGE_ENCODING_DELAY:], averaged_message_error[MESSAGE_ENCODING_DELAY:], label='Message Decoding Error', color='green', linewidth=1.5)
axes[2].set_xlabel('Time', fontsize=12)
axes[2].set_ylabel('Message Error', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

