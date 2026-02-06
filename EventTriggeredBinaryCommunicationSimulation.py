# EventTriggeredBinaryCommunicationSimulation.py
# Event-based version of BinaryCommunicationSimulation (variable symbol length)

import numpy as np
import matplotlib.pyplot as plt

from MasterLorenzSystem import MasterLorenzSystem
from DrivenLorenzSystem import DrivenLorenzSystem

# -----------------------------
# Simulation / System Parameters
# -----------------------------
TIME_STEP = 0.01
SIGMA = 16.0
RHO = 45.2
BETA = 4.0
SIGMA_OFFSET = 10.0

MASTER_INITIAL_STATE = np.array((20.0, 10.0, 9.9), dtype=float)
DRIVEN_INITIAL_STATE = np.array((20.1, 10.2, 9.9), dtype=float)

WHITE_NOISE_STDEV = 0.2

# Message
MESSAGE_LEN = 64
message = np.random.randint(0, 2, MESSAGE_LEN)

# -----------------------------
# Event Trigger Parameters
# -----------------------------
# Two-threshold hysteresis regions:
#   R0 (sync):   r(t) <= eps0
#   R1 (desync): r(t) >= eps1
# Must satisfy eps0 < eps1.
eps0 = 0.25      # "confident synchronized" (bit=0 completion condition)
eps1 = 1.25      # "confident desynchronized" (bit=1 completion condition)

# Dwell / confidence time: condition must hold this long before triggering next symbol
Tconf = 0.10     # seconds
Nconf = int(np.ceil(Tconf / TIME_STEP))

# Safety cap: prevents infinite waiting if thresholds too strict / noise too high
MAX_STEPS_PER_SYMBOL = 5000

# -----------------------------
# Initialize Systems
# -----------------------------
master = MasterLorenzSystem(MASTER_INITIAL_STATE.copy(), TIME_STEP, sigma=SIGMA, rho=RHO, beta=BETA)
driven = DrivenLorenzSystem(DRIVEN_INITIAL_STATE.copy(), TIME_STEP, sigma=SIGMA, rho=RHO, beta=BETA)

# -----------------------------
# Logging buffers (variable length)
# -----------------------------
times = [0.0]
master_x = [master.getXState()]
driven_x = [driven.getXState()]
errors = []  # r(t) = |x_m - x_d| at each time step

# For reconstructing square waves with variable symbol lengths
symbol_steps = []     # number of time steps spent per symbol
event_times = [0.0]   # absolute time when each symbol ends (t0=0)

# Decoded bits (receiver output)
received_bits = []

# -----------------------------
# Main event-triggered simulation
# -----------------------------
t = 0.0

for bit in message:
    good_count = 0
    steps_this_symbol = 0
    triggered_bit = None  # what receiver decides for this symbol

    while True:
        # Use current master x as the transmitted driving signal
        xm = master.getXState()

        # Measure sync error BEFORE stepping (consistent with your original test style)
        r = driven.getError(xm)
        errors.append(r)

        # Step driven system with noisy transmitted xm
        driven.nextState(xm + np.random.normal(0, WHITE_NOISE_STDEV))

        # Step master system, modulating sigma when transmitting a 1
        if bit == 1:
            master.nextState(SIGMA_OFFSET)
        else:
            master.nextState(0.0)

        # Advance time and log states AFTER stepping
        t += TIME_STEP
        times.append(t)
        master_x.append(master.getXState())
        driven_x.append(driven.getXState())

        steps_this_symbol += 1

        # Event logic with hysteresis + dwell time
        if bit == 0:
            if r <= eps0:
                good_count += 1
            else:
                good_count = 0

            if good_count >= Nconf:
                triggered_bit = 0  # receiver interprets this as a 0-region trigger
                break

        else:  # bit == 1
            if r >= eps1:
                good_count += 1
            else:
                good_count = 0

            if good_count >= Nconf:
                triggered_bit = 1  # receiver interprets this as a 1-region trigger
                break

        # Safety escape if event never triggers
        if steps_this_symbol >= MAX_STEPS_PER_SYMBOL:
            # Fallback: decide based on where we ended up relative to mid-threshold
            mid = 0.5 * (eps0 + eps1)
            triggered_bit = 1 if r > mid else 0
            break

    symbol_steps.append(steps_this_symbol)
    received_bits.append(triggered_bit)
    event_times.append(t)

received_bits = np.array(received_bits, dtype=int)

# -----------------------------
# Metrics
# -----------------------------
bit_errors = int(np.sum(message != received_bits))
ber = bit_errors / len(message)

print(f"Event-triggered decode complete")
print(f"Bits: {len(message)} | Bit errors: {bit_errors} | BER: {ber:.4f}")
print(f"Avg symbol duration (s): {np.mean(symbol_steps) * TIME_STEP:.3f}")
print(f"Min/Max symbol duration (s): {np.min(symbol_steps) * TIME_STEP:.3f} / {np.max(symbol_steps) * TIME_STEP:.3f}")

# -----------------------------
# Reconstruct square waves for plotting (variable symbol lengths)
# -----------------------------
def expand_bits(bits: np.ndarray, steps_per_symbol: list[int]) -> np.ndarray:
    """Expand per-symbol bits into a per-time-step waveform with variable symbol lengths."""
    if len(bits) != len(steps_per_symbol):
        raise ValueError("bits and steps_per_symbol must have same length")
    wave = np.empty(sum(steps_per_symbol), dtype=float)
    idx = 0
    for b, n in zip(bits, steps_per_symbol):
        wave[idx:idx+n] = float(b)
        idx += n
    return wave

m_wave = expand_bits(message.astype(int), symbol_steps)           # original bits as waveform
rm_wave = expand_bits(received_bits.astype(int), symbol_steps)    # received bits as waveform

# time axis for errors/waves (errors length equals total time steps)
err_t = np.arange(len(errors)) * TIME_STEP

# We logged states with an extra initial sample, so align for plotting:
state_t = np.array(times)

# Symbol boundary times for vertical lines
symbol_boundaries = np.array(event_times)

# -----------------------------
# Plot (same style as your original 4-panel output)
# -----------------------------
fig, axes = plt.subplots(4, 1, figsize=(15, 10))
fig.suptitle("Event-Triggered Lorenz Systems Comparison", fontsize=16)

# 1) Master vs Driven X state
axes[0].plot(state_t, master_x, label="Master System", linewidth=1.5)
axes[0].plot(state_t, driven_x, label="Driven System", linewidth=1.5, alpha=0.8)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("X State")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 2) Original vs Received message (variable-width symbols)
axes[1].plot(err_t, m_wave, label="Original Message", linewidth=1.5)
axes[1].plot(err_t, rm_wave, label="Received Message", linewidth=1.5, alpha=0.8)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Message")
axes[1].set_ylim(-0.1, 1.1)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# 3) Error with thresholds + symbol boundaries
axes[2].plot(err_t, errors, label="|x_m - x_d| (error)", linewidth=1.2)
axes[2].axhline(y=eps0, linestyle="--", label="eps0 (sync)", linewidth=1.2)
axes[2].axhline(y=eps1, linestyle="--", label="eps1 (desync)", linewidth=1.2)

# Symbol boundary lines
ymax = max(np.max(errors), eps1) * 1.05
for st in symbol_boundaries:
    axes[2].vlines(x=st, ymin=0, ymax=ymax, linestyle="--", alpha=0.25)

# Overlay original message for visual alignment (scaled)
axes[2].plot(err_t, m_wave * (0.6 * ymax), label="Original (scaled)", linewidth=0.6, alpha=0.7)

axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Error")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

# 4) Received message only
axes[3].plot(err_t, rm_wave, label="Received Message", linewidth=1.5)
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Received")
axes[3].set_ylim(-0.1, 1.1)
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.savefig("event_trigger_binary_communication.png", dpi=200)
plt.close()
