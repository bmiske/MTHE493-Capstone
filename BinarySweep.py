import numpy as np

from BinaryCommunication import BinaryCommunication
from BinaryCommunication import BinaryCommunicationParameters

def testConfig(symbolTime, threshold, sigma, noise, trials):
    errors = 0
    for _ in range(trials):
        params = BinaryCommunicationParameters(
            time_steps_per_symbol = symbolTime,
            error_threshold = threshold,
            sigma_offset = sigma,
            white_noise_std_dev = noise
        )
        errors += BinaryCommunication.BinaryCommunicationTest(params)
    return errors / (64 * trials)

# adjust parameters for testing
# be aware that the longer the lists, the longer the runtime
symbolTimeList = [120, 140, 160, 180, 200, 220, 240]
thresholdList = np.linspace(0.15, 0.6, 10)
sigmaList = [5, 7, 10, 12]
noise = 0.2
trials = 10

best = None
for i in symbolTimeList:
    for j in thresholdList:
        for k in sigmaList:
            bitErrorRate = testConfig(i, j, k, noise, trials)
            print(f"Time Per Symbol = {i}, Threshold = {j:.3f}, Sigma Offset = {k} → Bit Error Rate={bitErrorRate:.4f}")
            if best is None or (bitErrorRate < best[0]) or (bitErrorRate == best[0] and i < best[1]):
                best = (bitErrorRate, i, j, k)

print("\nBest Configuration:")
print(f"Bit Error Rate = {best[0]:.4f}")
print(f"Time Per Symbol = {best[1]}")
print(f"Threshold = {best[2]:.3f}")
print(f"Sigma Offset = {best[3]}")