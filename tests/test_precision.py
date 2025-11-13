import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DrivenLorenzSystem import DrivenLorenzSystem
from MasterLorenzSystem import MasterLorenzSystem
from TrialSimulation import run_simulation


class PrecisionTests(unittest.TestCase):
    def test_master_system_emits_64bit_state(self):
        """Master system should maintain double precision state values."""
        system = MasterLorenzSystem(initial_xyz=(1.0, -2.0, 3.5), timestep=0.01)

        state = system.nextState()

        self.assertEqual(state.dtype, np.float64)

    def test_driven_system_accepts_64bit_encoded_signal(self):
        """Driven system must accept externally encoded 64-bit data."""
        system = DrivenLorenzSystem(initial_xyz=(0.5, 0.2, -0.1), timestep=0.01)
        driving_signal = np.uint64(0x3FF0000000000000)  # 64-bit encoded representation of 1.0

        state = system.nextState(driving_signal)

        self.assertEqual(state.dtype, np.float64)

    def test_run_simulation_returns_float64_arrays(self):
        """Simulation outputs should stay in float64 to avoid precision loss when sending data."""
        results = run_simulation(num_steps=200, burn_in_steps=20, sample_stride=10)

        self.assertEqual(results["times"].dtype, np.float64)
        self.assertEqual(results["master_states"].dtype, np.float64)
        self.assertEqual(results["driven_states"].dtype, np.float64)
        self.assertEqual(results["error_times"].dtype, np.float64)
        self.assertEqual(results["errors"].dtype, np.float64)


class SynchronizationTests(unittest.TestCase):
    def test_master_and_driven_x_states_highly_correlated(self):
        """Strong correlation after burn-in demonstrates practical synchronization."""
        results = run_simulation(num_steps=4000, burn_in_steps=1000, sample_stride=50)
        master_states = results["master_states"]
        driven_states = results["driven_states"]

        tail_start = len(master_states)//2
        master_tail = master_states[tail_start:, 0]
        driven_tail = driven_states[tail_start:, 0]

        correlation_matrix = np.corrcoef(master_tail, driven_tail)
        correlation = correlation_matrix[0, 1]

        self.assertGreater(correlation, 0.99)

    def test_driven_system_converges_after_burn_in(self):
        """With standard parameters the driven system should synchronize after burn-in."""
        results = run_simulation(num_steps=4000, burn_in_steps=1000, sample_stride=50)

        final_error = results["errors"][-1]
        self.assertLess(final_error, 1.0)


if __name__ == "__main__":
    unittest.main()
