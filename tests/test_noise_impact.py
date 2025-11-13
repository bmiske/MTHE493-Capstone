import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TrialSimulation import run_simulation
from NoiseRobustnessStudy import sweep_noise_levels


class NoiseImpactTests(unittest.TestCase):
    def test_noise_increases_final_error(self):
        clean = run_simulation(num_steps=2000, burn_in_steps=500, noise_std=0.0, random_seed=123)
        noisy = run_simulation(num_steps=2000, burn_in_steps=500, noise_std=2.0, random_seed=123)

        self.assertGreater(noisy["errors"][-1], clean["errors"][-1])

    def test_random_seed_reproducibility(self):
        first = run_simulation(num_steps=2000, burn_in_steps=500, noise_std=1.0, random_seed=999)
        second = run_simulation(num_steps=2000, burn_in_steps=500, noise_std=1.0, random_seed=999)

        self.assertTrue(np.allclose(first["errors"], second["errors"]))

    def test_sweep_noise_levels_reports_all_entries(self):
        levels = [0.0, 0.5, 1.0]
        results = sweep_noise_levels(levels, num_trials=2, num_steps=1000, burn_in_steps=200, sample_stride=20)

        self.assertEqual(len(results), len(levels))
        self.assertTrue(all("mean_final_error" in r for r in results))


if __name__ == "__main__":
    unittest.main()
