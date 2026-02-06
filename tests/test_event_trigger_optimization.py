import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_triggered.event_triggered_simulation import EventTriggerParams, run_event_triggered_message
from event_triggered.threshold_optimization import (
    CEMConfig,
    build_random_bits,
    optimize_thresholds,
    optimize_thresholds_cem,
)


class EventTriggeredSimulationTests(unittest.TestCase):
    def test_confidence_window_controls_symbol_duration_in_immediate_region(self):
        params = EventTriggerParams(epsilon0=100.0, epsilon1=101.0, tconf=0.07, delta_sigma=0.0)
        timestep = 0.01
        expected_duration = np.ceil(params.tconf / timestep)*timestep

        result = run_event_triggered_message(
            [0],
            params,
            timestep=timestep,
            noise_std=0.0,
            random_seed=1,
            max_symbol_steps=1000,
        )

        self.assertAlmostEqual(result["symbol_durations"][0], expected_duration, places=8)
        self.assertEqual(result["tx_trigger_failure_rate"], 0.0)

    def test_output_shapes_and_ranges(self):
        params = EventTriggerParams(epsilon0=0.5, epsilon1=5.0, tconf=0.03, delta_sigma=8.0)
        bits = [0, 1, 1, 0, 1]

        result = run_event_triggered_message(
            bits,
            params,
            noise_std=0.1,
            random_seed=2,
            max_symbol_steps=1500,
        )

        self.assertEqual(len(result["transmitted_bits"]), len(bits))
        self.assertEqual(len(result["decoded_bits"]), len(bits))
        self.assertEqual(result["symbol_durations"].shape[0], len(bits))
        self.assertGreaterEqual(result["ber"], 0.0)
        self.assertLessEqual(result["ber"], 1.0)

    def test_abs_dx_floor_filter_prevents_metric_collapse(self):
        params = EventTriggerParams(epsilon0=1.0, epsilon1=5.0, tconf=0.03, delta_sigma=6.0)
        bits = [1, 1, 1]

        result = run_event_triggered_message(
            bits,
            params,
            metric="abs_dx",
            use_metric_filter=False,
            abs_dx_floor_ratio=1.0,
            noise_std=0.0,
            random_seed=4,
            max_symbol_steps=1000,
        )

        self.assertAlmostEqual(result["max_metric_error"], result["max_error"], places=8)


class ThresholdOptimizationTests(unittest.TestCase):
    def test_optimizer_returns_sorted_candidates(self):
        bits = build_random_bits(8, seed=7)

        search = optimize_thresholds(
            epsilon0_values=[0.5, 1.0],
            epsilon1_values=[5.0, 8.0],
            tconf_values=[0.02, 0.03],
            delta_sigma_values=[8.0],
            message_bits=bits,
            noise_std=0.1,
            random_seed=7,
            max_symbol_steps=1000,
            top_k=3,
        )

        all_scores = [row["score"] for row in search["all_results"]]
        self.assertEqual(all_scores, sorted(all_scores))
        self.assertEqual(search["best"]["score"], min(all_scores))
        self.assertEqual(len(search["top_k"]), 3)

    def test_cem_optimizer_returns_ranked_results(self):
        bits = build_random_bits(8, seed=13)
        search = optimize_thresholds_cem(
            bounds={
                "epsilon0": (0.5, 2.0),
                "epsilon1": (5.0, 10.0),
                "tconf": (0.02, 0.08),
                "delta_sigma": (2.0, 6.0),
            },
            message_bits=bits,
            noise_std=0.1,
            random_seed=13,
            evaluation_seeds=[13, 14],
            evaluation_noise_levels=[0.1, 0.2],
            max_symbol_steps=1000,
            top_k=3,
            config=CEMConfig(iterations=3, population_size=8, elite_fraction=0.25, smoothing=0.6),
        )

        self.assertIsNotNone(search["best"])
        all_scores = [row["score"] for row in search["all_results"]]
        self.assertEqual(all_scores, sorted(all_scores))
        self.assertEqual(search["best"]["score"], min(all_scores))
        self.assertGreaterEqual(len(search["history"]), 1)
        self.assertEqual(search["best"]["num_eval_runs"], 4)
        self.assertIn("ber_std", search["best"])


if __name__ == "__main__":
    unittest.main()
