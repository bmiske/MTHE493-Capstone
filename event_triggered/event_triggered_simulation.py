from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_MASTER_INITIAL_STATE = (10.0, -20.0, 30.0)
DEFAULT_DRIVEN_INITIAL_STATE = (-10.1, 10.2, 9.9)


@dataclass(frozen=True)
class EventTriggerParams:
    epsilon0: float
    epsilon1: float
    tconf: float
    delta_sigma: float

    def validate(self) -> None:
        if not (self.epsilon0 > 0.0):
            raise ValueError("epsilon0 must be > 0")
        if not (self.epsilon1 > self.epsilon0):
            raise ValueError("epsilon1 must be > epsilon0")
        if not (self.tconf > 0.0):
            raise ValueError("tconf must be > 0")
        if not (self.delta_sigma >= 0.0):
            raise ValueError("delta_sigma must be >= 0")


class _MasterLorenz:
    def __init__(self, initial_xyz, timestep, sigma=10.0, rho=28.0, beta=2.667):
        self.xyz = np.array(initial_xyz, dtype=float, copy=True)
        self.timestep = float(timestep)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)

    def next_state(self, sigma_offset: float) -> np.ndarray:
        self.xyz = self._rk4_step(self.xyz, sigma_offset)
        return self.xyz.copy()

    def _rk4_step(self, xyz: np.ndarray, sigma_offset: float) -> np.ndarray:
        dt = self.timestep
        k1 = self._derivatives(xyz, sigma_offset)
        k2 = self._derivatives(xyz + 0.5*dt*k1, sigma_offset)
        k3 = self._derivatives(xyz + 0.5*dt*k2, sigma_offset)
        k4 = self._derivatives(xyz + dt*k3, sigma_offset)
        return xyz + (dt / 6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def _derivatives(self, xyz: np.ndarray, sigma_offset: float) -> np.ndarray:
        x, y, z = xyz
        sigma = self.sigma + sigma_offset
        x_dot = sigma*(y - x)
        y_dot = self.rho*x - y - x*z
        z_dot = x*y - self.beta*z
        return np.array([x_dot, y_dot, z_dot], dtype=float)


class _DrivenLorenz:
    def __init__(self, initial_xyz, timestep, sigma=10.0, rho=28.0, beta=2.667):
        self.xyz = np.array(initial_xyz, dtype=float, copy=True)
        self.timestep = float(timestep)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)

    def next_state(self, driving_signal: float) -> np.ndarray:
        self.xyz = self._rk4_step(self.xyz, driving_signal)
        return self.xyz.copy()

    def _rk4_step(self, xyz: np.ndarray, driving_signal: float) -> np.ndarray:
        dt = self.timestep
        k1 = self._derivatives(xyz, driving_signal)
        k2 = self._derivatives(xyz + 0.5*dt*k1, driving_signal)
        k3 = self._derivatives(xyz + 0.5*dt*k2, driving_signal)
        k4 = self._derivatives(xyz + dt*k3, driving_signal)
        return xyz + (dt / 6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def _derivatives(self, xyz: np.ndarray, driving_signal: float) -> np.ndarray:
        x, y, z = xyz
        x_dot = self.sigma*(y - x)
        y_dot = self.rho*driving_signal - y - x*z
        z_dot = driving_signal*y - self.beta*z
        return np.array([x_dot, y_dot, z_dot], dtype=float)


def _error_metric(master_state: np.ndarray, driven_state: np.ndarray, metric: str) -> float:
    delta = master_state - driven_state
    if metric == "norm":
        return float(np.linalg.norm(delta))
    if metric == "abs_dx":
        return float(abs(delta[0]))
    raise ValueError("metric must be one of {'norm', 'abs_dx'}")


def _decode_from_error(error_value: float, params: EventTriggerParams) -> int:
    if error_value <= params.epsilon0:
        return 0
    if error_value >= params.epsilon1:
        return 1
    midpoint = 0.5*(params.epsilon0 + params.epsilon1)
    return int(error_value >= midpoint)


def run_event_triggered_message(
    message_bits,
    params: EventTriggerParams,
    *,
    timestep: float = 0.01,
    metric: str = "norm",
    use_metric_filter: bool = True,
    metric_filter_alpha: float = 0.3,
    abs_dx_floor_ratio: float = 0.2,
    noise_std: float = 0.0,
    random_seed: int | None = None,
    master_initial_state=DEFAULT_MASTER_INITIAL_STATE,
    driven_initial_state=DEFAULT_DRIVEN_INITIAL_STATE,
    max_symbol_steps: int = 12_000,
):
    """
    Simulate bit transmission where symbol duration is event-triggered by synchronization error.

    The transmitter switches to the next symbol once the confidence condition corresponding to
    the current bit is met. Receiver decoding is tracked independently using whichever acceptance
    region (R0 or R1) reaches confidence first.
    """
    params.validate()
    bits = [int(b) for b in message_bits]
    if any(b not in (0, 1) for b in bits):
        raise ValueError("message_bits must contain only 0/1 values")
    if max_symbol_steps <= 0:
        raise ValueError("max_symbol_steps must be > 0")
    if not (0.0 < metric_filter_alpha <= 1.0):
        raise ValueError("metric_filter_alpha must be in (0, 1]")
    if abs_dx_floor_ratio < 0.0:
        raise ValueError("abs_dx_floor_ratio must be >= 0")

    conf_steps = max(1, int(np.ceil(params.tconf / timestep)))
    master = _MasterLorenz(master_initial_state, timestep)
    driven = _DrivenLorenz(driven_initial_state, timestep)
    rng = np.random.default_rng(random_seed)

    decoded_bits = []
    symbol_durations = []
    tx_trigger_failures = 0
    rx_timeout_count = 0
    global_max_error = 0.0
    global_max_metric_error = 0.0

    for bit in bits:
        sigma_offset = params.delta_sigma*bit
        steps_taken = 0
        in_r0_count = 0
        in_r1_count = 0
        filtered_metric = None

        tx_finished = False
        rx_decoded = None

        while steps_taken < max_symbol_steps and not tx_finished:
            master_state = master.next_state(sigma_offset)
            driving_signal = master_state[0]
            if noise_std > 0.0:
                driving_signal = driving_signal + rng.normal(0.0, noise_std)
            driven_state = driven.next_state(driving_signal)

            delta = master_state - driven_state
            norm_error = float(np.linalg.norm(delta))
            raw_metric = _error_metric(master_state, driven_state, metric)
            if metric == "abs_dx":
                # Prevent false low-error decisions when x-components merely cross.
                raw_metric = max(raw_metric, abs_dx_floor_ratio*norm_error)

            if use_metric_filter:
                if filtered_metric is None:
                    filtered_metric = raw_metric
                else:
                    filtered_metric = (
                        metric_filter_alpha*raw_metric
                        + (1.0 - metric_filter_alpha)*filtered_metric
                    )
                error_value = filtered_metric
            else:
                error_value = raw_metric

            global_max_error = max(global_max_error, norm_error)
            global_max_metric_error = max(global_max_metric_error, error_value)

            if error_value <= params.epsilon0:
                in_r0_count += 1
            else:
                in_r0_count = 0

            if error_value >= params.epsilon1:
                in_r1_count += 1
            else:
                in_r1_count = 0

            if rx_decoded is None:
                if in_r0_count >= conf_steps:
                    rx_decoded = 0
                elif in_r1_count >= conf_steps:
                    rx_decoded = 1

            if bit == 0 and in_r0_count >= conf_steps:
                tx_finished = True
            elif bit == 1 and in_r1_count >= conf_steps:
                tx_finished = True

            steps_taken += 1

        if not tx_finished:
            tx_trigger_failures += 1

        if rx_decoded is None:
            rx_timeout_count += 1
            rx_decoded = _decode_from_error(error_value, params)

        decoded_bits.append(rx_decoded)
        symbol_durations.append(steps_taken*timestep)

    decoded = np.array(decoded_bits, dtype=int)
    transmitted = np.array(bits, dtype=int)
    durations = np.array(symbol_durations, dtype=float)

    return {
        "transmitted_bits": transmitted,
        "decoded_bits": decoded,
        "ber": float(np.mean(decoded != transmitted)) if transmitted.size else 0.0,
        "symbol_durations": durations,
        "avg_symbol_duration": float(np.mean(durations)) if durations.size else 0.0,
        "max_symbol_duration": float(np.max(durations)) if durations.size else 0.0,
        "tx_trigger_failure_rate": float(tx_trigger_failures / max(1, len(bits))),
        "rx_timeout_rate": float(rx_timeout_count / max(1, len(bits))),
        "max_error": float(global_max_error),
        "max_metric_error": float(global_max_metric_error),
        "params": params,
    }
