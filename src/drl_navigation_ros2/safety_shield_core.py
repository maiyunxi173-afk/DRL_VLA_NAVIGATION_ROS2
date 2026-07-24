"""Pure safety-shield logic shared by the ROS 2 node and unit tests."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ShieldDecision:
    """Velocity selected by the shield and the reason for intervention."""

    linear: float
    angular: float
    status: str
    intervened: bool
    minimum_range: float
    front_clearance: float


def _valid_ranges(ranges, maximum_range):
    values = np.asarray(ranges, dtype=np.float32)
    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=float(maximum_range),
        neginf=0.0,
    )


def _sector(values, angle_min, angle_increment, lower, upper):
    if values.size == 0 or not math.isfinite(angle_increment) or angle_increment == 0.0:
        return values
    angles = angle_min + np.arange(values.size) * angle_increment
    return values[(angles >= lower) & (angles <= upper)]


def evaluate_shield(
    ranges,
    angle_min,
    angle_increment,
    policy_linear,
    policy_angular,
    *,
    maximum_range=7.0,
    emergency_stop_distance=0.22,
    avoidance_distance=0.50,
    slow_down_distance=0.80,
    front_half_angle=0.52,
    avoidance_angular_speed=0.50,
    minimum_speed_scale=0.20,
):
    """Project a policy command onto a lightweight lidar-safe action set."""
    values = _valid_ranges(ranges, maximum_range)
    if values.size == 0:
        return ShieldDecision(0.0, 0.0, "invalid_scan", True, 0.0, 0.0)

    minimum_range = float(values.min())
    front = _sector(
        values,
        float(angle_min),
        float(angle_increment),
        -float(front_half_angle),
        float(front_half_angle),
    )
    front_clearance = float(front.min()) if front.size else minimum_range

    if minimum_range < emergency_stop_distance:
        return ShieldDecision(
            0.0,
            0.0,
            "emergency_stop",
            True,
            minimum_range,
            front_clearance,
        )

    if policy_linear > 0.0 and front_clearance < avoidance_distance:
        right = _sector(values, angle_min, angle_increment, -math.pi, 0.0)
        left = _sector(values, angle_min, angle_increment, 0.0, math.pi)
        right_clearance = float(np.percentile(right, 40.0)) if right.size else 0.0
        left_clearance = float(np.percentile(left, 40.0)) if left.size else 0.0
        turn_sign = 1.0 if left_clearance >= right_clearance else -1.0
        return ShieldDecision(
            0.0,
            turn_sign * float(avoidance_angular_speed),
            "avoidance",
            True,
            minimum_range,
            front_clearance,
        )

    if policy_linear > 0.0 and front_clearance < slow_down_distance:
        distance_span = max(slow_down_distance - avoidance_distance, 1e-6)
        ratio = (front_clearance - avoidance_distance) / distance_span
        scale = max(float(minimum_speed_scale), min(1.0, ratio))
        return ShieldDecision(
            float(policy_linear) * scale,
            float(policy_angular),
            "slow_down",
            True,
            minimum_range,
            front_clearance,
        )

    return ShieldDecision(
        float(policy_linear),
        float(policy_angular),
        "pass_through",
        False,
        minimum_range,
        front_clearance,
    )
