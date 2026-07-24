"""Unit tests for the ROS-independent safety-shield projection."""

from pathlib import Path
import sys

import numpy as np


SOURCE_DIRECTORY = Path(__file__).parents[1] / "src" / "drl_navigation_ros2"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from safety_shield_core import evaluate_shield  # noqa: E402


ANGLE_MIN = -np.pi
ANGLE_INCREMENT = 2.0 * np.pi / 360


def test_passes_through_clear_policy_action():
    decision = evaluate_shield(
        np.full(360, 2.0),
        ANGLE_MIN,
        ANGLE_INCREMENT,
        0.18,
        -0.20,
    )

    assert decision.status == "pass_through"
    assert not decision.intervened
    assert decision.linear == 0.18
    assert decision.angular == -0.20


def test_stops_for_critical_obstacle():
    scan = np.full(360, 2.0)
    scan[180] = 0.15
    decision = evaluate_shield(
        scan,
        ANGLE_MIN,
        ANGLE_INCREMENT,
        0.18,
        0.10,
    )

    assert decision.status == "emergency_stop"
    assert decision.intervened
    assert decision.linear == 0.0
    assert decision.angular == 0.0


def test_turns_toward_clearer_side():
    scan = np.full(360, 0.35)
    scan[180:] = 2.0
    scan[180] = 0.30
    decision = evaluate_shield(
        scan,
        ANGLE_MIN,
        ANGLE_INCREMENT,
        0.18,
        0.0,
    )

    assert decision.status == "avoidance"
    assert decision.linear == 0.0
    assert decision.angular > 0.0


def test_scales_forward_speed_near_obstacle():
    scan = np.full(360, 2.0)
    scan[180] = 0.65
    decision = evaluate_shield(
        scan,
        ANGLE_MIN,
        ANGLE_INCREMENT,
        0.20,
        0.10,
    )

    assert decision.status == "slow_down"
    assert 0.0 < decision.linear < 0.20
    assert decision.angular == 0.10
