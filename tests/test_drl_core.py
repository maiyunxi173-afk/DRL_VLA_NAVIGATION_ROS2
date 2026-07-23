"""Numerical checks for the navigation state and reward interfaces."""

from pathlib import Path
import sys

import numpy as np


SOURCE_DIRECTORY = Path(__file__).parents[1] / "src" / "drl_navigation_ros2"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from ros_python import ROS_env  # noqa: E402
from SAC.SAC import SAC  # noqa: E402
from inference import SacVlaController, target_features  # noqa: E402

from geometry_msgs.msg import PointStamped, Pose  # noqa: E402


def test_reward_prefers_progress_and_correct_turn_direction():
    scan = np.full(360, 2.0, dtype=np.float32)
    common = {
        "goal": False,
        "collision": False,
        "laser_scan": scan,
        "distance": 1.0,
        "previous_distance": 1.01,
        "cos": 0.5,
        "sin": 0.8,
    }

    correct_turn = ROS_env.get_reward(action=[0.25, 0.35], **common)
    wrong_turn = ROS_env.get_reward(action=[0.25, -0.35], **common)
    moving_away = ROS_env.get_reward(
        goal=False,
        collision=False,
        action=[0.25, 0.35],
        laser_scan=scan,
        cos=-0.5,
        sin=0.8,
        distance=1.01,
        previous_distance=1.0,
    )

    assert correct_turn > wrong_turn
    assert correct_turn > moving_away


def test_terminal_rewards_have_a_bounded_scale():
    scan = np.full(360, 2.0, dtype=np.float32)
    assert ROS_env.get_reward(True, False, [0.0, 0.0], scan) == 20.0
    assert ROS_env.get_reward(False, True, [0.0, 0.0], scan) == -20.0


def test_prepared_state_is_finite_bounded_and_fixed_size():
    state_builder = SAC.__new__(SAC)
    state_builder.state_dim = 25
    scan = np.concatenate(
        ([np.nan, np.inf], np.linspace(0.0, 8.0, 358, dtype=np.float32))
    )

    state, terminal = state_builder.prepare_state(
        scan,
        distance=2.5,
        cos=0.3,
        sin=-0.7,
        collision=False,
        goal=False,
        action=[0.175, -0.35],
    )

    assert terminal == 0
    assert len(state) == 25
    assert np.isfinite(state).all()
    assert 0.0 <= min(state[:20]) <= max(state[:20]) <= 1.0
    assert all(-1.0 <= value <= 1.0 for value in state[20:])


def test_vla_goal_heading_uses_the_odom_yaw_convention():
    pose = Pose()
    pose.orientation.w = 1.0
    goal = PointStamped()
    goal.point.x = 0.0
    goal.point.y = 2.0

    distance, cosine, sine = target_features(
        pose.position, pose.orientation, goal
    )

    assert distance == 2.0
    assert abs(cosine) < 1e-7
    assert sine > 0.99


def test_hard_avoidance_turns_toward_the_clearer_scan_side():
    controller = SacVlaController.__new__(SacVlaController)
    controller.latest_scan = [0.4] * 90 + [2.0] * 90
    controller.policy = type("Policy", (), {"LASER_MAX_RANGE": 7.0})()

    assert controller.safer_turn_sign() == 1.0
