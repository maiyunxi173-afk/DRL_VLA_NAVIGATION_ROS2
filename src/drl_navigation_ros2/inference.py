"""Run the trained SAC actor as a deterministic ROS 2 navigation controller.

The high-level planner (for example ``bpx_vla_navigation``) publishes an
``odom``-frame ``geometry_msgs/PointStamped`` on ``/vla_goal``.  This node
combines that goal with ``/scan`` and ``/odom``, builds exactly the same
25-dimensional state used during training, and publishes a configurable
velocity topic.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

from SAC.SAC import SAC


STATE_DIM = 25
ACTION_DIM = 2
DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parent / "models/SAC_epoch6_best"


def quaternion_to_yaw(orientation):
    """Return yaw without introducing another runtime dependency."""
    sin_yaw = 2.0 * (
        orientation.w * orientation.z + orientation.x * orientation.y
    )
    cos_yaw = 1.0 - 2.0 * (
        orientation.y * orientation.y + orientation.z * orientation.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def target_features(position, orientation, goal):
    """Compute distance and signed heading features in the odom frame."""
    dx = float(goal.point.x - position.x)
    dy = float(goal.point.y - position.y)
    distance = math.hypot(dx, dy)
    if distance < 1e-8:
        return distance, 1.0, 0.0

    yaw = quaternion_to_yaw(orientation)
    heading_x = math.cos(yaw)
    heading_y = math.sin(yaw)
    goal_x = dx / distance
    goal_y = dy / distance
    cosine = heading_x * goal_x + heading_y * goal_y
    sine = heading_x * goal_y - heading_y * goal_x
    return distance, float(cosine), float(sine)


def load_policy(model_directory, device):
    """Load and validate the saved prototype without creating training state."""
    model_directory = Path(model_directory).expanduser().resolve()
    checkpoint_path = model_directory / "SAC_checkpoint.pt"
    actor_path = model_directory / "SAC_actor.pth"
    if not checkpoint_path.exists() or not actor_path.exists():
        raise FileNotFoundError(
            "Expected SAC_checkpoint.pt and SAC_actor.pth in "
            f"{model_directory}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    schema = metadata.get("state_schema_version")
    if schema != SAC.STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Model state schema is {schema!r}; expected {SAC.STATE_SCHEMA_VERSION}"
        )

    actor_state = torch.load(actor_path, map_location="cpu")
    checkpoint_actor = checkpoint.get("actor")
    if checkpoint_actor is None or checkpoint_actor.keys() != actor_state.keys():
        raise RuntimeError("SAC_actor.pth does not match the checkpoint structure")
    if any(
        not torch.equal(checkpoint_actor[key], actor_state[key])
        for key in actor_state
    ):
        raise RuntimeError(
            "SAC_actor.pth and SAC_checkpoint.pt contain different actor weights"
        )

    # SAC owns the canonical state preparation and action scaling constants.
    # TensorBoard logging is disabled because inference never writes metrics.
    policy = SAC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=1.0,
        device=device,
        hidden_dim=256,
        save_every=0,
        load_model=False,
        save_directory=model_directory,
        enable_logging=False,
    )
    policy.actor.load_state_dict(actor_state)
    policy.actor.eval()
    policy.critic.eval()
    policy.critic_target.eval()
    policy.close()
    return policy, metadata


class SacVlaController(Node):
    def __init__(
        self,
        policy,
        metadata,
        model_directory,
        control_hz=10.0,
        goal_tolerance=0.5,
        emergency_stop_distance=0.22,
        hard_avoidance_distance=0.50,
        avoidance_angular_speed=0.50,
        policy_speed_scale=0.60,
        search_angular_speed=0.35,
        search_rotate_duration=60.0,
        search_observe_duration=0.0,
        sensor_timeout=1.0,
        goal_timeout=0.0,
        cmd_vel_topic="/cmd_vel",
        internal_safety=True,
    ):
        super().__init__("sac_vla_controller")
        self.policy = policy
        self.metadata = metadata
        self.model_directory = Path(model_directory)
        self.goal_tolerance = float(goal_tolerance)
        self.emergency_stop_distance = float(emergency_stop_distance)
        self.hard_avoidance_distance = float(hard_avoidance_distance)
        self.avoidance_angular_speed = float(avoidance_angular_speed)
        self.policy_speed_scale = min(
            1.0, max(0.1, float(policy_speed_scale))
        )
        self.search_angular_speed = float(search_angular_speed)
        self.search_rotate_duration = max(0.0, float(search_rotate_duration))
        self.search_observe_duration = max(0.0, float(search_observe_duration))
        self.sensor_timeout = float(sensor_timeout)
        self.goal_timeout = float(goal_timeout)
        self.cmd_vel_topic = str(cmd_vel_topic)
        self.internal_safety = bool(internal_safety)

        self.latest_scan = None
        self.latest_scan_message = None
        self.latest_odom = None
        self.goal = None
        self.scan_received_at = None
        self.odom_received_at = None
        self.goal_received_at = None
        self.last_physical_action = [0.0, 0.0]
        self.goal_reached_reported = False
        self.search_requested = False
        self.search_turn_sign = None
        self.search_phase = "observe"
        self.search_phase_started_at = None
        self.last_wait_message_at = 0.0
        self.last_status = None
        self.last_status_published_at = 0.0

        self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile_sensor_data
        )
        self.create_subscription(PointStamped, "/vla_goal", self.goal_callback, 10)
        self.create_subscription(
            Bool, "/vla_navigation/cancel", self.cancel_callback, 10
        )
        self.create_subscription(
            Bool, "/vla_navigation/search", self.search_callback, 10
        )
        self.cmd_publisher = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_publisher = self.create_publisher(
            String, "/sac_navigation/status", 10
        )
        marker_qos = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.marker_publisher = self.create_publisher(
            Marker, "/visualization_marker", marker_qos
        )
        self.create_timer(1.0 / float(control_hz), self.control_callback)
        self.create_timer(5.0, self.check_cmd_vel_publishers)

        epoch = metadata.get("epoch", "unknown")
        self.get_logger().info(
            f"Loaded deterministic SAC policy (epoch={epoch}) from "
            f"{self.model_directory}"
        )
        self.get_logger().info(
            f"SAC navigation speed scale={self.policy_speed_scale:.2f} "
            "(linear and angular policy commands)"
        )
        self.get_logger().info(
            f"Velocity output={self.cmd_vel_topic}; "
            f"internal safety={'enabled' if self.internal_safety else 'disabled'}"
        )
        self.get_logger().info(
            "Waiting for /scan, /odom and an odom-frame /vla_goal ..."
        )
        search_mode = (
            "continuous rotation"
            if self.search_observe_duration <= 0.0
            else "stop-and-look"
        )
        self.get_logger().info(
            f"Bounded search: {search_mode} "
            f"(observe={self.search_observe_duration:.1f}s, "
            f"rotate={self.search_rotate_duration:.1f}s, "
            f"angular={self.search_angular_speed:.2f}rad/s)"
        )
        self.publish_status("waiting")

    def check_cmd_vel_publishers(self):
        """Report accidental legacy controllers that compete for velocity."""
        endpoints = self.get_publishers_info_by_topic(self.cmd_vel_topic)
        foreign = sorted(
            {
                f"{endpoint.node_namespace.rstrip('/')}/{endpoint.node_name}"
                for endpoint in endpoints
                if endpoint.node_name != self.get_name()
            }
        )
        if foreign:
            self.get_logger().error(
                f"Velocity conflict: {self.cmd_vel_topic} also has publisher(s): "
                + ", ".join(foreign)
                + ". Stop legacy navigation/control nodes.",
                throttle_duration_sec=10.0,
            )

    def scan_callback(self, message):
        self.latest_scan = list(message.ranges)
        self.latest_scan_message = message
        self.scan_received_at = time.monotonic()

    def odom_callback(self, message):
        self.latest_odom = message
        self.odom_received_at = time.monotonic()

    def goal_callback(self, message):
        frame_id = message.header.frame_id.lstrip("/")
        if frame_id != "odom":
            self.get_logger().error(
                f"Rejected /vla_goal in frame '{message.header.frame_id}'. "
                "Transform the goal to odom before publishing."
            )
            self.goal = None
            self.publish_stop()
            self.publish_status("invalid_goal_frame")
            return
        if not np.isfinite([message.point.x, message.point.y]).all():
            self.get_logger().error("Rejected /vla_goal containing NaN or infinity")
            self.goal = None
            self.publish_stop()
            self.publish_status("invalid_goal")
            return

        if self.goal is not None:
            goal_shift = math.hypot(
                message.point.x - self.goal.point.x,
                message.point.y - self.goal.point.y,
            )
            if goal_shift < 0.05:
                # Vision/bridge nodes commonly republish the same target. Keep
                # the recurrent previous-action feature and reached state intact.
                self.goal.header = message.header
                self.goal_received_at = time.monotonic()
                return

        self.goal = message
        self.goal_received_at = time.monotonic()
        self.last_physical_action = [0.0, 0.0]
        self.goal_reached_reported = False
        self.get_logger().info(
            f"Accepted VLA goal: x={message.point.x:.3f}, "
            f"y={message.point.y:.3f}, frame=odom"
        )
        self.publish_status("active")

    def cancel_callback(self, message):
        if not message.data:
            return
        had_goal = self.goal is not None
        self.goal = None
        self.goal_received_at = None
        self.goal_reached_reported = False
        self.publish_stop()
        self.publish_status("waiting")
        if had_goal:
            self.get_logger().info("VLA goal cancelled; robot stopped")

    def search_callback(self, message):
        requested = bool(message.data)
        if requested and not self.search_requested:
            self.search_turn_sign = (
                self.safer_turn_sign() if self.latest_scan is not None else 1.0
            )
            # Cloud VLM inference is much slower than local vision.  Start
            # with a stationary observation window, then alternate bounded
            # turns and observation windows so the target cannot cross the
            # camera field of view between two Qwen requests.
            self.search_phase = "observe"
            self.search_phase_started_at = time.monotonic()
        elif not requested:
            self.search_turn_sign = None
            self.search_phase = "observe"
            self.search_phase_started_at = None
        self.search_requested = requested

    def inputs_ready(self, now):
        if self.latest_scan is None or self.latest_odom is None or self.goal is None:
            return False
        if now - self.scan_received_at > self.sensor_timeout:
            return False
        if now - self.odom_received_at > self.sensor_timeout:
            return False
        if self.goal_timeout > 0.0 and now - self.goal_received_at > self.goal_timeout:
            self.get_logger().warning("VLA goal expired; stopping the robot")
            self.goal = None
            return False
        return True

    def sensors_ready(self, now):
        return (
            self.latest_scan is not None
            and self.latest_odom is not None
            and now - self.scan_received_at <= self.sensor_timeout
            and now - self.odom_received_at <= self.sensor_timeout
        )

    def missing_inputs(self, now):
        missing = []
        if self.latest_scan is None:
            missing.append("/scan")
        elif now - self.scan_received_at > self.sensor_timeout:
            missing.append("fresh /scan")
        if self.latest_odom is None:
            missing.append("/odom")
        elif now - self.odom_received_at > self.sensor_timeout:
            missing.append("fresh /odom")
        if self.goal is None:
            missing.append("/vla_goal")
        return missing

    def control_callback(self):
        now = time.monotonic()
        if not self.sensors_ready(now):
            self.publish_stop()
            self.publish_status("sensor_stale")
            if now - self.last_wait_message_at > 5.0:
                missing = ", ".join(self.missing_inputs(now)) or "valid inputs"
                self.get_logger().info(f"Controller waiting for: {missing}")
                self.last_wait_message_at = now
            return

        if (
            self.goal is not None
            and self.goal_timeout > 0.0
            and now - self.goal_received_at > self.goal_timeout
        ):
            self.get_logger().warning("VLA goal expired; stopping the robot")
            self.goal = None


        if self.goal is None:
            if self.search_requested:
                self.run_search()
            else:
                self.publish_stop()
                self.publish_status("waiting")
                if now - self.last_wait_message_at > 5.0:
                    self.get_logger().info("Controller waiting for: /vla_goal")
                    self.last_wait_message_at = now
            return

        finite_scan = np.asarray(self.latest_scan, dtype=np.float32)
        finite_scan = finite_scan[np.isfinite(finite_scan)]
        if finite_scan.size == 0:
            self.get_logger().error("LaserScan contains no finite measurements")
            self.publish_stop()
            self.publish_status("invalid_scan")
            return
        minimum_range = float(finite_scan.min())
        if (
            self.internal_safety
            and minimum_range < self.emergency_stop_distance
        ):
            self.get_logger().warning(
                "Emergency stop: obstacle critically close",
                throttle_duration_sec=2.0,
            )
            self.publish_stop()
            self.publish_status("emergency_stop")
            return
        pose = self.latest_odom.pose.pose
        distance, cosine, sine = target_features(
            pose.position, pose.orientation, self.goal
        )
        self.publish_goal_marker(self.latest_odom.header.stamp)
        if distance < self.goal_tolerance:
            self.publish_stop()
            if not self.goal_reached_reported:
                self.get_logger().info(
                    f"VLA goal reached; final distance={distance:.3f} m"
                )
                self.goal_reached_reported = True
            self.publish_status("goal_reached")
            return

        if self.internal_safety and minimum_range < self.hard_avoidance_distance:
            self.run_hard_avoidance()
            return

        state, _ = self.policy.prepare_state(
            self.latest_scan,
            distance,
            cosine,
            sine,
            False,
            False,
            self.last_physical_action,
        )
        with torch.inference_mode():
            normalized_action = self.policy.get_action(state, add_noise=False)
        linear = float(
            (normalized_action[0] + 1.0)
            * self.policy.MAX_LINEAR_VELOCITY
            / 2.0
            * self.policy_speed_scale
        )
        angular = float(
            normalized_action[1]
            * self.policy.MAX_ANGULAR_VELOCITY
            * self.policy_speed_scale
        )
        self.publish_command(linear, angular)
        self.publish_status("active")
        self.last_physical_action = [linear, angular]

    def side_clearances(self):
        ranges = np.asarray(self.latest_scan, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=0.0,
            posinf=self.policy.LASER_MAX_RANGE,
            neginf=0.0,
        )
        midpoint = max(1, ranges.size // 2)
        # The current scan runs from negative (right) to positive (left) yaw.
        right = ranges[:midpoint]
        left = ranges[midpoint:]
        left_clearance = float(np.percentile(left, 40.0)) if left.size else 0.0
        right_clearance = float(np.percentile(right, 40.0)) if right.size else 0.0
        return left_clearance, right_clearance

    def safer_turn_sign(self):
        left, right = self.side_clearances()
        return 1.0 if left >= right else -1.0

    def run_hard_avoidance(self):
        angular = self.safer_turn_sign() * self.avoidance_angular_speed
        self.publish_command(0.0, angular)
        self.last_physical_action = [0.0, angular]
        self.publish_status("hard_avoidance")

    def run_search(self):
        finite = np.asarray(self.latest_scan, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            self.publish_stop()
            self.publish_status("invalid_scan")
            return
        minimum_range = float(finite.min())
        if (
            self.internal_safety
            and minimum_range < self.emergency_stop_distance
        ):
            self.publish_stop()
            self.publish_status("emergency_stop")
            return

        # With zero observation duration the camera remains live while the
        # robot rotates continuously. A positive value explicitly enables
        # the slower stop-and-look mode for strict cloud-detector experiments.
        if self.search_observe_duration > 0.0:
            now = time.monotonic()
            if self.search_phase_started_at is None:
                self.search_phase = "observe"
                self.search_phase_started_at = now
            elapsed = now - self.search_phase_started_at
            if self.search_phase == "observe":
                if elapsed < self.search_observe_duration:
                    self.publish_stop()
                    self.publish_status("search_observing")
                    return
                self.search_phase = "rotate"
                self.search_phase_started_at = now
            elif elapsed >= self.search_rotate_duration:
                self.search_phase = "observe"
                self.search_phase_started_at = now
                self.publish_stop()
                self.publish_status("search_observing")
                return

        if self.search_rotate_duration <= 0.0:
            self.publish_stop()
            self.publish_status("search_observing")
            return
        angular_scale = (
            self.avoidance_angular_speed
            if (
                self.internal_safety
                and minimum_range < self.hard_avoidance_distance
            )
            else self.search_angular_speed
        )
        turn_sign = self.search_turn_sign
        if turn_sign is None:
            turn_sign = self.safer_turn_sign()
            self.search_turn_sign = turn_sign
        angular = turn_sign * angular_scale
        self.publish_command(0.0, angular)
        self.last_physical_action = [0.0, angular]
        self.publish_status("searching")

    def publish_command(self, linear, angular):
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        self.cmd_publisher.publish(command)

    def publish_stop(self):
        self.publish_command(0.0, 0.0)
        self.last_physical_action = [0.0, 0.0]

    def publish_status(self, status):
        now = time.monotonic()
        if status == self.last_status and now - self.last_status_published_at < 1.0:
            return
        message = String()
        message.data = status
        self.status_publisher.publish(message)
        self.last_status = status
        self.last_status_published_at = now

    def publish_goal_marker(self, stamp):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = stamp
        marker.ns = "vla_goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.goal.point.x)
        marker.pose.position.y = float(self.goal.point.y)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.g = 1.0
        self.marker_publisher.publish(marker)

    def stop(self):
        self.publish_stop()


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="SAC controller for VLA goals")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY,
        help="directory containing SAC_checkpoint.pt and SAC_actor.pth",
    )
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.5)
    parser.add_argument("--emergency-stop-distance", type=float, default=0.22)
    parser.add_argument("--hard-avoidance-distance", type=float, default=0.50)
    parser.add_argument("--avoidance-angular-speed", type=float, default=0.50)
    parser.add_argument(
        "--policy-speed-scale",
        type=float,
        default=0.60,
        help="scale SAC linear/angular commands; clamped to [0.1, 1.0]",
    )
    parser.add_argument("--search-angular-speed", type=float, default=0.35)
    parser.add_argument(
        "--search-rotate-duration",
        type=float,
        default=60.0,
        help="seconds spent rotating during each search cycle",
    )
    parser.add_argument(
        "--search-observe-duration",
        type=float,
        default=0.0,
        help="seconds spent stationary for cloud VLM observation; 0 disables pauses",
    )
    parser.add_argument("--sensor-timeout", type=float, default=1.0)
    parser.add_argument(
        "--goal-timeout",
        type=float,
        default=0.0,
        help="seconds before a goal expires; 0 keeps it until replaced",
    )
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument(
        "--internal-safety",
        choices=["true", "false"],
        default="true",
        help="retain legacy emergency-stop and hard-avoidance logic",
    )
    return parser.parse_args(args)


def main(args=None):
    config = parse_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, metadata = load_policy(config.model_dir, device)
    rclpy.init()
    node = SacVlaController(
        policy=policy,
        metadata=metadata,
        model_directory=config.model_dir,
        control_hz=config.control_hz,
        goal_tolerance=config.goal_tolerance,
        emergency_stop_distance=config.emergency_stop_distance,
        hard_avoidance_distance=config.hard_avoidance_distance,
        avoidance_angular_speed=config.avoidance_angular_speed,
        policy_speed_scale=config.policy_speed_scale,
        search_angular_speed=config.search_angular_speed,
        search_rotate_duration=config.search_rotate_duration,
        search_observe_duration=config.search_observe_duration,
        sensor_timeout=config.sensor_timeout,
        goal_timeout=config.goal_timeout,
        cmd_vel_topic=config.cmd_vel_topic,
        internal_safety=config.internal_safety == "true",
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.stop()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            node.destroy_node()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except (Exception, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
