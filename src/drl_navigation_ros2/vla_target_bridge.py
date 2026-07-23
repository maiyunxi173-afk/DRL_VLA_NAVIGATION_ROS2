"""Bridge the existing BPX VLA PoseStamped target to the SAC goal interface.

The external workspace publishes ``/vla/target/map_pose``. Despite its legacy
topic name, the message may be in any TF frame. This node transforms the point
to ``odom`` and publishes ``geometry_msgs/PointStamped`` on ``/vla_goal``.
It never publishes velocity commands; SAC remains the sole local controller.
"""

import math
import time

import rclpy
import tf2_geometry_msgs  # noqa: F401 - registers geometry transformations
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, String
from tf2_ros import TransformException


class VlaTargetBridge(Node):
    def __init__(self):
        super().__init__("vla_target_bridge")
        self.declare_parameter("source_topic", "/vla/target/map_pose")
        self.declare_parameter("goal_topic", "/vla_goal/source/demo6")
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("minimum_goal_change", 0.05)
        self.declare_parameter("maximum_subgoal_distance", 1.2)
        # Simulation-only sanity check.  Ground truth never creates a goal; it
        # only rejects a perceptual 3-D point that is clearly not the spawned
        # red box (for example, Qwen selecting the robot body/floor).
        self.declare_parameter("truth_topic", "/vla/semantic_target/truth")
        self.declare_parameter("maximum_truth_error", 1.00)

        self.source_topic = str(self.get_parameter("source_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.minimum_goal_change = float(
            self.get_parameter("minimum_goal_change").value
        )
        self.maximum_subgoal_distance = float(
            self.get_parameter("maximum_subgoal_distance").value
        )
        self.maximum_truth_error = float(
            self.get_parameter("maximum_truth_error").value
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.goal_publisher = self.create_publisher(
            PointStamped, self.goal_topic, 10
        )
        self.create_subscription(
            PoseStamped, self.source_topic, self.pose_callback, 10
        )
        self.create_subscription(
            String,
            "/sac_navigation/status",
            self.status_callback,
            10,
        )
        self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("truth_topic").value),
            self.truth_callback,
            10,
        )
        self.create_subscription(
            Bool,
            "/vla_navigation/semantic_reset",
            self.semantic_reset_callback,
            10,
        )
        self.last_goal = None
        self.latest_odom = None
        self.latest_truth = None
        self.semantic_reached = False
        self.last_tf_warning_at = 0.0
        self.last_status = None

        self.get_logger().info(
            f"BPX VLA bridge ready: {self.source_topic} -> "
            f"{self.goal_topic} ({self.target_frame})"
        )
        self.get_logger().info(
            "SAC is the only /cmd_vel controller; do not run vla_agent_node_legacy.py"
        )

    def pose_callback(self, message):
        if self.latest_odom is None or self.semantic_reached:
            return
        source_frame = message.header.frame_id.lstrip("/")
        if not source_frame:
            self.get_logger().error("Rejected VLA target with empty frame_id")
            return
        if not math.isfinite(message.pose.position.x) or not math.isfinite(
            message.pose.position.y
        ):
            self.get_logger().error("Rejected VLA target containing NaN or infinity")
            return

        source = PointStamped()
        source.header = message.header
        source.header.frame_id = source_frame
        source.point.x = float(message.pose.position.x)
        source.point.y = float(message.pose.position.y)
        source.point.z = float(message.pose.position.z)

        try:
            if source_frame == self.target_frame:
                goal = source
            else:
                goal = self.tf_buffer.transform(
                    source,
                    self.target_frame,
                    timeout=Duration(seconds=0.2),
                )
        except TransformException as exception:
            now = time.monotonic()
            if now - self.last_tf_warning_at > 5.0:
                self.get_logger().warning(
                    f"Cannot transform VLA goal {source_frame} -> "
                    f"{self.target_frame}: {exception}"
                )
                self.last_tf_warning_at = now
            return

        goal.header.frame_id = self.target_frame
        goal.point.z = 0.0
        if self.latest_truth is not None:
            truth_error = math.hypot(
                float(goal.point.x - self.latest_truth.point.x),
                float(goal.point.y - self.latest_truth.point.y),
            )
            if truth_error > self.maximum_truth_error:
                self.get_logger().warning(
                    "Rejected false Demo6 target: perception=(%.2f, %.2f), "
                    "simulation_truth=(%.2f, %.2f), error=%.2f m"
                    % (
                        goal.point.x,
                        goal.point.y,
                        self.latest_truth.point.x,
                        self.latest_truth.point.y,
                        truth_error,
                    ),
                    throttle_duration_sec=2.0,
                )
                return
        robot = self.latest_odom.pose.pose.position
        dx = float(goal.point.x - robot.x)
        dy = float(goal.point.y - robot.y)
        object_distance = math.hypot(dx, dy)
        subgoal_distance = min(object_distance, self.maximum_subgoal_distance)
        scale = subgoal_distance / max(object_distance, 1e-6)
        goal.point.x = float(robot.x + dx * scale)
        goal.point.y = float(robot.y + dy * scale)
        if self.last_goal is not None:
            change = math.hypot(
                goal.point.x - self.last_goal.point.x,
                goal.point.y - self.last_goal.point.y,
            )
            if change < self.minimum_goal_change:
                return

        self.goal_publisher.publish(goal)
        self.last_goal = goal
        self.get_logger().info(
            f"Published SAC goal: x={goal.point.x:.3f}, "
            f"y={goal.point.y:.3f}, frame={self.target_frame}"
        )

    def odom_callback(self, message):
        self.latest_odom = message

    def truth_callback(self, message):
        if message.header.frame_id.lstrip("/") != self.target_frame:
            return
        self.latest_truth = message

    def semantic_reset_callback(self, message):
        if not message.data:
            return
        self.last_goal = None
        self.semantic_reached = False
        self.latest_truth = None

    def status_callback(self, message):
        if message.data == self.last_status:
            return
        self.last_status = message.data
        self.get_logger().info(f"SAC navigation status: {message.data}")


def main(args=None):
    rclpy.init(args=args)
    node = VlaTargetBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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
