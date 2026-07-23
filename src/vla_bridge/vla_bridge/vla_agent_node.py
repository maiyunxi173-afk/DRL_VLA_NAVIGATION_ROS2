#!/usr/bin/env python3
"""BPX VLA task input adapter for the hierarchical SAC navigation stack.

This node intentionally never publishes ``/cmd_vel`` and performs no local
navigation or goal arbitration. It converts an RViz/manual ``PoseStamped``
goal to an odom-frame candidate for ``vla_agent_supervisor.py`` and reports
high-level task/navigation state. Demo5 and Demo6 keep their own source topics.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
import tf2_geometry_msgs  # noqa: F401 - registers PointStamped transforms
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformException


class VlaTaskInputNode(Node):
    """Thin task-facing adapter; Supervisor remains the only goal arbiter."""

    def __init__(self) -> None:
        super().__init__("vla_task_input_node")
        self.declare_parameter("manual_goal_topic", "/goal_pose")
        self.declare_parameter(
            "manual_candidate_topic", "/vla_goal/source/manual"
        )
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("status_rate", 0.5)

        self.manual_goal_topic = str(
            self.get_parameter("manual_goal_topic").value
        )
        self.manual_candidate_topic = str(
            self.get_parameter("manual_candidate_topic").value
        )
        self.target_frame = str(self.get_parameter("target_frame").value)
        status_rate = max(0.1, float(self.get_parameter("status_rate").value))

        self.latest_instruction = ""
        self.supervisor_state = "unknown"
        self.sac_status = "unknown"
        self.last_manual_goal: Optional[PointStamped] = None
        self.last_tf_warning_at = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.manual_publisher = self.create_publisher(
            PointStamped, self.manual_candidate_topic, 10
        )
        self.status_publisher = self.create_publisher(
            String, "/vla/task/status", 10
        )

        self.create_subscription(
            PoseStamped, self.manual_goal_topic, self.manual_goal_callback, 10
        )
        self.create_subscription(
            String, "/vla/user_command", self.user_command_callback, 10
        )
        self.create_subscription(
            String, "/vla_agent/state", self.supervisor_state_callback, 10
        )
        self.create_subscription(
            String, "/sac_navigation/status", self.sac_status_callback, 10
        )
        self.create_timer(1.0 / status_rate, self.status_tick)

        self.get_logger().info(
            "BPX VLA task adapter started; this node does not publish /cmd_vel"
        )
        self.get_logger().info(
            f"Manual goals: {self.manual_goal_topic} -> "
            f"{self.manual_candidate_topic} ({self.target_frame})"
        )

    def manual_goal_callback(self, message: PoseStamped) -> None:
        source_frame = message.header.frame_id.lstrip("/")
        if not source_frame:
            self.get_logger().error("Rejected manual goal with empty frame_id")
            return

        point = PointStamped()
        point.header = message.header
        point.header.frame_id = source_frame
        point.point.x = float(message.pose.position.x)
        point.point.y = float(message.pose.position.y)
        point.point.z = float(message.pose.position.z)

        try:
            if source_frame == self.target_frame:
                transformed = point
            else:
                transformed = self.tf_buffer.transform(
                    point,
                    self.target_frame,
                    timeout=Duration(seconds=0.2),
                )
        except TransformException as exception:
            now = time.monotonic()
            if now - self.last_tf_warning_at > 5.0:
                self.get_logger().warning(
                    f"Cannot transform manual goal {source_frame} -> "
                    f"{self.target_frame}: {exception}"
                )
                self.last_tf_warning_at = now
            return

        transformed.header.frame_id = self.target_frame
        transformed.point.z = 0.0
        self.manual_publisher.publish(transformed)
        self.last_manual_goal = transformed
        self.get_logger().info(
            f"Published manual candidate: ({transformed.point.x:.2f}, "
            f"{transformed.point.y:.2f}) in {self.target_frame}"
        )

    def user_command_callback(self, message: String) -> None:
        # Demo4 also consumes the original topic. We cache it for task status
        # instead of republishing and creating a command feedback loop.
        self.latest_instruction = message.data.strip()

    def supervisor_state_callback(self, message: String) -> None:
        self.supervisor_state = message.data

    def sac_status_callback(self, message: String) -> None:
        self.sac_status = message.data

    def status_tick(self) -> None:
        status = {
            "instruction": self.latest_instruction,
            "supervisor_state": self.supervisor_state,
            "sac_status": self.sac_status,
            "manual_goal": None,
        }
        if self.last_manual_goal is not None:
            status["manual_goal"] = {
                "frame_id": self.last_manual_goal.header.frame_id,
                "x": self.last_manual_goal.point.x,
                "y": self.last_manual_goal.point.y,
            }
        message = String()
        message.data = json.dumps(status, ensure_ascii=False)
        self.status_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VlaTaskInputNode()
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
