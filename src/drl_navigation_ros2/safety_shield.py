"""Independent lidar safety shield between a navigation policy and TurtleBot3."""

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from safety_shield_core import evaluate_shield


class SafetyShield(Node):
    """Filter policy velocity commands using the latest lidar observation."""

    def __init__(
        self,
        input_topic="/policy_cmd_vel",
        output_topic="/cmd_vel",
        scan_topic="/scan",
        control_hz=10.0,
        command_timeout=0.5,
        sensor_timeout=1.0,
        emergency_stop_distance=0.22,
        avoidance_distance=0.50,
        slow_down_distance=0.80,
        front_half_angle=0.52,
        avoidance_angular_speed=0.50,
        minimum_speed_scale=0.20,
    ):
        super().__init__("safety_shield")
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.command_timeout = float(command_timeout)
        self.sensor_timeout = float(sensor_timeout)
        self.shield_parameters = {
            "emergency_stop_distance": float(emergency_stop_distance),
            "avoidance_distance": float(avoidance_distance),
            "slow_down_distance": float(slow_down_distance),
            "front_half_angle": float(front_half_angle),
            "avoidance_angular_speed": float(avoidance_angular_speed),
            "minimum_speed_scale": float(minimum_speed_scale),
        }
        self.latest_command = None
        self.latest_scan = None
        self.command_received_at = None
        self.scan_received_at = None
        self.last_status = None
        self.last_status_published_at = 0.0

        self.create_subscription(Twist, input_topic, self.command_callback, 10)
        self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.command_publisher = self.create_publisher(Twist, output_topic, 10)
        self.status_publisher = self.create_publisher(
            String, "/safety_shield/status", 10
        )
        self.intervention_publisher = self.create_publisher(
            Bool, "/safety_shield/intervention", 10
        )
        self.create_timer(1.0 / float(control_hz), self.control_callback)
        self.create_timer(5.0, self.check_output_publishers)

        self.get_logger().info(
            f"Safety shield active: {input_topic} -> {output_topic}"
        )
        self.publish_stop("waiting_for_inputs")

    def command_callback(self, message):
        self.latest_command = message
        self.command_received_at = time.monotonic()

    def scan_callback(self, message):
        self.latest_scan = message
        self.scan_received_at = time.monotonic()

    def check_output_publishers(self):
        endpoints = self.get_publishers_info_by_topic(self.output_topic)
        foreign = sorted(
            {
                f"{endpoint.node_namespace.rstrip('/')}/{endpoint.node_name}"
                for endpoint in endpoints
                if endpoint.node_name != self.get_name()
            }
        )
        if foreign:
            self.get_logger().error(
                f"Velocity conflict on {self.output_topic}: "
                + ", ".join(foreign),
                throttle_duration_sec=10.0,
            )

    def control_callback(self):
        now = time.monotonic()
        if self.latest_scan is None or self.latest_command is None:
            self.publish_stop("waiting_for_inputs")
            return
        if now - self.scan_received_at > self.sensor_timeout:
            self.publish_stop("sensor_stale")
            return
        if now - self.command_received_at > self.command_timeout:
            self.publish_stop("command_stale")
            return

        command = self.latest_command
        scan = self.latest_scan
        decision = evaluate_shield(
            scan.ranges,
            scan.angle_min,
            scan.angle_increment,
            command.linear.x,
            command.angular.z,
            maximum_range=scan.range_max,
            **self.shield_parameters,
        )
        output = Twist()
        output.linear.x = decision.linear
        output.angular.z = decision.angular
        self.command_publisher.publish(output)
        self.publish_state(decision.status, decision.intervened)

    def publish_stop(self, status):
        self.command_publisher.publish(Twist())
        self.publish_state(status, True)

    def publish_state(self, status, intervened):
        intervention = Bool()
        intervention.data = bool(intervened)
        self.intervention_publisher.publish(intervention)

        now = time.monotonic()
        if status == self.last_status and now - self.last_status_published_at < 1.0:
            return
        message = String()
        message.data = status
        self.status_publisher.publish(message)
        self.last_status = status
        self.last_status_published_at = now

    def stop(self):
        self.publish_stop("stopped")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Lidar policy-action safety shield")
    parser.add_argument("--input-topic", default="/policy_cmd_vel")
    parser.add_argument("--output-topic", default="/cmd_vel")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--sensor-timeout", type=float, default=1.0)
    parser.add_argument("--emergency-stop-distance", type=float, default=0.22)
    parser.add_argument("--avoidance-distance", type=float, default=0.50)
    parser.add_argument("--slow-down-distance", type=float, default=0.80)
    parser.add_argument("--front-half-angle", type=float, default=0.52)
    parser.add_argument("--avoidance-angular-speed", type=float, default=0.50)
    parser.add_argument("--minimum-speed-scale", type=float, default=0.20)
    return parser.parse_args(args)


def main(args=None):
    config = parse_args(args)
    rclpy.init()
    node = SafetyShield(**vars(config))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
