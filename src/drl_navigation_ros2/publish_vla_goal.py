"""Publish one test goal using the same interface expected from the VLA node."""

import argparse
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Publish a test /vla_goal")
    parser.add_argument("x", type=float, help="goal x in odom frame")
    parser.add_argument("y", type=float, help="goal y in odom frame")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--topic", default="/vla_goal")
    return parser.parse_args(args)


def main(args=None):
    config = parse_args(args)
    rclpy.init()
    node = Node("vla_goal_test_publisher")
    publisher = node.create_publisher(PointStamped, config.topic, 10)
    message = PointStamped()
    message.header.frame_id = "odom"
    message.point.x = float(config.x)
    message.point.y = float(config.y)
    message.point.z = 0.0

    try:
        for _ in range(max(1, config.repetitions)):
            message.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.1)
        node.get_logger().info(
            f"Published {config.topic}: x={message.point.x:.3f}, "
            f"y={message.point.y:.3f}, frame=odom"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
