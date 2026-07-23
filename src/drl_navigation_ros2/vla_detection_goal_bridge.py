"""Convert a BPX VLA image detection into a short-range SAC navigation goal.

This is the basic-navigation path for the current Gazebo model, which exposes
RGB, LaserScan and Odometry but no depth image. The detected bounding-box angle
is aligned with LaserScan to estimate range, then converted to an odom-frame
subgoal. It never publishes ``/cmd_vel``.
"""

import json
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, LaserScan
from std_msgs.msg import Bool, String
import tf2_ros
from tf2_ros import TransformException


def quaternion_to_yaw(orientation):
    sin_yaw = 2.0 * (
        orientation.w * orientation.z + orientation.x * orientation.y
    )
    cos_yaw = 1.0 - 2.0 * (
        orientation.y * orientation.y + orientation.z * orientation.z
    )
    return math.atan2(sin_yaw, cos_yaw)


class VlaDetectionGoalBridge(Node):
    def __init__(self):
        super().__init__("vla_detection_goal_bridge")
        self.declare_parameter("detection_topic", "/vla/detection/bbox")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("goal_topic", "/vla_goal/source/vision")
        self.declare_parameter("maximum_subgoal_distance", 1.2)
        self.declare_parameter("minimum_valid_range", 0.32)
        self.declare_parameter("sensor_timeout", 1.0)
        self.declare_parameter("minimum_goal_change", 0.10)
        self.declare_parameter("sensor_history_seconds", 15.0)
        self.declare_parameter("maximum_sync_error", 0.20)
        self.declare_parameter("semantic_reach_distance", 0.80)
        self.declare_parameter("bbox_inset_ratio", 0.08)

        param = lambda name: self.get_parameter(name).value
        self.maximum_subgoal_distance = float(param("maximum_subgoal_distance"))
        self.minimum_valid_range = float(param("minimum_valid_range"))
        self.sensor_timeout = float(param("sensor_timeout"))
        self.minimum_goal_change = float(param("minimum_goal_change"))
        self.sensor_history_seconds = float(param("sensor_history_seconds"))
        self.maximum_sync_error = float(param("maximum_sync_error"))
        self.semantic_reach_distance = float(param("semantic_reach_distance"))
        self.bbox_inset_ratio = float(param("bbox_inset_ratio"))

        self.camera_info = None
        self.scan = None
        self.odom = None
        self.scan_received_at = None
        self.odom_received_at = None
        self.last_goal = None
        self.consecutive_misses = 0
        self.scan_history = deque(maxlen=400)
        self.semantic_reached = False
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo,
            str(param("camera_info_topic")),
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            str(param("scan_topic")),
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            str(param("odom_topic")),
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(param("detection_topic")),
            self.detection_callback,
            10,
        )
        self.goal_publisher = self.create_publisher(
            PointStamped, str(param("goal_topic")), 10
        )
        self.semantic_reached_publisher = self.create_publisher(
            Bool, "/vla_navigation/semantic_reached", 10
        )

        self.get_logger().info(
            "Basic VLA bridge ready: time-synchronised bbox + TF-projected "
            "LaserScan -> /vla_goal/source/vision"
        )
        self.get_logger().info(
            "SAC remains the only /cmd_vel publisher"
        )

    def camera_info_callback(self, message):
        self.camera_info = message

    def scan_callback(self, message):
        self.scan = message
        self.scan_received_at = time.monotonic()
        self.scan_history.append(message)

    def odom_callback(self, message):
        self.odom = message
        self.odom_received_at = time.monotonic()

    def detection_callback(self, message):
        try:
            detection = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warning("Ignored malformed VLA detection JSON")
            self.register_miss()
            return

        if not bool(detection.get("success", False)):
            self.register_miss()
            return
        center = detection.get("center_pixel") or {}
        bbox = detection.get("bbox_pixel") or {}
        try:
            center_u = float(center["u"])
            x1 = float(bbox["x1"])
            x2 = float(bbox["x2"])
            y1 = float(bbox["y1"])
            y2 = float(bbox["y2"])
        except (KeyError, TypeError, ValueError):
            self.get_logger().warning("VLA detection has no valid pixel box")
            self.register_miss()
            return

        if not self.sensors_ready():
            self.get_logger().warning(
                "Waiting for fresh /camera/camera_info, /scan and /odom",
                throttle_duration_sec=5.0,
            )
            return

        scan = self.synchronised_scan(detection)
        if scan is None:
            self.get_logger().warning(
                "No LaserScan synchronized with the detection image",
                throttle_duration_sec=5.0,
            )
            self.register_miss()
            return

        target_point = self.project_scan_into_bbox(scan, x1, y1, x2, y2)
        if target_point is None:
            self.get_logger().warning(
                "No TF-projected LaserScan points fall inside the detected box",
                throttle_duration_sec=5.0,
            )
            self.register_miss()
            return

        self.consecutive_misses = 0
        object_x, object_y, range_m, bearing = target_point
        pose = self.odom.pose.pose
        dx = object_x - float(pose.position.x)
        dy = object_y - float(pose.position.y)
        object_distance = math.hypot(dx, dy)

        if object_distance <= self.semantic_reach_distance:
            if not self.semantic_reached:
                reached = Bool()
                reached.data = True
                self.semantic_reached_publisher.publish(reached)
                self.semantic_reached = True
                self.get_logger().info(
                    f"Semantic red-box target reached at {object_distance:.2f} m"
                )
            return
        if object_distance > self.semantic_reach_distance + 0.30:
            self.semantic_reached = False

        distance = min(object_distance, self.maximum_subgoal_distance)
        scale = distance / max(object_distance, 1e-6)
        goal = PointStamped()
        goal.header.stamp = self.odom.header.stamp
        goal.header.frame_id = "odom"
        goal.point.x = float(pose.position.x + dx * scale)
        goal.point.y = float(pose.position.y + dy * scale)
        goal.point.z = 0.0

        if self.last_goal is not None:
            change = math.hypot(
                goal.point.x - self.last_goal.point.x,
                goal.point.y - self.last_goal.point.y,
            )
            if change < self.minimum_goal_change:
                return

        self.goal_publisher.publish(goal)
        self.last_goal = goal
        target = str(detection.get("target", "object"))
        self.get_logger().info(
            f"{target}: projected_lidar_range={range_m:.2f} m, "
            f"bearing={math.degrees(bearing):.1f} deg -> "
            f"goal=({goal.point.x:.2f}, {goal.point.y:.2f})"
        )

    def sensors_ready(self):
        now = time.monotonic()
        return (
            self.camera_info is not None
            and self.scan is not None
            and self.odom is not None
            and now - self.scan_received_at <= self.sensor_timeout
            and now - self.odom_received_at <= self.sensor_timeout
        )

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def synchronised_scan(self, detection):
        stamp = detection.get("source_stamp") or {}
        try:
            source_time = float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
        except (KeyError, TypeError, ValueError):
            return self.scan
        if not self.scan_history:
            return None
        scan = min(
            self.scan_history,
            key=lambda item: abs(self.stamp_seconds(item.header.stamp) - source_time),
        )
        error = abs(self.stamp_seconds(scan.header.stamp) - source_time)
        return scan if error <= self.maximum_sync_error else None

    @staticmethod
    def transform_points(points, transform):
        q = transform.rotation
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        translation = np.asarray(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            dtype=np.float64,
        )
        return points @ rotation.T + translation

    def lookup_transform(self, target_frame, source_frame, stamp):
        source_frame = source_frame.lstrip("/")
        target_frame = target_frame.lstrip("/")
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.15),
            )
        except TransformException:
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.15),
                )
            except TransformException as exception:
                self.get_logger().warning(
                    f"TF unavailable {source_frame} -> {target_frame}: {exception}",
                    throttle_duration_sec=5.0,
                )
                return None

    def project_scan_into_bbox(self, scan, x1, y1, x2, y2):
        camera_frame = self.camera_info.header.frame_id.lstrip("/")
        if not camera_frame:
            self.get_logger().error("CameraInfo has an empty frame_id")
            return None
        camera_tf = self.lookup_transform(
            camera_frame, scan.header.frame_id, scan.header.stamp
        )
        odom_tf = self.lookup_transform("odom", scan.header.frame_id, scan.header.stamp)
        if camera_tf is None or odom_tf is None:
            return None

        ranges = np.asarray(scan.ranges, dtype=np.float64)
        angles = float(scan.angle_min) + np.arange(ranges.size) * float(
            scan.angle_increment
        )
        valid = (
            np.isfinite(ranges)
            & (ranges >= max(self.minimum_valid_range, float(scan.range_min)))
            & (ranges <= float(scan.range_max))
        )
        if not np.any(valid):
            return None
        ranges = ranges[valid]
        angles = angles[valid]
        scan_points = np.column_stack(
            (ranges * np.cos(angles), ranges * np.sin(angles), np.zeros_like(ranges))
        )
        camera_points = self.transform_points(scan_points, camera_tf.transform)
        z = camera_points[:, 2]
        in_front = z > 0.02
        fx, fy = float(self.camera_info.k[0]), float(self.camera_info.k[4])
        cx, cy = float(self.camera_info.k[2]), float(self.camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().error("CameraInfo focal length must be positive")
            return None
        u = fx * camera_points[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * camera_points[:, 1] / np.maximum(z, 1e-6) + cy
        x_low, x_high = sorted((float(x1), float(x2)))
        y_low, y_high = sorted((float(y1), float(y2)))
        inset_x = (x_high - x_low) * self.bbox_inset_ratio
        inset_y = (y_high - y_low) * self.bbox_inset_ratio
        inside = (
            in_front
            & (u >= x_low + inset_x)
            & (u <= x_high - inset_x)
            & (v >= y_low + inset_y)
            & (v <= y_high - inset_y)
        )
        indices = np.flatnonzero(inside)
        if indices.size == 0:
            return None

        selected_ranges = ranges[indices]
        nearest = float(np.min(selected_ranges))
        cluster = indices[selected_ranges <= nearest + 0.20]
        if cluster.size == 0:
            return None
        chosen_local = int(cluster[len(cluster) // 2])
        chosen_range = float(np.median(ranges[cluster]))
        chosen_angle = float(np.median(angles[cluster]))
        chosen_scan_point = np.asarray(
            [[chosen_range * math.cos(chosen_angle), chosen_range * math.sin(chosen_angle), 0.0]]
        )
        odom_point = self.transform_points(chosen_scan_point, odom_tf.transform)[0]
        return float(odom_point[0]), float(odom_point[1]), chosen_range, chosen_angle

    def register_miss(self):
        self.consecutive_misses += 1
        if self.consecutive_misses == 3:
            self.get_logger().info(
                "Vision target temporarily lost; supervisor fallback remains active"
            )


def main(args=None):
    rclpy.init(args=args)
    node = VlaDetectionGoalBridge()
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
