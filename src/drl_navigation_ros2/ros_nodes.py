import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Twist
from visualization_msgs.msg import Marker
from rclpy.logging import LoggingSeverity

SEVERITY = LoggingSeverity.ERROR


class SensorSubscriber(Node):
    def __init__(self):
        super().__init__("sensor_subscriber")
        self.get_logger().set_level(SEVERITY)
        self.scan_subscription = self.create_subscription(
            LaserScan, "scan", self.scan_listener_callback, 1
        )
        self.odom_subscription = self.create_subscription(
            Odometry, "odom", self.odom_listener_callback, 1
        )
        self.latest_position = None
        self.latest_heading = None
        self.latest_odom_stamp = None
        self.latest_scan = None
        self.scan_sequence = 0
        self.odom_sequence = 0

    def scan_listener_callback(self, msg):
        self.latest_scan = msg.ranges[:]
        self.scan_sequence += 1

    def odom_listener_callback(self, msg):
        self.latest_position = msg.pose.pose.position
        self.latest_heading = msg.pose.pose.orientation
        self.latest_odom_stamp = msg.header.stamp
        self.odom_sequence += 1

    def get_latest_sensor(self):
        return self.latest_scan, self.latest_position, self.latest_heading

    def get_latest_odom_stamp(self):
        return self.latest_odom_stamp

    def get_sequences(self):
        return self.scan_sequence, self.odom_sequence

    def has_fresh_sensor(self, previous_scan_sequence, previous_odom_sequence):
        return (
            self.scan_sequence > previous_scan_sequence
            and self.odom_sequence > previous_odom_sequence
            and self.latest_scan is not None
            and self.latest_position is not None
            and self.latest_heading is not None
        )


class ScanSubscriber(Node):
    def __init__(self):
        super().__init__("scan_subscriber")
        self.get_logger().set_level(SEVERITY)
        self.subscriber_ = self.create_subscription(
            LaserScan, "scan", self.listener_callback, 1
        )
        self.latest_scan = None

    def listener_callback(self, msg):
        self.latest_scan = msg.ranges[:]

    def get_latest_scan(self):
        return self.latest_scan


class OdomSubscriber(Node):
    def __init__(self):
        super().__init__("odom_subscriber")
        self.get_logger().set_level(SEVERITY)
        self.subscriber_ = self.create_subscription(
            Odometry, "odom", self.listener_callback, 1
        )
        self.latest_position = None
        self.latest_heading = None

    def listener_callback(self, msg):
        self.latest_position = msg.pose.pose.position
        self.latest_heading = msg.pose.pose.orientation

    def get_latest_odom(self):
        return self.latest_position, self.latest_heading


class ResetWorldClient(Node):
    def __init__(self):
        super().__init__("reset_world_client")
        self.get_logger().set_level(SEVERITY)
        self.reset_client = self.create_client(Empty, "/reset_world")

        self.wait_for_service(self.reset_client, "reset_world")

    def wait_for_service(self, client, service_name, timeout=10.0):
        self.get_logger().info(f"Waiting for {service_name} service...")
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service {service_name} not available after waiting."
            )
            raise RuntimeError(f"Service {service_name} not available.")

    def reset_world(self, timeout=5.0):
        self.get_logger().info("Calling /gazebo/reset_world service...")
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError("Timed out waiting for /reset_world")
        if future.exception() is not None:
            raise RuntimeError(f"/reset_world failed: {future.exception()}")
        self.get_logger().info("World reset successfully.")


class PhysicsClient(Node):
    def __init__(self):
        super().__init__("physics_client")
        self.get_logger().set_level(SEVERITY)
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.pause_client = self.create_client(Empty, "/pause_physics")

        self.wait_for_service(self.unpause_client, "unpause_physics")
        self.wait_for_service(self.pause_client, "pause_physics")

    def wait_for_service(self, client, service_name, timeout=10.0):
        self.get_logger().info(f"Waiting for {service_name} service...")
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service {service_name} not available after waiting."
            )
            raise RuntimeError(f"Service {service_name} not available.")

    def _call(self, client, service_name, timeout=5.0):
        request = Empty.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError(f"Timed out waiting for /{service_name}")
        if future.exception() is not None:
            raise RuntimeError(f"/{service_name} failed: {future.exception()}")

    def pause_physics(self, timeout=5.0):
        self.get_logger().info("Calling /gazebo/pause_physics service...")
        self._call(self.pause_client, "pause_physics", timeout)

    def unpause_physics(self, timeout=5.0):
        self.get_logger().info("Calling /gazebo/unpause_physics service...")
        self._call(self.unpause_client, "unpause_physics", timeout)


class SetModelStateClient(Node):
    def __init__(self):
        super().__init__("set_entity_state_client")
        self.get_logger().set_level(SEVERITY)
        self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        self.request = SetEntityState.Request()

    def set_state(self, name, new_pose, timeout=5.0):
        self.request.state.name = name
        self.request.state.pose = new_pose
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError(f"Timed out moving Gazebo entity: {name}")
        if future.exception() is not None:
            raise RuntimeError(f"Failed moving Gazebo entity {name}: {future.exception()}")
        response = future.result()
        if response is None or not response.success:
            status = "no response" if response is None else response.status_message
            raise RuntimeError(f"Failed moving Gazebo entity {name}: {status}")
        return response


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__("cmd_vel_publisher")
        self.get_logger().set_level(SEVERITY)
        self.publisher_ = self.create_publisher(Twist, "cmd_vel", 1)
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)

    def publish_cmd_vel(self, linear_velocity=0.0, angular_velocity=0.0):
        twist_msg = Twist()
        # Set linear and angular velocities
        twist_msg.linear.x = float(linear_velocity)  # Example linear velocity (m/s)
        twist_msg.angular.z = float(
            angular_velocity
        )  # Example angular velocity (rad/s)
        self.publisher_.publish(twist_msg)


class MarkerPublisher(Node):
    def __init__(self):
        super().__init__("marker_publisher")
        self.get_logger().set_level(SEVERITY)
        marker_qos = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(
            Marker, "visualization_marker", marker_qos
        )

    def publish(self, x, y, stamp=None):
        marker = Marker()
        marker.header.frame_id = "odom"
        # The target and robot pose are both expressed in odom. Reuse the
        # corresponding Odometry timestamp so RViz never mixes wall and
        # Gazebo simulation time. A zero stamp remains a safe fallback.
        if stamp is not None:
            marker.header.stamp = stamp
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.1

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.publisher.publish(marker)
        self.get_logger().info("Publishing Marker")


def run_scan(args=None):
    rclpy.init()
    reading_laser = ScanSubscriber()
    reading_laser.get_logger().info("Hello friend!")
    rclpy.spin(reading_laser)

    reading_laser.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    run_scan()
