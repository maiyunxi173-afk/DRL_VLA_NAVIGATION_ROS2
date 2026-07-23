"""Manage a random physical red-box target for the VLA simulation stack.

The RViz marker is only a diagnostic mirror.  The actual target is the
``vla_red_box`` Gazebo entity, so RGB/depth cameras and LaserScan can observe
the same object.  This node never publishes ``/cmd_vel`` or a navigation goal.
"""

import math
import random
import time

import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker


# Conservative horizontal bounding radii derived from the collision geometry
# in turtlebot3_gazebo/models/obstacle*/model.sdf.  Using circumscribed circles
# is deliberately stricter than centre-distance-only checks and remains valid
# if a box obstacle is rotated in Gazebo.
OBSTACLE_BOUNDING_RADII = {
    "obstacle1": math.hypot(0.50, 0.50),       # box 1.0 x 1.0
    "obstacle2": math.hypot(0.15, 0.35),       # box 0.3 x 0.7
    "obstacle3": math.hypot(0.125, 0.125),     # box 0.25 x 0.25
    "obstacle4": math.hypot(1.00, 0.75),       # box 2.0 x 1.5
    "obstacle5": 0.50,                         # cylinder
    "obstacle6": 0.25,                         # cylinder
    "obstacle7": math.hypot(0.35, 0.10),       # box 0.7 x 0.2
    "obstacle8": 0.75,                         # cylinder
}


def yaw_from_quaternion(q):
    sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
    cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(sin_yaw, cos_yaw)


class RandomRedBoxManager(Node):
    """Move the semantic target safely and expose ground truth for diagnostics."""

    def __init__(self):
        super().__init__("random_red_box_manager")
        self.declare_parameter("entity_name", "vla_red_box")
        self.declare_parameter("robot_entity_name", "turtlebot3_waffle")
        # A negative seed produces a genuinely random integration-test target;
        # set a non-negative seed from ROS parameters for reproducible tests.
        self.declare_parameter("random_seed", -1)
        self.declare_parameter("minimum_spawn_distance", 4.0)
        self.declare_parameter("maximum_spawn_distance", 5.5)
        self.declare_parameter("minimum_spawn_bearing", 0.55)
        self.declare_parameter("maximum_spawn_bearing", 1.20)
        self.declare_parameter("world_limit", 6.0)
        self.declare_parameter("obstacle_clearance", 0.15)
        self.declare_parameter("red_box_half_extent", 0.20)
        self.declare_parameter("robot_bounding_radius", 0.22)
        self.declare_parameter("robot_obstacle_clearance", 0.10)
        self.declare_parameter("navigation_path_clearance", 0.15)
        self.declare_parameter("line_of_sight_clearance", 0.05)
        self.declare_parameter("require_avoidance_challenge", True)
        self.declare_parameter("wait_for_environment_ready", True)
        self.declare_parameter("episode_timeout", 60.0)
        # The target is an obstacle, so a semantic episode is complete at a
        # safe stand-off distance rather than after touching its centre.
        self.declare_parameter("semantic_reach_center_distance", 0.80)
        self.declare_parameter("semantic_reach_dwell", 0.50)
        self.declare_parameter("marker_topic", "/visualization_marker")
        self.declare_parameter("truth_topic", "/vla/semantic_target/truth")

        param = lambda name: self.get_parameter(name).value
        self.entity_name = str(param("entity_name"))
        self.robot_entity_name = str(param("robot_entity_name"))
        self.minimum_spawn_distance = float(param("minimum_spawn_distance"))
        self.maximum_spawn_distance = float(param("maximum_spawn_distance"))
        self.minimum_spawn_bearing = abs(float(param("minimum_spawn_bearing")))
        self.maximum_spawn_bearing = abs(float(param("maximum_spawn_bearing")))
        if self.minimum_spawn_bearing > self.maximum_spawn_bearing:
            raise ValueError("minimum_spawn_bearing must not exceed maximum_spawn_bearing")
        self.world_limit = float(param("world_limit"))
        self.obstacle_clearance = float(param("obstacle_clearance"))
        self.red_box_half_extent = float(param("red_box_half_extent"))
        self.red_box_bounding_radius = math.sqrt(2.0) * self.red_box_half_extent
        self.robot_bounding_radius = float(param("robot_bounding_radius"))
        self.robot_obstacle_clearance = float(param("robot_obstacle_clearance"))
        self.navigation_path_clearance = float(param("navigation_path_clearance"))
        self.line_of_sight_clearance = float(param("line_of_sight_clearance"))
        self.require_avoidance_challenge = bool(param("require_avoidance_challenge"))
        self.environment_ready = not bool(param("wait_for_environment_ready"))
        self.episode_timeout = max(0.0, float(param("episode_timeout")))
        self.semantic_reach_center_distance = float(
            param("semantic_reach_center_distance")
        )
        self.semantic_reach_dwell = float(param("semantic_reach_dwell"))
        random_seed = int(param("random_seed"))
        self.rng = random.Random(None if random_seed < 0 else random_seed)

        self.model_states = None
        self.odom = None
        self.target_world = None
        self.target_odom = None
        self.pending_pose = None
        self.pending_future = None
        self.next_spawn_at = time.monotonic()
        self.spawn_count = 0
        self.reach_started_at = None
        self.episode_started_at = None
        self.episode_finished = False
        self.episode_result = None

        self.create_subscription(ModelStates, "/gazebo/model_states", self.states_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(
            Bool,
            "/vla_navigation/semantic_reached",
            self.semantic_reached_cb,
            10,
        )
        ready_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Bool,
            "/vla_environment/ready",
            self.environment_ready_cb,
            ready_qos,
        )
        self.set_state_client = self.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )

        marker_qos = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.marker_publisher = self.create_publisher(
            Marker, str(param("marker_topic")), marker_qos
        )
        self.truth_publisher = self.create_publisher(
            PointStamped, str(param("truth_topic")), 10
        )
        self.semantic_reset_publisher = self.create_publisher(
            Bool, "/vla_navigation/semantic_reset", 10
        )
        self.semantic_reached_publisher = self.create_publisher(
            Bool, "/vla_navigation/semantic_reached", 10
        )
        self.semantic_timeout_publisher = self.create_publisher(
            Bool, "/vla_navigation/semantic_timeout", 10
        )
        result_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.result_publisher = self.create_publisher(
            String, "/vla_navigation/result", result_qos
        )
        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.publish_diagnostics)

        self.get_logger().info(
            "Random semantic target ready: physical Gazebo red box + RViz CUBE; "
            "this node never publishes /cmd_vel or /vla_goal"
        )

    def states_cb(self, message):
        self.model_states = message

    def odom_cb(self, message):
        self.odom = message

    def environment_ready_cb(self, message):
        if message.data and not self.environment_ready:
            self.environment_ready = True
            self.get_logger().info(
                "Random obstacle environment ready; target sampling enabled"
            )

    def semantic_reached_cb(self, message):
        if not message.data or self.episode_finished:
            return
        self.finish_episode("SUCCESS", "semantic target reached")

    def tick(self):
        if self.episode_finished:
            return
        if not self.environment_ready:
            return
        if self.pending_future is not None:
            if not self.pending_future.done():
                return
            future = self.pending_future
            pose = self.pending_pose
            self.pending_future = None
            self.pending_pose = None
            try:
                response = future.result()
            except Exception as exception:
                self.get_logger().error(f"Failed to move red box: {exception}")
                self.next_spawn_at = time.monotonic() + 1.0
                return
            if response is None or not response.success:
                status = "no response" if response is None else response.status_message
                self.get_logger().error(f"Gazebo rejected red-box pose: {status}")
                self.next_spawn_at = time.monotonic() + 1.0
                return
            self.commit_target(pose)
            return

        if self.target_world is not None:
            self.check_semantic_reach()
            if self.target_world is not None:
                self.check_episode_timeout()
            return
        if time.monotonic() < self.next_spawn_at:
            return
        if self.model_states is None or self.odom is None:
            return
        if not self.set_state_client.service_is_ready():
            self.set_state_client.wait_for_service(timeout_sec=0.0)
            return

        pose = self.sample_pose()
        if pose is None:
            self.get_logger().warning(
                "No collision-free red-box position found; retrying",
                throttle_duration_sec=5.0,
            )
            return
        request = SetEntityState.Request()
        request.state.name = self.entity_name
        request.state.pose.position.x = pose["world_x"]
        request.state.pose.position.y = pose["world_y"]
        request.state.pose.position.z = 0.2
        request.state.pose.orientation.w = 1.0
        request.state.reference_frame = "world"
        self.pending_pose = pose
        self.pending_future = self.set_state_client.call_async(request)

    def check_semantic_reach(self):
        """End a simulation episode reliably even if close-range vision drops.

        Demo5 remains responsible for finding the object and producing the
        navigation subgoals.  This ground-truth check is only the Gazebo task
        completion signal, analogous to a real robot's task verifier.
        """
        if self.target_odom is None or self.odom is None:
            self.reach_started_at = None
            return
        pose = self.odom.pose.pose.position
        distance = math.hypot(
            float(pose.x) - self.target_odom[0],
            float(pose.y) - self.target_odom[1],
        )
        if distance > self.semantic_reach_center_distance:
            self.reach_started_at = None
            return
        now = time.monotonic()
        if self.reach_started_at is None:
            self.reach_started_at = now
            return
        if now - self.reach_started_at < self.semantic_reach_dwell:
            return
        reached = Bool()
        reached.data = True
        self.get_logger().info(
            f"Red-box episode complete at safe stand-off: center_distance={distance:.2f} m"
        )
        self.semantic_reached_publisher.publish(reached)
        # Finish synchronously; delivery of our own reached message is async.
        self.finish_episode(
            "SUCCESS", f"semantic target reached at center distance {distance:.2f} m"
        )

    def check_episode_timeout(self):
        """Stop and report a failed one-shot integration-test episode."""
        if self.episode_timeout <= 0.0 or self.episode_started_at is None:
            return
        now = time.monotonic()
        if now - self.episode_started_at < self.episode_timeout:
            return
        timed_out = Bool()
        timed_out.data = True
        self.semantic_timeout_publisher.publish(timed_out)
        self.finish_episode(
            "ERROR", f"semantic episode timed out after {self.episode_timeout:.1f}s"
        )

    def finish_episode(self, outcome, detail):
        if self.episode_finished:
            return
        self.episode_finished = True
        self.episode_result = outcome
        self.reach_started_at = None
        self.episode_started_at = None
        result = String()
        result.data = f"{outcome}: {detail}"
        self.result_publisher.publish(result)
        log = self.get_logger().info if outcome == "SUCCESS" else self.get_logger().error
        log(f"VLA-DRL integration result: {result.data}; controller stopped")

    def sample_pose(self):
        try:
            robot_index = self.model_states.name.index(self.robot_entity_name)
        except ValueError:
            self.get_logger().warning(
                f"Waiting for Gazebo entity {self.robot_entity_name!r}",
                throttle_duration_sec=5.0,
            )
            return None

        robot_world = self.model_states.pose[robot_index]
        world_yaw = yaw_from_quaternion(robot_world.orientation)
        odom_pose = self.odom.pose.pose
        odom_yaw = yaw_from_quaternion(odom_pose.orientation)
        obstacles = []
        for name, pose in zip(self.model_states.name, self.model_states.pose):
            if name.startswith("obstacle"):
                radius = OBSTACLE_BOUNDING_RADII.get(name, 0.75)
                obstacles.append(
                    (float(pose.position.x), float(pose.position.y), radius)
                )

        robot_x = float(robot_world.position.x)
        robot_y = float(robot_world.position.y)
        if not self.is_robot_pose_safe(robot_x, robot_y, obstacles):
            self.get_logger().error(
                "Robot starts too close to an obstacle; refusing to place semantic target",
                throttle_duration_sec=5.0,
            )
            return None

        for _attempt in range(2000):
            distance = self.rng.uniform(
                self.minimum_spawn_distance, self.maximum_spawn_distance
            )
            bearing_magnitude = self.rng.uniform(
                self.minimum_spawn_bearing, self.maximum_spawn_bearing
            )
            relative_bearing = self.rng.choice((-1.0, 1.0)) * bearing_magnitude
            world_bearing = world_yaw + relative_bearing
            world_x = robot_world.position.x + distance * math.cos(world_bearing)
            world_y = robot_world.position.y + distance * math.sin(world_bearing)
            if abs(world_x) > self.world_limit or abs(world_y) > self.world_limit:
                continue
            if not self.is_collision_free(world_x, world_y, obstacles):
                continue
            if not self.has_clear_line_of_sight(
                robot_x,
                robot_y,
                world_x,
                world_y,
                obstacles,
            ):
                continue
            if self.require_avoidance_challenge and not self.has_avoidance_challenge(
                robot_x, robot_y, world_x, world_y, obstacles
            ):
                continue

            # Preserve the same robot-relative vector while expressing it in odom.
            odom_bearing = odom_yaw + relative_bearing
            return {
                "world_x": float(world_x),
                "world_y": float(world_y),
                "odom_x": float(
                    odom_pose.position.x + distance * math.cos(odom_bearing)
                ),
                "odom_y": float(
                    odom_pose.position.y + distance * math.sin(odom_bearing)
                ),
            }
        return None

    def is_robot_pose_safe(self, x, y, obstacles):
        for obstacle_x, obstacle_y, obstacle_radius in obstacles:
            required = (
                obstacle_radius
                + self.robot_bounding_radius
                + self.robot_obstacle_clearance
            )
            if math.hypot(x - obstacle_x, y - obstacle_y) < required:
                return False
        return abs(x) <= self.world_limit and abs(y) <= self.world_limit

    def is_collision_free(self, x, y, obstacles):
        for obstacle_x, obstacle_y, obstacle_radius in obstacles:
            required = (
                obstacle_radius
                + self.red_box_bounding_radius
                + self.obstacle_clearance
            )
            if math.hypot(x - obstacle_x, y - obstacle_y) < required:
                return False
        return True

    def has_clear_line_of_sight(self, start_x, start_y, end_x, end_y, obstacles):
        """Keep the first semantic detection visible to the forward camera."""
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        length_squared = segment_x * segment_x + segment_y * segment_y
        if length_squared <= 1e-9:
            return False
        for obstacle_x, obstacle_y, obstacle_radius in obstacles:
            projection = (
                (obstacle_x - start_x) * segment_x
                + (obstacle_y - start_y) * segment_y
            ) / length_squared
            # Only obstacles strictly between robot and semantic target matter.
            if projection <= 0.05 or projection >= 0.95:
                continue
            closest_x = start_x + projection * segment_x
            closest_y = start_y + projection * segment_y
            required = obstacle_radius + self.line_of_sight_clearance
            if math.hypot(obstacle_x - closest_x, obstacle_y - closest_y) < required:
                return False
        return True

    def has_avoidance_challenge(self, start_x, start_y, end_x, end_y, obstacles):
        """Require a visible obstacle near the straight driving corridor.

        The red-box centre ray stays visually clear, while the Waffle footprint
        cannot safely follow that ray without steering around the obstacle.
        """
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        length_squared = segment_x * segment_x + segment_y * segment_y
        if length_squared <= 1e-9:
            return False
        for obstacle_x, obstacle_y, obstacle_radius in obstacles:
            projection = (
                (obstacle_x - start_x) * segment_x
                + (obstacle_y - start_y) * segment_y
            ) / length_squared
            if projection <= 0.15 or projection >= 0.85:
                continue
            closest_x = start_x + projection * segment_x
            closest_y = start_y + projection * segment_y
            lateral_distance = math.hypot(
                obstacle_x - closest_x, obstacle_y - closest_y
            )
            visually_clear = (
                lateral_distance
                >= obstacle_radius + self.line_of_sight_clearance
            )
            driving_corridor_blocked = (
                lateral_distance
                <= obstacle_radius
                + self.robot_bounding_radius
                + self.navigation_path_clearance
            )
            if visually_clear and driving_corridor_blocked:
                return True
        return False

    def commit_target(self, pose):
        self.target_world = (pose["world_x"], pose["world_y"])
        self.target_odom = (pose["odom_x"], pose["odom_y"])
        self.spawn_count += 1
        self.reach_started_at = None
        self.episode_started_at = time.monotonic()
        reset = Bool()
        reset.data = True
        self.semantic_reset_publisher.publish(reset)
        self.publish_diagnostics()
        self.get_logger().info(
            f"Random red box #{self.spawn_count}: odom=({pose['odom_x']:.2f}, "
            f"{pose['odom_y']:.2f}), Gazebo=({pose['world_x']:.2f}, "
            f"{pose['world_y']:.2f})"
        )

    def publish_diagnostics(self):
        if self.target_odom is None or self.odom is None:
            return
        x, y = self.target_odom
        stamp = self.odom.header.stamp

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = stamp
        marker.ns = "semantic_target"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.a = 0.85
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_publisher.publish(marker)

        truth = PointStamped()
        truth.header.frame_id = "odom"
        truth.header.stamp = stamp
        truth.point.x = x
        truth.point.y = y
        truth.point.z = 0.2
        self.truth_publisher.publish(truth)


def main(args=None):
    rclpy.init(args=args)
    node = RandomRedBoxManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
