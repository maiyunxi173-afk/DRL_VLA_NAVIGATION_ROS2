"""Create one collision-free randomized obstacle layout for the VLA test."""

import math
import random
import time

import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, String


OBSTACLE_BOUNDING_RADII = {
    "obstacle1": math.hypot(0.50, 0.50),
    "obstacle2": math.hypot(0.15, 0.35),
    "obstacle3": math.hypot(0.125, 0.125),
    "obstacle4": math.hypot(1.00, 0.75),
    "obstacle5": 0.50,
    "obstacle6": 0.25,
    "obstacle7": math.hypot(0.35, 0.10),
    "obstacle8": 0.75,
}


def yaw_from_quaternion(q):
    sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
    cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(sin_yaw, cos_yaw)


class RandomObstacleManager(Node):
    def __init__(self):
        super().__init__("random_obstacle_manager")
        self.declare_parameter("random_seed", -1)
        self.declare_parameter("layout_limit", 5.8)
        self.declare_parameter("obstacle_clearance", 0.30)
        self.declare_parameter("robot_bounding_radius", 0.22)
        self.declare_parameter("robot_clearance", 0.45)
        self.declare_parameter("randomize_robot", True)
        self.declare_parameter("robot_entity_name", "turtlebot3_waffle")
        self.declare_parameter("robot_spawn_limit", 2.75)
        self.declare_parameter("settling_time", 1.0)
        self.declare_parameter("target_minimum_distance", 4.0)
        self.declare_parameter("target_maximum_distance", 5.5)
        self.declare_parameter("target_minimum_bearing", 0.55)
        self.declare_parameter("target_maximum_bearing", 1.20)
        self.declare_parameter("target_world_limit", 6.0)

        param = lambda name: self.get_parameter(name).value
        seed = int(param("random_seed"))
        self.rng = random.Random(None if seed < 0 else seed)
        self.layout_limit = float(param("layout_limit"))
        self.obstacle_clearance = float(param("obstacle_clearance"))
        self.robot_bounding_radius = float(param("robot_bounding_radius"))
        self.robot_clearance = float(param("robot_clearance"))
        self.randomize_robot = bool(param("randomize_robot"))
        self.robot_entity_name = str(param("robot_entity_name"))
        self.robot_spawn_limit = float(param("robot_spawn_limit"))
        self.settling_time = float(param("settling_time"))
        self.target_minimum_distance = float(param("target_minimum_distance"))
        self.target_maximum_distance = float(param("target_maximum_distance"))
        self.target_minimum_bearing = float(param("target_minimum_bearing"))
        self.target_maximum_bearing = float(param("target_maximum_bearing"))
        self.target_world_limit = float(param("target_world_limit"))

        self.model_states = None
        self.odom = None
        self.move_queue = []
        self.pending_future = None
        self.pending_name = None
        self.layout_sampled = False
        self.settle_until = None
        self.finished = False

        self.create_subscription(ModelStates, "/gazebo/model_states", self.states_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")

        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.ready_publisher = self.create_publisher(
            Bool, "/vla_environment/ready", latched_qos
        )
        self.status_publisher = self.create_publisher(
            String, "/vla_environment/status", latched_qos
        )
        self.create_timer(0.1, self.tick)
        self.get_logger().info(
            "Waiting for Gazebo before creating a constrained random obstacle layout"
        )

    def states_cb(self, message):
        self.model_states = message

    def odom_cb(self, message):
        self.odom = message

    def tick(self):
        if self.finished:
            return
        if self.pending_future is not None:
            if not self.pending_future.done():
                return
            future = self.pending_future
            name = self.pending_name
            self.pending_future = None
            self.pending_name = None
            try:
                response = future.result()
            except Exception as exception:
                self.fail(f"failed to move {name}: {exception}")
                return
            if response is None or not response.success:
                detail = "no response" if response is None else response.status_message
                self.fail(f"Gazebo rejected {name}: {detail}")
                return

        if self.move_queue:
            self.move_next_obstacle()
            return

        if self.layout_sampled:
            if self.settle_until is None:
                self.settle_until = time.monotonic() + self.settling_time
                return
            if time.monotonic() < self.settle_until:
                return
            ready = Bool()
            ready.data = True
            self.ready_publisher.publish(ready)
            self.publish_status("READY: randomized obstacle layout is collision-free")
            self.finished = True
            self.get_logger().info(
                "Random obstacle layout ready; semantic target generation unlocked"
            )
            return

        if self.model_states is None or self.odom is None:
            return
        if not self.client.service_is_ready():
            self.client.wait_for_service(timeout_sec=0.0)
            return
        sampled = self.sample_valid_layout()
        if sampled is None:
            self.fail("could not sample a solvable collision-free obstacle layout")
            return
        layout, robot_target = sampled
        # Move obstacles first.  Their sampled positions avoid both the old and
        # new robot poses, so the final robot teleport cannot create overlap.
        self.move_queue = [
            (name, x, y, None) for name, (x, y, _radius) in layout.items()
        ]
        if self.randomize_robot:
            self.move_queue.append(
                (
                    self.robot_entity_name,
                    robot_target[0],
                    robot_target[1],
                    robot_target[2],
                )
            )
        self.layout_sampled = True
        self.get_logger().info(
            "Valid randomized layout sampled; moving obstacle1..8"
            + (" and robot" if self.randomize_robot else "")
        )

    def sample_valid_layout(self):
        try:
            robot_index = self.model_states.name.index(self.robot_entity_name)
        except ValueError:
            return None
        robot_pose = self.model_states.pose[robot_index]
        initial_robot_x = float(robot_pose.position.x)
        initial_robot_y = float(robot_pose.position.y)

        ordered = sorted(
            OBSTACLE_BOUNDING_RADII.items(), key=lambda item: item[1], reverse=True
        )
        for _layout_attempt in range(60):
            if self.randomize_robot:
                robot_x = self.rng.uniform(
                    -self.robot_spawn_limit, self.robot_spawn_limit
                )
                robot_y = self.rng.uniform(
                    -self.robot_spawn_limit, self.robot_spawn_limit
                )
                robot_yaw = self.rng.uniform(-math.pi, math.pi)
            else:
                robot_x = initial_robot_x
                robot_y = initial_robot_y
                robot_yaw = yaw_from_quaternion(robot_pose.orientation)
            layout = {}
            for name, radius in ordered:
                position = self.sample_obstacle_position(
                    radius,
                    layout,
                    robot_x,
                    robot_y,
                    initial_robot_x,
                    initial_robot_y,
                )
                if position is None:
                    break
                layout[name] = (position[0], position[1], radius)
            if len(layout) != len(ordered):
                continue
            if self.layout_supports_target(layout, robot_x, robot_y, robot_yaw):
                return layout, (robot_x, robot_y, robot_yaw)
        return None

    def sample_obstacle_position(
        self,
        radius,
        layout,
        robot_x,
        robot_y,
        initial_robot_x,
        initial_robot_y,
    ):
        limit = self.layout_limit - radius
        for _attempt in range(2000):
            x = self.rng.uniform(-limit, limit)
            y = self.rng.uniform(-limit, limit)
            robot_required = radius + self.robot_bounding_radius + self.robot_clearance
            if math.hypot(x - robot_x, y - robot_y) < robot_required:
                continue
            if math.hypot(x - initial_robot_x, y - initial_robot_y) < robot_required:
                continue
            overlap = any(
                math.hypot(x - other_x, y - other_y)
                < radius + other_radius + self.obstacle_clearance
                for other_x, other_y, other_radius in layout.values()
            )
            if not overlap:
                return x, y
        return None

    def layout_supports_target(self, layout, robot_x, robot_y, robot_yaw):
        obstacles = list(layout.values())
        red_radius = math.sqrt(2.0) * 0.20
        for _attempt in range(4000):
            distance = self.rng.uniform(
                self.target_minimum_distance, self.target_maximum_distance
            )
            bearing = self.rng.choice((-1.0, 1.0)) * self.rng.uniform(
                self.target_minimum_bearing, self.target_maximum_bearing
            )
            target_angle = robot_yaw + bearing
            x = robot_x + distance * math.cos(target_angle)
            y = robot_y + distance * math.sin(target_angle)
            if abs(x) > self.target_world_limit or abs(y) > self.target_world_limit:
                continue
            if any(
                math.hypot(x - ox, y - oy) < radius + red_radius + 0.15
                for ox, oy, radius in obstacles
            ):
                continue
            if self.target_has_visible_challenge(
                robot_x, robot_y, x, y, obstacles
            ):
                return True
        return False

    def target_has_visible_challenge(self, start_x, start_y, end_x, end_y, obstacles):
        dx = end_x - start_x
        dy = end_y - start_y
        length_squared = dx * dx + dy * dy
        visible = True
        challenge = False
        for obstacle_x, obstacle_y, radius in obstacles:
            projection = (
                (obstacle_x - start_x) * dx + (obstacle_y - start_y) * dy
            ) / max(length_squared, 1e-9)
            if projection <= 0.05 or projection >= 0.95:
                continue
            closest_x = start_x + projection * dx
            closest_y = start_y + projection * dy
            lateral = math.hypot(obstacle_x - closest_x, obstacle_y - closest_y)
            if lateral < radius + 0.05:
                visible = False
                break
            if (
                0.15 < projection < 0.85
                and lateral <= radius + self.robot_bounding_radius + 0.15
            ):
                challenge = True
        return visible and challenge

    def move_next_obstacle(self):
        name, x, y, requested_yaw = self.move_queue.pop(0)
        try:
            index = self.model_states.name.index(name)
            z = max(0.0, float(self.model_states.pose[index].position.z))
        except ValueError:
            self.fail(f"Gazebo entity {name!r} is missing")
            return
        yaw = (
            self.rng.uniform(-math.pi, math.pi)
            if requested_yaw is None
            else requested_yaw
        )
        request = SetEntityState.Request()
        request.state.name = name
        request.state.pose.position.x = x
        request.state.pose.position.y = y
        request.state.pose.position.z = z
        request.state.pose.orientation.z = math.sin(yaw / 2.0)
        request.state.pose.orientation.w = math.cos(yaw / 2.0)
        request.state.reference_frame = "world"
        self.pending_name = name
        self.pending_future = self.client.call_async(request)
        if name == self.robot_entity_name:
            self.get_logger().info(
                f"Random robot start: Gazebo=({x:.2f}, {y:.2f}), "
                f"yaw={math.degrees(yaw):.1f} deg"
            )

    def publish_status(self, text):
        message = String()
        message.data = text
        self.status_publisher.publish(message)

    def fail(self, detail):
        self.finished = True
        self.publish_status(f"ERROR: {detail}")
        self.get_logger().error(detail)


def main(args=None):
    rclpy.init(args=args)
    node = RandomObstacleManager()
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
