import time
import rclpy
from ros_nodes import (
    ResetWorldClient,
    SetModelStateClient,
    CmdVelPublisher,
    MarkerPublisher,
    PhysicsClient,
    SensorSubscriber,
)
import numpy as np
from geometry_msgs.msg import Pose
from squaternion import Quaternion


class ROS_env:
    def __init__(
        self,
        init_target_distance=1.0,
        target_dist_increase=0.02,
        max_target_dist=4.5,
        min_target_distance=0.8,
        target_reached_delta=0.5,
        collision_delta=0.32,
        control_period=0.1,
        sensor_timeout=2.0,
        args=None,
    ):
        rclpy.init(args=args)
        self.cmd_vel_publisher = CmdVelPublisher()
        self.robot_state_publisher = SetModelStateClient()
        self.world_reset = ResetWorldClient()
        self.physics_client = PhysicsClient()
        self.publish_target = MarkerPublisher()
        self.element_positions = [
            [-2.93, 3.17],
            [2.86, -3.0],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]
        self.sensor_subscriber = SensorSubscriber()
        self.target_dist = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.max_target_dist = max_target_dist
        self.min_target_distance = min_target_distance
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        self.control_period = control_period
        self.sensor_timeout = sensor_timeout
        self.target = self.set_target_position([0.0, 0.0])
        self.previous_distance = None
        self.progress_reward_scale = 15.0
        self.heading_reward_scale = 0.25
        self.turn_alignment_scale = 0.15
        self.turn_penalty_scale = 0.03
        self.obstacle_penalty_scale = 0.5
        self.obstacle_penalty_distance = 0.8
        self.step_penalty = 0.01

    def step(
        self,
        lin_velocity=0.0,
        ang_velocity=0.1,
        update_curriculum=True,
    ):
        previous_scan_sequence, previous_odom_sequence = (
            self.sensor_subscriber.get_sequences()
        )
        self.cmd_vel_publisher.publish_cmd_vel(lin_velocity, ang_velocity)
        self.physics_client.unpause_physics()
        started_at = time.monotonic()
        deadline = started_at + self.sensor_timeout
        try:
            while time.monotonic() < deadline:
                rclpy.spin_once(self.sensor_subscriber, timeout_sec=0.05)
                period_elapsed = time.monotonic() - started_at >= self.control_period
                sensors_fresh = self.sensor_subscriber.has_fresh_sensor(
                    previous_scan_sequence,
                    previous_odom_sequence,
                )
                if period_elapsed and sensors_fresh:
                    break
            else:
                raise TimeoutError(
                    "Timed out waiting for fresh /scan and /odom frames"
                )
        finally:
            self.physics_client.pause_physics()

        (
            latest_scan,
            latest_position,
            latest_orientation,
        ) = self.sensor_subscriber.get_latest_sensor()

        distance, cos, sin, _ = self.get_dist_sincos(
            latest_position, latest_orientation
        )
        collision = self.check_collision(latest_scan)
        goal = self.check_target(
            distance,
            collision,
            update_curriculum=update_curriculum,
        )
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(
            goal=goal,
            collision=collision,
            action=action,
            laser_scan=latest_scan,
            cos=cos,
            sin=sin,
            distance=distance,
            previous_distance=self.previous_distance,
            progress_reward_scale=self.progress_reward_scale,
            heading_reward_scale=self.heading_reward_scale,
            turn_alignment_scale=self.turn_alignment_scale,
            turn_penalty_scale=self.turn_penalty_scale,
            obstacle_penalty_scale=self.obstacle_penalty_scale,
            obstacle_penalty_distance=self.obstacle_penalty_distance,
            step_penalty=self.step_penalty,
        )
        self.previous_distance = distance

        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def reset(self):
        self.previous_distance = None
        self.world_reset.reset_world()
        action = [0.0, 0.0]
        self.cmd_vel_publisher.publish_cmd_vel(
            linear_velocity=action[0], angular_velocity=action[1]
        )

        self.element_positions = [
            [-2.93, 3.17],
            [2.86, -3.0],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]
        self.set_positions()

        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0],
            ang_velocity=action[1],
            update_curriculum=False,
        )
        self.publish_target.publish(
            self.target[0],
            self.target[1],
            stamp=self.sensor_subscriber.get_latest_odom_stamp(),
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    def eval(self, scenario):
        self.previous_distance = None
        self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)

        self.target = [scenario[-1].x, scenario[-1].y]

        for element in scenario[:-1]:
            self.set_position(element.name, element.x, element.y, element.angle)

        latest_scan, distance, cos, sin, _, _, a, reward = self.step(
            lin_velocity=0.0,
            ang_velocity=0.0,
            update_curriculum=False,
        )
        self.publish_target.publish(
            self.target[0],
            self.target[1],
            stamp=self.sensor_subscriber.get_latest_odom_stamp(),
        )
        return latest_scan, distance, cos, sin, False, False, a, reward

    def set_target_position(self, robot_position):
        """Sample a target in an annulus around the robot.

        This prevents targets from spawning inside the success radius and makes
        target_dist an actual radial curriculum distance.
        """
        pos = False
        while not pos:
            max_distance = max(self.target_dist, self.min_target_distance)
            distance = np.random.uniform(self.min_target_distance, max_distance)
            angle = np.random.uniform(-np.pi, np.pi)
            x = robot_position[0] + distance * np.cos(angle)
            y = robot_position[1] + distance * np.sin(angle)
            inside_world = -4.0 <= x <= 4.0 and -4.0 <= y <= 4.0
            pos = inside_world and self.check_position(x, y, 1.2)
        self.element_positions.append([x, y])
        return [x, y]

    def set_random_position(self, name):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        while not pos:
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            pos = self.check_position(x, y, 1.8)
        self.element_positions.append([x, y])
        self.set_position(name, x, y, angle)

    def set_robot_position(self):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        while not pos:
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            pos = self.check_position(x, y, 1.8)
        self.set_position("turtlebot3_waffle", x, y, angle)
        return x, y

    def set_position(self, name, x, y, angle):
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = quaternion.x
        pose.orientation.y = quaternion.y
        pose.orientation.z = quaternion.z
        pose.orientation.w = quaternion.w

        self.robot_state_publisher.set_state(name, pose)

    def set_positions(self):
        for i in range(4, 8):
            name = "obstacle" + str(i + 1)
            self.set_random_position(name)

        robot_position = self.set_robot_position()
        self.target = self.set_target_position(robot_position)

    def check_position(self, x, y, min_dist):
        pos = True
        for element in self.element_positions:
            distance_vector = [element[0] - x, element[1] - y]
            distance = np.linalg.norm(distance_vector)
            if distance < min_dist:
                pos = False
        return pos

    def check_collision(self, laser_scan):
        finite_scan = np.asarray(laser_scan, dtype=np.float32)
        finite_scan = finite_scan[np.isfinite(finite_scan)]
        if finite_scan.size == 0:
            raise RuntimeError("LaserScan contains no finite measurements")
        if float(finite_scan.min()) < self.collision_delta:
            return True
        return False

    def check_target(self, distance, collision, update_curriculum=True):
        if distance < self.target_reached_delta and not collision:
            if update_curriculum:
                self.target_dist += self.target_dist_increase
                if self.target_dist > self.max_target_dist:
                    self.target_dist = self.max_target_dist
            return True
        return False

    def get_dist_sincos(self, odom_position, odom_orientation):
        # Calculate robot heading from odometry data
        odom_x = odom_position.x
        odom_y = odom_position.y
        quaternion = Quaternion(
            odom_orientation.w,
            odom_orientation.x,
            odom_orientation.y,
            odom_orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        pose_vector = [np.cos(angle), np.sin(angle)]
        goal_vector = [self.target[0] - odom_x, self.target[1] - odom_y]

        distance = np.linalg.norm(goal_vector)
        cos, sin = self.cossin(pose_vector, goal_vector)

        return distance, cos, sin, angle

    @staticmethod
    def get_reward(
        goal,
        collision,
        action,
        laser_scan,
        cos=1.0,
        sin=0.0,
        distance=None,
        previous_distance=None,
        progress_reward_scale=15.0,
        heading_reward_scale=0.25,
        turn_alignment_scale=0.15,
        turn_penalty_scale=0.03,
        obstacle_penalty_scale=0.5,
        obstacle_penalty_distance=0.8,
        step_penalty=0.01,
    ):
        if goal:
            return 20.0
        elif collision:
            return -20.0
        else:
            progress = 0.0
            if distance is not None and previous_distance is not None:
                progress = previous_distance - distance

            finite_scan = np.asarray(laser_scan, dtype=np.float32)
            finite_scan = finite_scan[np.isfinite(finite_scan)]
            min_scan = float(finite_scan.min()) if finite_scan.size else 0.0
            obstacle_margin = max(0.0, obstacle_penalty_distance - min_scan)
            obstacle_ratio = obstacle_margin / obstacle_penalty_distance
            linear_normalized = float(np.clip(action[0] / 0.35, 0.0, 1.0))
            angular_normalized = float(np.clip(action[1] / 0.7, -1.0, 1.0))
            progress_reward = progress_reward_scale * progress
            heading_reward = heading_reward_scale * linear_normalized * cos
            turn_alignment_reward = (
                turn_alignment_scale * angular_normalized * sin
            )
            # Turning is only wasteful when the robot already faces the goal.
            turn_penalty = (
                turn_penalty_scale
                * abs(angular_normalized)
                * max(0.0, cos)
            )
            obstacle_penalty = obstacle_penalty_scale * obstacle_ratio
            reward = (
                progress_reward
                + heading_reward
                + turn_alignment_reward
                - turn_penalty
                - obstacle_penalty
                - step_penalty
            )
            return float(np.clip(reward, -20.0, 20.0))

    @staticmethod
    def cossin(vec1, vec2):
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        if vec1_norm < 1e-8 or vec2_norm < 1e-8:
            return 1.0, 0.0
        vec1 = vec1 / vec1_norm
        vec2 = vec2 / vec2_norm
        cos = np.dot(vec1, vec2)
        sin = float(vec1[0] * vec2[1] - vec1[1] * vec2[0])

        return cos, sin

    def close(self):
        self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)
        if rclpy.ok():
            rclpy.shutdown()
