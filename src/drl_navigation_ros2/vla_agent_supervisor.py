"""Low-frequency VLA brain: arbitrate goals and manage bounded fallbacks.

All high-frequency velocity control remains in ``inference.py``. This node
only selects among cached goal sources and requests search/stop modes, so there
is exactly one publisher of ``/cmd_vel``.
"""

import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, String


@dataclass
class GoalCandidate:
    message: PointStamped
    received_at: float
    priority: int
    ttl: float


class VlaAgentSupervisor(Node):
    def __init__(self):
        super().__init__("vla_agent_supervisor")
        self.declare_parameter("brain_rate", 2.0)
        self.declare_parameter("cached_goal_hold", 10.0)
        # 0.35 rad/s needs about 18 seconds for a full revolution. Keep search
        # long enough for several revolutions when the target spawns outside
        # the RGB camera field of view.
        self.declare_parameter("search_duration", 60.0)
        self.declare_parameter("initial_wait", 4.0)
        self.declare_parameter("minimum_goal_change", 0.05)
        self.declare_parameter("completed_goal_rearm_distance", 0.50)
        self.declare_parameter("subgoal_refresh_wait", 2.0)
        self.declare_parameter("new_target_observation_wait", 2.0)
        self.declare_parameter("target_mode", "vision")
        self.declare_parameter("wait_for_environment_ready", True)

        self.brain_rate = max(0.1, float(self.get_parameter("brain_rate").value))
        self.cached_goal_hold = max(
            0.0, float(self.get_parameter("cached_goal_hold").value)
        )
        self.search_duration = max(
            0.0, float(self.get_parameter("search_duration").value)
        )
        self.initial_wait = max(0.0, float(self.get_parameter("initial_wait").value))
        self.minimum_goal_change = max(
            0.0, float(self.get_parameter("minimum_goal_change").value)
        )
        self.completed_goal_rearm_distance = max(
            0.0,
            float(self.get_parameter("completed_goal_rearm_distance").value),
        )
        self.subgoal_refresh_wait = max(
            0.0, float(self.get_parameter("subgoal_refresh_wait").value)
        )
        self.new_target_observation_wait = max(
            0.0, float(self.get_parameter("new_target_observation_wait").value)
        )
        self.target_mode = str(self.get_parameter("target_mode").value)
        self.environment_ready = not bool(
            self.get_parameter("wait_for_environment_ready").value
        )
        if self.target_mode not in ("vision", "demo6"):
            raise ValueError("target_mode must be 'vision' or 'demo6'")

        self.candidates = {}
        self.active_source = None
        self.active_goal = None
        self.active_last_seen_at = None
        self.active_expiry_at = None
        self.search_started_at = None
        self.started_at = time.monotonic()
        self.ever_had_goal = False
        self.goal_completed = False
        # A semantic completion closes the current target episode.  Demo5 may
        # continue publishing the old bbox for a few frames while Gazebo moves
        # the box; those stale detections must not re-arm navigation.
        self.awaiting_semantic_reset = False
        self.subgoal_wait_until = None
        self.cancel_sent = False
        self.completed_goals = {}
        self.last_state = None
        self.last_sac_status = None

        self.create_subscription(
            PointStamped,
            "/vla_goal/source/manual",
            lambda message: self.cache_goal("manual", message, 300, 30.0),
            10,
        )
        self.create_subscription(
            PointStamped,
            "/vla_goal/source/demo6",
            lambda message: self.cache_goal("demo6", message, 200, 5.0),
            10,
        )
        self.create_subscription(
            PointStamped,
            "/vla_goal/source/vision",
            lambda message: self.cache_goal("vision", message, 100, 15.0),
            10,
        )
        self.create_subscription(
            String, "/sac_navigation/status", self.sac_status_callback, 10
        )
        self.create_subscription(
            Bool,
            "/vla_navigation/semantic_reached",
            self.semantic_reached_callback,
            10,
        )
        self.create_subscription(
            Bool,
            "/vla_navigation/semantic_reset",
            self.semantic_reset_callback,
            10,
        )
        self.create_subscription(
            Bool,
            "/vla_navigation/semantic_timeout",
            self.semantic_timeout_callback,
            10,
        )
        environment_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Bool,
            "/vla_environment/ready",
            self.environment_ready_callback,
            environment_qos,
        )
        self.goal_publisher = self.create_publisher(PointStamped, "/vla_goal", 10)
        self.search_publisher = self.create_publisher(
            Bool, "/vla_navigation/search", 10
        )
        self.semantic_reached_publisher = self.create_publisher(
            Bool, "/vla_navigation/semantic_reached", 10
        )
        self.cancel_publisher = self.create_publisher(
            Bool, "/vla_navigation/cancel", 10
        )
        self.state_publisher = self.create_publisher(
            String, "/vla_agent/state", 10
        )
        self.create_timer(1.0 / self.brain_rate, self.brain_tick)

        self.get_logger().info(
            f"VLA brain ready: target_mode={self.target_mode}, manual has highest "
            "priority; SAC owns /cmd_vel"
        )

    def cache_goal(self, source, message, priority, ttl):
        if source in ("vision", "demo6") and source != self.target_mode:
            return
        if source in ("vision", "demo6") and self.awaiting_semantic_reset:
            return
        if message.header.frame_id.lstrip("/") != "odom":
            self.get_logger().warning(
                f"Rejected {source} goal outside odom frame: "
                f"{message.header.frame_id!r}"
            )
            return
        if not math.isfinite(message.point.x) or not math.isfinite(message.point.y):
            self.get_logger().warning(f"Rejected invalid {source} goal")
            return
        completed = self.completed_goals.get(source)
        if completed is not None and math.hypot(
            message.point.x - completed.point.x,
            message.point.y - completed.point.y,
        ) < self.completed_goal_rearm_distance:
            return
        now = time.monotonic()
        self.candidates[source] = GoalCandidate(message, now, priority, ttl)
        self.goal_completed = False
        self.subgoal_wait_until = None
        if source == self.active_source:
            self.active_last_seen_at = now

    def sac_status_callback(self, message):
        self.last_sac_status = message.data
        if message.data == "goal_reached" and self.active_source is not None:
            # A vision goal is deliberately capped to a short local subgoal.
            # Reaching it is not equivalent to reaching the semantic object.
            if self.active_source in ("vision", "demo6"):
                self.candidates.pop(self.active_source, None)
                self.active_source = None
                self.active_goal = None
                self.active_last_seen_at = None
                self.active_expiry_at = None
                self.goal_completed = False
                self.subgoal_wait_until = (
                    time.monotonic() + self.subgoal_refresh_wait
                )
                self.cancel_sent = False
                return
            if self.active_goal is not None:
                self.completed_goals[self.active_source] = self.active_goal
            self.candidates.pop(self.active_source, None)
            self.active_source = None
            self.active_goal = None
            self.active_last_seen_at = None
            self.active_expiry_at = None
            self.goal_completed = True

    def semantic_reached_callback(self, message):
        if not message.data:
            return
        if self.active_source is not None and self.active_goal is not None:
            self.completed_goals[self.active_source] = self.active_goal
        self.candidates.pop("vision", None)
        self.active_source = None
        self.active_goal = None
        self.active_last_seen_at = None
        self.active_expiry_at = None
        self.subgoal_wait_until = None
        self.goal_completed = True
        self.awaiting_semantic_reset = True
        self.cancel_sent = False
        self.get_logger().info("Semantic target reached; task completed")

    def semantic_reset_callback(self, message):
        if not message.data:
            return
        self.candidates.pop("vision", None)
        self.candidates.pop("demo6", None)
        self.completed_goals.pop("vision", None)
        self.completed_goals.pop("demo6", None)
        self.active_source = None
        self.active_goal = None
        self.active_last_seen_at = None
        self.active_expiry_at = None
        self.goal_completed = False
        self.awaiting_semantic_reset = False
        self.search_started_at = None
        self.cancel_sent = False
        self.subgoal_wait_until = (
            time.monotonic() + self.new_target_observation_wait
        )
        self.get_logger().info("New semantic target available; task state reset")

    def semantic_timeout_callback(self, message):
        if not message.data:
            return
        self.candidates.pop("vision", None)
        self.candidates.pop("demo6", None)
        self.active_source = None
        self.active_goal = None
        self.active_last_seen_at = None
        self.active_expiry_at = None
        self.subgoal_wait_until = None
        self.goal_completed = True
        self.awaiting_semantic_reset = True
        self.search_started_at = None
        self.cancel_sent = False
        self.set_search(False)
        self.publish_cancel()
        self.publish_state("episode_timeout")
        self.get_logger().warning(
            "Semantic episode timed out; old goal cancelled, terminal ERROR"
        )

    def environment_ready_callback(self, message):
        if message.data and not self.environment_ready:
            self.environment_ready = True
            self.started_at = time.monotonic()
            self.get_logger().info(
                "Randomized robot/obstacle environment ready; VLA control enabled"
            )

    def brain_tick(self):
        now = time.monotonic()
        if not self.environment_ready:
            self.set_search(False)
            self.publish_cancel()
            self.publish_state("environment_setup")
            return
        fresh = [
            (source, candidate)
            for source, candidate in self.candidates.items()
            if now - candidate.received_at <= candidate.ttl
        ]
        if fresh:
            source, candidate = max(fresh, key=lambda item: item[1].priority)
            self.activate_goal(source, candidate, now)
            return

        if self.goal_completed:
            self.set_search(False)
            self.publish_cancel()
            self.publish_state("completed")
            return

        if self.subgoal_wait_until is not None and now <= self.subgoal_wait_until:
            self.set_search(False)
            self.publish_cancel()
            self.publish_state("waiting_for_next_subgoal")
            return
        self.subgoal_wait_until = None

        if not self.ever_had_goal and now - self.started_at <= self.initial_wait:
            self.set_search(False)
            self.publish_state("waiting_for_goal")
            return

        if (
            self.active_goal is not None
            and self.active_expiry_at is not None
            and now <= self.active_expiry_at + self.cached_goal_hold
        ):
            self.set_search(False)
            self.publish_state("cached_goal")
            return

        if self.search_started_at is None:
            self.search_started_at = now
            self.publish_cancel()
        if now - self.search_started_at <= self.search_duration:
            self.set_search(True)
            self.publish_state("search")
            return

        self.set_search(False)
        self.publish_cancel()
        self.publish_state("stopped")

    def activate_goal(self, source, candidate, now):
        changed = self.active_source != source or self.goal_changed(
            self.active_goal, candidate.message
        )
        self.active_source = source
        self.active_goal = candidate.message
        self.active_last_seen_at = candidate.received_at
        self.active_expiry_at = candidate.received_at + candidate.ttl
        self.search_started_at = None
        self.ever_had_goal = True
        self.goal_completed = False
        self.subgoal_wait_until = None
        self.cancel_sent = False
        self.set_search(False)
        if changed:
            self.goal_publisher.publish(candidate.message)
            self.get_logger().info(
                f"Selected {source} goal: ({candidate.message.point.x:.2f}, "
                f"{candidate.message.point.y:.2f})"
            )
        self.publish_state(f"navigating:{source}")

    def goal_changed(self, previous, current):
        if previous is None:
            return True
        return math.hypot(
            current.point.x - previous.point.x,
            current.point.y - previous.point.y,
        ) >= self.minimum_goal_change

    def set_search(self, enabled):
        message = Bool()
        message.data = bool(enabled)
        self.search_publisher.publish(message)

    def publish_cancel(self):
        if self.cancel_sent:
            return
        message = Bool()
        message.data = True
        self.cancel_publisher.publish(message)
        self.active_goal = None
        self.active_source = None
        self.active_last_seen_at = None
        self.active_expiry_at = None
        self.cancel_sent = True

    def publish_state(self, state):
        if state == self.last_state:
            return
        message = String()
        message.data = state
        self.state_publisher.publish(message)
        self.last_state = state
        self.get_logger().info(f"Brain state: {state}")


def main(args=None):
    rclpy.init(args=args)
    node = VlaAgentSupervisor()
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
