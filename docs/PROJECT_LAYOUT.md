# Project Layout

## Runtime Data Flow

```text
VLA / vision bridge
  -> /vla_goal/source/*
  -> vla_agent_supervisor
  -> /vla_goal (PointStamped in odom)

/scan + /odom + /vla_goal
  -> deterministic SAC controller
  -> /policy_cmd_vel
  -> independent LiDAR safety shield
  -> /cmd_vel
  -> TurtleBot3 Waffle in Gazebo
```

When the shield is disabled, the SAC controller can publish `/cmd_vel` directly. The launch file must keep exactly one final velocity owner.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `src/drl_navigation_ros2/train.py` | SAC training and checkpoint lifecycle |
| `src/drl_navigation_ros2/inference.py` | Deterministic policy loading, state construction and ROS 2 control |
| `src/drl_navigation_ros2/ros_python.py` | Gazebo environment, rewards and terminal conditions |
| `src/drl_navigation_ros2/SAC/` | Actor, critic and SAC update implementation |
| `src/drl_navigation_ros2/safety_shield.py` | ROS 2 shield node and timeout handling |
| `src/drl_navigation_ros2/safety_shield_core.py` | Testable LiDAR action projection |
| `src/drl_navigation_ros2/vla_agent_supervisor.py` | Goal arbitration and bounded fallback |
| `src/vla_bridge/vla_bridge/` | VLA parsing, detection and target-coordinate bridge nodes |
| `src/turtlebot3_simulations/` | Gazebo worlds, launch files and TurtleBot3 simulation assets |
| `src/turtlebot3_description/` | Retained TurtleBot3 URDF and meshes |
| `tests/` | Pure-Python and ROS-dependent tests |
| `docs/media/` | Curated GIF/MP4 evidence used by the README |
| `docs/EXPERIMENT_RESULTS.md` | Metric definitions and pending evaluation protocol |
| `docs/REPRODUCTION.md` | Build, training, inference and VLA launch commands |
| `docs/SOURCE_ATTRIBUTION.md` | Original/upstream scope and licenses |

## Evidence Boundary

- The committed checkpoint supports deterministic inference review.
- GIFs and the MP4 support qualitative workflow review only.
- Numerical navigation claims require saved outputs from the protocol in `EXPERIMENT_RESULTS.md`.
- Physical-robot deployment is outside this simulation repository.

## Generated Files

Do not commit build trees, ROS logs, bags, TensorBoard event files, replay buffers, API keys or raw result directories. Curated files under `docs/media/` are the only intentional media exception.
