# Project Layout

This document maps the consolidated single-workspace layout. It explains which
files belong to training, inference, VLA integration, simulation, and old
references that should stay outside the main repository.

## Recommended Workspace Names

| Current path | Role | Open-source naming |
| --- | --- | --- |
| `/home/mai/DRL-Robot-Navigation-ROS2` | Main project | `drl-vla-navigation-ros2` |
| `/home/mai/DRL-Robot-Navigation-ROS2/src/vla_bridge` | Integrated VLA Demo4/5/6 scripts | `src/vla_bridge` |
| `/home/mai/DRL_robot_navigation_ros2` | Legacy TD3/Velodyne reference | Not required for this repository |

The main project should remain the only required repository for SAC training,
Gazebo simulation, RViz visualization, deterministic SAC inference, and the
Demo4/5/6 VLA chain.

## Runtime Data Flow

```text
Gazebo
  -> /scan, /odom, /camera/*
  -> random red box + random obstacles

VLA brain, optional
  Demo4 instruction parsing
  Demo5 red-box bbox detection
  Demo6 bbox/depth/TF target projection
  -> /vla/target/map_pose

Goal arbitration
  /vla_goal/source/manual
  /vla_goal/source/vision
  /vla_goal/source/demo6
  -> vla_agent_supervisor
  -> /vla_goal

SAC cerebellum
  /scan + /odom + /vla_goal
  -> lidar safety + deterministic SAC actor
  -> /cmd_vel
```

Only `src/drl_navigation_ros2/inference.py` should publish `/cmd_vel` during VLA
navigation.

## Main Workspace

Root: `/home/mai/DRL-Robot-Navigation-ROS2`

| Path | Content | Function |
| --- | --- | --- |
| `README.md` | Open-source quick guide | Project overview and core commands |
| `.gitignore` | Git ignore rules | Keeps build/cache/log/API/runtime files out of Git |
| `pyproject.toml` | Python metadata | Python package/test/lint configuration |
| `poetry.lock` | Python lock file | Reproducible Poetry dependency lock |
| `Makefile` | Helper commands | Local development helper targets |
| `LICENSE` | License text | Project license |
| `docs/PROJECT_LAYOUT.md` | This file | Workspace and file mapping |
| `tests/test_drl_core.py` | Core tests | Reward, replay buffer, control logic sanity checks |
| `tests/test_default.py` | Template/default test | Existing package test |

## DRL And SAC Code

Root: `/home/mai/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2`

| Path | Content | Function |
| --- | --- | --- |
| `train.py` | SAC training loop | Clean training, resume, eval, checkpoints |
| `inference.py` | SAC inference controller | Loads `SAC_epoch6_best`, subscribes `/scan`, `/odom`, `/vla_goal`, publishes `/cmd_vel` |
| `ros_python.py` | RL environment | State construction, reward, target sampling, collision and goal detection |
| `ros_nodes.py` | ROS helper node | Gazebo services and topic I/O for training |
| `replay_buffer.py` | Replay buffer | Stores and samples transitions |
| `utils.py` | Shared utilities | Math, seeding, tensor/log helpers |
| `pretrain_utils.py` | Legacy pretraining helpers | Compatibility support for old offline data |
| `hardcoded_model.py` | Simple policy baseline | Deterministic hand-coded/reference behavior |
| `publish_vla_goal.py` | Manual goal publisher | Publishes a test `PointStamped` to `/vla_goal` or source topics |
| `SAC/SAC.py` | SAC algorithm | Actor/critic update logic |
| `SAC/SAC_actor.py` | Actor network | Policy network and action distribution |
| `SAC/SAC_critic.py` | Critic network | Q-network implementation |
| `SAC/SAC_utils.py` | SAC helpers | Model loading/saving and SAC utility functions |

## Released Model

Root: `/home/mai/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/models`

| Path | Content | Function |
| --- | --- | --- |
| `SAC_epoch6_best/SAC_actor.pth` | Actor weights | Main inference policy |
| `SAC_epoch6_best/SAC_checkpoint.pt` | Full checkpoint | Training state snapshot for provenance |
| `SAC_epoch6_best/SAC_critic.pth` | Critic weights | Training/evaluation reference |
| `SAC_epoch6_best/SAC_critic_target.pth` | Target critic weights | SAC target network snapshot |

Runtime training checkpoints should go to `models/SAC/` and should not be
treated as the released demo model.

## VLA Integration Code In Main Workspace

Root: `/home/mai/DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2`

| Path | Content | Function |
| --- | --- | --- |
| `vla_agent_supervisor.py` | Goal supervisor | Arbitrates `manual > demo6 > vision`, caches goals, manages search/timeout |
| `vla_target_bridge.py` | Demo6 target bridge | Converts `/vla/target/map_pose` to `/vla_goal/source/demo6` |
| `vla_detection_goal_bridge.py` | Vision goal bridge | Converts Demo5 bbox + LaserScan/TF/Odometry into `/vla_goal/source/vision` |
| `random_red_box_manager.py` | Semantic target manager | Spawns/randomizes Gazebo red box and publishes truth/result topics |
| `random_obstacle_manager.py` | Obstacle manager | Randomizes obstacle placement with safety spacing |

## Integrated VLA Demo4/5/6 Code

Root: `/home/mai/DRL-Robot-Navigation-ROS2/src/vla_bridge`

| Path | Content | Function |
| --- | --- | --- |
| `README.md` | VLA bridge notes | Explains the integrated VLA scripts |
| `vla_bridge/__init__.py` | Python package marker | Makes the directory importable |
| `vla_bridge/demo4_test_batch.py` | Demo4 VLM task node | Uses Qwen/DashScope to parse high-level scene/task information |
| `vla_bridge/demo5_ros_detector.py` | Demo5 detector | Detects target bbox from `/camera/image_raw`, supports Qwen-VL and local color fallback |
| `vla_bridge/demo5_ros_bridge.py` | Demo5 compatibility bridge | Compatibility helper for bbox/topic integration |
| `vla_bridge/demo6_vla_coord_node.py` | Demo6 coordinate node | Converts bbox + depth + camera info + TF into target pose |
| `vla_bridge/vla_agent_node.py` | Task adapter | Converts manual/RViz goals and reports task state; does not publish `/cmd_vel` |

Runtime VLA results now live under:

```text
/home/mai/DRL-Robot-Navigation-ROS2/results/vla_bridge/
```

`results/` is ignored by Git.

## Simulation And Visualization

Root: `/home/mai/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo`

| Path | Content | Function |
| --- | --- | --- |
| `launch/ros2_drl.launch.py` | Training launch | Starts normal DRL Gazebo world |
| `launch/ros2_drl_vla.launch.py` | VLA world launch | Starts Gazebo world with camera/depth/red-box support |
| `launch/vla_sac_stack.launch.py` | Full stack launch | Starts Gazebo, randomizers, bridges, VLA scripts, SAC controller |
| `launch/robot_state_publisher.launch.py` | Robot state publisher | Publishes robot TF from URDF/model |
| `rviz/tb3_gazebo.rviz` | RViz config | Fixed TF, LaserScan, Odometry, Marker displays |
| `worlds/turtlebot3_drl/` | Training world assets | Original DRL training worlds |
| `worlds/turtlebot3_drl_vla/waffle.model` | VLA robot/world model | Waffle with camera/depth support for VLA tests |
| `models/14by14/` | Larger arena model | Larger environment for longer navigation demos |
| `models/10by10/` | Smaller arena model | Original/smaller environment |
| `models/obstacle1` to `models/obstacle10` | Obstacle models | Gazebo obstacle entities |
| `models/turtlebot3_waffle/` | Robot Gazebo model | TurtleBot3 Waffle model files |

## TurtleBot Description

Root: `/home/mai/DRL-Robot-Navigation-ROS2/src/turtlebot3_description`

| Path | Content | Function |
| --- | --- | --- |
| `urdf/turtlebot3_waffle.urdf` | Waffle URDF | Robot description used by ROS/Gazebo |
| `urdf/turtlebot3_burger.urdf` | Burger URDF | Alternate robot model |
| `urdf/turtlebot3_waffle_pi.urdf` | Waffle Pi URDF | Alternate robot model |
| `meshes/` | STL/DAE/JPG assets | Visual and collision meshes |
| `rviz/model.rviz` | Model RViz config | Robot model visualization |
| `package.xml`, `CMakeLists.txt` | ROS package metadata | Build and package metadata |

## Legacy Reference Workspace

Root: `/home/mai/DRL_robot_navigation_ros2`

| Path | Content | Function |
| --- | --- | --- |
| `src/td3/` | TD3 navigation code | Legacy/reference implementation, not active in the SAC+VLA stack |
| `src/td3/scripts/train_velodyne_node.py` | TD3 training script | Reference only |
| `src/td3/scripts/test_velodyne_node.py` | TD3 test script | Reference only |
| `src/td3/worlds/td3.world` | Legacy Gazebo world | Reference only |
| `src/velodyne_simulator/` | Velodyne simulator packages | Reference dependency for the legacy TD3 project |

Recommendation: do not merge this into the main source tree unless you
explicitly want to publish a second, separate TD3/Velodyne baseline. It is not
required for the SAC+VLA project.

## Files To Keep In Git

```text
README.md
LICENSE
pyproject.toml
poetry.lock
Makefile
docs/
tests/
src/drl_navigation_ros2/
src/vla_bridge/
src/turtlebot3_simulations/
src/turtlebot3_description/
```

## Files To Exclude From Git

```text
build/
install/
log/
.venv/
.python_deps/
runs/
artifacts/
results/
__pycache__/
*.bag
*.db3
events.out.tfevents*
replay_buffer.pkl
*.mp4
*.avi
*.mkv
.env
.env.*
```

## Recommended Future Rename Plan

Do this only when you are ready to make a clean GitHub repository:

```text
DRL-Robot-Navigation-ROS2 -> drl-vla-navigation-ros2
```

The current launch file already allows overriding this path:

```bash
ros2 launch turtlebot3_gazebo vla_sac_stack.launch.py \
  vla_workspace:=/absolute/path/to/another/vla_workspace
```
