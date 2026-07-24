# DRL Robot Navigation ROS 2

Research code for mobile robot navigation in ROS 2 Humble and Gazebo Classic. The project trains and deploys a Soft Actor-Critic (SAC) local navigation policy for TurtleBot3 Waffle, and connects it to a vision-language-action (VLA) goal interface: the VLA layer decides where the robot should go, while the SAC controller decides how to move safely through lidar observations.

The released deterministic policy is stored in `src/drl_navigation_ros2/models/SAC_epoch6_best`.

## Motivation

Language-conditioned robot navigation is difficult because high-level semantic instructions and low-level collision avoidance operate at different frequencies and levels of abstraction. A vision-language model can identify a target object or infer a navigation intent, but direct velocity control from that model is brittle in cluttered indoor simulation.

This repository studies a two-layer alternative:

- A low-frequency VLA layer interprets the scene and publishes candidate goals.
- A high-frequency SAC policy consumes lidar, odometry, and the active goal to publish velocity commands.
- A supervisor arbitrates goal sources and applies safety fallbacks for target loss, stale goals, and near-obstacle cases.

The result is a reproducible ROS 2 navigation stack for studying semantic goal grounding and learned local control in Gazebo.

## Method

### System Overview

```text
Gazebo world
  -> /scan + /odom
  -> SAC state encoder
  -> SAC actor
  -> lidar safety layer
  -> /cmd_vel

VLA demos / manual goal / vision bridge
  -> /vla_goal/source/*
  -> vla_agent_supervisor
  -> /vla_goal
```

Only `sac_vla_controller` should publish `/cmd_vel`. VLA components publish goals, not robot velocities.

### Reinforcement Learning Policy

The local navigation policy is trained with SAC using:

- LaserScan observations from TurtleBot3.
- Odometry-derived relative target features.
- Previous action context.
- Reward terms for goal progress, goal completion, collision, timeout, and inefficient motion.

Training entry point: `src/drl_navigation_ros2/train.py`.

Deterministic inference entry point: `src/drl_navigation_ros2/inference.py`.

### VLA Goal Grounding

The VLA stack is integrated under `src/vla_bridge/vla_bridge`. Demo nodes can publish semantic observations, detection results, or target poses. These candidate goals are normalized through source topics and then selected by `vla_agent_supervisor.py`.

Goal priority:

```text
manual > demo6 target pose > vision detection fallback
```

The full launch file is:

```text
src/turtlebot3_simulations/turtlebot3_gazebo/launch/vla_sac_stack.launch.py
```

### Repository Layout

| Path | Purpose |
| --- | --- |
| `src/drl_navigation_ros2/train.py` | SAC training script |
| `src/drl_navigation_ros2/inference.py` | SAC/VLA inference controller |
| `src/drl_navigation_ros2/ros_python.py` | Gazebo environment, state construction, rewards |
| `src/drl_navigation_ros2/SAC/` | SAC actor, critic, replay buffer, update logic |
| `src/drl_navigation_ros2/models/SAC_epoch6_best/` | Released stable inference checkpoint |
| `src/drl_navigation_ros2/vla_agent_supervisor.py` | Goal arbitration and fallback behavior |
| `src/drl_navigation_ros2/random_obstacle_manager.py` | Randomized obstacle spawning |
| `src/drl_navigation_ros2/random_red_box_manager.py` | Random semantic target spawning |
| `src/vla_bridge/vla_bridge/` | Demo4/Demo5/Demo6 VLA bridge nodes |
| `docs/PROJECT_LAYOUT.md` | Detailed file map |
| `docs/EXPERIMENT_RESULTS.md` | Current metric log and evaluation protocol |

## Results

The current released model is `SAC_epoch6_best`. Latest tracked metrics and the evaluation template are maintained in:

```text
docs/EXPERIMENT_RESULTS.md
```

Primary metrics:

| Metric | Meaning |
| --- | --- |
| `eval/avg_reward` | Mean evaluation return |
| `eval/avg_goal` | Goal success rate |
| `eval/avg_col` | Collision rate |
| `eval/timeout_rate` | Timeout rate |
| `eval/avg_steps` | Mean steps per episode |

No TensorBoard event files or runtime result JSON files are committed in this repository. Re-run evaluation locally and record the final metrics in `docs/EXPERIMENT_RESULTS.md`.

## Video

Recommended demo video contents:

1. Gazebo world reset with randomized obstacles and target object.
2. RViz view showing TF, LaserScan, Odometry, and active goal marker.
3. VLA goal publication on `/vla_goal`.
4. SAC controller driving TurtleBot3 to the target while avoiding obstacles.
5. Terminal output for `/sac_navigation/status` and `/vla_navigation/result`.

Add the final video link here after upload:

```text
Video: TBD
```

## Installation

Tested target environment:

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- Python 3.10
- PyTorch

Install system dependencies:

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-gazebo-ros-pkgs
```

Install Python dependencies into a project-local directory:

```bash
cd ~/DRL-Robot-Navigation-ROS2
python3 -m pip install --target .python_deps \
  numpy pyyaml tqdm tensorboard squaternion torch dashscope opencv-python
```

Build the ROS 2 workspace:

```bash
cd ~/DRL-Robot-Navigation-ROS2
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build --symlink-install --allow-overriding turtlebot3_description
source install/setup.bash
```

## Training

Terminal 1:

```bash
cd ~/DRL-Robot-Navigation-ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH="$PWD"
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH="$PWD/src/turtlebot3_simulations/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}"

ros2 launch turtlebot3_gazebo ros2_drl.launch.py
```

Terminal 2:

```bash
cd ~/DRL-Robot-Navigation-ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH="$PWD"
export TURTLEBOT3_MODEL=waffle
export PYTHONPATH="$PWD/.python_deps:$PWD/src:$PWD/src/drl_navigation_ros2:${PYTHONPATH:-}"

mkdir -p src/drl_navigation_ros2/models/SAC
python3 src/drl_navigation_ros2/train.py
```

Resume from a checkpoint:

```bash
python3 src/drl_navigation_ros2/train.py --resume
```

TensorBoard:

```bash
cd ~/DRL-Robot-Navigation-ROS2
export PYTHONPATH="$PWD/.python_deps:${PYTHONPATH:-}"
export PATH="$PWD/.python_deps/bin:${PATH:-}"
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

## Inference

Start Gazebo, then run deterministic SAC inference:

```bash
cd ~/DRL-Robot-Navigation-ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH="$PWD"
export TURTLEBOT3_MODEL=waffle
export PYTHONPATH="$PWD/.python_deps:$PWD/src:$PWD/src/drl_navigation_ros2:${PYTHONPATH:-}"

python3 src/drl_navigation_ros2/inference.py
```

Publish a manual odom-frame goal:

```bash
python3 src/drl_navigation_ros2/publish_vla_goal.py 1.0 0.0 --topic /vla_goal
```

Check runtime topics:

```bash
ros2 topic echo /sac_navigation/status
ros2 topic echo /cmd_vel --once
```

## Full VLA + SAC Stack

```bash
cd ~/DRL-Robot-Navigation-ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH="$PWD"
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH="$PWD/src/turtlebot3_simulations/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}"
export PYTHONPATH="$PWD/.python_deps:$PWD/src:$PWD/src/drl_navigation_ros2:$PWD/src/vla_bridge:${PYTHONPATH:-}"

export DASHSCOPE_API_KEY="your_dashscope_api_key"
export BPX_QWEN_VL_MODEL=qwen3-vl-32b-thinking

ros2 launch turtlebot3_gazebo vla_sac_stack.launch.py \
  target_mode:=demo6 \
  start_gazebo:=true \
  start_random_obstacles:=true \
  start_random_target:=true \
  start_demo4:=true \
  start_demo5:=true \
  start_demo6:=true \
  start_task_adapter:=true \
  enable_color_fallback:=true \
  policy_speed_scale:=0.60 \
  search_angular_speed:=0.35 \
  search_rotate_duration:=60.0 \
  search_observe_duration:=0.0 \
  search_duration:=60.0 \
  episode_timeout:=120.0 \
  demo5_detect_rate:=0.2 \
  vlm_model:=qwen3-vl-32b-thinking
```

Useful checks:

```bash
ros2 topic echo /vla/detection/bbox
ros2 topic echo /vla/target/map_pose
ros2 topic echo /vla_goal
ros2 topic echo /sac_navigation/status
ros2 topic echo /vla_navigation/result --qos-durability transient_local
ros2 topic info /cmd_vel -v
```

## Citation

If this repository supports academic work, cite the repository and the underlying SAC and TurtleBot3/ROS 2 components used by your experiment. A BibTeX entry can be added after publication.

## Acknowledgements

This project builds on ROS 2, Gazebo Classic, TurtleBot3 simulation assets, SAC implementations, and earlier DRL navigation examples.

Related references:

- <https://github.com/tomasvr/turtlebot3_drlnav>
- <https://github.com/reiniscimurs/DRL-robot-navigation>
- <https://github.com/denisyarats/pytorch_sac>

## License

This repository is released under the MIT License. See `LICENSE` for details.
