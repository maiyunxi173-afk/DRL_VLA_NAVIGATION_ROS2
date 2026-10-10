# Reproduction Guide

This guide separates commands verified from the repository layout from commands that require a ROS 2/Gazebo runtime and external services.

## Environment

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- Python 3.10
- TurtleBot3 Waffle

## Install and Build

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep ros-humble-gazebo-ros-pkgs

git clone https://github.com/maiyunxi173-afk/DRL_VLA_NAVIGATION_ROS2.git
cd DRL_VLA_NAVIGATION_ROS2
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build --symlink-install --allow-overriding turtlebot3_description
source install/setup.bash
```

Install Python runtime packages in your chosen virtual environment:

```bash
python3 -m pip install numpy pyyaml tqdm tensorboard squaternion torch dashscope opencv-python
```

## Common Environment

```bash
export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH="$PWD"
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH="$PWD/src/turtlebot3_simulations/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}"
export PYTHONPATH="$PWD/src:$PWD/src/drl_navigation_ros2:$PWD/src/vla_bridge:${PYTHONPATH:-}"
```

## Training

Terminal 1:

```bash
ros2 launch turtlebot3_gazebo ros2_drl.launch.py
```

Terminal 2:

```bash
mkdir -p src/drl_navigation_ros2/models/SAC
python3 src/drl_navigation_ros2/train.py
```

Resume an existing checkpoint with:

```bash
python3 src/drl_navigation_ros2/train.py --resume
```

## Deterministic Inference

After Gazebo is running:

```bash
python3 src/drl_navigation_ros2/inference.py
python3 src/drl_navigation_ros2/publish_vla_goal.py 1.0 0.0 --topic /vla_goal
```

Useful checks:

```bash
ros2 topic echo /sac_navigation/status
ros2 topic echo /cmd_vel --once
```

## Full VLA + SAC + Shield Stack

The cloud VLM path requires a valid DashScope key. Do not commit the key.

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export BPX_QWEN_VL_MODEL=qwen3-vl-32b-thinking

ros2 launch turtlebot3_gazebo vla_sac_stack.launch.py \
  target_mode:=demo6 \
  start_gazebo:=true \
  start_random_obstacles:=true \
  start_random_target:=true \
  enable_safety_shield:=true \
  internal_safety:=false \
  policy_speed_scale:=0.60 \
  episode_timeout:=120.0
```

Verify that only the shield publishes the final command:

```bash
ros2 topic info /cmd_vel -v
ros2 topic echo /safety_shield/status
ros2 topic echo /safety_shield/intervention
```

## Tests

Pure-Python safety logic:

```bash
python3 -m pytest -q tests/test_safety_shield_core.py
```

All tests:

```bash
python3 -m pytest -q
```

The full suite imports ROS 2 modules and requires a sourced ROS environment. Coverage options in `pyproject.toml` also require `pytest-cov`.

## Evaluation Boundary

Do not infer success rate, collision rate or localisation accuracy from the GIFs. Run the fixed-seed protocol in [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md), retain raw outputs, and record the commit SHA before publishing numerical claims.
