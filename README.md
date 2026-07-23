# DRL Robot Navigation ROS 2

ROS 2 Humble + Gazebo Classic + PyTorch 的移动机器人导航项目。项目使用 SAC 训练 TurtleBot3 Waffle 的局部导航策略，并提供 VLA 大模型目标理解接口：大模型负责“去哪里”，SAC 负责“怎么避障开过去”。

当前稳定推理模型为 `src/drl_navigation_ros2/models/SAC_epoch6_best`。

## Features

- SAC 局部导航：输入 `/scan`、`/odom` 和目标点，输出 `/cmd_vel`
- Gazebo 随机障碍物、随机 red box 语义目标
- RViz 固定配置：TF、LaserScan、Odometry、Marker
- TensorBoard 训练指标
- VLA 双层架构：低频大脑生成/仲裁目标，高频小脑执行 SAC 控制
- 雷达安全层：紧急停车、硬避障、目标丢失后的有界搜索

## Architecture

```text
Training:
Gazebo -> /scan + /odom -> ROS_env reward/state -> SAC -> /cmd_vel

VLA inference:
Demo4/Demo5/Demo6 or manual goal
        -> /vla_goal/source/*
        -> vla_agent_supervisor
        -> /vla_goal
        -> SAC inference + lidar safety
        -> /cmd_vel
```

系统原则：只有 `sac_vla_controller` 发布 `/cmd_vel`。VLA 节点只发布目标，不直接控制速度。

## Main Files

Detailed workspace and file mapping: [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md).

| Path | Function |
| --- | --- |
| `src/drl_navigation_ros2/train.py` | SAC 训练入口 |
| `src/drl_navigation_ros2/inference.py` | SAC 确定性推理控制器 |
| `src/drl_navigation_ros2/ros_python.py` | Gazebo 环境、reward、目标与碰撞判断 |
| `src/drl_navigation_ros2/ros_nodes.py` | ROS topic/service 封装 |
| `src/drl_navigation_ros2/SAC/` | SAC actor/critic/algorithm |
| `src/drl_navigation_ros2/models/SAC_epoch6_best/` | 稳定推理模型 |
| `src/drl_navigation_ros2/random_obstacle_manager.py` | 随机障碍物 |
| `src/drl_navigation_ros2/random_red_box_manager.py` | 随机 red box 语义目标 |
| `src/drl_navigation_ros2/vla_agent_supervisor.py` | VLA 目标仲裁与 fallback |
| `src/drl_navigation_ros2/vla_target_bridge.py` | Demo6 目标转 `/vla_goal` |
| `src/drl_navigation_ros2/vla_detection_goal_bridge.py` | Demo5 bbox 转局部目标 |
| `src/vla_bridge/vla_bridge/` | 集成后的 Demo4/5/6 VLA 脚本 |
| `src/turtlebot3_simulations/turtlebot3_gazebo/launch/vla_sac_stack.launch.py` | VLA + SAC 一键启动 |

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- Python 3.10
- PyTorch
- TurtleBot3 Gazebo dependencies

Install system dependencies:

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-gazebo-ros-pkgs
```

Install Python dependencies into the project-local `.python_deps` directory:

```bash
cd ~/DRL-Robot-Navigation-ROS2

python3 -m pip install --target .python_deps \
  numpy pyyaml tqdm tensorboard squaternion torch dashscope opencv-python
```

Install ROS dependencies and build:

```bash
cd ~/DRL-Robot-Navigation-ROS2

source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build --symlink-install --allow-overriding turtlebot3_description
source install/setup.bash
```

## Train SAC

Terminal 1, start Gazebo:

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

Terminal 2, start clean training:

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

Continue from checkpoint:

```bash
python3 src/drl_navigation_ros2/train.py --resume
```

Clean old training logs and runtime checkpoints:

```bash
rm -rf runs/*
rm -rf src/drl_navigation_ros2/models/SAC
```

Do not delete `src/drl_navigation_ros2/models/SAC_epoch6_best` unless you intentionally want to remove the released demo model.

## TensorBoard

```bash
cd ~/DRL-Robot-Navigation-ROS2
export PYTHONPATH="$PWD/.python_deps:${PYTHONPATH:-}"
export PATH="$PWD/.python_deps/bin:${PATH:-}"
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

Open:

```text
http://127.0.0.1:6006
```

Important metrics:

| Metric | Meaning |
| --- | --- |
| `eval/avg_reward` | Higher is better |
| `eval/avg_goal` | Goal success rate |
| `eval/avg_col` | Collision rate |
| `eval/timeout_rate` | Timeout rate |
| `eval/avg_steps` | Average episode steps |

## RViz

```bash
cd ~/DRL-Robot-Navigation-ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1

rviz2 -d "$PWD/src/turtlebot3_simulations/turtlebot3_gazebo/rviz/tb3_gazebo.rviz"
```

## SAC Inference Only

Start Gazebo first, then run:

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

Check status:

```bash
ros2 topic echo /sac_navigation/status
ros2 topic echo /cmd_vel --once
```

## VLA Full Stack

The VLA Demo4/5/6 scripts are integrated in:

```text
src/vla_bridge/vla_bridge
```

Launch Demo4/5/6 + random environment + SAC:

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

`/cmd_vel` should have only one publisher: `/sac_vla_controller`.

## GitHub Notes

Upload checklist and repository structure: [`docs/GITHUB_UPLOAD.md`](docs/GITHUB_UPLOAD.md).

Keep:

```text
README.md
LICENSE
pyproject.toml
poetry.lock
Makefile
tests/
docs/
src/
```

Do not commit local runtime outputs:

```text
build/
install/
log/
.venv/
.python_deps/
runs/
artifacts/
results/
*.bag
*.db3
events.out.tfevents*
replay_buffer.pkl
```

Never commit real API keys. Use environment variables such as `DASHSCOPE_API_KEY`.

## Sources

- ROS 2 adaptation: <https://github.com/tomasvr/turtlebot3_drlnav>
- TD3 reference: <https://github.com/reiniscimurs/DRL-robot-navigation>
- SAC reference: <https://github.com/denisyarats/pytorch_sac>

## License

See `LICENSE`.
