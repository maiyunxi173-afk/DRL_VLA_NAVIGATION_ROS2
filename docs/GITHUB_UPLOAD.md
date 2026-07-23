# GitHub Upload Guide

This repository is now organized as a single workspace for SAC training,
deterministic SAC inference, Gazebo simulation, RViz visualization, and the
integrated VLA Demo4/5/6 chain.

## Repository Name

Recommended GitHub repository name:

```text
drl-vla-navigation-ros2
```

Current local path:

```text
/home/mai/DRL-Robot-Navigation-ROS2
```

## GitHub Sections

| Section | Path | Purpose |
| --- | --- | --- |
| Project overview | `README.md` | Quick start, commands, architecture |
| Layout document | `docs/PROJECT_LAYOUT.md` | Full path and file mapping |
| Upload guide | `docs/GITHUB_UPLOAD.md` | GitHub preparation and upload checklist |
| SAC navigation | `src/drl_navigation_ros2/` | SAC training, inference, reward, ROS environment |
| VLA bridge | `src/vla_bridge/` | Demo4/5/6 scripts integrated into the main repo |
| Simulation | `src/turtlebot3_simulations/` | Gazebo models, worlds, launch files, RViz config |
| Robot description | `src/turtlebot3_description/` | TurtleBot3 URDF and meshes |
| Tests | `tests/` | Core regression tests |
| Released model | `src/drl_navigation_ros2/models/SAC_epoch6_best/` | Stable demo policy |

## Files To Keep

```text
README.md
LICENSE
pyproject.toml
poetry.lock
Makefile
.gitignore
.github/
docs/
tests/
src/drl_navigation_ros2/
src/vla_bridge/
src/turtlebot3_simulations/
src/turtlebot3_description/
```

## Files To Exclude

These are already covered by `.gitignore`:

```text
build/
install/
log/
.venv/
.python_deps/
runs/
results/
artifacts/
__pycache__/
*.pyc
*.bag
*.db3
*.mcap
events.out.tfevents*
replay_buffer.pkl
*.mp4
*.avi
*.mkv
.env
.env.*
```

## Not Required

Do not upload these old local workspaces as part of this repository:

```text
/home/mai/bpx_vla_navigation
/home/mai/DRL_robot_navigation_ros2
```

The useful VLA scripts from `bpx_vla_navigation` are already integrated under:

```text
src/vla_bridge/vla_bridge/
```

The old `DRL_robot_navigation_ros2` workspace is a TD3/Velodyne reference and is
not required for the current SAC+VLA project.

## Pre-Upload Checks

Run these before pushing:

```bash
cd ~/DRL-Robot-Navigation-ROS2

git status --short --ignored
git diff --check

source /opt/ros/humble/setup.bash
source install/setup.bash

export PYTHONPATH="$PWD/.python_deps:$PWD/src:$PWD/src/drl_navigation_ros2:$PWD/src/vla_bridge:${PYTHONPATH:-}"

python3 -m py_compile \
  src/drl_navigation_ros2/*.py \
  src/drl_navigation_ros2/SAC/*.py \
  src/vla_bridge/vla_bridge/*.py \
  src/turtlebot3_simulations/turtlebot3_gazebo/launch/*.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -o addopts='' tests/test_drl_core.py
```

Clean generated Python caches after checks:

```bash
find . -path './.python_deps' -prune -o -type d -name '__pycache__' -exec rm -rf {} +
```

## Upload Commands

For a new GitHub repository:

```bash
cd ~/DRL-Robot-Navigation-ROS2

git add .
git status --short
git commit -m "Prepare integrated SAC VLA navigation workspace"

git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin <your-github-repo-url>
git push -u origin main
```

## Security

Never commit real API keys. Use environment variables:

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export BPX_QWEN_VL_MODEL=qwen3-vl-32b-thinking
```
