# Experiment Results

This document records the latest public metrics for the ROS 2 SAC navigation and VLA-goal navigation stack.

## Current Release

| Item | Value |
| --- | --- |
| Released policy | `src/drl_navigation_ros2/models/SAC_epoch6_best` |
| Robot | TurtleBot3 Waffle |
| Simulator | Gazebo Classic 11 |
| ROS distribution | ROS 2 Humble |
| Local controller | Soft Actor-Critic |
| Goal interface | `/vla_goal` as `geometry_msgs/PointStamped` in `odom` frame |
| Velocity output | `/cmd_vel` from `sac_vla_controller` only |

## Latest Metrics

No committed TensorBoard event files, result JSON files, or evaluation CSV files are available in the repository snapshot. Fill this table after running the evaluation protocol below.

| Date | Checkpoint | Episodes | `eval/avg_reward` | `eval/avg_goal` | `eval/avg_col` | `eval/timeout_rate` | `eval/avg_steps` | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TBD | `SAC_epoch6_best` | TBD | TBD | TBD | TBD | TBD | TBD | Latest released model; metrics pending local evaluation export |

## Method Comparison

This is the canonical comparison table for the project. Only values supported by saved evaluation outputs should be moved into the verified columns.

| Method | Episodes | Success Rate | Collisions | Mean Time | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| DWA | TBD | TBD | TBD | TBD | Evaluation pending |
| SAC | TBD | TBD | TBD | TBD | Evaluation pending |
| VLA + SAC | TBD | TBD | TBD | TBD | Evaluation pending |

For each completed benchmark, record the date, commit SHA, map set, random seeds, checkpoint, and raw-result artifact location in the notes. Use identical scenarios across methods.

## Evaluation Protocol

1. Build and source the workspace.
2. Launch the TurtleBot3 Gazebo environment with randomized obstacles.
3. Load `src/drl_navigation_ros2/models/SAC_epoch6_best`.
4. Run deterministic inference with fixed environment settings.
5. Evaluate across a fixed number of episodes and record success, collision, timeout, reward, and step count.
6. Store raw runtime outputs outside Git-tracked paths such as `runs/` or `results/`.

Recommended environment variables:

```bash
export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH="$PWD"
export TURTLEBOT3_MODEL=waffle
export PYTHONPATH="$PWD/.python_deps:$PWD/src:$PWD/src/drl_navigation_ros2:${PYTHONPATH:-}"
```

Recommended metrics:

| Metric | Definition |
| --- | --- |
| `eval/avg_reward` | Mean return across evaluation episodes |
| `eval/avg_goal` | Fraction of episodes ending in goal success |
| `eval/avg_col` | Fraction of episodes ending in collision |
| `eval/timeout_rate` | Fraction of episodes ending by timeout |
| `eval/avg_steps` | Mean number of control steps per episode |

## VLA Stack Observations

Track VLA runs separately from pure SAC local-navigation runs because perception, target grounding, and fallback search can change the failure distribution.

| Date | Mode | VLM | Episodes | Success Rate | Collision Rate | Timeout Rate | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| TBD | `demo6` | `qwen3-vl-32b-thinking` | TBD | TBD | TBD | TBD | Pending reproducible VLA benchmark run |

## Artifact Policy

The following runtime artifacts should not be committed:

```text
runs/
results/
artifacts/
events.out.tfevents*
*.bag
*.db3
replay_buffer.pkl
```

Commit only curated summaries, plots, or tables that are small enough for Git and useful for academic review.
