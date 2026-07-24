# Contributing

Thank you for contributing to the Vision-Language Guided Reinforcement Learning Navigation System.

## Development Setup

1. Fork the repository and create a focused branch from `main`.
2. Follow the installation and workspace build instructions in `README.md`.
3. Keep generated ROS 2, Gazebo, training, and evaluation artifacts out of Git.
4. Run the relevant tests before opening a pull request.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
pytest
```

## Research Contributions

Changes to learning algorithms, observations, rewards, environments, or evaluation should document:

- the research question or expected improvement;
- training and evaluation seeds;
- checkpoint and configuration identifiers;
- number of evaluation episodes;
- success, collision, timeout, reward, and path-efficiency metrics;
- hardware, ROS, simulator, and dependency versions.

Record verified aggregate results in `docs/EXPERIMENT_RESULTS.md`. Do not present preliminary or unverified values as final results.

## Code Contributions

- Keep changes scoped and compatible with ROS 2 Humble unless a migration is proposed.
- Add or update tests for behavior changes.
- Preserve the rule that only the SAC navigation controller publishes `/cmd_vel`.
- Never commit API keys, credentials, rosbags, model caches, replay buffers, or large runtime artifacts.
- Use clear commit messages and explain user-visible changes in the pull request.

## Reporting Issues

Open a GitHub issue with reproduction steps, expected and observed behavior, relevant logs, and environment details. Remove credentials and private data before attaching output.

## Pull Requests

A pull request should contain a concise motivation, implementation summary, verification evidence, and any limitations. By contributing, you agree that your contribution is licensed under the repository's MIT License.
