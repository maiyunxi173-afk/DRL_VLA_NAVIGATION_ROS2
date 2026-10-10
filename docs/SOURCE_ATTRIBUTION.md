# Source Attribution

This repository combines original integration work with established open-source robotics components.

## Portfolio Contribution

The portfolio-specific contribution covers the ROS 2 SAC inference path, VLA goal arbitration and bridge wiring, policy-action safety shield, launch integration, tests, experiment protocol and documentation.

## Retained and Adapted Components

- TurtleBot3 description and Gazebo simulation assets originate from ROBOTIS TurtleBot3 projects and retain their upstream notices.
- The SAC implementation and training structure were informed by earlier public SAC and DRL-navigation implementations.
- ROS 2, Gazebo Classic, PyTorch and Qwen/DashScope are external runtime dependencies.

Relevant upstream references:

- https://github.com/tomasvr/turtlebot3_drlnav
- https://github.com/reiniscimurs/DRL-robot-navigation
- https://github.com/denisyarats/pytorch_sac
- https://github.com/ROBOTIS-GIT/turtlebot3
- https://github.com/ROBOTIS-GIT/turtlebot3_simulations

The top-level MIT license applies to original material in this repository. Retained third-party files remain subject to their own licenses and notices.
