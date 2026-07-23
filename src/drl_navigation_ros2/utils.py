from dataclasses import dataclass
import numpy as np


@dataclass
class pos_data:
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0


def check_position(x, y, element_positions, min_dist):
    pos = True
    for element in element_positions:
        distance_vector = [element[0] - x, element[1] - y]
        distance = np.linalg.norm(distance_vector)
        if distance < min_dist:
            pos = False
    return pos


def set_random_position(name, element_positions, rng):
    angle = rng.uniform(-np.pi, np.pi)
    pos = False
    while not pos:
        x = rng.uniform(-4.0, 4.0)
        y = rng.uniform(-4.0, 4.0)
        pos = check_position(x, y, element_positions, 1.8)
    element_positions.append([x, y])
    eval_element = pos_data()
    eval_element.name = name
    eval_element.x = x
    eval_element.y = y
    eval_element.angle = angle
    return eval_element


def set_target_near_robot(
    robot,
    obstacle_positions,
    rng,
    min_target_distance,
    max_target_distance,
):
    """Generate a reproducible curriculum-level evaluation target."""
    while True:
        distance = rng.uniform(min_target_distance, max_target_distance)
        angle = rng.uniform(-np.pi, np.pi)
        x = robot.x + distance * np.cos(angle)
        y = robot.y + distance * np.sin(angle)
        if not (-4.0 <= x <= 4.0 and -4.0 <= y <= 4.0):
            continue
        if not check_position(x, y, obstacle_positions, 1.2):
            continue
        return pos_data(name="target", x=x, y=y, angle=0.0)


def record_eval_positions(
    n_eval_scenarios=10,
    seed=2026,
    target_distance_range=(0.8, 1.5),
):
    """Create fixed scenarios without changing NumPy's global RNG state.

    target_distance_range=None produces the original full-map evaluation.
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n_eval_scenarios):
        eval_scenario = []
        element_positions = [[-2.93, 3.17], [2.86, -3.0], [-2.77, -0.96], [2.83, 2.93]]
        for i in range(4, 8):
            name = "obstacle" + str(i + 1)
            eval_element = set_random_position(name, element_positions, rng)
            eval_scenario.append(eval_element)

        robot = set_random_position("turtlebot3_waffle", element_positions, rng)
        eval_scenario.append(robot)

        if target_distance_range is None:
            target = set_random_position("target", element_positions, rng)
        else:
            target = set_target_near_robot(
                robot=robot,
                obstacle_positions=element_positions[:-1],
                rng=rng,
                min_target_distance=target_distance_range[0],
                max_target_distance=target_distance_range[1],
            )
        eval_scenario.append(target)

        scenarios.append(eval_scenario)

    return scenarios
