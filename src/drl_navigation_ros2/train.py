import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from SAC.SAC import SAC
from pretrain_utils import Pretraining
from replay_buffer import ReplayBuffer
from ros_python import ROS_env
from utils import record_eval_positions


MODEL_DIRECTORY = Path("src/drl_navigation_ros2/models/SAC")
CHECKPOINT_NAME = "SAC_checkpoint.pt"
REPLAY_BUFFER_NAME = "replay_buffer.pkl"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train SAC navigation policy")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume networks, optimizers and replay buffer from models/SAC",
    )
    parser.add_argument(
        "--use-legacy-pretraining",
        action="store_true",
        help=(
            "use assets/data.yml despite its old velocity/action scale; "
            "not recommended for the current environment"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(args)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_state(model, replay_buffer, epoch, episode, target_dist):
    metadata = {
        "epoch": epoch,
        "episode": episode,
        "target_dist": target_dist,
        "state_schema_version": model.STATE_SCHEMA_VERSION,
        "reward_schema_version": 2,
    }
    model.save_checkpoint(
        filename=CHECKPOINT_NAME,
        directory=MODEL_DIRECTORY,
        metadata=metadata,
    )
    replay_path = MODEL_DIRECTORY / REPLAY_BUFFER_NAME
    replay_buffer.save(replay_path)
    print(f"Saved replay buffer: {replay_path} ({replay_buffer.size()} transitions)")


def main(args=None):
    config = parse_args(args)
    set_random_seeds(config.seed)

    action_dim = 2
    max_action = 1
    state_dim = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nr_eval_episodes = 10
    nr_full_eval_episodes = 5
    full_eval_every = 5
    episodes_per_epoch = 70
    train_every_n = 8
    training_iterations = 80
    batch_size = 256
    min_replay_size = 5000
    max_steps = 500
    # One evaluation step advances Gazebo for about 0.1 s.  A 120-step cap
    # keeps the 10-scenario evaluation bounded while still allowing enough
    # time to turn toward the short-range curriculum targets.
    eval_max_steps = 120
    pretraining_iterations = 10
    save_every = 10
    random_near_obstacle = True
    near_obstacle_distance = 0.6

    epoch = 0
    episode = 0
    steps = 0
    episode_reward = 0.0
    count_rand_actions = 0
    random_action = np.zeros(action_dim, dtype=np.float32)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        hidden_dim=256,
        actor_update_frequency=2,
        # The main loop owns checkpointing so model metadata and replay remain
        # synchronized with the network weights.
        save_every=0,
        load_model=False,
        save_directory=MODEL_DIRECTORY,
    )
    ros = ROS_env(
        init_target_distance=1.0,
        target_dist_increase=0.01,
        max_target_dist=3.5,
        min_target_distance=0.8,
        target_reached_delta=0.5,
        collision_delta=0.32,
    )
    replay_buffer = ReplayBuffer(buffer_size=1e6, random_seed=config.seed)

    if config.resume:
        metadata = model.load_checkpoint(
            filename=CHECKPOINT_NAME,
            directory=MODEL_DIRECTORY,
            required_metadata={
                "state_schema_version": model.STATE_SCHEMA_VERSION,
                "reward_schema_version": 2,
            },
        )
        replay_path = MODEL_DIRECTORY / REPLAY_BUFFER_NAME
        if not replay_path.exists():
            raise FileNotFoundError(
                f"Checkpoint exists but replay buffer is missing: {replay_path}"
            )
        replay_buffer.load(replay_path)
        epoch = int(metadata.get("epoch", 0))
        episode = int(metadata.get("episode", 0))
        ros.target_dist = float(metadata.get("target_dist", ros.target_dist))
        print(
            f"Resuming at epoch={epoch}, episode={episode}, "
            f"target_dist={ros.target_dist:.2f}, transitions={replay_buffer.size()}"
        )
    elif config.use_legacy_pretraining:
        print(
            "WARNING: assets/data.yml uses the old action/velocity scale. "
            "Regenerating offline data is recommended."
        )
        pretraining = Pretraining(
            file_names=["src/drl_navigation_ros2/assets/data.yml"],
            model=model,
            replay_buffer=replay_buffer,
            reward_function=ros.get_reward,
        )
        replay_buffer = pretraining.load_buffer()
        pretraining.train(
            pretraining_iterations=pretraining_iterations,
            replay_buffer=replay_buffer,
            iterations=training_iterations,
            batch_size=batch_size,
        )
    else:
        print(
            "Starting clean training: no old weights and no legacy offline data.",
            flush=True,
        )

    # Two deterministic evaluation sets: one matching early curriculum and one
    # measuring long-range generalization.  Local RNGs keep training randomness
    # independent from evaluation generation.
    print("Preparing fixed evaluation scenarios...", flush=True)
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes,
        seed=config.seed + 1,
        target_distance_range=(0.8, 1.5),
    )
    full_eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_full_eval_episodes,
        seed=config.seed + 2,
        target_distance_range=None,
    )

    print("Resetting Gazebo world and waiting for first sensor frame...", flush=True)
    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
    print(
        f"Initial state ready: distance={distance:.3f}, min_scan={min(latest_scan):.3f}",
        flush=True,
    )

    try:
        # TensorBoard cannot draw a line from a single scalar point.  Record a
        # real, deterministic baseline at step 0 (or at the restored epoch when
        # resuming), so the first post-training evaluation forms a real curve.
        print("Recording evaluation baseline for TensorBoard...", flush=True)
        evaluate(
            model=model,
            env=ros,
            scenarios=eval_scenarios,
            epoch=epoch,
            max_steps=eval_max_steps,
            tag_prefix="eval",
        )
        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
        steps = 0
        episode_reward = 0.0

        while True:
            state, _ = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = model.get_action(state, True)
            if replay_buffer.size() < min_replay_size:
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(
                    np.float32
                )

            if random_near_obstacle:
                if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action = random_action.copy()
                elif (
                    min(latest_scan) < near_obstacle_distance
                    and np.random.random() > 0.85
                ):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.array(
                        [
                            -1.0,
                            np.random.choice([-1.0, 1.0])
                            * np.random.uniform(0.5, 1.0),
                        ],
                        dtype=np.float32,
                    )
                    action = random_action.copy()

            # Policy/replay action stays normalized in [-1, 1].  Only the command
            # sent to the robot is converted to physical velocity units.
            command = [(action[0] + 1) * 0.175, action[1] * 0.7]
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
                lin_velocity=command[0],
                ang_velocity=command[1],
                update_curriculum=True,
            )
            next_state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )

            steps += 1
            episode_reward += reward
            timeout = steps >= max_steps
            done = bool(terminal or timeout)
            # A time limit truncates data collection but is not an MDP terminal
            # state. Keep bootstrapping through it; only goal/collision is
            # terminal in the replay buffer.
            replay_buffer.add(state, action, reward, bool(terminal), next_state)

            if done:
                completed_episode = epoch * episodes_per_epoch + episode + 1
                model.writer.add_scalar(
                    "train_episode/reward", episode_reward, completed_episode
                )
                model.writer.add_scalar(
                    "train_episode/length", steps, completed_episode
                )
                model.writer.add_scalar(
                    "train_episode/goal", int(goal), completed_episode
                )
                model.writer.add_scalar(
                    "train_episode/collision", int(collision), completed_episode
                )
                model.writer.add_scalar(
                    "train_episode/timeout", int(timeout and not terminal), completed_episode
                )
                model.writer.add_scalar(
                    "train_episode/final_distance", distance, completed_episode
                )
                latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
                episode += 1
                steps = 0
                episode_reward = 0.0
                count_rand_actions = 0
                random_action.fill(0.0)

                if (
                    replay_buffer.size() >= min_replay_size
                    and episode % train_every_n == 0
                ):
                    model.train(
                        replay_buffer=replay_buffer,
                        iterations=training_iterations,
                        batch_size=batch_size,
                    )
                    if model.train_cycle % save_every == 0:
                        save_training_state(
                            model=model,
                            replay_buffer=replay_buffer,
                            epoch=epoch,
                            episode=episode,
                            target_dist=ros.target_dist,
                        )

            if episode >= episodes_per_epoch:
                episode = 0
                epoch += 1
                evaluate(
                    model=model,
                    env=ros,
                    scenarios=eval_scenarios,
                    epoch=epoch,
                    max_steps=eval_max_steps,
                    tag_prefix="eval",
                )
                if epoch % full_eval_every == 0:
                    evaluate(
                        model=model,
                        env=ros,
                        scenarios=full_eval_scenarios,
                        epoch=epoch,
                        max_steps=eval_max_steps,
                        tag_prefix="eval_full",
                    )

                # Evaluation moves the Gazebo world.  Always establish a fresh,
                # matching training state before continuing the replay stream.
                latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
                steps = 0
                episode_reward = 0.0
                count_rand_actions = 0
                random_action.fill(0.0)
                save_training_state(
                    model=model,
                    replay_buffer=replay_buffer,
                    epoch=epoch,
                    episode=episode,
                    target_dist=ros.target_dist,
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user; saving before exit...")
        save_training_state(
            model=model,
            replay_buffer=replay_buffer,
            epoch=epoch,
            episode=episode,
            target_dist=ros.target_dist,
        )
    finally:
        model.close()
        ros.close()


def evaluate(model, env, scenarios, epoch, max_steps, tag_prefix="eval"):
    """Evaluate deterministically without changing the training curriculum."""
    started_at = time.monotonic()
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating {len(scenarios)} scenarios ({tag_prefix})")
    total_reward = 0.0
    collisions = 0
    goals = 0
    timeouts = 0
    total_steps = 0
    linear_action_sum = 0.0
    angular_action_sum = 0.0
    turn_matches = 0
    turn_opportunities = 0

    for scenario_index, scenario in enumerate(scenarios, start=1):
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(
            scenario=scenario
        )
        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            if terminal:
                break
            action = model.get_action(state, False)
            linear_action_sum += float(action[0])
            angular_action_sum += float(action[1])
            if abs(sin) > 0.05:
                turn_opportunities += 1
                turn_matches += int(np.sign(action[1]) == np.sign(sin))
            command = [(action[0] + 1) * 0.175, action[1] * 0.7]
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=command[0],
                ang_velocity=command[1],
                update_curriculum=False,
            )
            total_reward += reward
            count += 1
            total_steps += 1
            collisions += int(collision)
            goals += int(goal)
        status = "goal" if goal else "collision" if collision else "timeout"
        timeouts += int(status == "timeout")
        print(
            f"  {tag_prefix} scenario {scenario_index}/{len(scenarios)} "
            f"finished: {status}, steps={count}, distance={distance:.3f}",
            flush=True,
        )

    avg_reward = total_reward / len(scenarios)
    avg_col = collisions / len(scenarios)
    avg_goal = goals / len(scenarios)
    timeout_rate = timeouts / len(scenarios)
    avg_steps = total_steps / len(scenarios)
    mean_linear_action = linear_action_sum / max(total_steps, 1)
    mean_angular_action = angular_action_sum / max(total_steps, 1)
    turn_agreement = turn_matches / max(turn_opportunities, 1)
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print(f"Average Timeout rate: {timeout_rate}")
    print(f"Average Episode steps: {avg_steps}")
    print(f"Turn-direction agreement: {turn_agreement}")
    print(f"Evaluation duration: {time.monotonic() - started_at:.1f} seconds")
    print("..............................................")
    model.writer.add_scalar(f"{tag_prefix}/avg_reward", avg_reward, epoch)
    model.writer.add_scalar(f"{tag_prefix}/avg_col", avg_col, epoch)
    model.writer.add_scalar(f"{tag_prefix}/avg_goal", avg_goal, epoch)
    model.writer.add_scalar(f"{tag_prefix}/timeout_rate", timeout_rate, epoch)
    model.writer.add_scalar(f"{tag_prefix}/avg_steps", avg_steps, epoch)
    model.writer.add_scalar(
        f"{tag_prefix}/mean_linear_action", mean_linear_action, epoch
    )
    model.writer.add_scalar(
        f"{tag_prefix}/mean_angular_action", mean_angular_action, epoch
    )
    model.writer.add_scalar(
        f"{tag_prefix}/turn_agreement", turn_agreement, epoch
    )
    model.writer.flush()
    return {
        "avg_reward": avg_reward,
        "avg_col": avg_col,
        "avg_goal": avg_goal,
        "timeout_rate": timeout_rate,
        "avg_steps": avg_steps,
        "turn_agreement": turn_agreement,
    }


if __name__ == "__main__":
    main()
