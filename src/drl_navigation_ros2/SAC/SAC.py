from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from statistics import mean
import SAC.SAC_utils as utils
from SAC.SAC_critic import DoubleQCritic as critic_model
from SAC.SAC_actor import DiagGaussianActor as actor_model
from torch.utils.tensorboard import SummaryWriter


class _NullWriter:
    """TensorBoard-compatible no-op writer for deterministic inference."""

    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


class SAC(object):
    """SAC algorithm."""

    STATE_SCHEMA_VERSION = 2
    LASER_MAX_RANGE = 7.0
    DISTANCE_SCALE = 5.0
    MAX_LINEAR_VELOCITY = 0.35
    MAX_ANGULAR_VELOCITY = 0.7

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=1e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=1e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=2,
        hidden_dim=256,
        learnable_temperature=True,
        save_every=0,
        load_model=False,
        log_dist_and_hist = False,
        save_directory=Path("src/drl_navigation_ros2/models/SAC"),
        model_name="SAC",
        load_directory=Path("src/drl_navigation_ros2/models/SAC"),
        enable_logging=True,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = (-max_action, max_action)
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.hidden_dim = hidden_dim
        self.learnable_temperature = learnable_temperature
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.log_dist_and_hist = log_dist_and_hist

        self.train_metrics_dict = { "train_critic/loss_av": [],
                                    "train_actor/loss_av": [],
                                    "train_actor/target_entropy_av": [],
                                    "train_actor/entropy_av": [],
                                    "train_alpha/loss_av": [],
                                    "train_alpha/value_av": [],
                                    "train/batch_reward_av": []
        }

        self.critic = critic_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target = critic_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = actor_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            hidden_depth=2,
            log_std_bounds=[-5, 2],
        ).to(self.device)

        if load_model:
            self.load(filename=model_name, directory=load_directory)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.critic_target.train()

        self.actor.train(True)
        self.critic.train(True)
        # Keep environment training cycles and gradient updates separate.  Update
        # frequencies must be based on gradient updates, not on model.train calls.
        self.train_cycle = 0
        self.update_step = 0
        self.writer = SummaryWriter() if enable_logging else _NullWriter()

    def save(self, filename, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), directory / f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), directory / f"{filename}_critic.pth")
        torch.save(
            self.critic_target.state_dict(),
            directory / f"{filename}_critic_target.pth",
        )
        self.writer.flush()

    def load(self, filename, directory):
        directory = Path(directory)
        self.actor.load_state_dict(
            torch.load(directory / f"{filename}_actor.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(directory / f"{filename}_critic.pth", map_location=self.device)
        )
        self.critic_target.load_state_dict(
            torch.load(
                directory / f"{filename}_critic_target.pth",
                map_location=self.device,
            )
        )
        print(f"Loaded weights from: {directory}")

    def save_checkpoint(self, filename="SAC_checkpoint.pt", directory=None, metadata=None):
        """Save everything required to continue training without losing state."""
        directory = self.save_directory if directory is None else Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / filename
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "train_cycle": self.train_cycle,
                "update_step": self.update_step,
                "metadata": metadata or {},
            },
            temporary_path,
        )
        temporary_path.replace(checkpoint_path)
        # Keep the three simple weight files for inference/backward compatibility.
        self.save(filename=self.model_name, directory=directory)
        print(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        filename="SAC_checkpoint.pt",
        directory=None,
        required_metadata=None,
    ):
        """Restore networks, optimizers, temperature and counters."""
        directory = self.save_directory if directory is None else Path(directory)
        checkpoint_path = directory / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        for key, expected_value in (required_metadata or {}).items():
            if metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"Checkpoint {key}={metadata.get(key)!r}, expected "
                    f"{expected_value!r}. Start without --resume."
                )
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.train_cycle = int(checkpoint.get("train_cycle", 0))
        self.update_step = int(checkpoint.get("update_step", 0))
        print(f"Loaded checkpoint: {checkpoint_path}")
        return metadata

    def train(self, replay_buffer, iterations, batch_size):
        for _ in range(iterations):
            self.update(
                replay_buffer=replay_buffer,
                step=self.update_step,
                batch_size=batch_size,
            )
            self.update_step += 1

        for key, value in self.train_metrics_dict.items():
            if len(value):
                self.writer.add_scalar(key, mean(value), self.train_cycle)
            self.train_metrics_dict[key] = []
        self.train_cycle += 1
        self.writer.flush()

        if self.save_every > 0 and self.train_cycle % self.save_every == 0:
            self.save_checkpoint()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs, add_noise):
        if add_noise:
            return self.act(obs, sample=True)
        return self.act(obs)

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, done, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + ((1 - done) * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        self.train_metrics_dict["train_critic/loss_av"].append(critic_loss.item())
        self.writer.add_scalar("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        if self.log_dist_and_hist:
            self.critic.log(self.writer, step)

    def update_actor_and_alpha(self, obs, step):
        for param in self.critic.parameters():
            param.requires_grad_(False)
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        self.train_metrics_dict["train_actor/loss_av"].append(actor_loss.item())
        self.train_metrics_dict["train_actor/target_entropy_av"].append(self.target_entropy)
        self.train_metrics_dict["train_actor/entropy_av"].append(-log_prob.mean().item())
        self.writer.add_scalar("train_actor/loss", actor_loss, step)
        self.writer.add_scalar("train_actor/target_entropy", self.target_entropy, step)
        self.writer.add_scalar("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        for param in self.critic.parameters():
            param.requires_grad_(True)
        if self.log_dist_and_hist:
            self.actor.log(self.writer, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.train_metrics_dict["train_alpha/loss_av"].append(alpha_loss.item())
            self.train_metrics_dict["train_alpha/value_av"].append(self.alpha.item())
            self.writer.add_scalar("train_alpha/loss", alpha_loss, step)
            self.writer.add_scalar("train_alpha/value", self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step, batch_size):
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_dones,
            batch_next_states,
        ) = replay_buffer.sample_batch(batch_size)

        state = torch.Tensor(batch_states).to(self.device)
        next_state = torch.Tensor(batch_next_states).to(self.device)
        action = torch.Tensor(batch_actions).to(self.device)
        reward = torch.Tensor(batch_rewards).to(self.device)
        done = torch.Tensor(batch_dones).to(self.device)
        self.train_metrics_dict["train/batch_reward_av"].append(batch_rewards.mean().item())
        self.writer.add_scalar("train/batch_reward", batch_rewards.mean(), step)

        self.update_critic(state, action, reward, next_state, done, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(state, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def close(self):
        self.writer.flush()
        self.writer.close()

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """Build a bounded state vector from lidar, target and last command."""
        latest_scan = np.asarray(latest_scan, dtype=np.float32)
        if latest_scan.size == 0:
            raise ValueError("LaserScan is empty")
        latest_scan = np.nan_to_num(
            latest_scan,
            nan=self.LASER_MAX_RANGE,
            posinf=self.LASER_MAX_RANGE,
            neginf=0.0,
        )
        latest_scan = np.clip(latest_scan, 0.0, self.LASER_MAX_RANGE)
        max_bins = self.state_dim - 5
        if latest_scan.size < max_bins:
            raise ValueError(
                f"LaserScan has {latest_scan.size} rays; at least {max_bins} required"
            )
        laser_bins = [
            float(segment.min() / self.LASER_MAX_RANGE)
            for segment in np.array_split(latest_scan, max_bins)
        ]
        distance_normalized = float(np.clip(distance / self.DISTANCE_SCALE, 0.0, 1.0))
        linear_normalized = float(
            np.clip(
                2.0 * action[0] / self.MAX_LINEAR_VELOCITY - 1.0,
                -1.0,
                1.0,
            )
        )
        angular_normalized = float(
            np.clip(
                action[1] / self.MAX_ANGULAR_VELOCITY,
                -1.0,
                1.0,
            )
        )
        state = laser_bins + [
            distance_normalized,
            float(np.clip(cos, -1.0, 1.0)),
            float(np.clip(sin, -1.0, 1.0)),
            linear_normalized,
            angular_normalized,
        ]

        assert len(state) == self.state_dim, len(state)
        if not np.isfinite(state).all():
            raise ValueError("Prepared state contains NaN or infinity")
        terminal = 1 if collision or goal else 0

        return state, terminal
