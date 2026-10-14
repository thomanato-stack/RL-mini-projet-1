import numpy as np
import torch.nn as nn
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.td3.policies import TD3Policy


def add_layernorm(q):
    layers = []
    for m in q[:-1]:
        layers.append(m)
        if isinstance(m, nn.Linear):
            layers.append(nn.LayerNorm(m.out_features))
    return nn.Sequential(*layers, q[-1])


class Critic(ContinuousCritic):
    def __init__(self, *args, layernorm=False, **kwargs):
        super().__init__(*args, **kwargs)
        if layernorm:
            # Wraps SB3's own Linear layers (LayerNorm draws no random numbers), so for a
            # given seed both variants start from the same weights.
            for i, q in enumerate(self.q_networks):
                self.q_networks[i] = add_layernorm(q)
                self.add_module(f"qf{i}", self.q_networks[i])


class Policy(TD3Policy):
    def __init__(self, *args, layernorm=False, **kwargs):
        self.layernorm = layernorm  # TD3Policy.__init__ builds the critic, so set it first
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return Critic(**kw, layernorm=self.layernorm).to(self.device)


def make_model(cfg, env):
    n = env.action_space.shape[0]
    policy_kwargs = dict(net_arch=list(cfg.net_arch), layernorm=cfg.layernorm)
    kw = dict(
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        action_noise=NormalActionNoise(np.zeros(n), cfg.action_noise * np.ones(n)),
        seed=cfg.seed,
        device="cpu",
    )
    if cfg.algo == "ddpg":
        return DDPG(Policy, env, policy_kwargs=policy_kwargs, **kw)
    return TD3(
        Policy,
        env,
        policy_kwargs=dict(policy_kwargs, n_critics=cfg.n_critics),
        policy_delay=cfg.policy_delay,
        target_policy_noise=cfg.target_policy_noise,
        target_noise_clip=cfg.target_noise_clip,
        **kw,
    )
