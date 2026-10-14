import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    algo: str = "td3"
    layernorm: bool = False
    seed: int = 0
    env_id: str = "LunarLanderContinuous-v3"

    total_timesteps: int = 300_000
    eval_every: int = 10_000
    n_eval_episodes: int = 10
    mc_horizon: int = 500

    # RL Zoo
    gamma: float = 0.98
    learning_rate: float = 1e-3
    buffer_size: int = 200_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: float = 0.1  # std of the Gaussian exploration noise
    net_arch: tuple = (400, 300)
    # SB3 defaults (not set by the Zoo).
    batch_size: int = 256
    tau: float = 0.005
    # TD3 only: SB3's DDPG hardcodes n_critics=1, policy_delay=1 and no target smoothing.
    n_critics: int = 2
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5

    def __post_init__(self):
        assert self.algo in ("ddpg", "td3"), self.algo
        assert self.total_timesteps >= self.eval_every, "no checkpoint would be logged"
        assert self.n_critics in (1, 2), "bias.py logs q1 and q2 only"

    @property
    def name(self):
        # A factor is tagged only when it differs from its default, so earlier run names are unchanged.
        default = {f.name: f.default for f in fields(self)}
        tags = [f"{t}{getattr(self, f):g}" for f, t in TAGS.items()
                if getattr(self, f) != default[f] and (self.algo == "td3" or f not in TD3_ONLY)]
        return "_".join([self.algo, "ln" if self.layernorm else "noln", *tags, f"seed{self.seed}"])


TAGS = dict(learning_rate="lr", n_critics="c", policy_delay="d", target_policy_noise="tpn", target_noise_clip="tnc")
TD3_ONLY = ("n_critics", "policy_delay", "target_policy_noise", "target_noise_clip")


def _bool(s):
    if s.lower() in ("1", "true"):
        return True
    if s.lower() in ("0", "false"):
        return False
    raise argparse.ArgumentTypeError(f"expected 0/1, got {s!r}")


def _ints(s):
    return tuple(int(x) for x in s.split(","))


def add_args(parser):
    for f in fields(Config):
        d = f.default
        typ = _bool if isinstance(d, bool) else _ints if isinstance(d, tuple) else type(d)
        parser.add_argument(f"--{f.name}", type=typ, default=d)
    return parser


def from_args(args):
    return Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
