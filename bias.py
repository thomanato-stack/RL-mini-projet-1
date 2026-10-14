import gzip
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.callbacks import BaseCallback


def discounted_returns(r, gamma):
    # r[t] est la récompense apres avoir pris a_t de s_t,  r_{t+1}.
    g = np.empty(len(r))
    acc = 0.0
    for t in reversed(range(len(r))):
        acc = r[t] + gamma * acc
        g[t] = acc
    return g


def rollout(env, act, seed):
    obs, _ = env.reset(seed=seed)
    S, A, R = [], [], []
    while True:
        a = act(obs)
        S.append(obs)
        A.append(a)
        obs, r, term, trunc, _ = env.step(a)
        R.append(r)
        if term or trunc:
            return np.array(S), np.array(A), np.array(R), trunc


def keep(t, max_steps, horizon):
    return t <= max_steps - horizon


def evaluate(model, env, seeds, gamma):
    act = lambda o: model.predict(o, deterministic=True)[0]
    eps = []
    for i, seed in enumerate(seeds):
        S, A, R, trunc = rollout(env, act, seed)
        obs = th.as_tensor(S, dtype=th.float32)
        acts = th.as_tensor(model.policy.scale_action(A), dtype=th.float32)
        with th.no_grad():
            q = model.critic(obs, acts)
        eps.append(pd.DataFrame({
            "episode": i,
            "t": np.arange(len(R)),
            "T": len(R),
            "truncated": trunc,
            "reward": R,
            "g": discounted_returns(R, gamma),
            "q1": q[0].squeeze(1).numpy(),
            "q2": q[1].squeeze(1).numpy() if len(q) > 1 else np.nan,
        }))
    return pd.concat(eps, ignore_index=True)


def summarize(pts, max_steps, horizon):
    ep = pts.groupby("episode").agg(ret=("reward", "sum"), T=("T", "first"), trunc=("truncated", "first"))
    sel = pts[keep(pts["t"], max_steps, horizon)]
    q = sel[["q1", "q2"]].min(axis=1)  
    g = sel.g
    d = q - g
    return dict(
        eval_return_mean=ep.ret.mean(),
        eval_return_std=ep.ret.std(ddof=0),
        eval_len_mean=ep["T"].mean(),
        trunc_frac=ep.trunc.mean(),
        n_points=len(sel),
        n_small_g=int((g.abs() < 1).sum()),
        q_mean=q.mean(),
        q1_mean=sel.q1.mean(),
        q2_mean=sel.q2.mean(),
        g_mean=g.mean(),
        q_minus_g_mean=d.mean(),
        q_minus_g_std=d.std(ddof=0),
        normalized_redq=d.mean() / abs(g.mean()),
        normalized_pointwise=(d / g.abs()).mean(),
    )


class BiasEval(BaseCallback):
    def __init__(self, cfg, csv_path, points_path):
        super().__init__()
        self.cfg, self.csv_path, self.points_path = cfg, csv_path, points_path
        self.env = gym.make(cfg.env_id)
        # Same start states at every checkpoint and across configs sharing a seed, so curves
        # move because the policy changed, not because the evaluation draws did.
        self.seeds = [10_000 * (cfg.seed + 1) + i for i in range(cfg.n_eval_episodes)]
        self.losses = []

    def _on_training_start(self):
        self.t0 = self.t_last = time.time()

    def _on_step(self):
        # SB3 records the mean critic loss of each train() call; collected here to average it
        # over the whole interval between checkpoints.
        loss = self.logger.name_to_value.get("train/critic_loss")
        if loss is not None:
            self.losses.append(loss)
        if self.num_timesteps % self.cfg.eval_every == 0:
            self._checkpoint()
        return True

    def _checkpoint(self):
        c, n = self.cfg, self.num_timesteps
        t = time.time()
        fps = c.eval_every / (t - self.t_last)
        pts = evaluate(self.model, self.env, self.seeds, c.gamma)
        pts.insert(0, "timesteps", n)
        row = dict(
            timesteps=n, algo=c.algo, layernorm=int(c.layernorm), seed=c.seed,
            **summarize(pts, self.env.spec.max_episode_steps, c.mc_horizon),
            critic_loss=np.mean(self.losses) if self.losses else np.nan,
            fps=fps, wall_s=time.time() - self.t0,
        )
        self.losses = []
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=not self.csv_path.exists(), index=False)
        header = not self.points_path.exists()
        # newline="": otherwise Windows text mode turns pandas' \r\n into \r\r\n.
        with gzip.open(self.points_path, "at", newline="") as f:
            pts.to_csv(f, header=header, index=False, float_format="%.6g")

        self.t_last = time.time()
        ev = self.t_last - t
        eta = (c.total_timesteps - n) * (1 / fps + ev / c.eval_every) / 60
        print(
            f"[{c.name}] {n:>7d} | return {row['eval_return_mean']:7.1f} ± {row['eval_return_std']:5.1f}"
            f" | Q {row['q_mean']:7.2f}  G {row['g_mean']:7.2f}  Q-G {row['q_minus_g_mean']:+7.2f}"
            f" | pts {row['n_points']:5d}  trunc {row['trunc_frac']:.1f} | loss {row['critic_loss']:.3g}"
            f" | {fps:4.0f} fps  eval {ev:3.0f}s  eta {eta:4.0f} min",
            flush=True,
        )
