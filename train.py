import argparse
import json
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import torch as th

from bias import BiasEval
from configuration import add_args, from_args
from model import make_model


def main():
    p = add_args(argparse.ArgumentParser())
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    cfg = from_args(args)

    # One process per run: intra-op threads would only fight over the cores.
    th.set_num_threads(1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv, part, points = out / f"{cfg.name}.csv", out / f"{cfg.name}.csv.part", out / f"{cfg.name}_points.csv.gz"
    for f in (csv, part, points):
        f.unlink(missing_ok=True)
    (out / f"{cfg.name}.json").write_text(json.dumps(asdict(cfg), indent=1))

    model = make_model(cfg, gym.make(cfg.env_id))
    print(f"[{cfg.name}] actor: {model.actor.mu}\n[{cfg.name}] critic: {model.critic}", flush=True)
    model.learn(cfg.total_timesteps, callback=BiasEval(cfg, part, points))
    # Renamed only once complete, so the launcher can tell finished runs from interrupted ones.
    part.rename(csv)
    print(f"[{cfg.name}] done -> {csv}", flush=True)


if __name__ == "__main__":
    main()
