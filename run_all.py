import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from configuration import add_args, from_args

TRAIN = Path(__file__).with_name("train.py")


def main():
    p = argparse.ArgumentParser(description="Extra arguments are forwarded to every train.py run.")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--algos", nargs="+", choices=["ddpg", "td3"], default=["ddpg", "td3"])
    p.add_argument("--layernorms", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    p.add_argument("--configs", nargs="+", metavar="'KEY=VAL ...'",
                   help="explicit configs, each a quoted list of Config overrides; replaces --algos x --layernorms")
    p.add_argument("--out_dir", default="results")
    args, extra = p.parse_known_args()

    if args.configs:
        specs = [[f"--{kv}" for kv in c.split()] for c in args.configs]
    else:
        specs = [["--algo", a, "--layernorm", str(ln)] for a in args.algos for ln in args.layernorms]
    # Seed-major order: if we have to stop early, every config has the same number of seeds.
    # Parsed with train.py's own parser, so names match the files it writes and typos fail here.
    cli = add_args(argparse.ArgumentParser(prog="train.py"))
    jobs = [[*extra, *s, "--seed", str(seed)] for seed in args.seeds for s in specs]
    names = [from_args(cli.parse_args(j)).name for j in jobs]
    assert len(set(names)) == len(names), f"configs collide on a run name: {names}"

    out = Path(args.out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    todo = [(n, j) for n, j in zip(names, jobs) if not (out / f"{n}.csv").exists()]
    print(f"{len(jobs) - len(todo)} runs already done, {len(todo)} to go on {args.workers} workers", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(job):
        name, argv = job
        print(f"start {name}", flush=True)
        t = time.time()
        with open(out / "logs" / f"{name}.log", "w") as log:
            cmd = [sys.executable, str(TRAIN), *argv, "--out_dir", str(out)]
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print(f"{'done' if code == 0 else 'FAILED'} {name} in {(time.time() - t) / 60:.0f} min", flush=True)
        return name, code

    

    with ThreadPoolExecutor(args.workers) as ex:
        failed = [name for name, code in ex.map(run, todo) if code]
    if failed:
        sys.exit(f"failed: {', '.join(failed)} (see {out / 'logs'})")


if __name__ == "__main__":
    main()
