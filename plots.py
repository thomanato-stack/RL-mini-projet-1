import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from rliable import library as rly
from scipy.stats import trim_mean

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ALGOS = {"ddpg": "DDPG", "td3": "TD3"}
LN = {0: dict(color=BLUE, ls="--", label="No LayerNorm"), 1: dict(color=ORANGE, ls="-", label="LayerNorm in critic")}
QS = {
    "q1_mean": dict(color=INK2, ls="-", lw=0.9, label="Q1"),
    "q2_mean": dict(color=INK2, ls=":", lw=0.9, label="Q2"),
    "q_mean": dict(color=AQUA, ls="-", lw=1.5, label="min(Q1, Q2)"),
    "g_mean": dict(color=INK, ls="--", lw=1.5, label="Monte-Carlo return G"),
}
FINAL = {"eval_return_mean": "Return", "q_minus_g_mean": "Q - G", "normalized_redq": "(Q - G) / |G| (REDQ)"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "pdf.fonttype": 42,  # embed TrueType: some venues reject Type 3 fonts
})


def iqm(x):
    # (runs, tasks, checkpoints) -> IQM over runs and tasks at each checkpoint
    return trim_mean(x.reshape(-1, x.shape[-1]), 0.25, axis=0)


def load(d):
    df = pd.concat(map(pd.read_csv, sorted(Path(d).glob("*.csv"))), ignore_index=True)
    cfgs = [json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))]
    for c in cfgs:
        for k in ("algo", "layernorm", "seed"):
            c.pop(k)
    assert all(c == cfgs[0] for c in cfgs), f"runs in {d} do not share the same hyperparameters"
    n = df.groupby(["algo", "layernorm"]).seed.nunique()
    assert n.nunique() == 1, f"unequal number of seeds per config:\n{n}"
    # Mean over states of (Q1 + Q2)/2 - min(Q1, Q2) = |Q1 - Q2|/2: how much the min lowers the estimate.
    df["q_gap"] = (df.q1_mean + df.q2_mean) / 2 - df.q_mean
    return df, cfgs[0], n.iloc[0]


def curves(df, col, reps):
    m = {k: g.pivot(index="seed", columns="timesteps", values=col) for k, g in df.groupby(["algo", "layernorm"])}
    assert not any(v.isna().any().any() for v in m.values()), f"missing checkpoints for {col}"
    point, ci = rly.get_interval_estimates({k: v.to_numpy()[:, None, :] for k, v in m.items()}, iqm, reps=reps)
    return {k: (m[k].columns.to_numpy(), point[k], ci[k]) for k in m}


def draw(ax, est, color, ls="-", lw=1.5, band=True):
    x, p, ci = est
    if band:
        ax.fill_between(x, ci[0], ci[1], color=color, alpha=0.18, lw=0)
    ax.plot(x, p, color=color, ls=ls, lw=lw)


def steps_axis(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Environment steps")


def ci_patch():
    return Patch(color=MUTED, alpha=0.3, lw=0, label="95% CI (stratified bootstrap)")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out / name}.pdf")


def by_algo(est, ylabel, out, name, sharey=True, ref=None, symlog=False):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=sharey, layout="constrained")
    for ax, algo in zip(axes, ALGOS):
        if ref is not None:
            ax.axhline(ref[0], color=MUTED, lw=0.8, zorder=1)
            if ref[1]:
                ax.text(1, ref[0], ref[1], transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=MUTED, fontsize=8)
        for ln, st in LN.items():
            draw(ax, est[algo, ln], st["color"], st["ls"])
        if symlog:
            ax.set_yscale("symlog", linthresh=1)
        ax.set_title(ALGOS[algo])
        steps_axis(ax)
    axes[0].set_ylabel(ylabel)
    handles = [Line2D([], [], color=s["color"], ls=s["ls"], lw=1.5, label=s["label"]) for s in LN.values()]
    fig.legend(handles=handles + [ci_patch()], loc="outside upper center", ncol=3, frameon=False)
    save(fig, out, name)


def td3_critics(df, n, reps, out):
    est = {c: curves(df[df.algo == "td3"], c, reps) for c in [*QS, "q_gap"]}
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.4), sharex=True, sharey="row", height_ratios=[2, 1], layout="constrained")
    for j, ln in enumerate(LN):
        top, bot = axes[:, j]
        for c, st in QS.items():
            # Bands only on the min and on G: Q1 and Q2 sit on top of the min and would hide it.
            draw(top, est[c]["td3", ln], st["color"], st["ls"], st["lw"], band=c in ("q_mean", "g_mean"))
        draw(bot, est["q_gap"]["td3", ln], AQUA)
        top.set_title(f"TD3 – {LN[ln]['label']}")
        steps_axis(bot)
    axes[0, 0].set_ylabel(f"Value (return units)\nIQM over {n} seeds")
    axes[1, 0].set_ylabel("(Q1 + Q2)/2 − min\n(return units)")
    handles = [Line2D([], [], color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"]) for s in QS.values()]
    fig.legend(handles=handles + [ci_patch()], loc="outside upper center", ncol=3, frameon=False)
    save(fig, out, "td3_critics")


def final_table(df, last, reps, out):
    ts = np.sort(df.timesteps.unique())[-last:]
    fin = df[df.timesteps.isin(ts)].groupby(["algo", "layernorm", "seed"])[list(FINAL)].mean()
    rows = []
    for (algo, ln), g in fin.groupby(level=[0, 1]):
        point, ci = rly.get_interval_estimates({c: g[c].to_numpy()[:, None, None] for c in FINAL}, iqm, reps=reps)
        row = {"config": ALGOS[algo] + (" + LN" if ln else "")}
        for c in FINAL:
            row |= {c: point[c][0], f"{c}_lo": ci[c][0, 0], f"{c}_hi": ci[c][1, 0]}
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "final.csv", index=False)

    def fmt(r, c):
        d = 2 if c == "normalized_redq" else 1
        return f"{r[c]:.{d}f} [{r[c + '_lo']:.{d}f}, {r[c + '_hi']:.{d}f}]"

    txt = pd.DataFrame({"config": tab.config, **{name: [fmt(r, c) for _, r in tab.iterrows()] for c, name in FINAL.items()}})
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Config & Return & $Q - G$ & $\overline{Q-G}\,/\,|\overline{G}|$ \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in txt.to_numpy()]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "final.tex").write_text("\n".join(lines) + "\n")
    print(f"\nFinal values: per-seed mean over checkpoints {', '.join(f'{t // 1000}k' for t in ts)}, "
          f"then IQM [95% CI] across seeds\n{txt.to_string(index=False)}\nwrote {out / 'final.tex'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", default="results/main")
    p.add_argument("--out", default="figures")
    p.add_argument("--reps", type=int, default=50_000, help="bootstrap replications")
    p.add_argument("--last", type=int, default=3, help="checkpoints averaged per seed for the final table")
    args = p.parse_args()
    # rliable's StratifiedBootstrap draws from the global numpy RNG, not from random_state.
    np.random.seed(0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, cfg, n = load(args.results)
    # Up to learning_starts the critic has never been updated: Q is its initial output (~0), not an estimate.
    crit = df[df.timesteps > cfg["learning_starts"]]

    by_algo(curves(df, "eval_return_mean", args.reps), f"Evaluation return (sum of rewards)\nIQM over {n} seeds",
            out, "learning_curves", ref=(200, "solved (200)"))
    by_algo(curves(crit, "q_minus_g_mean", args.reps), f"Bias Q − G (return units)\nIQM over {n} seeds",
            out, "bias", sharey=False, ref=(0, None))
    by_algo(curves(crit, "normalized_redq", args.reps), f"mean(Q − G) / |mean(G)|\nIQM over {n} seeds, symlog scale",
            out, "bias_normalized", sharey=False, ref=(0, None), symlog=True)
    td3_critics(crit, n, args.reps, out)
    final_table(df, args.last, args.reps, out)


if __name__ == "__main__":
    main()
