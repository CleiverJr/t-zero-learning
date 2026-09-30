"""Reads the A2C assignment runs from the wandb project and draws the overlaid
sweep figures for the report.

    python scripts/plot_report_a2c.py      # -> report_a2c/figs/*.png + report_a2c/runs_summary.csv

The project defaults to WANDB_ENTITY/WANDB_PROJECT (or PROJECT below); histories
are cached under report_a2c/cache/ so re-plotting does not re-download.
"""
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

PROJECT = os.environ.get(
    "WANDB_REPORT_PROJECT", "jr-cleiver-federal-university-of-goi-s/a2c-assignment"
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_a2c" / "figs"

# categorical slots in fixed order (small -> large sweep value)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

TITLES = {
    "charts/episodic_return_mean_last100": "Retorno médio (últimos 100 ep.)",
    "losses/policy_loss": "Policy loss",
    "losses/value_loss": "Value loss",
    "losses/explained_variance": "Explained variance",
    "losses/entropy": "Entropia da política",
    "charts/advantage_std": "Desvio-padrão dos pesos A",
    "charts/advantage_mean": "Média dos pesos A",
    "charts/SPS": "Passos de ambiente por segundo",
}
# EMA smoothing per metric (noisy on-policy signals)
SMOOTH = {
    "losses/value_loss": 0.9,
    "losses/explained_variance": 0.9,
    "charts/advantage_std": 0.9,
    "charts/advantage_mean": 0.9,
}
# metrics plotted raw: for the policy loss the per-update noise IS the finding
RAW_BAND = set()

BASELINE = {"num_envs": 8, "num_steps": 5, "ent_coef": 0.01, "use_baseline": True}


CACHE = ROOT / "report_a2c" / "cache"


def load_runs(project: str = PROJECT):
    """Every finished run of the project as (cfg, history, summary)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for run in wandb.Api().runs(project):
        if run.state != "finished":  # skip crashed/interrupted runs
            continue
        cached = CACHE / f"{run.id}.pkl"
        if cached.exists():
            out.append(pickle.loads(cached.read_bytes()))
            continue
        hist = defaultdict(list)
        # sampled history: one request for up to `samples` evenly spaced points
        # (scan_history streams every row and is far too slow for a sweep)
        for row in run.history(keys=list(TITLES), samples=1500, pandas=False):
            step = row.get("global_step", row.get("_step"))
            for k, v in row.items():
                if isinstance(v, (int, float)) and not k.startswith("_") and k != "global_step":
                    hist[k].append((step, v))
        for k in hist:
            hist[k].sort()
        cfg = dict(run.config)
        cfg["_display_name"], cfg["_run_id"] = run.name, run.id
        cfg["_total"] = cfg.get("total_timesteps", 0)
        item = (cfg, dict(hist), dict(run.summary))
        cached.write_bytes(pickle.dumps(item))
        out.append(item)
        print("fetched", run.id, run.name)
    return out


def ema(y, alpha):
    out, acc = np.empty(len(y)), y[0]
    for i, v in enumerate(y):
        acc = alpha * acc + (1 - alpha) * v
        out[i] = acc
    return out


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel("global step", fontsize=8, color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))


def sps_bar(ax, values):
    """Clean SPS benchmark: one run at a time, 200k steps, no wandb — the SPS
    logged during the sweep is contaminated by runs competing for CPU."""
    bench = {}
    with open(ROOT / "report_a2c" / "sps_benchmark.csv") as fh:
        for row in csv.DictReader(fh):
            bench[int(row["num_envs"])] = int(row["sps"])
    vals = [v for v in values if v in bench]
    ax.bar(range(len(vals)), [bench[v] for v in vals],
           color=[PALETTE[values.index(v)] for v in vals], width=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{bench[v]:,}".replace(",", " "), (i, bench[v]), ha="center",
                    textcoords="offset points", xytext=(0, 3), fontsize=8, color=INK2)
    ax.set_xlabel("num_envs", fontsize=8, color=INK2)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(len(vals))))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(v) for v in vals]))
    ax.set_ylim(0, max(bench.values()) * 1.18)


def figure(runs, key, values, label, metrics, fname, logy=(), fmt=str):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        if m == "charts/SPS":
            style(ax, "SPS (medido isoladamente)")  # before the bars: style() sets a k-formatter
            sps_bar(ax, values)
            continue
        for ci, val in enumerate(values):
            for (v, seed), (_, hist, _) in sorted(runs.items(), key=lambda kv: str(kv[0])):
                if v != val or m not in hist:
                    continue
                x, y = map(np.asarray, zip(*hist[m]))
                lbl = None
                if seed == 1:
                    lbl = f"{label}={fmt(val)}" + (" (baseline)" if val == BASELINE.get(key) else "")
                if m in RAW_BAND:  # raw trace behind the smoothed line: the noise IS the finding
                    ax.plot(x, y, color=PALETTE[ci], linewidth=0.5, alpha=0.25)
                if m in SMOOTH:
                    y = ema(y, SMOOTH[m])
                ax.plot(x, y, color=PALETTE[ci], linewidth=1.6 if seed == 1 else 1.2,
                        linestyle="-" if seed == 1 else "--", alpha=1 if seed == 1 else 0.85, label=lbl)
        if m in logy:
            ax.set_yscale("log")
        if m == "losses/policy_loss":  # spikes span orders of magnitude, both signs
            ax.set_yscale("symlog", linthresh=1)
        if m == "losses/explained_variance":
            ax.set_ylim(-1.05, 1.05)
        if m == "losses/entropy":  # ln 2 = uniform policy over CartPole's 2 actions
            ax.axhline(np.log(2), color=INK2, linewidth=1, linestyle=":")
            ax.annotate("ln 2 (uniforme)", (0.02, np.log(2)), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 3), fontsize=7, color=INK2)
        if m == "charts/advantage_mean":
            ax.axhline(0.0, color=INK2, linewidth=1, linestyle=":")
        style(ax, TITLES.get(m, m))
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, linestyle="--"))
    labels.append("seed 2 (tracejado)")
    fig.legend(handles, labels, loc="outside lower center",
               ncol=min(len(labels), 4 if len(metrics) < 3 else 7), fontsize=8, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)
    print("wrote", OUT / fname)


def main():
    all_runs = [
        (cfg["_run_id"], cfg, hist, summary)
        for cfg, hist, summary in load_runs()
        if cfg.get("algorithm") == "a2c" and hist.get("charts/episodic_return_mean_last100")
    ]
    if not all_runs:
        sys.exit(f"no finished a2c runs found in {PROJECT}")

    rows = []
    for name, cfg, hist, summary in all_runs:
        a = cfg["a2c"]
        ret = [v for _, v in hist["charts/episodic_return_mean_last100"]]
        sps = [v for _, v in hist.get("charts/SPS", [(0, 0)])]
        rows.append(dict(
            run_name=cfg["_display_name"], run_id=cfg["_run_id"], seed=cfg["seed"],
            env_id=cfg["env_id"], num_envs=cfg["num_envs"], num_steps=a["num_steps"],
            ent_coef=a["ent_coef"], use_baseline=a["use_baseline"], batch_size=cfg["num_envs"] * a["num_steps"],
            final_return_last100=round(ret[-1], 1), max_return_last100=round(max(ret), 1),
            eval_mean_return=summary.get("eval/mean_return"), eval_std_return=summary.get("eval/std_return"),
            sps_median=int(np.median(sps)),
            final_entropy=round(hist["losses/entropy"][-1][1], 3) if "losses/entropy" in hist else None,
        ))
    (ROOT / "report_a2c").mkdir(exist_ok=True)
    with open(ROOT / "report_a2c" / "runs_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["env_id"], r["num_envs"], r["num_steps"],
                                                r["ent_coef"], not r["use_baseline"], r["seed"])))

    def sweep(key, extra_filter=None):
        """CartPole runs that differ from the baseline only in `key`."""
        sel = {}
        for (_, cfg, hist, summary), r in zip(all_runs, rows):
            if r["env_id"] != "CartPole-v1":
                continue
            if any(r[k] != BASELINE[k] for k in BASELINE if k != key):
                continue
            if extra_filter and not extra_filter(r):
                continue
            sel[(r[key], r["seed"])] = (cfg, hist, summary)
        return sel, sorted({v for v, _ in sel})

    runs, vals = sweep("num_envs")
    figure(runs, "num_envs", vals, "num_envs",
           ["charts/episodic_return_mean_last100", "losses/policy_loss", "charts/SPS"],
           "q1_num_envs.png")

    runs, vals = sweep("num_steps")
    figure(runs, "num_steps", vals, "num_steps",
           ["charts/episodic_return_mean_last100", "losses/value_loss", "losses/explained_variance"],
           "q2_num_steps.png", logy=("losses/value_loss",))

    runs, vals = sweep("ent_coef")
    figure(runs, "ent_coef", vals, "ent_coef",
           ["charts/episodic_return_mean_last100", "losses/entropy"], "q3_ent_coef.png")

    # ---- extra: LunarLander (every run, labelled by what was changed) ----
    lunar = [(cfg, hist) for _, cfg, hist, _ in all_runs if cfg["env_id"] == "LunarLander-v3"]
    if lunar:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
        # one representative run per step of the tuning story (the full list is
        # in runs_summary.csv); more than ~5 lines would reuse palette colors
        KEEP = {(7e-4, 16, 16, 0.01), (3e-4, 32, 16, 0.01), (1e-4, 32, 16, 0.001),
                (1e-4, 64, 16, 0.001)}
        lunar = [(c, h) for c, h in lunar
                 if (c["a2c"]["learning_rate"], c["a2c"]["num_steps"], c["num_envs"],
                     c["a2c"]["ent_coef"]) in KEEP]
        order = sorted(lunar, key=lambda ch: (-ch[0]["a2c"]["learning_rate"], ch[0]["a2c"]["num_steps"],
                                              -ch[0]["a2c"]["gamma"], -ch[0]["_total"]))
        for ci, (cfg, hist) in enumerate(order):
            a = cfg["a2c"]
            lbl = (f"lr={a['learning_rate']:g} n={a['num_steps']} envs={cfg['num_envs']} "
                   f"ent={a['ent_coef']:g} ({cfg['_total']/1e6:g}M passos)")
            for ax, m in zip(axes, ["charts/episodic_return_mean_last100", "losses/entropy"]):
                if m not in hist:
                    continue
                x, y = map(np.asarray, zip(*sorted(hist[m])))
                ax.plot(x, ema(y, 0.6), color=PALETTE[ci % len(PALETTE)], linewidth=1.4,
                        label=lbl if m.startswith("charts/") else None)
        axes[0].axhline(200, color=INK2, linewidth=1, linestyle=":")
        axes[0].annotate("resolvido = 200", (0.02, 200), xycoords=("axes fraction", "data"),
                         textcoords="offset points", xytext=(0, 3), fontsize=7, color=INK2)
        axes[1].axhline(np.log(4), color=INK2, linewidth=1, linestyle=":")
        axes[1].annotate("ln 4 (uniforme)", (0.02, np.log(4)), xycoords=("axes fraction", "data"),
                         textcoords="offset points", xytext=(0, 3), fontsize=7, color=INK2)
        for ax in axes:
            style(ax, "")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:g}M"))
        axes[0].set_title(TITLES["charts/episodic_return_mean_last100"], loc="left", fontsize=10, color=INK)
        axes[1].set_title(TITLES["losses/entropy"], loc="left", fontsize=10, color=INK)
        fig.legend(loc="outside lower center", ncol=2, fontsize=7, frameon=False)
        fig.savefig(OUT / "extra_lunarlander.png", dpi=200)
        plt.close(fig)
        print("wrote", OUT / "extra_lunarlander.png")

    runs, vals = sweep("use_baseline")
    figure(runs, "use_baseline", sorted(vals, reverse=True), "baseline",
           ["charts/episodic_return_mean_last100", "charts/advantage_std",
            "charts/advantage_mean", "losses/entropy"],
           "q4_use_baseline.png", logy=("charts/advantage_std",),
           fmt=lambda v: "sim (R−V)" if v else "não (R)")


if __name__ == "__main__":
    main()
