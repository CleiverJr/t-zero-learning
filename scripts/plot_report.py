"""Reads the local wandb run files (online or offline) and draws the overlaid
sweep figures used in the DQN report.

    python scripts/plot_report.py            # -> report/figs/*.png + report/runs_summary.csv
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figs"

# categorical slots in fixed order (small -> large sweep value)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

TITLES = {
    "charts/episodic_return_mean_last100": "Retorno médio (últimos 100 ep.)",
    "losses/td_loss": "TD loss (EMA 0.9)",
    "losses/q_values": "Q médio predito",
    "charts/epsilon": "Epsilon",
}
SMOOTH = {"losses/td_loss": 0.9, "losses/q_values": 0.6}


def load_run(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    cfg, hist, summary = {}, defaultdict(list), {}
    while (data := ds.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(data)
        kind = rec.WhichOneof("record_type")
        if kind == "run":
            cfg = {u.key: json.loads(u.value_json) for u in rec.run.config.update}
            cfg["_display_name"], cfg["_run_id"] = rec.run.display_name, rec.run.run_id
        elif kind == "history":
            row = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in rec.history.item}
            step = row.get("global_step", row.get("_step"))
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    hist[k].append((step, v))
        elif kind == "summary":
            for i in rec.summary.update:
                summary[i.key or "/".join(i.nested_key)] = json.loads(i.value_json)
    return cfg, hist, summary


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


def figure(runs, key, values, label, metrics, fname, logy=()):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        for ci, val in enumerate(values):
            for (v, seed), (_, hist, _) in sorted(runs.items()):
                if v != val or m not in hist:
                    continue
                x, y = map(np.asarray, zip(*hist[m]))
                if m in SMOOTH:
                    y = ema(y, SMOOTH[m])
                ax.plot(x, y, color=PALETTE[ci], linewidth=1.6 if seed == 1 else 1.2,
                        linestyle="-" if seed == 1 else "--", alpha=1 if seed == 1 else 0.8,
                        label=f"{label}={val}" + (" (baseline)" if val == BASELINE[key] else "") if seed == 1 else None)
        if m == "losses/q_values" and key != "gamma":
            # r=1 per step -> no true Q can exceed sum_k gamma^k = 1/(1-gamma)
            q_max = 1 / (1 - BASELINE["gamma"])
            ax.axhline(q_max, color=INK2, linewidth=1, linestyle=":")
            ax.annotate(f"1/(1-γ) = {q_max:.0f}", (0.02, q_max), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 3), fontsize=7, color=INK2)
        if m == "losses/q_values" and key == "gamma":
            for ci, val in enumerate(values):
                ax.axhline(1 / (1 - val), color=PALETTE[ci], linewidth=1, linestyle=":")
        if m in logy:
            if m == "losses/q_values":
                ax.set_yscale("symlog", linthresh=10)
                ax.set_ylim(bottom=0)
            else:
                ax.set_yscale("log")
        style(ax, TITLES.get(m, m))
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, linestyle="--"))
    labels.append("seed 2 (tracejado)")
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 4 if len(metrics) < 3 else 7), fontsize=8, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)
    print("wrote", OUT / fname)


BASELINE = {"target_network_frequency": 500, "buffer_size": 10000, "gamma": 0.99}


def main():
    all_runs = []
    for f in sorted(glob.glob(str(ROOT / "wandb" / "*run-*" / "run-*.wandb"))):
        cfg, hist, summary = load_run(f)
        if cfg.get("algorithm") != "dqn" or not hist.get("charts/episodic_return_mean_last100"):
            continue
        all_runs.append((Path(f).parent.name, cfg, hist, summary))
    if not all_runs:
        sys.exit("no dqn runs found under wandb/")

    rows = []
    for name, cfg, hist, summary in all_runs:
        d = cfg["dqn"]
        ret = [v for _, v in hist["charts/episodic_return_mean_last100"]]
        rows.append(dict(wandb_dir=name, run_name=cfg["_display_name"], run_id=cfg["_run_id"], seed=cfg["seed"],
                         target_network_frequency=d["target_network_frequency"], buffer_size=d["buffer_size"],
                         gamma=d["gamma"], final_return_last100=round(ret[-1], 1), max_return_last100=round(max(ret), 1),
                         eval_mean_return=summary.get("eval/mean_return"), eval_std_return=summary.get("eval/std_return"),
                         final_q=round(hist["losses/q_values"][-1][1], 2) if "losses/q_values" in hist else None))
    (ROOT / "report").mkdir(exist_ok=True)
    with open(ROOT / "report" / "runs_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["target_network_frequency"], r["buffer_size"], r["gamma"], r["seed"])))

    def sweep(key):
        """runs that differ from the baseline only in `key` (baseline included)."""
        sel = {}
        for (_, cfg, hist, summary), r in zip(all_runs, rows):
            if all(r[k] == BASELINE[k] for k in BASELINE if k != key):
                sel[(r[key], r["seed"])] = (cfg, hist, summary)
        return sel, sorted({v for v, _ in sel})

    runs, vals = sweep("target_network_frequency")
    figure(runs, "target_network_frequency", vals, "tnf",
           ["charts/episodic_return_mean_last100", "losses/td_loss", "losses/q_values"], "q1_target_network_frequency.png",
           logy=("losses/td_loss", "losses/q_values"))
    runs, vals = sweep("buffer_size")
    figure(runs, "buffer_size", vals, "buffer",
           ["charts/episodic_return_mean_last100", "losses/q_values"], "q2_buffer_size.png")
    runs, vals = sweep("gamma")
    figure(runs, "gamma", vals, "γ",
           ["charts/episodic_return_mean_last100", "losses/q_values", "losses/td_loss"], "q3_gamma.png",
           logy=("losses/q_values", "losses/td_loss"))


if __name__ == "__main__":
    main()
