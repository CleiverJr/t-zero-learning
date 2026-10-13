"""Figures for the PPO report (Parts 4 and 5), from the wandb project.

    python scripts/plot_report_ppo.py   # -> report_ppo/figs/*.png + runs_summary.csv

Histories are cached under report_ppo/cache/ so re-plotting does not re-download.
"""
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

PROJECT = "jr-cleiver-federal-university-of-goi-s/ppo-assignment"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_ppo" / "figs"
CACHE = ROOT / "report_ppo" / "cache"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

KEYS = ["charts/episodic_return_mean_last100", "losses/approx_kl", "losses/clipfrac",
        "losses/entropy", "losses/explained_variance", "losses/value_loss",
        "losses/policy_loss", "charts/SPS", "charts/num_updates", "losses/grad_norm",
        "charts/advantage_std", "charts/learning_rate"]
TITLES = {
    "charts/episodic_return_mean_last100": "Retorno médio (últimos 100 ep.)",
    "losses/approx_kl": "approx_kl",
    "losses/clipfrac": "clipfrac",
    "losses/entropy": "Entropia da política",
    "losses/explained_variance": "Explained variance",
    "losses/value_loss": "Value loss",
    "charts/num_updates": "Passos de gradiente acumulados",
}
SMOOTH = {"losses/approx_kl": 0.8, "losses/clipfrac": 0.8, "losses/value_loss": 0.8,
          "losses/explained_variance": 0.8, "charts/episodic_return_mean_last100": 0.3}


def load_runs():
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for run in wandb.Api().runs(PROJECT):
        if run.state != "finished":
            continue
        cached = CACHE / f"{run.id}.pkl"
        if cached.exists():
            out.append(pickle.loads(cached.read_bytes()))
            continue
        hist = defaultdict(list)
        # history(keys=...) only returns rows that contain *every* key, and the
        # A2C runs log no approx_kl/clipfrac: ask each run for what it has
        keys = [k for k in KEYS if k in run.summary]
        for row in run.history(keys=keys, samples=1200, pandas=False):
            step = row.get("global_step", row.get("_step"))
            for k, v in row.items():
                if isinstance(v, (int, float)) and not k.startswith("_") and k != "global_step":
                    hist[k].append((step, v))
        # wall-clock curve: seconds since the run started
        rt = defaultdict(list)
        for row in run.history(keys=["charts/episodic_return_mean_last100", "_runtime"],
                               samples=1200, pandas=False):
            if "_runtime" in row and "charts/episodic_return_mean_last100" in row:
                rt["ret_vs_time"].append((row["_runtime"], row["charts/episodic_return_mean_last100"]))
        cfg = dict(run.config)
        cfg.update(_group=run.group or run.name, _id=run.id, _name=run.name,
                   _runtime=run.summary.get("_runtime", float("nan")))
        item = (cfg, {k: sorted(v) for k, v in {**hist, **rt}.items()}, dict(run.summary))
        cached.write_bytes(pickle.dumps(item))
        out.append(item)
        print("fetched", run.id, cfg["_group"])
    return out


def ema(y, alpha):
    out, acc = np.empty(len(y)), y[0]
    for i, v in enumerate(y):
        acc = alpha * acc + (1 - alpha) * v
        out[i] = acc
    return out


def style(ax, title, xlabel="global step", k_axis=True):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    if k_axis:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))


def series(hist, metric):
    v = hist.get(metric)
    if not v:
        return None, None
    x, y = map(np.asarray, zip(*v))
    return x, (ema(y, SMOOTH[metric]) if metric in SMOOTH else y)


def panel(ax, runs, metric, label_of, solved=None):
    # one colour per label (a group may have one seed or two)
    order = []
    for cfg, _, _ in runs:
        if label_of(cfg) not in order:
            order.append(label_of(cfg))
    for cfg, hist, _ in runs:
        x, y = series(hist, metric)
        if x is None:
            continue
        seed = cfg["seed"]
        ax.plot(x, y, color=PALETTE[order.index(label_of(cfg)) % len(PALETTE)],
                linewidth=1.5 if seed == 1 else 1.2,
                linestyle="-" if seed == 1 else "--", alpha=1 if seed == 1 else 0.85,
                label=label_of(cfg) if seed == 1 else None)
    if solved is not None:
        ax.axhline(solved, color=INK2, linewidth=1, linestyle=":")
        ax.annotate("resolvido = 200", (0.02, solved), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 3), fontsize=7, color=INK2)
    style(ax, TITLES.get(metric, metric))


def figure(runs, metrics, fname, label_of, width=4.0, solved_on=("charts/episodic_return_mean_last100",),
           logy=(), ncol=4):
    fig, axes = plt.subplots(1, len(metrics), figsize=(width * len(metrics), 3.0), constrained_layout=True)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        panel(ax, runs, m, label_of, solved=200 if m in solved_on else None)
        if m in logy:
            ax.set_yscale("log")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, linestyle="--"))
    labels.append("seed 2 (tracejado)")
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, fontsize=8, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)
    print("wrote", OUT / fname)


def main():
    runs = load_runs()
    for cfg, hist, _ in runs:
        # A2C logs no num_updates: it does exactly one gradient step per rollout
        if cfg["algorithm"] == "a2c" and "charts/num_updates" not in hist:
            batch = cfg["num_envs"] * cfg["a2c"]["num_steps"]
            hist["charts/num_updates"] = [(x, x / batch)
                                          for x, _ in hist["charts/episodic_return_mean_last100"]]
    by_group = defaultdict(list)
    for cfg, hist, summ in runs:
        by_group[cfg["_group"]].append((cfg, hist, summ))

    def pick(*groups):
        """runs of these groups, in the given order, seed 1 first."""
        out = []
        for g in groups:
            for item in sorted(by_group.get(g, []), key=lambda t: t[0]["seed"]):
                out.append(item)
        return out

    # --- summary table -----------------------------------------------------
    rows = []
    for cfg, hist, summ in sorted(runs, key=lambda t: (t[0]["_group"], t[0]["seed"])):
        algo, p = cfg["algorithm"], cfg.get("ppo") or cfg.get("a2c")
        ret = [v for _, v in hist.get("charts/episodic_return_mean_last100", [(0, np.nan)])]
        late = ret[int(len(ret) * 0.7):] or [np.nan]
        kl = [v for _, v in hist.get("losses/approx_kl", [])]
        cf = [v for _, v in hist.get("losses/clipfrac", [])]
        rows.append(dict(
            grupo=cfg["_group"], algoritmo=algo, seed=cfg["seed"], run_id=cfg["_id"],
            passos=cfg["total_timesteps"], num_envs=cfg["num_envs"], num_steps=p["num_steps"],
            lote=cfg["num_envs"] * p["num_steps"],
            learning_rate=p["learning_rate"],
            update_epochs=p.get("update_epochs", 1), num_minibatches=p.get("num_minibatches", 1),
            gamma=p["gamma"],
            retorno_final=round(float(np.mean(late)), 1),
            eval_mean=summ.get("eval/mean_return"), eval_std=summ.get("eval/std_return"),
            approx_kl_mediano=round(float(np.median(kl)), 5) if kl else None,
            clipfrac_mediano=round(float(np.median(cf)), 4) if cf else None,
            num_updates=hist.get("charts/num_updates", [(0, None)])[-1][1],
            sps_mediano=int(np.median([v for _, v in hist.get("charts/SPS", [(0, 0)])])),
            tempo_s=round(float(cfg["_runtime"]), 1),
        ))
    (ROOT / "report_ppo").mkdir(exist_ok=True)
    with open(ROOT / "report_ppo" / "runs_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- Part 4: the tuning path ------------------------------------------
    lbl_tuning = lambda c: {
        "base_seed1": "inicial", "base_seed2": "inicial",
        "t1_envs16": "+ num_envs=16", "t2_gamma999": "+ gamma=0.999",
        "t3_epochs10_mb32": "+ epochs=10, mb=32", "t4_lr1e3": "+ lr=1e-3",
        "c2_lr1e3_ep10_envs16": "final (as três juntas)",
    }.get(c["_group"], c["_group"])
    figure(pick("base_seed1", "t1_envs16", "t2_gamma999", "t3_epochs10_mb32", "t4_lr1e3"),
           ["charts/episodic_return_mean_last100", "losses/approx_kl", "losses/clipfrac"],
           "p4_triagem.png", lbl_tuning, logy=("losses/approx_kl",), ncol=5)

    lbl_final = lambda c: "config inicial" if c["_group"].startswith("base") else "config final"
    figure(pick("base_seed1", "base_seed2", "final_seed1", "final_seed2"),
           ["charts/episodic_return_mean_last100", "losses/clipfrac",
            "losses/explained_variance", "losses/entropy"],
           "p4_inicial_vs_final.png", lbl_final, width=3.3, ncol=3)

    # --- Part 5: PPO vs A2C -----------------------------------------------
    lbl_cmp = lambda c: ("PPO (config final)" if c["algorithm"] == "ppo" else "A2C (melhor em 1M)")
    cmp_runs = pick("final_seed1", "final_seed2", "a2c_s1_default")
    figure(cmp_runs, ["charts/episodic_return_mean_last100", "charts/num_updates"],
           "p5_ppo_vs_a2c.png", lbl_cmp, width=4.0, ncol=3)

    # retorno x tempo de relógio (o eixo "Relative Time" do wandb)
    bench = pick("bench_ppo", "bench_a2c") or cmp_runs
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    for ci, (cfg, hist, _) in enumerate(bench):
        v = hist.get("ret_vs_time")
        if not v:
            continue
        x, y = map(np.asarray, zip(*v))
        ax.plot(x, ema(y, 0.3), color=PALETTE[0 if cfg["algorithm"] == "ppo" else 1],
                linewidth=1.5 if cfg["seed"] == 1 else 1.2,
                linestyle="-" if cfg["seed"] == 1 else "--",
                label=lbl_cmp(cfg) if cfg["seed"] == 1 else None)
    ax.axhline(200, color=INK2, linewidth=1, linestyle=":")
    style(ax, "Retorno × tempo de relógio", xlabel="segundos", k_axis=False)
    fig.legend(loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    fig.savefig(OUT / "p5_tempo.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "p5_tempo.png")


if __name__ == "__main__":
    main()
