"""Figures for the SAC report (Parts 3, 4 and 5), from the wandb project.

    python scripts/plot_report_sac.py   # -> report_sac/figs/*.png + runs_summary.csv

Histories are cached under report_sac/cache/ (delete it to re-download).
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

PROJECT = "jr-cleiver-federal-university-of-goi-s/sac-assignment"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_sac" / "figs"
CACHE = ROOT / "report_sac" / "cache"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

KEYS = ["charts/episodic_return_mean_last100", "losses/alpha", "losses/entropy",
        "charts/target_entropy", "charts/q_minus_return", "losses/qf1_values",
        "losses/qf2_values", "losses/qf1_loss", "charts/SPS", "charts/num_updates",
        "charts/q_estimate_ep_start", "charts/soft_return_ep_start"]
TITLES = {
    "charts/episodic_return_mean_last100": "Retorno médio (últimos 100 ep.)",
    "losses/alpha": "Temperatura α",
    "losses/entropy": "Entropia estimada (−log π)",
    "charts/q_minus_return": "Q(s₀,a₀) − retorno suave realizado",
    "losses/qf1_values": "Q médio do crítico 1 (buffer)",
    "charts/num_updates": "Passos de gradiente acumulados",
}
SMOOTH = {"charts/q_minus_return": 0.85, "losses/entropy": 0.6, "losses/qf1_values": 0.3}


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
        # history(keys=...) keeps only rows holding *every* key: ask each run
        # only for what it logs, and the sparse episode metrics separately
        dense = [k for k in KEYS if k in run.summary and not k.startswith("charts/q_") and "soft_return" not in k]
        sparse = [k for k in ("charts/q_minus_return", "charts/q_estimate_ep_start", "charts/soft_return_ep_start")
                  if k in run.summary]
        for keys in (dense, sparse):
            if not keys:
                continue
            for row in run.history(keys=keys, samples=1500, pandas=False):
                step = row.get("global_step", row.get("_step"))
                for k in keys:
                    if isinstance(row.get(k), (int, float)):
                        hist[k].append((step, row[k]))
        for row in run.history(keys=["charts/episodic_return_mean_last100", "_runtime"], samples=1500, pandas=False):
            if "_runtime" in row and "charts/episodic_return_mean_last100" in row:
                hist["ret_vs_time"].append((row["_runtime"], row["charts/episodic_return_mean_last100"]))
        cfg = dict(run.config)
        cfg.update(_group=run.group or run.name, _id=run.id, _runtime=run.summary.get("_runtime", np.nan))
        item = (cfg, {k: sorted(v) for k, v in hist.items()}, dict(run.summary))
        cached.write_bytes(pickle.dumps(item))
        out.append(item)
        print("fetched", run.id, cfg["_group"], "seed", cfg["seed"])
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


def band(ax, runs, metric, color, label):
    """Mean over seeds with a min–max band, on a common step grid."""
    curves = [series(h, metric) for _, h, _ in runs]
    curves = [(x, y) for x, y in curves if x is not None]
    if not curves:
        return
    grid = np.linspace(max(x.min() for x, _ in curves), min(x.max() for x, _ in curves), 300)
    ys = np.array([np.interp(grid, x, y) for x, y in curves])
    ax.fill_between(grid, ys.min(0), ys.max(0), color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, ys.mean(0), color=color, linewidth=1.8, label=f"{label} (média de {len(curves)} seeds, faixa mín–máx)")


def seed_lines(ax, runs, metric, color, label):
    for cfg, hist, _ in sorted(runs, key=lambda t: t[0]["seed"]):
        x, y = series(hist, metric)
        if x is None:
            continue
        ls = {1: "-", 2: "--", 3: ":"}.get(cfg["seed"], "-")
        ax.plot(x, y, color=color, linewidth=1.4, linestyle=ls, label=label if cfg["seed"] == 1 else None)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    runs = load_runs()
    by = defaultdict(list)
    for item in runs:
        by[item[0]["_group"]].append(item)

    # ---- summary table (every run) ----
    rows = []
    for cfg, hist, summ in sorted(runs, key=lambda t: (t[0]["_group"], t[0]["seed"])):
        algo = cfg["algorithm"]
        p = cfg.get("sac") or cfg.get("ppo_continuous_action") or {}
        ret = hist.get("charts/episodic_return_mean_last100", [(0, np.nan)])
        rows.append(dict(
            grupo=cfg["_group"], algoritmo=algo, seed=cfg["seed"], run_id=cfg["_id"],
            passos=cfg["total_timesteps"], updates_per_step=p.get("updates_per_step"),
            learning_starts=p.get("learning_starts"), autotune=p.get("autotune"), alpha=p.get("alpha"),
            clipped_double_q=p.get("clipped_double_q"),
            eval_mean=summ.get("eval/mean_return"), eval_mean_sampled=summ.get("eval/mean_return_sampled"),
            retorno_last100_fim=ret[-1][1],
            num_updates=(hist.get("charts/num_updates") or [(0, None)])[-1][1],
            sps_mediano=int(np.median([v for _, v in hist.get("charts/SPS", [(0, 0)])])),
            tempo_min=round(float(cfg["_runtime"]) / 60, 1),
        ))
    (ROOT / "report_sac").mkdir(exist_ok=True)
    with open(ROOT / "report_sac" / "runs_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    R = "charts/episodic_return_mean_last100"

    # ---- Part 3: initial vs final, 3 seeds each ----
    fig, ax = plt.subplots(figsize=(6.2, 3.0), constrained_layout=True)
    band(ax, by["p3_base"], R, PALETTE[0], "inicial")
    band(ax, by["p3_utd4"], R, PALETTE[1], "final: updates_per_step=4")
    style(ax, TITLES[R])
    fig.legend(loc="outside lower center", ncol=1, fontsize=8, frameon=False)
    save(fig, "p3_inicial_vs_final.png")

    # ---- Part 4, ablation 1: fixed alpha ----
    final = by["p3_utd4"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), constrained_layout=True)
    for ax, m in zip(axes, ["losses/alpha", "losses/entropy", R]):
        seed_lines(ax, final, m, PALETTE[1], "final da Parte 3 (autotune)")
        seed_lines(ax, by["p4_alpha_fixo"], m, PALETTE[2], "ablação: α fixo = 0,08")
        if m == "losses/entropy":
            ax.axhline(-3, color=INK2, linewidth=1, linestyle=":")
            ax.annotate("target_entropy = −3", (0.02, -3), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 3), fontsize=7, color=INK2)
        if m == "losses/alpha":
            ax.set_yscale("log")
        style(ax, TITLES[m])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], color=INK2, linestyle="--"), plt.Line2D([], [], color=INK2, linestyle=":")],
               l + ["seed 2 (tracejado)", "seed 3 (pontilhado)"], loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    save(fig, "p4_alpha_fixo.png")

    # ---- Part 4, ablation 2: single critic ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), constrained_layout=True)
    for ax, m in zip(axes, ["charts/q_minus_return", "losses/qf1_values", R]):
        seed_lines(ax, final, m, PALETTE[1], "final da Parte 3 (min(Q1,Q2))")
        seed_lines(ax, by["p4_um_critico"], m, PALETTE[4], "ablação: um crítico só (Q1)")
        if m == "charts/q_minus_return":
            ax.axhline(0, color=INK2, linewidth=1, linestyle=":")
        style(ax, TITLES[m])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], color=INK2, linestyle="--"), plt.Line2D([], [], color=INK2, linestyle=":")],
               l + ["seed 2 (tracejado)", "seed 3 (pontilhado)"], loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    save(fig, "p4_um_critico.png")

    # ---- Part 5: SAC vs PPO ----
    sac, ppo = [r for r in final if r[0]["seed"] in (1, 2)], by.get("p5_ppo", [])
    if ppo:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), constrained_layout=True)
        seed_lines(axes[0], sac, R, PALETTE[1], "SAC (final da Parte 3)")
        seed_lines(axes[0], ppo, R, PALETTE[0], "PPO contínuo (sem ajuste)")
        style(axes[0], "Retorno × passos de ambiente")
        # wall clock: the SAC sweep runs shared the CPU (3 at a time, ~2x slower
        # with updates_per_step=4); use the run measured alone when it exists
        sac_clock = by.get("p5_bench_sac") or sac
        for item, color in [(x, PALETTE[1]) for x in sac_clock] + [(x, PALETTE[0]) for x in ppo]:
            cfg, hist, _ = item
            v = hist.get("ret_vs_time")
            if v:
                x, y = map(np.asarray, zip(*v))
                axes[1].plot(x / 60, y, color=color, linewidth=1.4,
                             linestyle="-" if cfg["seed"] == 1 else "--")
        axes[1].set_xscale("symlog", linthresh=1)
        style(axes[1], "Retorno × tempo de relógio (SAC medido sozinho)", xlabel="minutos (escala log)", k_axis=False)
        ticks = [0, 0.3, 1, 2, 5, 10, 18]
        axes[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axes[1].xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
        axes[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        seed_lines(axes[2], sac, "charts/num_updates", PALETTE[1], None)
        seed_lines(axes[2], ppo, "charts/num_updates", PALETTE[0], None)
        axes[2].set_yscale("log")
        style(axes[2], TITLES["charts/num_updates"] + " (log)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h + [plt.Line2D([], [], color=INK2, linestyle="--")], l + ["seed 2 (tracejado)"],
                   loc="outside lower center", ncol=3, fontsize=8, frameon=False)
        save(fig, "p5_sac_vs_ppo.png")


if __name__ == "__main__":
    main()
