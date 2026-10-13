"""Prints one diagnostic line per finished run of the PPO assignment project.

    python scripts/ppo_diag.py [substring-do-grupo ...]

The columns are the ones the handout asks you to read together: return,
eval, approx_kl / clipfrac (how far the policy moves per update and whether
clipping is active), entropy, explained_variance, and the update budget.
"""
import sys

import numpy as np
import wandb

PROJECT = "jr-cleiver-federal-university-of-goi-s/ppo-assignment"
KEYS = ["charts/episodic_return_mean_last100", "losses/approx_kl", "losses/clipfrac",
        "losses/entropy", "losses/explained_variance", "losses/value_loss",
        "charts/SPS", "charts/num_updates", "losses/grad_norm"]


def main(filters):
    rows = []
    for run in wandb.Api().runs(PROJECT):
        group = run.group or run.name
        if run.state != "finished" or (filters and not any(f in group for f in filters)):
            continue
        h = {k: [] for k in KEYS}
        for row in run.history(keys=KEYS, samples=800, pandas=False):
            for k in KEYS:
                if isinstance(row.get(k), (int, float)):
                    h[k].append(row[k])
        if not h["charts/episodic_return_mean_last100"]:
            continue
        late = lambda k: np.array(h[k])[int(len(h[k]) * 0.7):] if h[k] else np.array([np.nan])
        cfg, p = run.config, run.config["ppo"]
        rows.append((group, run.config["seed"], dict(
            ret=np.mean(late("charts/episodic_return_mean_last100")),
            ev=np.mean(late("losses/explained_variance")),
            ent=np.mean(late("losses/entropy")),
            eval=run.summary.get("eval/mean_return", float("nan")),
            kl=np.median(h["losses/approx_kl"]), cf=np.median(h["losses/clipfrac"]),
            cfmax=np.max(h["losses/clipfrac"]), sps=int(np.median(h["charts/SPS"])),
            upd=h["charts/num_updates"][-1], steps=cfg["total_timesteps"],
            envs=cfg["num_envs"], batch=cfg["num_envs"] * p["num_steps"],
            id=run.id)))
    for group, seed, d in sorted(rows):
        print(f"{group:<26} s{seed} | ret={d['ret']:7.1f} eval={d['eval']:7.1f} | "
              f"kl={d['kl']:.4f} clipfrac={d['cf']:.3f} (max {d['cfmax']:.2f}) | "
              f"ent={d['ent']:.2f} ev={d['ev']:5.2f} | lote={d['batch']:>5} upd={d['upd']:>6.0f} "
              f"sps={d['sps']:>6} passos={d['steps']/1e6:g}M | {d['id']}")


if __name__ == "__main__":
    main(sys.argv[1:])
