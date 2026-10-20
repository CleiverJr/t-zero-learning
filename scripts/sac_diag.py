"""One line per config (grouped over seeds) for the SAC assignment project.

    python scripts/sac_diag.py [substring-do-grupo ...]

Prints, per group, the end-of-training eval/mean_return and
episodic_return_mean_last100 as mean (min-max) over seeds — the numbers the
Part 3 table asks for — plus the diagnostics used to pick the next change.
"""
import sys
from collections import defaultdict

import numpy as np
import wandb

PROJECT = "jr-cleiver-federal-university-of-goi-s/sac-assignment"
KEYS = ["charts/episodic_return_mean_last100", "losses/alpha", "losses/entropy",
        "charts/target_entropy", "charts/q_minus_return", "losses/qf1_values",
        "losses/qf1_loss", "charts/SPS", "charts/num_updates"]


def first_step_below(steps, vals, thr):
    for s, v in zip(steps, vals):
        if v <= thr:
            return s
    return None


def main(filters):
    groups = defaultdict(list)
    for run in wandb.Api().runs(PROJECT):
        g = run.group or run.name
        if run.state != "finished" or (filters and not any(f in g for f in filters)):
            continue
        keys = [k for k in KEYS if k in run.summary]
        h = defaultdict(list)
        for row in run.history(keys=keys, samples=1000, pandas=False):
            for k in keys:
                if isinstance(row.get(k), (int, float)):
                    h[k].append((row.get("global_step", row.get("_step")), row[k]))
        late = lambda k: np.array([v for _, v in h[k]][int(len(h[k]) * 0.8):] or [np.nan])
        a_steps, a_vals = zip(*h["losses/alpha"]) if h["losses/alpha"] else ((), ())
        groups[g].append(dict(
            seed=run.config["seed"], id=run.id,
            eval=run.summary.get("eval/mean_return", np.nan),
            eval_s=run.summary.get("eval/mean_return_sampled", np.nan),
            ret=h["charts/episodic_return_mean_last100"][-1][1] if h["charts/episodic_return_mean_last100"] else np.nan,
            alpha_end=a_vals[-1] if a_vals else np.nan,
            alpha_min=min(a_vals) if a_vals else np.nan,
            alpha_lt_005=first_step_below(a_steps, a_vals, 0.05),
            ent=np.mean(late("losses/entropy")), tgt=run.summary.get("charts/target_entropy", np.nan),
            qmr=np.nanmedian(late("charts/q_minus_return")),
            qv=np.mean(late("losses/qf1_values")),
            sps=int(np.median([v for _, v in h["charts/SPS"]] or [0])),
            upd=h["charts/num_updates"][-1][1] if h["charts/num_updates"] else np.nan,
            runtime=run.summary.get("_runtime", np.nan),
        ))
    for g in sorted(groups):
        rs = sorted(groups[g], key=lambda d: d["seed"])
        f = lambda k: np.array([d[k] for d in rs], dtype=float)
        mm = lambda k: f"{np.nanmean(f(k)):7.1f} ({np.nanmin(f(k)):.0f}–{np.nanmax(f(k)):.0f})"
        print(f"{g:<22} n={len(rs)} | eval {mm('eval')} | r_last100 {mm('ret')} | "
              f"alpha fim {np.nanmean(f('alpha_end')):.3f} | ent {np.nanmean(f('ent')):.2f} (alvo {rs[0]['tgt']:.0f}) | "
              f"Q-ret {np.nanmean(f('qmr')):6.1f} | Q {np.nanmean(f('qv')):6.1f} | "
              f"upd {np.nanmean(f('upd')):.0f} | {np.nanmean(f('runtime'))/60:.1f} min | "
              + " ".join(f"s{d['seed']}:{d['id']}" for d in rs))
        for d in rs:
            print(f"    s{d['seed']} eval={d['eval']:7.1f} eval_amostrado={d['eval_s']:7.1f} r_last100={d['ret']:7.1f} "
                  f"alpha<0.05 em {d['alpha_lt_005']} alpha_min={d['alpha_min']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
