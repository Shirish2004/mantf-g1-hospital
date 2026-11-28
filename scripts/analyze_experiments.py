"""Charts and tables of the evaluation from runs/<sim>/cooperative/exp/<group>/<value>/<time>/results.json.

  analyze_experiments.py [--sim isaac] [--out runs/<sim>/cooperative/exp/report]
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 7, "figure.dpi": 130})


def latest(group_dir):
    """The newest finished run of each value folder of a group."""
    out = {}
    for d in sorted(glob.glob(os.path.join(group_dir, "*"))):
        runs = sorted(glob.glob(os.path.join(d, "*", "results.json")))
        if runs:
            out[os.path.basename(d)] = runs[-1]
    return out


def load(p):
    r = json.load(open(p))
    r["folder"] = os.path.dirname(p)
    return r


def table(folder, name):
    p = os.path.join(folder, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def vals(d):
    return [v for v in (d or {}).values() if v is not None]


def save(fig, out, name, title):
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def agents_charts(runs, out):
    ns = sorted(runs, key=lambda k: int(k[1:]))
    R = [runs[k] for k in ns]
    n = [r["n_agents"] for r in R]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for x, r in zip(n, R):
        t = vals(r["time_to_reach_s"])
        ax[0].scatter([x] * len(t), t, s=18, color=SERIES[0], alpha=0.45, lw=0)
    ttc = [r["time_to_complete_s"] for r in R]
    ax[0].plot(n, [np.nan if v is None else v for v in ttc], "-o", color=SERIES[1], label="time to complete (all reached)")
    ax[0].plot(n, [np.nan if r["mean_time_to_reach_s"] is None else r["mean_time_to_reach_s"] for r in R], "-o",
               color=SERIES[0], label="mean time to reach")
    for x, r in zip(n, R):
        ax[0].annotate(r["reached"], (x, 0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=INK2)
    ax[0].set_xlabel("agents n"); ax[0].set_ylabel("sim time (s)"); ax[0].legend(loc="upper left")
    ax[0].set_ylim(bottom=0); ax[0].set_xticks(n)
    parts = ["track_s", "rows_s", "train_s", "plan_s"]
    bottom = np.zeros(len(R))
    for i, k in enumerate(parts):
        v = np.array([r["clock_s"].get(k) or 0 for r in R])
        ax[1].bar(n, v, bottom=bottom, color=SERIES[i], width=0.7, label=k[:-2], edgecolor="white", linewidth=1)
        bottom += v
    ax[1].set_xlabel("agents n"); ax[1].set_ylabel("wall s per model call"); ax[1].legend(ncol=4, loc="upper left")
    ax[1].set_xticks(n)
    save(fig, out, "agents_time.png", "Time to complete vs number of agents (dots: each agent's time to reach) | clock per call")
    fig, ax = plt.subplots(figsize=(6, 4))
    ca = [np.mean(vals(r["collision_avoidance_s"])) if vals(r["collision_avoidance_s"]) else np.nan for r in R]
    tn = [np.mean(vals(r["time_to_next_position_s"])) if vals(r["time_to_next_position_s"]) else np.nan for r in R]
    ax.plot(n, ca, "-o", color=SERIES[0], label="collision avoidance s per agent")
    ax.plot(n, tn, "-o", color=SERIES[2], label="time to next position s")
    ax.set_xlabel("agents n"); ax.set_ylabel("sim time (s)"); ax.set_xticks(n); ax.legend(); ax.set_ylim(bottom=0)
    save(fig, out, "agents_avoid_next.png", "Avoidance and time to next position vs number of agents")
    big = R[-1]
    ids = list(big["time_to_reach_s"])
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(ids)), 3.6))
    t = [big["time_to_reach_s"][k] or 0 for k in ids]
    ax.bar(ids, t, color=[SERIES[0] if big["time_to_reach_s"][k] else "#bbbbbb" for k in ids], width=0.65)
    ax.set_ylabel("time to reach (s)")
    save(fig, out, "agents_per_agent.png", f"Time for each agent to reach its goal, n = {big['n_agents']} (grey: not reached)")


def speed_charts(runs, out):
    ks = sorted(runs, key=lambda k: float(k[1:]))
    R = [runs[k] for k in ks]
    v = [float(k[1:]) for k in ks]
    ratio = [(r["robot_max_v_mps"] or 0.75) / x for r, x in zip(R, v)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(v, [np.nan if r["time_to_complete_s"] is None else r["time_to_complete_s"] for r in R], "-o", color=SERIES[1],
               label="time to complete")
    ax[0].plot(v, [np.nan if r["mean_time_to_reach_s"] is None else r["mean_time_to_reach_s"] for r in R], "-o",
               color=SERIES[0], label="mean time to reach")
    ax[0].set_ylabel("sim time (s)"); ax[0].legend(); ax[0].set_ylim(bottom=0)
    ax[1].plot(v, [np.mean(vals(r["collision_avoidance_s"])) for r in R], "-o", color=SERIES[0], label="collision avoidance")
    ax[1].plot(v, [np.mean(vals(r["in_tube_s"])) for r in R], "-o", color=SERIES[6], label="inside a person's swept tube")
    ax[1].plot(v, [np.mean(vals(r["time_to_next_position_s"])) if vals(r["time_to_next_position_s"]) else np.nan
                   for r in R], "-o", color=SERIES[2], label="time to next position")
    ax[1].set_ylabel("s per agent"); ax[1].legend(); ax[1].set_ylim(bottom=0)
    ax[2].plot(v, [r["min_people_m"] for r in R], "-o", color=SERIES[0])
    ax[2].axhline(0.55, color=SERIES[7], ls=":", lw=1.2)
    ax[2].text(v[-1], 0.57, "close call 0.55 m", ha="right", fontsize=8, color=INK2)
    ax[2].set_ylabel("closest robot-person distance (m)"); ax[2].set_ylim(bottom=0)
    for a in ax:
        a.set_xticks(v)
        a.set_xticklabels([f"{x:g}\n({q:.2g}x)" for x, q in zip(v, ratio)])
        a.set_xlabel("person speed m/s (robot / obstacle speed)")
    save(fig, out, "speed.png", "Robot speed vs obstacle speed (robot max 0.75 m/s, 4 agents, 5 people)")
    return ratio


def spread_charts(runs, out):
    ks = sorted(runs, key=lambda k: float(k[1:]))
    R = [runs[k] for k in ks]
    r_ = [float(k[1:]) for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(r_, [np.nan if r["time_to_complete_s"] is None else r["time_to_complete_s"] for r in R], "-o", color=SERIES[1],
               label="time to complete")
    ax[0].plot(r_, [np.nan if r["mean_time_to_reach_s"] is None else r["mean_time_to_reach_s"] for r in R], "-o",
               color=SERIES[0], label="mean time to reach")
    ax[0].set_ylabel("sim time (s)"); ax[0].legend(); ax[0].set_ylim(bottom=0)
    ax[1].plot(r_, [np.mean(vals(r["collision_avoidance_s"])) for r in R], "-o", color=SERIES[0], label="collision avoidance s")
    ax[1].plot(r_, [r["min_robot_m"] for r in R], "-o", color=SERIES[2], label="closest robot-robot m")
    ax[1].legend(); ax[1].set_ylim(bottom=0)
    for a in ax:
        a.set_xlabel("teammates spawned within (m) of the G1")
    save(fig, out, "spread.png", "Spawning agents nearer (4 agents, seed 0)")


def explore_charts(runs, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, k in enumerate(sorted(runs)):
        cov = table(runs[k]["folder"], "coverage_log.csv")
        if not cov:
            continue
        t = np.array([float(c["t"]) for c in cov]); t -= t[0]
        ax.plot(t, [float(c["known_m2"]) for c in cov], color=SERIES[i], label=f"{k} (90% at {runs[k]['exploration_time_s']} s)")
    ax.set_xlabel("sim time (s)"); ax.set_ylabel("mapped area (m²)"); ax.legend(); ax.set_ylim(bottom=0)
    save(fig, out, "explore.png", "Exploration with dynamic objects: shared map growth, SE(2) tracker on vs off (8 people)")


def tracking_charts(all_runs, out):
    rows = []
    for r in all_runs:
        rows += table(r["folder"], "tracking_log.csv")
    if not rows:
        return None
    pe = np.array([float(q["pos_err_m"]) for q in rows])
    ok = [q for q in rows if q["vel_err_mps"] not in ("", "nan")]
    ve = np.array([float(q["vel_err_mps"]) for q in ok])
    sg = np.array([float(q["speed_gt"]) for q in ok])
    se = np.array([float(q["speed_est"]) for q in ok])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].hist(pe, bins=40, color=SERIES[0], edgecolor="white", linewidth=0.5)
    ax[0].axvline(np.median(pe), color=INK, lw=1, ls="--")
    ax[0].text(np.median(pe), ax[0].get_ylim()[1] * 0.92, f" median {np.median(pe):.3f} m", fontsize=8, color=INK)
    ax[0].set_xlabel("centroid error (m)"); ax[0].set_ylabel("detections")
    ax[1].hist(ve, bins=40, color=SERIES[2], edgecolor="white", linewidth=0.5)
    ax[1].axvline(np.median(ve), color=INK, lw=1, ls="--")
    ax[1].text(np.median(ve), ax[1].get_ylim()[1] * 0.92, f" median {np.median(ve):.3f} m/s", fontsize=8, color=INK)
    ax[1].set_xlabel("twist error |v_est - v_true| (m/s)")
    ax[2].scatter(sg, se, s=8, color=SERIES[0], alpha=0.35, lw=0)
    m = max(sg.max(), se.max()) * 1.05
    ax[2].plot([0, m], [0, m], color=INK2, lw=1, ls=":")
    ax[2].set_xlabel("true person speed (m/s)"); ax[2].set_ylabel("tracked speed (m/s)")
    save(fig, out, "se2_tracking.png", f"SE(2) tracking vs ground truth ({len(rows)} detections, {len(ok)} with a twist; all runs)")
    return {"detections": len(rows), "with_twist": len(ok), "pos_err_median_m": float(np.median(pe)), "pos_err_mean_m": float(pe.mean()),
            "vel_err_median_mps": float(np.median(ve)), "vel_err_mean_mps": float(ve.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="isaac")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None, help="default runs/<sim>/cooperative/exp")
    a = ap.parse_args()
    base = a.base or os.path.join(ROOT, "runs", a.sim, "cooperative", "exp")
    out = a.out or os.path.join(base, "report")
    os.makedirs(out, exist_ok=True)
    groups = {g: {k: load(p) for k, p in latest(os.path.join(base, g)).items()} for g in ("agents", "speed", "spread", "explore")}
    print("runs:", {g: sorted(v) for g, v in groups.items()})
    if groups["agents"]:
        agents_charts(groups["agents"], out)
    if groups["speed"]:
        speed_charts(groups["speed"], out)
    if groups["spread"]:
        spread_charts(groups["spread"], out)
    if groups["explore"]:
        explore_charts(groups["explore"], out)
    all_runs = [r for g in groups.values() for r in g.values()]
    trk = tracking_charts(all_runs, out)
    keys = ["n_agents", "people", "people_speed_mps", "spread_m", "detect", "reached", "time_to_complete_s",
            "mean_time_to_reach_s", "exploration_time_s", "known_m2", "wall_contacts", "min_people_m", "min_robot_m", "calls"]
    lines = ["| group | value | " + " | ".join(keys) + " | avoid s/agent | next pos s | tracking pos err m | twist err m/s |",
             "|" + "---|" * (len(keys) + 6)]
    rows = []
    for g, rs in groups.items():
        for k, r in sorted(rs.items()):
            av = np.mean(vals(r["collision_avoidance_s"])) if vals(r["collision_avoidance_s"]) else None
            tn = np.mean(vals(r["time_to_next_position_s"])) if vals(r["time_to_next_position_s"]) else None
            st = r["se2_tracking"]
            row = [g, k] + [r.get(q) for q in keys] + [av, tn, st["pos_err_mean_m"], st["vel_err_mean_mps"]]
            rows.append(row)
            lines.append("| " + " | ".join("-" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)) for v in row) + " |")
    with open(os.path.join(out, "summary.csv"), "w") as f:
        f.write(",".join(["group", "value"] + keys + ["avoid_s", "next_pos_s", "track_pos_err_m", "track_vel_err_mps"]) + "\n")
        for row in rows:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")
    md = "# Evaluation summary\n\n" + "\n".join(lines) + "\n"
    if trk:
        md += "\n## SE(2) tracking, all runs\n\n```\n" + json.dumps(trk, indent=2) + "\n```\n"
    open(os.path.join(out, "summary.md"), "w").write(md)
    json.dump({g: {k: r["folder"] for k, r in rs.items()} for g, rs in groups.items()}, open(os.path.join(out, "runs.json"), "w"),
              indent=2)
    print(md)


if __name__ == "__main__":
    main()
