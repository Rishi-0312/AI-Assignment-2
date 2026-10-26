"""Make every figure of the README and the report from results/ and runs/.

    python scripts/make_figures.py

Run scripts/evaluate.py first.  Figures are written to results/figures/ as
PDF (vector, used by the LaTeX report) and PNG (used by the README).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wakesteer.config import config_from_dict  # noqa: E402
from wakesteer.farm import make_floris_model  # noqa: E402
from wakesteer.ppo import load_policy  # noqa: E402

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
REPORT_FIGS = ROOT / "report" / "figures"

# One fixed colour per controller (categorical slots 1-3; the greedy reference is neutral).
C_PPO, C_LUT, C_PLAIN, C_GREEDY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
C_LUT_LIGHT = "#f5a986"  # tint of the LUT orange for the filtered-LUT variant (also dashed / open markers)
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
WAKE_CMAP = LinearSegmentedColormap.from_list(
    "wake", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#fcfcfb"]
)
COL_W = 3.45  # inches, one column of the two-column report


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    REPORT_FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(REPORT_FIGS / f"{name}.pdf")  # the report folder stays self-contained (e.g. for Overleaf)
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {name}")


def ppo_runs() -> list[str]:
    """The main-method runs, as evaluated by scripts/evaluate.py."""
    return json.loads((RESULTS / "evaluation.json").read_text())["policies"]


# --------------------------------------------------------------------------- figures
def fig_flow(direction=275.0, speed=8.0) -> None:
    """Hub-height flow for greedy control vs. the PPO agent's settled yaw offsets."""
    from floris.layout_visualization import plot_turbine_rotors

    sweep = np.load(RESULTS / "steady_state.npz")
    run = ppo_runs()[0]
    i = int(np.argmin(np.abs(sweep["directions"] - direction)))
    _, payload = load_policy(ROOT / "runs" / run / "policy.pt")
    cfg = config_from_dict(payload["config"])
    fm = make_floris_model(cfg.farm)
    panels = [("Greedy", np.zeros(fm.n_turbines)), ("PPO agent", sweep[f"{run}_offsets"][i])]
    xlim, ylim = (-250, 2150), (-330, 1590)

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.38), sharey=True, gridspec_kw={"wspace": 0.08})
    powers = []
    for ax, (title, yaw) in zip(axes, panels):
        fm.set(wind_directions=[direction], wind_speeds=[speed], turbulence_intensities=[cfg.farm.turbulence_intensity],
               yaw_angles=yaw[None, :])
        fm.run()
        powers.append(fm.get_farm_power()[0] / 1e6)
        # FLORIS samples the plane on a wind-aligned grid (given in map coordinates), hence tripcolor.
        plane = fm.calculate_horizontal_plane(height=90.0, x_resolution=340, y_resolution=280,
                                              x_bounds=(xlim[0] - 250, xlim[1] + 250), y_bounds=(ylim[0] - 250, ylim[1] + 250))
        df = plane.df
        mesh = ax.tripcolor(df.x1, df.x2, df.u, cmap=WAKE_CMAP, vmin=3.0, vmax=speed, shading="gouraud", rasterized=True)
        plot_turbine_rotors(fm, ax=ax, color=INK, wd=np.array([direction]), yaw_angles=yaw[None, :])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([0, 1000, 2000])
        ax.set_xlabel("x [m]")
        ax.set_title(title, color=INK, pad=3)
    axes[0].set_ylabel("y [m]")
    gain = 100 * (powers[1] / powers[0] - 1)
    for ax, p, extra in zip(axes, powers, ["", f" ({gain:+.1f}%)"]):
        ax.text(0.03, 0.97, f"{p:.2f} MW{extra}", transform=ax.transAxes, va="top", fontsize=6.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85))
    cb = fig.colorbar(mesh, ax=axes, orientation="vertical", fraction=0.025, pad=0.02, aspect=18)
    cb.set_label("wind speed [m/s]", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "fig_flow")


def training_runs() -> list[dict]:
    """Every finished run in runs/ with its settings (read from its own config.yaml)."""
    import yaml

    out = []
    for cfg_path in sorted(ROOT.glob("runs/*/config.yaml")):
        run = cfg_path.parent
        if run.name.startswith("_") or not (run / "progress.csv").exists():
            continue
        c = yaml.safe_load(cfg_path.read_text())
        df = pd.read_csv(run / "progress.csv").dropna(subset=["val_gain_pct"])
        out.append({
            "name": run.name, "gamma": c["ppo"]["gamma"], "symmetry": c["ppo"].get("symmetry", False),
            "default": c["env"].get("observe_powers", True) and c["env"]["yaw_cost"] == 0.005,
            "steps": df["env_steps"].to_numpy() / 1e6, "gain": df["val_gain_pct"].to_numpy(),
        })
    return out


def _mean_curve(runs):
    n = min(len(r["steps"]) for r in runs)
    return runs[0]["steps"][:n], np.mean([r["gain"][:n] for r in runs], axis=0)


def fig_training() -> None:
    """Validation energy gain during training: (a) symmetry ablation, (b) discount-factor study."""
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    lut = ev["validation_reference"]["LUT"]
    runs = [r for r in training_runs() if r["default"]]
    main_gamma = yaml_gamma(ROOT / "runs" / ppo_runs()[0] / "config.yaml")
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.22), sharey=True, gridspec_kw={"wspace": 0.08})

    sym = [r for r in runs if r["symmetry"] and r["gamma"] == 0.95]
    plain = [r for r in runs if not r["symmetry"] and r["gamma"] == 0.95]
    for group, col, label in [(sym, C_PPO, "C$_4$-symmetric"), (plain, C_PLAIN, "plain MLP")]:
        if group:
            for r in group if len(group) > 1 else []:
                a.plot(r["steps"], r["gain"], color=col, lw=0.6, alpha=0.4)
            a.plot(*_mean_curve(group), color=col, lw=1.4, label=label)
    a.set_title("(a) symmetry, $\\gamma$ = 0.95", loc="left", color=INK2, pad=2)

    ramp = {0.0: "#86b6ef", 0.5: "#3987e5", 0.8: "#1c5cab", 0.95: "#0d366b"}  # ordinal: darker = longer horizon
    for g in sorted({r["gamma"] for r in runs if r["symmetry"]}):
        group = [r for r in runs if r["symmetry"] and r["gamma"] == g]
        tag = " (ours)" if g == main_gamma else ""
        b.plot(*_mean_curve(group), color=ramp.get(g, C_PPO), lw=1.4, label=f"$\\gamma$ = {g:g}{tag}")
    b.set_title("(b) discount factor", loc="left", color=INK2, pad=2)

    for ax in (a, b):
        ax.axhline(lut, color=C_LUT, lw=1.1, ls=(0, (4, 2)), label="LUT")
        ax.axhline(0.0, color=C_GREEDY, lw=0.9)
        ax.set_xlabel("env. steps [M]")
        ax.set_xlim(0, 5.0)
        ax.legend(loc="lower right", fontsize=5.8, handlelength=1.4, borderaxespad=0.2)
    a.set_ylabel("val. energy gain [%]")
    save(fig, "fig_training")


def yaml_gamma(path: Path) -> float:
    import yaml

    return yaml.safe_load(Path(path).read_text())["ppo"]["gamma"]


def fig_sweep(speed_label="8 m/s", turbine=0) -> None:
    """Steady-state gain over greedy and yaw schedule of one turbine vs. wind direction."""
    sw = np.load(RESULTS / "steady_state.npz")
    d = sw["directions"]
    runs = [r for r in ppo_runs() if f"{r}_power" in sw]
    opt_gain = 100 * (sw["lut_optimal"] / sw["greedy"] - 1)
    ppo_gain = np.stack([100 * (sw[f"{r}_power"] / sw["greedy"] - 1) for r in runs])

    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 1.7), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    a.plot(d, opt_gain, color=C_LUT, lw=1.1, label="FLORIS-SR optimum (model-based)")
    a.plot(d, ppo_gain.mean(0), color=C_PPO, lw=1.1, label=f"PPO (mean of {len(runs)} seeds)")
    a.set_ylabel("gain [%]")
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.1)
    a.set_ylim(bottom=min(-1.0, float(ppo_gain.mean(0).min()) - 0.5), top=float(opt_gain.max()) * 1.08)

    b.plot(d, sw["lut_offsets"][:, turbine], color=C_LUT, lw=1.1)
    b.plot(d, sw[f"{runs[0]}_offsets"][:, turbine], color=C_PPO, lw=1.1)
    b.set_ylabel(f"T{turbine + 1} yaw [deg]")
    b.set_xlabel(f"wind direction [deg]  (U = {speed_label}, constant)")
    b.set_xticks(np.arange(0, 361, 45))
    b.set_xlim(0, 359)
    save(fig, "fig_sweep")


def fig_robustness() -> None:
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    if "noise_robustness" not in ev:
        return
    runs = ev["policies"]
    nz = ev["noise_robustness"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.15), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.38})
    sig = np.array(nz["sigma_deg"])
    ppo = np.array([nz[r]["gain"] for r in runs])
    a.plot(sig, nz["LUT"]["gain"], color=C_LUT, marker="s", ms=4, label="LUT")
    if "LUT_filtered" in nz:
        a.plot(sig, nz["LUT_filtered"]["gain"], color=C_LUT, ls=(0, (3, 1.5)), marker="^", ms=4.5, mfc="white",
               label="LUT + filter")
    a.errorbar(sig, ppo.mean(0), yerr=ppo.std(0, ddof=1) if len(runs) > 1 else None, color=C_PPO, marker="o", ms=4,
               capsize=2, lw=1.4, label="PPO (ours)")
    a.axhline(0, color=C_GREEDY, lw=1.0)
    a.set_xlabel("wind-direction noise $\\sigma$ [deg]")
    a.set_ylabel("energy gain [%]")
    a.set_xticks(sig)

    mm = ev["model_mismatch"]
    labels = list(mm)
    short = {"GCH (training model)": "GCH\n(training)", "Cumulative-curl": "cumulative\ncurl", "GCH, TI = 10%": "GCH,\nTI = 10%"}
    x = np.arange(len(labels))
    ppo_vals = np.array([[mm[k][r] for r in runs] for k in labels])
    bars = [("LUT", C_LUT)] + ([("LUT_filtered", C_LUT_LIGHT)] if "LUT_filtered" in mm[labels[0]] else [])
    w = 0.8 / (len(bars) + 1)
    for j, (key, col) in enumerate(bars):
        b.bar(x + (j - len(bars) / 2) * (w + 0.02), [mm[k][key] for k in labels], width=w, color=col)
    b.bar(x + (len(bars) / 2) * (w + 0.02), ppo_vals.mean(1), width=w, color=C_PPO,
          yerr=ppo_vals.std(1, ddof=1) if len(runs) > 1 else None, error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
    b.axhline(0, color=AXIS, lw=0.6)
    b.set_xticks(x)
    b.set_xticklabels([short.get(lbl, lbl) for lbl in labels], fontsize=6.3)
    b.set_ylabel("energy gain [%]")
    b.grid(axis="x", visible=False)
    handles, names = a.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=len(names))
    save(fig, "fig_robustness")


def fig_direction() -> None:
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    bd = ev["by_direction"]
    runs = ev["policies"]
    c = np.array(bd["center_deg"])
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    ax.plot(c, bd["LUT"], color=C_LUT, marker="s", ms=3, label="LUT")
    ppo = np.array([bd[r] for r in runs], dtype=float)
    ax.plot(c, ppo.mean(0), color=C_PPO, marker="o", ms=3, label="PPO")
    ax.axhline(0, color=C_GREEDY, lw=1.0)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlim(0, 360)
    ax.set_xlabel("mean wind direction of episode [deg]")
    ax.set_ylabel("energy gain [%]")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "fig_direction")


def fig_hysteresis(turbine=0) -> None:
    """T1 yaw offset while the wind veers slowly (0.5 deg/min, noise-free) through the 270 deg alignment."""
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    if "hysteresis" not in ev:
        return
    hy = ev["hysteresis"]
    run = ev["policies"][0]
    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    for name, col, lbl in [("LUT", C_LUT, "LUT"), (run, C_PPO, "PPO")]:
        for leg, ls in [("forward", "-"), ("backward", (0, (3, 1.5)))]:
            d = hy[name][leg]
            ax.plot(d["wind_direction"], np.array(d["yaw_offsets"])[:, turbine], color=col, ls=ls,
                    label=f"{lbl}, wind veering {'right' if leg == 'forward' else 'left'}")
    ax.axvline(270, color=AXIS, lw=0.6)
    ax.set_xlabel("wind direction [deg] (slow veer, 0.5 deg/min, U = 8 m/s)")
    ax.set_ylabel(f"yaw offset T{turbine + 1} [deg]")
    ax.set_xlim(250, 290)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    save(fig, "fig_hysteresis")


def fig_episode() -> None:
    """One test episode in which the wind meanders across the westerly row alignment (270 deg)."""
    ex = np.load(RESULTS / "example_episodes.npz")
    wd = ex["Greedy_wind_direction"]  # (T, n)
    near = np.abs(((wd - 270.0 + 180) % 360) - 180)
    crossings = np.sum(np.diff(np.sign(((wd - 270.0 + 180) % 360) - 180), axis=0) != 0, axis=0)
    score = np.where(near.max(0) < 20, crossings, -1)
    k = int(np.argmax(score))
    t = np.arange(wd.shape[0]) + 1
    g = ex["Greedy_farm_power"][:, k]
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(COL_W + 0.9, 3.0), sharex=True, gridspec_kw={"hspace": 0.18})
    a.plot(t, ex["Greedy_measured_direction"][:, k], color=C_GREEDY, lw=0.8, marker="o", ms=1.5, label="measured")
    a.plot(t, wd[:, k], color=INK, lw=1.2, label="true")
    a.axhline(270, color=AXIS, lw=0.6)
    a.set_ylabel("wind dir. [deg]")
    for name, col, marker in [("LUT", C_LUT, "s"), ("PPO", C_PPO, "o")]:
        b.plot(t, ex[f"{name}_yaw_offsets"][:, k, 0], color=col, marker=marker, ms=1.8, label=name)
        c.plot(t, 100 * (ex[f"{name}_farm_power"][:, k] / g - 1), color=col, marker=marker, ms=1.8, label=name)
    b.set_ylabel("yaw T1 [deg]")
    c.axhline(0, color=C_GREEDY, lw=1.0, label="Greedy")
    c.set_ylabel("gain vs greedy [%]")
    c.set_xlabel("time [min]")
    for ax in (a, b, c):
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    save(fig, "fig_episode")


def main() -> None:
    style()
    fig_training()
    fig_sweep()
    fig_robustness()
    fig_direction()
    fig_episode()
    fig_hysteresis()
    fig_flow()


if __name__ == "__main__":
    main()
