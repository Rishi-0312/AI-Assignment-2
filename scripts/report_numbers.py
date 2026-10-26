"""Export the numbers quoted in the report from results/evaluation.json.

    python scripts/report_numbers.py

Writes report/numbers.tex (LaTeX macros) and report/table_main.tex (results
table body), so every figure quoted in the PDF comes straight from the
evaluation output.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
sys.path.insert(0, str(ROOT))

from wakesteer.evaluation import switch_direction  # noqa: E402

ABLATIONS = [  # run-name prefix (all seeds are averaged), table label, macro stem
    ("gamma_0.95", r"\ -- long horizon ($\gamma$\,=\,0.95)", "LongHorizon"),
    ("ablation_no_symmetry", r"\ -- plain MLP ($\gamma$\,=\,0.95)", "NoSymmetry"),
    ("ablation_no_powers", r"\ -- no power inputs", "NoPowers"),
    ("ablation_no_yaw_cost", r"\ -- no wear penalty", "NoYawCost"),
]
GAMMA_TAGS = {0.0: "Zero", 0.5: "Half", 0.8: "PointEight", 0.95: "NinetyFive"}


def best_validation_rewards() -> dict[float, list[float]]:
    """Best validation reward of every symmetric run with default settings, grouped by its discount factor."""
    out: dict[float, list[float]] = {}
    for cfg_path in sorted(ROOT.glob("runs/*/config.yaml")):
        if cfg_path.parent.name.startswith("_") or not (cfg_path.parent / "progress.csv").exists():
            continue
        c = yaml.safe_load(cfg_path.read_text())
        if not c["ppo"].get("symmetry") or not c["env"].get("observe_powers", True) or c["env"]["yaw_cost"] != 0.005:
            continue
        df = pd.read_csv(cfg_path.parent / "progress.csv")
        out.setdefault(float(c["ppo"]["gamma"]), []).append(float(df["val_reward"].max()))
    return out


def group_mean(main_: dict, prefix: str) -> dict | None:
    members = [k for k in main_ if k.startswith(prefix + "_seed")]
    if not members:
        return None
    return {key: float(np.mean([main_[m][key] for m in members])) for key in main_[members[0]]}


def fmt(x: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "\\textminus{}") if s.startswith("-") else s


def main() -> None:
    ev = json.loads((ROOT / "results" / "evaluation.json").read_text())
    main_ = ev["main"]
    runs = ev["policies"]
    agg = ev["ppo_aggregate"]

    def ppo(key):
        vals = np.array([main_[r][key] for r in runs])
        return vals.mean(), (vals.std(ddof=1) if len(vals) > 1 else 0.0)

    macros = {
        "NumEpisodes": f"{ev['episodes']}",
        "NumSeeds": f"{len(runs)}",
        "GainLUT": fmt(main_["LUT"]["energy_gain_pct"]),
        "GainPPO": fmt(agg["energy_gain_pct_mean"]),
        "GainPPOstd": fmt(agg["energy_gain_pct_std"], 3),
        "GainPPOoverLUT": fmt(100 * (agg["energy_gain_pct_mean"] / main_["LUT"]["energy_gain_pct"] - 1), 0),
        "TravelGreedy": fmt(main_["Greedy"]["yaw_travel_deg_per_turbine_h"], 0),
        "TravelLUT": fmt(main_["LUT"]["yaw_travel_deg_per_turbine_h"], 0),
        "TravelPPO": fmt(agg["yaw_travel_mean"], 0),
        "TravelCutVsLUT": fmt(100 * (1 - agg["yaw_travel_mean"] / main_["LUT"]["yaw_travel_deg_per_turbine_h"]), 0),
        "PowerGreedy": fmt(main_["Greedy"]["mean_power_MW"]),
        "GainRandomAbs": fmt(abs(main_["Random"]["energy_gain_pct"])),
        "SwitchLUT": fmt(main_["LUT"]["side_switches_per_turbine_h"], 1),
        "GainLUTf": fmt(main_["LUT_filtered"]["energy_gain_pct"]),
        "TravelLUTf": fmt(main_["LUT_filtered"]["yaw_travel_deg_per_turbine_h"], 0),
        "RewardLUTf": fmt(main_["LUT_filtered"]["mean_reward"]),
        "RewardLUT": fmt(main_["LUT"]["mean_reward"]),
        "RewardPPO": fmt(agg["mean_reward_mean"]),
        "AlphaLUTf": f"{ev['filtered_lut_tuning']['alpha']:g}",
        "SwitchPPO": fmt(np.mean([main_[r]["side_switches_per_turbine_h"] for r in runs]), 1),
        "BetterEpisodesPPO": fmt(np.mean([main_[r]["episodes_better_pct"] for r in runs]), 0),
        "SSoptimal": fmt(ev["steady_state_8ms"]["LUT_optimal"]["mean_gain_pct"]),
        "SSppo": fmt(np.mean([ev["steady_state_8ms"][r]["mean_gain_pct"] for r in runs])),
        "SSfraction": fmt(100 * np.mean([ev["steady_state_8ms"][r]["fraction_of_optimal_gain"] for r in runs]), 0),
        "DecisionPPO": fmt(ev["decision_time_ms"]["ppo_policy"], 2),
        "DecisionSR": fmt(ev["decision_time_ms"]["serial_refine_one_condition"], 0),
        "ValLUT": fmt(ev["validation_reference"]["LUT"]),
    }
    for base, tag in (("LUT", "LUT"), ("LUT_filtered", "LUTf")):
        p = ev["paired_vs_baselines"][base]
        macros[f"Paired{tag}Diff"] = fmt(p["mean_diff_pct_points"])
        macros[f"Paired{tag}SEM"] = fmt(p["sem_pct_points"], 3)
        macros[f"Paired{tag}Better"] = fmt(p["episodes_ppo_better_pct"], 0)
    if "hysteresis" in ev:  # slow veer through 270 deg: where does T1 change deflection side?
        def band(name):
            h = ev["hysteresis"][name]
            fwd = switch_direction(h["forward"]["wind_direction"], np.array(h["forward"]["yaw_offsets"])[:, 0])
            bwd = switch_direction(h["backward"]["wind_direction"], np.array(h["backward"]["yaw_offsets"])[:, 0])
            return fwd, bwd

        lut_f, lut_b = band("LUT")
        ppo_fb = [band(r) for r in runs]
        macros["HystLUT"] = fmt(lut_f - lut_b, 1)
        macros["HystPPO"] = fmt(np.mean([f - b for f, b in ppo_fb]), 1)
        macros["HystPPOfwd"] = fmt(np.mean([f for f, _ in ppo_fb]), 1)
        macros["HystPPObwd"] = fmt(np.mean([b for _, b in ppo_fb]), 1)
    if "noise_robustness" in ev:
        nz = ev["noise_robustness"]
        for i, s in enumerate(nz["sigma_deg"]):
            tag = ["Zero", "Two", "Four", "Six"][i] if i < 4 else str(i)
            macros[f"NoiseLUT{tag}"] = fmt(nz["LUT"]["gain"][i])
            macros[f"NoiseLUTf{tag}"] = fmt(nz["LUT_filtered"]["gain"][i])
            macros[f"NoisePPO{tag}"] = fmt(np.mean([nz[r]["gain"][i] for r in runs]))
        for label, key in [("GCH (training model)", "GCH"), ("Cumulative-curl", "CC"), ("GCH, TI = 10%", "TI")]:
            mm = ev["model_mismatch"][label]
            macros[f"Mismatch{key}LUT"] = fmt(mm["LUT"])
            macros[f"Mismatch{key}LUTf"] = fmt(mm["LUT_filtered"])
            macros[f"Mismatch{key}PPO"] = fmt(np.mean([mm[r] for r in runs]))
    # Discount-factor study: chosen by best validation reward; test results per gamma.
    main_gamma = float(yaml.safe_load((ROOT / "runs" / runs[0] / "config.yaml").read_text())["ppo"]["gamma"])
    macros["GammaMain"] = f"{main_gamma:g}"
    for g, rewards in best_validation_rewards().items():
        tag = GAMMA_TAGS.get(g)
        if tag:
            macros[f"ValReward{tag}"] = fmt(float(np.mean(rewards)))
    for g, tag in GAMMA_TAGS.items():
        s = group_mean(main_, f"gamma_{g:g}" if g != 0 else "gamma_0.0")
        if g == main_gamma:
            s = {k: float(np.mean([main_[r][k] for r in runs])) for k in main_[runs[0]]}
        macros[f"Gamma{tag}Gain"] = "??" if s is None else fmt(s["energy_gain_pct"])
        macros[f"Gamma{tag}Reward"] = "??" if s is None else fmt(s["mean_reward"])
        macros.setdefault(f"ValReward{tag}", "??")

    for prefix, _, stem in ABLATIONS:
        s = group_mean(main_, prefix)
        missing = s is None  # not trained yet: keep the draft compilable, with visible placeholders
        macros[f"Abl{stem}Gain"] = "??" if missing else fmt(s["energy_gain_pct"])
        macros[f"Abl{stem}Travel"] = "??" if missing else fmt(s["yaw_travel_deg_per_turbine_h"], 0)
        macros[f"Abl{stem}Reward"] = "??" if missing else fmt(s["mean_reward"])

    REPORT.mkdir(exist_ok=True)
    with open(REPORT / "numbers.tex", "w") as fh:
        fh.write("% generated by scripts/report_numbers.py -- do not edit\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    def row(label, s, bold=False):
        cells = [label, fmt(s["energy_gain_pct"], sign=True), fmt(s["yaw_travel_deg_per_turbine_h"], 0),
                 fmt(s["side_switches_per_turbine_h"], 1), fmt(s["mean_reward"], sign=True)]
        if bold:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        return " & ".join(cells) + r" \\"

    alpha = ev["filtered_lut_tuning"]["alpha"]
    lines = [row("Greedy (baseline)", main_["Greedy"]),  # random set-points are quoted in the text only
             row("LUT (FLORIS-SR)", main_["LUT"]), row(f"LUT + filter ($\\alpha$={alpha:g})", main_["LUT_filtered"]),
             r"\midrule"]
    g, gs = ppo("energy_gain_pct")
    t, ts = ppo("yaw_travel_deg_per_turbine_h")
    sw, _ = ppo("side_switches_per_turbine_h")
    r, _ = ppo("mean_reward")
    lines.append(" & ".join(f"\\textbf{{{c}}}" for c in [
        f"PPO (ours, {len(runs)} seeds)", f"{fmt(g, sign=True)}$\\pm${fmt(gs, 3)}", f"{fmt(t, 0)}$\\pm${fmt(ts, 0)}",
        fmt(sw, 1), fmt(r, sign=True)]) + r" \\")
    for prefix, label, _ in ABLATIONS:
        s = group_mean(main_, prefix)
        if s is not None:
            lines.append(row(label, s))
    # \bottomrule lives in this file: after \input inside a tabular it would raise "Misplaced \noalign".
    lines.append(r"\bottomrule")
    (REPORT / "table_main.tex").write_text("% generated by scripts/report_numbers.py -- do not edit\n" + "\n".join(lines) + "\n")
    # Same table in Markdown for the README.
    def md_row(label, s, bold=False):
        cells = [label, f"{s['energy_gain_pct']:+.2f} %", f"{s['yaw_travel_deg_per_turbine_h']:.0f}",
                 f"{s['side_switches_per_turbine_h']:.1f}", f"{s['mean_reward']:+.3f}"]
        return "| " + " | ".join(f"**{c}**" if bold else c for c in cells) + " |"

    md = ["| Controller | Energy gain vs greedy | Yaw travel [°/turbine/h] | Wake-side switches [/turbine/h] | Reward / step |",
          "|---|---:|---:|---:|---:|",
          md_row("Greedy (industry baseline)", main_["Greedy"]), md_row("Random set-points", main_["Random"]),
          md_row("LUT (FLORIS Serial-Refine, model-based)", main_["LUT"]),
          md_row(f"LUT + exponential filter (α = {alpha:g}, tuned on validation)", main_["LUT_filtered"])]
    agg_row = {k: float(np.mean([main_[r][k] for r in runs])) for k in main_[runs[0]]}
    md.append(md_row(f"PPO agent (ours, mean of {len(runs)} seeds)", agg_row, bold=True))
    for prefix, label, _ in ABLATIONS:
        s = group_mean(main_, prefix)
        if s is not None:
            plain = label.replace(r"\ -- ", "").replace(r"$\gamma$\,=\,", "γ = ").replace("C$_4$", "C4")
            md.append(md_row("ablation: " + plain, s))
    (ROOT / "results" / "results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {len(macros)} macros and {len(lines)} table rows")
    for k, v in macros.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
