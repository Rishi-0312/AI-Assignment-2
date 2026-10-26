| Controller | Energy gain vs greedy | Yaw travel [°/turbine/h] | Wake-side switches [/turbine/h] | Reward / step |
|---|---:|---:|---:|---:|
| Greedy (industry baseline) | +0.00 % | 143 | 0.4 | -0.146 |
| Random set-points | -2.68 % | 719 | 15.8 | -1.775 |
| LUT (FLORIS Serial-Refine, model-based) | +2.53 % | 290 | 2.2 | +0.874 |
| LUT + exponential filter (α = 0.6, tuned on validation) | +2.72 % | 185 | 1.4 | +1.040 |
| **PPO agent (ours, mean of 3 seeds)** | **+2.92 %** | **147** | **1.3** | **+1.154** |
| ablation: long horizon (γ = 0.95) | +2.52 % | 169 | 1.5 | +0.963 |
| ablation: plain MLP (γ = 0.95) | +2.25 % | 169 | 1.7 | +0.840 |
| ablation: no power inputs | +2.74 % | 159 | 1.4 | +1.068 |
| ablation: no wear penalty | +2.92 % | 173 | 1.4 | +1.138 |
