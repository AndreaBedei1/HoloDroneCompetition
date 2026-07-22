# Frozen Stage-1 Evaluations — Verdict

Two independent, frozen held-out evaluations of the combined 34-demo BC model
(`bc/model_hash.json`), run on the **real HoloOcean adapter** (no fallback), the
**unchanged Stage-1 track and referee**, onboard-only observations, with the
corrected metric/randomization code. Seeds are disjoint from all demonstration,
validation and prior development seeds (0–33, 300–319, 400–419).

| Evaluation | Condition | Seeds | Result | Wilson 95% CI | Collisions | OOB | Wrong-dir | Verdict |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| **A** `evaluation_fixed_50/` | fixed start (**Stage 1**) | 1000–1049 | **50/50 = 100%** | [0.929, 1.000] | 0 | 0 | 0 | **Stage 1 PASS** |
| **B** `evaluation_randomized_50/` | randomized start (**Stage 2**) | 1100–1149 | **48/50 = 96%** | [0.865, 0.989] | 0 | 124 | 1 | Stage 2 PASS (point est.) |

Mean inference ≈ 42 ms/step (≈24 Hz, comfortably real-time at dt=0.1 s) in both.

## Verdict (matching-condition rule)

- **Stage 1** is judged **only** by Evaluation A (fixed start): **PASS** — 100% over 50
  held-out seeds; the Wilson 95% lower bound (92.9%) is above the 90% criterion.
- **Stage 2** is judged **only** by Evaluation B (randomized start): **PASS on the point
  estimate** (96% ≥ 90%), but the Wilson 95% lower bound (86.5%) dips **below** 90%, so
  the pass is not statistically ironclad at 95% confidence. Reported honestly rather
  than rounded up.

Randomized-start demonstrations were used, but Stage 2 is credited on the held-out
**randomized** evaluation (B), not merely because training was randomized.

## Failure analysis (Evaluation B)

Both failures (`evaluation_randomized_50/failure_analysis.json`) occur at
**near-maximum** randomization offsets:

- seed 1103 — lateral −0.94 m, yaw −13.9°: drifts out of bounds (103 OOB), 0 gates.
- seed 1139 — lateral +0.97 m, yaw +14.1°: drifts out of bounds (21 OOB), 1 wrong-dir.

The policy is robust across the interior of the Stage-2 envelope but fails at its
extreme corners (large lateral offset combined with large yaw). Natural next steps:
add demonstrations at extreme offsets (or DAgger-style corrective demos), which is a
data-quantity remedy — no gate or referee margin was relaxed.
