# Refinement Report
**Direction**: DT + intelligent manufacturing, CAS Q2 mechanical paper
**Final title**: Mechanics-Anchored Monotone Residual Diffusion for Tool-Wear Path Correction under Label Scarcity
**Rounds**: 3/5 | **Final Score**: 9.1/10 | **Verdict**: READY | **Date**: 2026-08-01

## Score Evolution
| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|-------|----------|-------------|--------------|----------|-------------|------------|-------|---------|---------|
| 1     | 6        | 6           | 6            | 7        | 6           | 8          | 7     | 6.3     | REVISE  |
| 2     | 9        | 8           | 8            | 8        | 7           | 8          | 8     | 8.1     | REVISE  |
| 3     | 9        | 10          | 9            | 9        | 8           | 9          | 8     | 9.1     | READY   |

## Final Thesis (5 bullets)
- Twin: textbook wear ODE + force model, MAP-calibrated from source full-life + target early 15% of life.
- Core (M-DRDC): ~0.8M-param 1-D DDPM over increment sequence u (T'=64); w_corr = w_early + cumsum(softplus(u)); conditioning [twin increments, condition p, misfit m]; monotonicity structural; endpoint learned from source real crossings, guidance weight 0 in primary tables.
- Force features: mechanistic F_twin(w,p) + closed-form ridge residual (~20 coefs); force-only TCN predictor, identical schema for synthetic/real.
- UQ wrapper: weighted split-CP (logistic weights on [p, m]), empirical late-change-rate report.
- Validation: PHM2010 rotations x k∈{5,10,20}% labels x 5 seeds; B1/B2/B3/B5/B5+/A1/ours; wear MAE, RUL PHM-score, crossing-time CRPS/coverage; appendix: role-swapped SciData transfer, h_psi isolation, guidance sensitivity.

## Method Evolution Highlights
1. Round 1: penalty-based multichannel window generation -> structural monotone wear-path diffusion (the decisive move).
2. Round 2: fairness architecture for the kill-shot comparison (B5/B5+/A1 separates residualization vs information budget vs structure).
3. Throughout: parts count shrank every round (3 proxy losses -> 0; MLP -> ridge; multichannel -> force-only; 3M -> 0.8M params).

## Pushback / Drift Log
| Round | Reviewer Said | Response | Outcome |
|-------|---------------|----------|---------|
| 1     | 2 drift warnings (L_end twin bias; multichannel fiction) | accepted, both fixed | resolved |
| 2-3   | none | — | — |

## Remaining Weaknesses (honest)
- Small-data variance risk (2 PHM2010 + ~6 SciData source paths); mitigations in place but results may be noisy.
- Venue Readiness 8/10: JIM depends on clean empirical execution; Measurement/EAAI safer.

## Next Steps
- /experiment-plan for the detailed execution roadmap, then implement (PHM2010 download + twin calibration first).
