# FINAL PROPOSAL (READY 9.1/10, 3 rounds, GPT-5.4 xhigh reviewed)

# Research Proposal: Mechanics-Anchored Monotone Residual Diffusion for Tool-Wear Path Correction under Label Scarcity

## Problem Anchor
- Bottom-line problem: Predict milling tool flank wear and RUL under a TARGET cutting condition where real labeled run-to-failure data is scarce (<=20%), and output calibrated prediction intervals that support tool-change decisions.
- Must-solve bottleneck: (a) degradation-inconsistent generative augmentation causes negative transfer; (b) UQ coverage collapses under sim-to-real + cross-condition shift.
- Non-goals: no online control; no new diffusion architecture; no bearings; no LLM.
- Constraints: public data only (PHM2010 + Nature SciData 2024 coated end-mill); single consumer GPU (<=100 GPU-h); ~3 months; JIM/Measurement/EAAI (CAS Q2).
- Success condition: <=20% target labels -> wear MAE within 10% of full-supervision bound; coverage >= nominal-2% under shift; components justified by ablation.

## Method Thesis
Model the sim-to-real gap of an early-cycle-calibrated mechanistic tool-wear twin as a monotone-structured residual diffusion process on the latent wear path; train the downstream force-only wear/RUL predictor on corrected paths; wrap decisions with weighted split-conformal intervals (deployment wrapper, not a novelty claim).

## Contribution Focus
- Dominant: M-DRDC mechanism + evidence that residualization (vs direct generation, B5/B5+) and monotone structure (vs A1) each contribute under label scarcity.
- Wrapper: weighted split-CP with explicit-descriptor weights; empirical late-change-rate report.
- Non-contributions: twin models, DDPM backbone, TCN, CP theory.

## Proposed Method
### Pipeline
```
S1 TWIN    calibrate wear ODE + force model per condition (MAP; source full-life + target early 15%)
           outputs: w_twin(t;p), F_twin(w,p), misfit scalar m
S2 M-DRDC  1-D DDPM (~0.8M params) over u in R^64 (spline-resampled increment grid)
           w_corr = w_early + cumsum(softplus(u)); cond = [Delta w_twin seq, p, m]; CFG
           trained on source pool: 2 PHM2010 + >=6 SciData full-life paths, random-crop segments,
           dropout + early stop, 5 seeds; endpoint guidance weight 0 in primary results
S3 FORCE   f(t) = F_twin(w_corr(t), p) + ridge([w, w^2, w*vc, w*fz, p])  (closed-form fit on source)
S4 PREDICT force-only TCN, identical feature schema for synthetic & real; Huber + RUL score loss;
           trained on corrected synthetic + k% real target labels
S5 UQ      weighted split-CP; weights: logistic on [p, m]; clip + ESS check;
           decision rule w_hi >= w_th; late-change rate reported empirically
```
### Failure Modes & Diagnostics
- Thin source signal -> capacity cut + crops + seeds (above); report learning curves.
- Twin misfit at target -> m in conditioning; early-cycle R2 diagnostic; hierarchical MAP fallback.
- h_psi suspected as gain source -> isolation checks (i)/(ii) in appendix.
- Guidance bias -> weight-0 primary + appendix sensitivity.

## Validation Matrix (main paper = PHM2010 rotations, force-only)
| ID | System | Purpose |
|----|--------|---------|
| B1 | TCN, 100% target labels | upper bound |
| B2 | TCN, k% labels only | scarcity floor |
| B3 | twin-only augmentation (+ridge force map) | is generation needed at all? |
| B5 | direct monotone path DDPM, cond [p,m], same budget | residualization delta |
| B5+| B5 + twin curve as conditioning channel | information-budget-matched variant |
| A1 | non-monotone raw-residual diffusion | structure delta |
| OURS | M-DRDC | headline |
- k in {5,10,20}%; 3 rotations x 5 seeds; metrics: wear MAE/RMSE, RUL PHM-score, crossing-time CRPS/coverage, monotonicity-violation rate (0 for ours/B5 by construction, reported for A1).
- Claim 2 (wrapper): PICP@90 / NMPIW / late-change rate: weighted CP vs vanilla CP vs MC-dropout under rotation shift.
- Diagnostic figure: twin calibration fraction {5,10,15,25}% -> twin RMSE + downstream MAE.
- Appendix: role-swapped transfer (SciData target), h_psi isolation, guidance sensitivity, vib/AE practitioner note.

## Compute & Timeline
<=30 GPU-h total. W1-2 twin + B1/B2/B3; W3-5 M-DRDC + B5/B5+/A1; W6 CP; W7-8 appendix; W9-12 writing (JIM).
