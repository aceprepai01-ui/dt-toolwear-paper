# Round 2 Refinement

## Problem Anchor (verbatim, unchanged from round 0)
- Bottom-line problem: Predict milling tool flank wear and RUL under a TARGET cutting condition where real labeled run-to-failure data is scarce (<=20%), and output calibrated prediction intervals that support tool-change decisions.
- Must-solve bottleneck: (a) degradation-inconsistent generative augmentation causes negative transfer; (b) UQ coverage collapses under sim-to-real + cross-condition shift.
- Non-goals: no online control; no new diffusion architecture; no bearings; no LLM.
- Constraints: public data only; single consumer GPU (<=100 GPU-h); ~3 months; JIM/Measurement/EAAI.
- Success condition: <=20% target labels -> wear MAE within 10% of full-supervision bound; coverage >= nominal-2% under shift; components justified by ablation.

## Anchor Check
- Still on anchor (reviewer confirmed round-2 fidelity 9/10). This round only tightens interfaces and experimental fairness; no mechanism change.
- Rejected as drift: NONE.

## Simplicity Check
- Dominant contribution unchanged: M-DRDC.
- Simplified further this round: predictor is FORCE-ONLY in main paper (vib/AE removed entirely from headline path); h_psi downgraded from MLP to a linear-in-features ridge correction; endpoint guidance weight = 0 in all primary tables; DDPM shrunk to ~0.8M params on a spline-resampled T'=64 increment grid.
- Rejected as unnecessary complexity: modality-mask fusion design (dropped instead, per reviewer's simpler option).

## Changes Made (mapping reviewer's 5 action items)
1. B5 fairness (blocking): B5 redefined as DIRECT monotone wear-path generator — identical U-Net backbone, identical parameter count, identical conditioning [p, m] (twin increments withheld since it is non-residual; twin curve given as an extra conditioning channel in variant B5+ so information budget is matched both ways). Both B5 variants reported. The single ablation A1 (non-monotone raw-residual) isolates the monotone structure. Together: residualization delta = ours vs B5/B5+; structure delta = ours vs A1.
2. Predictor interface (blocking): force-only TCN in main paper for BOTH synthetic and real samples — identical feature schema (per-cut force summaries), no domain cue leakage. Vib/AE relegated to an appendix "practitioner note", not in any headline table.
3. h_psi isolation: h_psi is now ridge regression on [w, w^2, w*vc, w*fz, p] (closed-form, ~20 coefficients). Appendix sanity check: (i) ours with h_psi=0 (pure mechanistic force map); (ii) B3 + h_psi (twin-only augmentation with the same force corrector). If gains persist in (i) and do not appear in (ii), attribution to M-DRDC is clean.
4. Training-signal thinness (blocking): stated explicitly — PHM2010 gives only 2 source wear trajectories per rotation. Plan: (a) source pool ENLARGED with the Nature SciData 2024 coated-end-mill whole-life paths (multiple tools/conditions, public) as additional SOURCE training data (this is training-pool composition, not a cross-dataset transfer claim; the appendix transfer check now swaps roles instead); (b) random-crop subsequence training on the increment grid (T'=64 -> hundreds of overlapping segments per path); (c) model capacity cut to ~0.8M params + dropout + early stopping on held-out segments; (d) 5 seeds with std reported. Effective source count stated in the paper: 2 PHM2010 + >=6 SciData full-life paths.
5. Endpoint guidance sensitivity: primary tables use guidance weight 0; appendix curve over {0, 0.1, 0.5, 1.0} with bias check (does guidance drag corrected endpoints toward biased twin life?).
6. (Modernization items) New generative-object metric in main paper: crossing-time calibration — distribution of threshold-crossing times from N=50 sampled w_corr paths vs real crossing time (CRPS + coverage of empirical crossing-time interval). Novelty language narrowed to: "mechanics-anchored monotone residual diffusion for tool-wear path correction" (no universal 'nobody' claims).

## Revised Proposal (v2)

# Research Proposal: Mechanics-Anchored Monotone Residual Diffusion for Tool-Wear Path Correction under Label Scarcity

## Problem Anchor
[verbatim round-0 anchor — see top of this file]

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
