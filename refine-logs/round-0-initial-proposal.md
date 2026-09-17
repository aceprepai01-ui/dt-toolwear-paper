# Research Proposal: Degradation-Consistent Residual Diffusion Correction of an Early-Cycle-Identifiable Tool Digital Twin for Label-Scarce Wear/RUL Prediction with Shift-Aware Conformal Risk Control

## Problem Anchor
- Bottom-line problem: Predict milling tool flank wear and RUL under a TARGET cutting condition where real labeled run-to-failure data is scarce (<=20% of tools/labels), and output calibrated prediction intervals that support tool-change decisions.
- Must-solve bottleneck: (a) Existing generative augmentation for tool wear (incl. the July-2026 Research Square preprint: cutting-mechanics-conditioned diffusion w/ FiLM) generates signals WITHOUT degradation consistency — no wear monotonicity, no cross-sensor coherence, no life-endpoint consistency — causing negative transfer in downstream regressors. (b) Existing UQ for tool wear is an afterthought (vanilla split CP / MC-dropout) whose coverage collapses under sim-to-real + cross-condition distribution shift.
- Non-goals: no online process control; no new diffusion architecture invention; no bearing/gearbox experiments; no LLM component.
- Constraints: public data only (PHM2010 milling c1/c4/c6; Nature Sci Data 2024 coated end-mill whole-life multi-feature dataset), single consumer GPU (<=100 GPU-h total), ~3 months, target journals JIM / Measurement / EAAI (CAS Q2).
- Success condition: With <=20% target-condition labels, wear MAE within 10% of the full-supervision upper bound; empirical interval coverage >= nominal-2% under cross-condition shift while baselines' coverage degrades >=5%; each mechanism component justified by ablation.

## Technical Gap
Current pipeline failure point: physics-conditioned generation (preprint A) treats each signal window independently — the generator can emit a "worn" window followed by a "fresh" window, force channels inconsistent with vibration channels, and trajectories that never cross the wear threshold. Downstream regressors trained on such data inherit these artifacts (negative transfer documented in the preprint's own GAN baselines). Naive fixes fail: more sim data amplifies the sim bias; bigger conditional generators do not learn degradation structure they are never supervised on; vanilla CP after the fact cannot repair coverage under covariate shift because calibration and test distributions differ.
Missing mechanism: a corrector that (1) starts from a mechanistic trajectory that is identifiable from cheap early-cycle data, (2) learns only the RESIDUAL between twin and reality at trajectory level, under explicit degradation-consistency constraints, and (3) propagates the remaining error into a shift-corrected decision-level uncertainty guarantee.

## Method Thesis
- One-sentence thesis: Learn the sim-to-real gap as a degradation-consistent residual process on top of an early-cycle-identifiable cutting-mechanics twin, instead of generating wear signals from physics conditions — then certify tool-change decisions with shift-aware conformal risk control.
- Smallest adequate intervention: we reuse a textbook mechanistic wear model, a standard 1D conditional DDPM backbone, a standard TCN regressor, and published weighted-CP theory; the only new pieces are the residual formulation + 3 constraint losses + the decision-level risk wrapper.
- Timeliness: diffusion-era generative priors and conformal risk control are both 2024-2026 primitives; their combination at the degradation-trajectory level is unoccupied (verified by novelty check 2026-08-01).

## Contribution Focus
- Dominant contribution: degradation-consistent residual diffusion correction (DRDC) of an identifiable tool digital twin — a mechanism-level claim: residual + trajectory-level constraints beats condition-injected direct generation (preprint A style) for downstream few-label wear/RUL learning.
- Supporting contribution: shift-aware conformal risk control (SA-CRC) on the tool-change decision, calibrated with likelihood-ratio-weighted target/source scores.
- Explicit non-contributions: the twin itself (textbook models), the regressor, the DDPM backbone.

## Proposed Method

### Complexity Budget
- Reused/frozen: extended-Taylor/Usui wear ODE + mechanistic milling force model (twin); 1D U-Net DDPM backbone; TCN regressor; weighted split-CP machinery.
- New trainable components (2): (1) residual diffusion corrector with constraint losses; (2) a small condition-embedding density-ratio head (logistic regression on embeddings) for weighting — arguably not even "trainable component" scale.
- Tempting additions intentionally excluded: DANN/MMD adversarial alignment (redundant with residual correction), multi-task wear-stage classifier, attention-based sensor fusion, LLM knowledge injection.

### System Overview
```
Stage 1  TWIN        cutting params p, early-cycle data D_early(target) + full source data
                     -> calibrate theta of wear ODE + force model (MAP, per-condition)
                     -> emit twin feature trajectories X_twin(p) over full life + wear curve w_twin
Stage 2  DRDC        train conditional residual DDPM: r = X_real - X_twin on SOURCE conditions
                     conditioning: [X_twin window, p, wear-stage embedding]; CFG guidance
                     constraints: L_mono + L_coh + L_end
                     -> generate corrected synthetic full-life trajectories for TARGET condition
Stage 3  PREDICTOR   TCN trained on {corrected synthetic} + {<=20% real target labels} (mixing ratio swept)
Stage 4  SA-CRC      weighted split-CP intervals on wear; conformal risk control on
                     threshold-crossing decision (bound P[late tool change] <= alpha)
```

### Core Mechanism (DRDC)
- Input/output: input = twin feature-trajectory window (L=64 cuts, C=6-10 per-cut features: force RMS/peak per axis, vib RMS, AE energy), condition vector p (vc, fz, ap, ae, material one-hot), wear-stage embedding (sinusoidal on w_twin/w_max); output = residual window r_hat added to twin trajectory.
- Architecture: standard 1D conditional U-Net DDPM (~8M params), FiLM conditioning (same backbone class as preprint A — deliberately, to isolate the residual+constraints delta), classifier-free guidance scale swept in {1,2,4}.
- Training signal / loss: L = L_ddpm(eps-pred) + lam1*L_mono + lam2*L_coh + lam3*L_end, where on x_hat = X_twin + r_hat:
  - L_mono = mean ReLU(-(g(x_hat_{t+1}) - g(x_hat_t))) with g = frozen auxiliary wear-proxy head pretrained on source (penalizes implied-wear decreases);
  - L_coh = || Corr_channels(x_hat) - Corr_channels(x_real) ||_F on matched wear stages (cross-sensor coherence);
  - L_end = hinge on |t_cross(x_hat) - t_cross(w_twin)| / T_life (threshold-crossing endpoint consistency, tolerance 10%).
  Constraint losses computed on x0-parameterization at sampled timesteps (standard practice, no architecture change).
- Why main novelty: supervision target (residual process) + trajectory-level degradation constraints are absent from preprint A (window-level FiLM generation), from bearing DT-diffusion works (B/C/D — classification-oriented, no degradation trajectory), and from CycleGAN mapping works (no constraints, mode issues).

### Supporting Component (SA-CRC)
- Input/output: predictor residuals on calibration set -> per-cut wear interval [w_lo, w_hi]; decision rule "change tool when w_hi >= w_th" with certified risk.
- Mechanism: weighted split conformal (Tibshirani et al. 2019 covariate-shift CP) with density ratio dP_tgt/dP_src estimated by logistic head on TCN penultimate embeddings; on top, conformal risk control (Angelopoulos et al. 2024) selects the decision threshold to bound the late-change (false-healthy) rate <= alpha = 0.1.
- No sprawl: zero new trained models beyond the logistic head; pure calibration-time machinery.

### Integration & Inference Path
At test time on target condition: twin(p) -> DRDC sampling (N=50 trajectories, gives generative ensemble spread) -> TCN point prediction per cut -> SA-CRC interval + change decision. Twin recalibration is one-shot from early cycles (first 15% of life, wear labels from standard microscope measurements already present in both datasets).

### Training Plan
1. Calibrate twin per condition (least squares + MAP priors from literature ranges; ~minutes, CPU).
2. Train DRDC on source conditions only (PHM2010: train on c1,c4 -> target c6, rotate; cross-dataset: PHM2010 -> SciData). 4-8 GPU-h per config.
3. Train TCN on corrected synthetic + k% real target (k in {5,10,20}); Huber loss on wear + PHM2010-score loss on RUL. <1 GPU-h.
4. Calibrate SA-CRC on the k% real target data (split from training folds properly: calibration never seen by TCN).

### Failure Modes & Diagnostics
- Twin badly mis-calibrated for unseen condition -> diagnostic: early-cycle fit R2; fallback: widen MAP priors / hierarchical pooling across conditions.
- DRDC memorizes source residuals (doesn't transfer) -> diagnostic: MMD(corrected, real) on held-out target unlabeled data vs MMD(twin, real); if no reduction, report honestly + analyze.
- Constraint losses conflict with DDPM loss -> lambda sweep + constraint-violation-rate curves (reviewer-friendly).
- Density ratio unstable with tiny target sets -> clip weights, effective-sample-size check.

### Novelty & Elegance Argument
Closest work and exact deltas: preprint A (condition-injected direct generation; we: residual target + trajectory constraints; we use A-style generation as ablation B5); Measurement'25/26 + ASCE bearing DT-diffusion (classification/bearing; no degradation trajectory, no decision-level UQ); MSSP'26 staged conformal tool monitoring (no generative twin, vanilla CP, no shift correction); AEI'26 physics-guided UDA (feature alignment, no generation, no UQ). One dominant mechanism, one supporting wrapper, everything else reused — parts count is LOWER than preprint A + separate UQ papers combined.

## Claim-Driven Validation Sketch (3 core blocks)
### Claim 1 (dominant): Residual + degradation constraints beat direct physics-conditioned generation for few-label downstream learning.
- Minimal experiment: PHM2010 leave-one-tool-out x k in {5,10,20}% labels; identical backbone everywhere.
- Baselines: B2 few-shot-only TCN; B3 twin-only augmentation (no DRDC); B4 CycleGAN mapping; B5 direct conditional DDPM w/ FiLM (preprint-A surrogate, same U-Net); B6 physics-informed regressor (no generation); B1 full-label upper bound.
- Ablations: drop each of L_mono/L_coh/L_end; direct (non-residual) generation with constraints.
- Metrics: wear MAE/RMSE, RUL PHM-score; MMD + spectral distance (generation quality); constraint-violation rates.
- Expected evidence: B5 < ours by clear margin at k=5-10%; each constraint contributes; ours within 10% of B1 at k=20%.
### Claim 2: SA-CRC keeps coverage under cross-condition shift where vanilla CP fails.
- Minimal experiment: cross-condition (PHM2010 rotations) + cross-dataset (PHM2010->SciData) calibration/test splits.
- Baselines: vanilla split CP, MC-dropout, deep ensembles; SA-CRC w/o weighting (ablation).
- Metrics: PICP vs nominal (90%), NMPIW, late-change rate vs alpha.
- Expected evidence: baselines lose >=5% coverage under shift; SA-CRC within 2%; late-change rate <= alpha empirically.
### Claim 3: Early-cycle identifiability — 15% of life suffices to calibrate the twin for useful extrapolation.
- Minimal experiment: calibration-fraction sweep {5,10,15,25}% x parameter-error vs downstream-MAE curve; hierarchical pooling on/off.
- Metric: twin trajectory RMSE on remaining life; downstream sensitivity.
- Expected evidence: knee at ~15%; downstream robust to moderate twin error (shows DRDC absorbs it — ties back to Claim 1).

## Experiment Handoff Inputs
- Must-prove: the three claims above; must-run: B5 surrogate comparison (this kills or crowns the paper).
- Datasets: PHM2010 (c1,c4,c6, 315 cuts each, wear microscope labels); Nature Sci Data 2024 coated end-mill (multi-condition, force+vib+AE).
- Highest-risk assumptions: twin feature synthesis realistic enough that residual is learnable; SciData conditions overlap enough for cross-dataset transfer.

## Compute & Timeline
- <=100 GPU-h total (single 3090/4090). Twin: CPU-only. 
- Weeks 1-2 twin + baselines; 3-6 DRDC; 7-8 SA-CRC + ablations; 9-12 writing (JIM first).
