# Round 1 Refinement

## Problem Anchor (verbatim, unchanged)
- Bottom-line problem: Predict milling tool flank wear and RUL under a TARGET cutting condition where real labeled run-to-failure data is scarce (<=20%), and output calibrated prediction intervals that support tool-change decisions.
- Must-solve bottleneck: (a) existing generative augmentation generates signals without degradation consistency, causing negative transfer; (b) existing UQ coverage collapses under sim-to-real + cross-condition shift.
- Non-goals: no online control; no new diffusion architecture; no bearings; no LLM.
- Constraints: public data only; single consumer GPU (<=100 GPU-h); ~3 months; JIM/Measurement/EAAI.
- Success condition: <=20% target labels -> wear MAE within 10% of full-supervision bound; coverage >= nominal-2% under shift; components justified by ablation.

## Anchor Check
- Original bottleneck: degradation-inconsistent generation + shift-fragile UQ.
- Revised method still addresses it — more directly: consistency is now STRUCTURAL (monotone parameterization) rather than penalty-based, and the generated object is exactly the degradation path the anchor cares about.
- Reviewer suggestions rejected as drift: NONE. Both drift warnings (L_end preserving twin bias; multichannel synthesis fiction) are accepted and fixed.

## Simplicity Check
- Dominant contribution after revision: monotone-structured residual diffusion on the latent wear path (M-DRDC), headline claim = beats direct condition-injected generation (preprint-A style) for scarce-label wear/RUL.
- Components removed: AE/vibration synthesis; 3 proxy constraint losses (L_mono, L_end structural now; L_coh gone with multichannel synthesis); DANN/MMD (already excluded); conformal risk control theorem claims (demoted to empirical deployment report).
- Rejected as unnecessary complexity: nothing added; every reviewer fix REDUCES parts count.
- Smallest adequate route: diffusion now models a T-length 1-D increment sequence (not L x C windows) — smaller model, structural constraints, cleaner story.

## Changes Made
1. Generative target (CRITICAL fix): was multichannel sensor windows; now the latent wear path only. w_corr(t) = w_early + cumsum(softplus(u_t)), diffusion models u = (u_1..u_T) conditioned on [twin increment sequence, condition p, early-cycle misfit summary m]. Monotonicity is structural. Force summaries come from the twin's mechanistic wear->force map plus a small residual MLP (h_psi, <0.1M params) fitted on source — NOT diffused. Vibration/AE: optional observed covariates for the downstream regressor on real data only; never synthesized.
2. Endpoint consistency (CRITICAL fix): was hinge to twin's t_cross (preserves bias); now learned from SOURCE real threshold crossings via the DDPM likelihood itself + optional endpoint guidance at sampling where twin life L_twin(p) enters only as a PRIOR (guidance weight swept, can be 0).
3. Contribution focus (IMPORTANT fix): SA-CRC demoted to deployment wrapper. Single UQ deliverable: weighted split-conformal wear intervals; late-tool-change rate reported empirically (no second guarantee claim).
4. Shift weights (fix): logistic regression on [explicit condition descriptors (vc, fz, ap, ae), early-cycle twin-misfit RMSE m] — no deep embeddings.
5. Scope (IMPORTANT fix): main paper = PHM2010 cross-condition rotations only, baselines B1/B2/B3/B5 + ours + one ablation (non-monotone raw-residual variant). PHM2010->SciData moved to appendix sanity check. Early-cycle identifiability demoted to prerequisite diagnostic (one calibration-fraction figure).

## Revised Proposal

# Research Proposal (v1): Monotone-Structured Residual Diffusion Correction of a Mechanistic Tool-Wear Twin for Label-Scarce Wear/RUL Prediction

## Problem Anchor
[identical to round 0 — see above]

## Technical Gap
Direct physics-conditioned generation (July-2026 preprint A) emits signal windows with no degradation structure: implied wear can decrease, trajectories need not terminate at the wear threshold, and synthesis of channels the physics cannot produce (AE/vibration) injects fiction. Downstream regressors inherit these artifacts. Naive fixes (more sim data, bigger conditional generators, post-hoc CP) do not repair this: the missing piece is a generative object that IS the degradation path, corrected from a cheap mechanistic prior, with consistency guaranteed by construction.

## Method Thesis
- One sentence: Model the sim-to-real gap as a monotone-structured residual process on the latent wear path of an early-cycle-calibrated mechanistic twin — not as free-form conditional signal generation — and wrap decisions with weighted conformal intervals.
- Smallest adequate: the diffusion object is a 1-D increment sequence; monotonicity/endpoint realism are structural/likelihood-based, not penalty-based; force features come from the twin's own mechanistic map + tiny residual MLP.

## Contribution Focus
- Dominant: M-DRDC — monotone-structured residual diffusion on the wear path; headline claim: beats same-backbone direct condition-injected DDPM (preprint-A surrogate) for scarce-label wear/RUL prediction.
- Supporting (wrapper, not novelty claim): weighted split-CP intervals with condition-descriptor + twin-misfit weights; empirical late-change-rate report.
- Non-contributions: twin models, DDPM backbone, TCN regressor, CP theory.

## Proposed Method
### Complexity Budget
- Reused: extended-Taylor/Usui wear ODE + mechanistic milling force model; 1-D U-Net DDPM (~3M params now); TCN regressor; weighted split-CP.
- New trainable: (1) M-DRDC increment diffusion; (2) residual force-map MLP h_psi. (Logistic weight head is calibration machinery.)
- Excluded on purpose: multichannel synthesis, adversarial alignment, risk-control theorems, multi-task heads.

### System Overview
```
S1 TWIN     calibrate theta on source full-life + target early 15% (MAP)
            -> w_twin(t;p), F_twin(w,p), misfit summary m
S2 M-DRDC   diffusion over u in R^T; w_corr = w_early + cumsum(softplus(u))
            cond = [Delta w_twin seq, p, m]; CFG; endpoint guidance from L_twin(p) (weight >=0)
            trained on SOURCE real wear paths (residual increments vs twin)
S3 FORCE    f(t) = F_twin(w_corr(t), p) + h_psi(w_corr(t), p)   [MLP fitted on source]
S4 PREDICT  TCN: force summaries (+real vib/AE where available) -> wear; trained on
            {(f, w_corr)} synthetic + k% real target; Huber + RUL PHM-score loss
S5 UQ       weighted split-CP on wear residuals; weights: logistic on [p, m];
            tool-change rule w_hi >= w_th; late-change rate reported empirically
```

### Core Mechanism (M-DRDC)
- Object: u in R^T (T = number of cuts, ~315 for PHM2010), 1-D; increments delta_t = softplus(u_t) guarantee monotone w_corr.
- Conditioning: twin increment sequence (the mechanistic prior trajectory), condition vector p, early-cycle misfit m (scalar RMSE + bias of twin on first 15%).
- Training: standard eps-prediction DDPM loss on source conditions' real wear paths expressed as u* = softplus^{-1}(diff(w_real)); no auxiliary penalty losses remain.
- Sampling for target: CFG on [p, m]; optional endpoint guidance nudging cumsum toward crossing w_th near L_twin(p) with tunable (possibly zero) weight — twin as prior, not constraint.
- Why novel: prior works diffuse signal windows conditioned on physics variables (A) or generate class-conditional signals (bearing B/C/D). Nobody diffuses the monotone degradation object itself as a residual over an identifiable mechanistic twin. The structure makes degradation consistency exact and the diffusion problem 100x smaller.

### Failure Modes & Diagnostics
- Twin misfit too large at target -> m is in the conditioning; diagnostic: early-cycle R2; fallback: hierarchical MAP pooling.
- Wear-path realism insufficient to train predictor alone -> mixing ratio sweep with k% real anchors it; report sensitivity.
- softplus^{-1} numerical issues at plateaus -> epsilon-floor on increments (implementation note).
- Weighted CP unstable (tiny calibration set) -> weight clipping + ESS check; report.

### Novelty & Elegance Argument
Delta vs A: generative object (wear path vs signal windows), supervision (residual vs direct), constraints (structural vs none). Delta vs bearing DT-diffusion: degradation-trajectory-level, regression task, decision UQ. Delta vs MSSP'26 staged conformal: generative twin + shift-aware weights. Parts count: one small diffusion + one MLP + textbook twin — fewer than A.

## Claim-Driven Validation (main paper = PHM2010 only)
### Claim 1 (headline): M-DRDC beats direct condition-injected generation for scarce-label learning.
- Setup: PHM2010 rotations (train 2 tools, target 1), k in {5,10,20}% target labels.
- Baselines: B1 full-label UB; B2 few-shot only; B3 twin-only augmentation; B5 direct FiLM-DDPM surrogate (same U-Net budget); ours.
- Ablation (single): non-monotone raw-residual diffusion (removes structure, keeps residual).
- Metrics: wear MAE/RMSE, RUL PHM-score; wear-path realism (monotonicity violations = 0 by construction for ours; endpoint error distribution).
### Claim 2 (wrapper): weighted CP holds coverage under cross-condition shift.
- Baselines: vanilla split-CP, MC-dropout. Metrics: PICP@90, NMPIW, empirical late-change rate.
### Diagnostic (not a claim): early-cycle calibration fraction sweep {5,10,15,25}% -> twin RMSE + downstream MAE (one figure).
### Appendix: PHM2010 -> SciData coated-end-mill sanity transfer.

## Compute & Timeline
- Diffusion now 1-D/3M params: <=2 GPU-h per config; total <=40 GPU-h.
- W1-2 twin + B1/B2/B3; W3-5 M-DRDC + B5; W6 CP wrapper; W7-8 ablation + appendix; W9-12 writing.
