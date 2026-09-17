# Experiment Plan

**Problem**: Label-scarce (<=20%) milling tool wear/RUL prediction under target-condition shift, with calibrated decision intervals.
**Method Thesis**: Mechanics-anchored monotone residual diffusion (M-DRDC) corrects the twin's latent wear path; force-only TCN learns from corrected paths; weighted split-CP wraps decisions.
**Date**: 2026-08-01 | **Budget**: <=30 GPU-h, single consumer GPU, 12 weeks | **Venue**: JIM (stretch) / Measurement / EAAI

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|------------------------------|---------------|
| C1 (headline): residualization over a mechanistic twin beats direct generation, and monotone structure contributes, under label scarcity | This is the delta vs the July-2026 preprint; the paper lives or dies here | ours > B5 AND ours > B5+ AND ours > A1 at k=5,10% on wear MAE + RUL score, 3 rotations x 5 seeds, non-overlapping std | K1, K2 |
| C2 (wrapper): weighted CP keeps coverage under rotation shift where vanilla CP / MC-dropout degrade | Makes the method deployable; secondary table | PICP@90 >= 88% for ours; baselines drop >= 5 pts; late-change rate reported | K3 |
| Anti-claim 1: "gain comes from h_psi force corrector" | Attribution attack from R2 review | ours w/ h_psi=0 still beats B3+h_psi | K4 |
| Anti-claim 2: "gain comes from twin-endpoint guidance bias" | Drift attack from R1 review | primary tables at guidance weight 0; sensitivity curve {0,.1,.5,1} | K4 |
| Anti-claim 3: "diffusion is decorative for a 64-length sequence" | Likely JIM reviewer question | GP/bootstrap residual sampler underperforms M-DRDC (nice-to-have) | K5 |

## Paper Storyline
- Main paper must prove: C1 (Tables 1-2), C2 (Table 3), monotonicity/crossing-time realism (Fig: generative-object quality), early-cycle identifiability diagnostic (Fig).
- Appendix: h_psi isolation, guidance sensitivity, SciData role-swap transfer, vib/AE practitioner note, GP-sampler necessity check.
- Intentionally cut: multichannel synthesis, DANN/MMD alignment, conformal risk-control theorems, cross-machine generalization claims.

## Experiment Blocks

### K1: Main anchor table (MUST-RUN) — Table 1
- Claim: C1 part 1 — corrected synthetic paths rescue scarcity.
- Data: PHM2010 c1/c4/c6, 3 leave-one-condition-out rotations; k in {5,10,20}% target labels; SciData paths in source pool.
- Systems: B1 (100% labels UB), B2 (k% only), B3 (twin-only + ridge), OURS.
- Metrics: wear MAE/RMSE (decisive), RUL PHM-score; 5 seeds, mean±std.
- Success: ours - B3 gap > seed std at k=5,10%; ours within 10% of B1 at k=20%.
- Failure reading: ours ≈ B3 → generation adds nothing → diagnose path realism (K2 metrics) before any further runs.

### K2: Novelty isolation (MUST-RUN) — Table 2 + Fig
- Claim: C1 part 2 — residualization delta (vs B5/B5+) and structure delta (vs A1).
- Systems: B5 (direct monotone path DDPM, cond [p,m], same 0.8M budget), B5+ (B5 + twin curve as conditioning channel), A1 (non-monotone raw-residual), OURS. Identical U-Net, identical training regime.
- Metrics: downstream wear MAE (decisive); generative-object metrics: crossing-time CRPS + empirical coverage, monotonicity-violation rate (A1 only), path RMSE vs held-out real.
- Success: ours > B5+ (information-matched) with non-overlapping std at k=5,10%.
- Failure reading: ours ≈ B5+ → residualization not needed → CONTINGENCY: re-center narrative on structure delta (ours vs A1) + twin identifiability; still publishable at Measurement/EAAI.

### K3: UQ wrapper (MUST-RUN) — Table 3
- Claim: C2. Systems: weighted split-CP (ours), vanilla split-CP, MC-dropout. Metrics: PICP@90 (decisive), NMPIW, empirical late-change rate at w_th; weight ESS reported.
- Success: coverage within 2 pts of nominal under rotation shift; baselines degrade.
- Failure reading: weighting no help (tiny calibration sets) → report honestly, demote to appendix, paper still stands on C1.

### K4: Attribution & bias defenses (MUST-RUN) — Appendix A/B
- h_psi isolation: (i) ours with h_psi=0; (ii) B3 + h_psi. Guidance sensitivity: weight {0,0.1,0.5,1.0}, endpoint-bias check (corrected life vs twin life vs real life).
- Success: (i) retains most of the gain; (ii) does not close the gap; guidance-0 already wins.

### K5: Diffusion necessity (NICE-TO-HAVE) — Appendix C
- GP-over-increments residual sampler (closed-form, CPU) as the strongest simple alternative.
- Success: M-DRDC > GP sampler downstream; if not → honest discussion (diffusion still wins on crossing-time calibration or multi-modality of residuals, else acknowledge).

### K6: Transfer + diagnostics + failure analysis (MUST-RUN, cheap) — Figs + Appendix D
- Early-cycle calibration sweep {5,10,15,25}% → twin RMSE + downstream MAE (main-paper diagnostic figure).
- SciData role-swap (train source incl. PHM2010, target = SciData condition) — appendix sanity.
- Failure analysis: worst rotation, largest-misfit tool; qualitative corrected-vs-real path figure.

## Run Order and Milestones (12 weeks)
| Milestone | Weeks | Goal | Key Runs | Decision Gate | GPU-h | Risk / Mitigation |
|-----------|-------|------|----------|---------------|-------|-------------------|
| M0 sanity | W1 | Data pipeline, wear labels, PHM-score metric verified vs literature; twin MAP calibration | twin fits, 1 overfit TCN run | twin early-15% fit R2 >= 0.8 per source condition; else widen priors/hierarchical MAP | ~1 | PHM2010 label quirks → cross-check 3 published MAE numbers |
| M1 baselines | W2 | B1/B2/B3 complete | 3 systems x 3 rot x 3 k x 5 seeds (TCN minutes each) | B1 MAE in literature range (≈10-20 μm); B2 degrades monotonically with k | ~3 | If B2 barely degrades, scarcity story weak → drop to k∈{2,5,10}% |
| M2 core | W3-4 | M-DRDC trained; generation quality | 3 rot x 5 seeds diffusion + sampling | corrected paths beat raw twin on held-out source crossing-time CRPS | ~8 | Thin data → crops/capacity/early-stop already in design; watch val curves |
| M2.5 first blood | W5 | OURS downstream vs B3 | ours full grid | ours > B3 at k=10% beyond std | ~2 | If fails: fix path realism BEFORE touching K2 (do not run kill-shot on a broken generator) |
| M3 decision | W6 | K2 kill-shot: B5/B5+/A1 | 3 variants x 3 rot x 5 seeds + downstream | C1 verdict; trigger contingency if B5+ ≈ ours | ~10 | Highest-stakes week; freeze all hyperparams from M2 for fairness |
| M3.5 UQ | W7 | K3 table | CP calibration (CPU) + MC-dropout runs | coverage gate | ~2 | ESS too low → widen calibration split |
| M4 defenses | W8 | K4 (+K5 if time) | isolation + sensitivity grids | anti-claims closed | ~3 | — |
| M4.5 extras | W9 | K6 transfer, diagnostics, failure figs | role-swap + sweeps | none (appendix) | ~1 | — |
| M5 paper | W10-12 | Writing, figures, /auto-review-loop, submit | — | internal review >= 8/10 before submission | 0 | Target JIM first; Measurement fallback ready in cover letter |

Total: ~30 GPU-h (fits budget). All TCN/CP work is minutes-scale; diffusion training dominates.

## Compute and Data Budget
- GPU: ~30 h single consumer GPU. CPU: twin calibration, ridge, CP, GP sampler.
- Data prep: PHM2010 download + wear-label parsing (W1); SciData download + increment-grid resampling to T'=64 (W1); spline resampling utilities shared by all systems.
- Human eval: none. Biggest bottleneck: W6 decision week — protect it by freezing configs in W3-4.

## Risks and Mitigations
- B5+ matches ours (headline fails) → contingency narrative: structure delta + identifiability (K2/A1 + K6), venue drops to Measurement/EAAI.
- Small-data variance swamps gaps → 5 seeds + paired-per-rotation reporting (report per-rotation deltas, not only pooled means).
- Twin misfit large on c6-style conditions → misfit m is conditioning input; failure analysis (K6) turns it into content, not a hidden weakness.
- Preprint A gets published mid-project → cite the journal version; our B5/B5+ comparison is already designed as the direct answer.

## Final Checklist
- [x] Main tables covered (K1-K3)  [x] Novelty isolated (K2)  [x] Simplicity defended (K4, h_psi/guidance)  [x] Frontier justified (K5)  [x] MUST vs NICE separated
