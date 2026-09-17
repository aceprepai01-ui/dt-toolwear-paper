# FINAL PROPOSAL v3 (reframed 2026-08-13; novelty cross-verified 7.0/10 PROCEED WITH CAUTION)

# When Scarce Labels Favor Pooled Physics over Deep Learning: an Empirical-Bayes Mechanistic Twin Benchmark for Milling Tool-Wear Prognosis

## Problem Anchor (v3)
- Bottom-line problem: predict milling tool flank wear and RUL on the decision-critical last 20% of life when the target tool has only k<=10% uniformly-sampled wear labels and NO requirement for an online sensor stream.
- Must-solve bottleneck: (a) DL/generative pipelines are data-hungry and, as our pre-registered arbitration shows, add no downstream value in this regime; (b) single-tool mechanistic calibration cannot see late-life regime knowledge that exists only in other tools' full lives.
- Non-goals: no new Bayesian machinery (HB is a borrowed tool — cite arXiv 2601.15942, JMP 2026-06 drilling upfront); no online updating; no universal anti-diffusion claim.
- Constraints: public data only (PHM2010 + QIT-CEMC); CPU-deterministic main method; JIM (stretch) / Measurement / EAAI.
- Success condition: HB twin at k=10% within 20% of the DL upper bound trained at 80% labels; >=1.8x better than every DL system at matched budget; conclusions stable under leave-one-source-out and prior-scale x{0.5,1,2}.

## Method Thesis
Fit the three-phase wear-rate ODE independently on each full-life SOURCE tool (2 PHM2010 rotation tools + QIT cross-machine tool); form an empirical-Bayes hyperprior (per-parameter mean/sd of source MAP fits); calibrate the TARGET tool by MAP under this hyperprior from its k% labels; forecast deterministically; wrap wear intervals with weighted split-conformal. Selectively flag targets whose labeled-window misfit signals an out-of-population regime (c4-type coating breakdown) instead of silently extrapolating.

## Contribution Focus (per reviewer framing)
1. Dominant (evidence): the first matched-budget, public-data, pre-registered head-to-head showing pooled mechanistic transfer beats TCN few-shot / twin-augmentation / diffusion-corrected augmentation ~2x on the decision window at k in {5,10}%, while approaching the 80%-label DL upper bound.
2. Supporting (mechanism combination, MEDIUM novelty, claimed narrowly): milling wear-trajectory ODE + EB cross-tool prior transfer + sparse-early-label MAP + stream-free deployment — the combination absent from Karandikar'14 / Niaki'15-16 / arXiv'26-01 / JMP'26-06.
3. Supporting (negative result): 10-iteration diagnosis chain showing residual-diffusion correction of the twin adds no downstream value in this public-data regime (scoped claim, matched budgets, 3 seeds).
4. Supporting (boundary): empirical indistinguishability analysis of regime-change tools (c4) from early labels + misfit-flag detector (NOT claimed as provable unless formalized).

## Experiment Matrix (all CPU except reused W2/W5 tables)
| ID | Experiment | Purpose |
|----|-----------|---------|
| H1 | HB twin vs no-pool twin vs handbook-prior twin, 3 rotations x k{5,10}% | pooling delta |
| H2 | HB vs B1/B2/B3/OURS-diffusion (reuse W2/W5 tables, same protocol/seeds) | headline table |
| H3 | Leave-one-source-out hyperprior (drop each of 3 source tools) | EB fragility defense |
| H4 | Prior-scale robustness: sd x {0.5, 1, 2} | reviewer defense |
| H5 | Misfit-flag ROC: labeled-window residual -> flags c4 across rotations? | boundary claim |
| H6 | Weighted split-CP coverage on HB residuals (PICP/NMPIW, rotation shift) | UQ wrapper |
| H7 | Role-swap: QIT as target, PHM as sources | cross-machine sanity (appendix) |
- Metrics: wear MAE/RMSE (last-20%), RUL PHM-score, PICP@90/NMPIW; 3 rotations; deterministic (no seeds needed except DL comparisons).

## Pre-empted Rejections (from cross-verification)
- "domain instantiation of HBM-PHM" -> cite Jan+Jun 2026 upfront; HB borrowed, claim is the benchmark.
- "Bayesian mechanistic milling existed" -> Karandikar/Niaki cited; delta = pooled cross-tool transfer under matched scarcity.
- "your diffusion was just weak" -> pre-registered rule, matched inputs/budgets/seeds, scoped claim.
- "c4 is anecdote" -> H5 turns it into a detector with operating characteristics; wording stays empirical.
- "EB from 3 source tools is fragile" -> H3+H4.

## Timeline (revised)
- Day 1-2: hierarchical.py + H1/H3/H4/H5 (CPU-fast); H6 CP wrapper.
- Day 3: assemble headline tables (H2 reuses stored results), figures.
- Week 2-3: writing (Measurement format), /auto-review-loop, submit.
