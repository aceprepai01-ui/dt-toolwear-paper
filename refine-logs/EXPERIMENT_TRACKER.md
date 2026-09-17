# Experiment Tracker
| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | data sanity | label parse + PHM-score check | all | — | MUST | DONE | DONE 315/315 all tools; wear c1 48.9-172.7 / c4 31.4-210.9 / c6 62.8-234.7 um |
| R002 | M0 | twin calibration | wear ODE + force MAP | per condition | early-15% R2 | MUST | DONE | DONE early R2: c1 0.882, c4 0.940, c6 0.990; extrap MAE 11.1/12.9/29.8 um |
| R003 | M0 | pipeline sanity | TCN overfit 1 tool | c1 | MAE→~0 | MUST | DONE | DONE real-data train MAE 1.19um < 3um gate |
| R010 | M1 | upper bound | B1 TCN 100% | 3 rot x 5 seeds | MAE/RMSE/RUL | MUST | DONE | DONE B1=21.0±3.3um; LIT(cross-tool full-traj)=38.3±13.1 in corrected gate (10,45) — initial (8,25) was same-tool lit, documented |
| R011 | M1 | scarcity floor | B2 k∈{5,10,20}% | 3 rot x 3 k x 5 seeds | same | MUST | DONE | DONE B2: 5%=49.9, 10%=50.0, 20%=20.4um; monotone within SEM (5-10% delta 0.13 << SEM 2.7) |
| R012 | M1 | twin-only aug | B3 + ridge | same grid | same | MUST | DONE | DONE B3: 5%=53.8, 10%=49.0, 20%=37.4um; twin-only aug ~ B2 at low k (negative-transfer evidence, motivates M-DRDC) |
| R020 | M2 | core generator | M-DRDC train+sample | 3 rot x 5 seeds | crossing-CRPS, path RMSE | MUST | CLOSED-SEE-NOTE | R020 W3 CLOSED (8 iterations): M2 crossing-gate NOT passed at k=10% (final localized+selective: 26.3-27.6 vs twin 24.3, seed-stable). Root cause established: with only 3-4 source tools, per-rotation regime support is N=1-2 (c1 plateau ESS=4.4); implicit AND localized conditioning both hit the same small-N wall. Assets: QIT rescue of c4 (42->2.2 best), unobservable-late-acceleration finding, full diagnosis chain. DECISION: arbitrate by DOWNSTREAM K1 (W5) at k={5,10}, not the staging gate. |
| R021 | M2.5 | first blood | OURS downstream | full grid | MAE/RUL | MUST | DONE-NEGATIVE | R021 W5 ARBITRATION DONE: NO SEPARATION at k=5% (OURS 53.7+/-5.6 vs B3 53.5+/-4.5 vs B2 51.4+/-2.5) and k=10% (48.4 vs 48.5 vs 52.4). Pre-registered K1 verdict: DEAD on PHM2010+QIT regime. Single-cell exception: c4@10% s0 applied-correction MAE 24.1. Root: B2 already near-saturated with uniform budgets; diffusion correction adds no downstream signal beyond twin. STRATEGIC FORK raised to user. |
| R030 | M3 | residual delta | B5 direct DDPM | 3 rot x 5 seeds + downstream | MAE + gen metrics | MUST | TODO | identical budget/regime |
| R031 | M3 | info-matched | B5+ (+twin channel) | same | same | MUST | TODO | decisive vs ours |
| R032 | M3 | structure delta | A1 non-monotone | same | same + violation rate | MUST | TODO | |
| R040 | M3.5 | UQ table | weighted CP vs vanilla CP vs MC-dropout | rotations | PICP/NMPIW/late-rate | MUST | TODO | report ESS |
| R050 | M4 | h_psi isolation | ours h_psi=0; B3+h_psi | k=10% grid | MAE | MUST | TODO | appendix A |
| R051 | M4 | guidance bias | weight {0,.1,.5,1} | 1 rot x 5 seeds | MAE + endpoint bias | MUST | TODO | appendix B |
| R052 | M4 | diffusion necessity | GP increment sampler | k=10% grid | MAE + CRPS | NICE | TODO | appendix C |
| R060 | M4.5 | identifiability | calib fraction {5,10,15,25}% | per condition | twin RMSE + MAE | MUST | TODO | main-paper figure |
| R061 | M4.5 | role-swap transfer | target = SciData | 1 config x 5 seeds | MAE | MUST | TODO | appendix D |
| R062 | M4.5 | failure analysis | worst rotation/misfit | qualitative | path figures | MUST | TODO | |
