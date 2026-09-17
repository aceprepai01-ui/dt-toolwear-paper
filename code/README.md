# M-DRDC / Pooled-Twin Code

Paper: *When Scarce Labels Favor Pooled Physics over Deep Learning in Public
Milling Prognosis* (working title, FINAL_PROPOSAL_v3). Plan & tracker:
`../refine-logs/EXPERIMENT_PLAN.md`, `../refine-logs/EXPERIMENT_TRACKER.md`.

> REBUILT 2026-09-17 from session records after the working tree was deleted
> (~2026-08-22). All 31 unit tests pass on the rebuild. Result JSONs and raw
> datasets were NOT recoverable — re-download data (below) and re-run the
> pipeline before submission so every table regenerates from this exact code.

## Layout
```
mdrdc/
  data/phm2010.py     PHM2010 parsing (c1/c4/c6, wear labels, per-cut signals)
  data/qit.py         QIT-CEMC wear labels (source-pool role only)
  data/features.py    12-dim force-only per-cut features
  data/resample.py    isotonize -> PCHIP T'=64 -> softplus codec (ENC_EPS=0.05)
  metrics.py          MAE/RMSE/R2, PHM score, PICP/NMPIW, CRPS, crossing time
  twin/wear_ode.py    three-phase wear-rate ODE twin
  twin/calibrate.py   MAP calibration (handbook or custom prior; arbitrary indices)
  twin/hierarchical.py EB pooled twin (v3 MAIN METHOD): hyperprior, pooled MAP, misfit flag
  twin/force_model.py mechanistic K(VB) + closed-form ridge (7 basis coefs/feature)
  models/tcn.py       shared TCN regressor (x and y normalized)
  models/ddpm.py      residual DDPM (negative-result baseline; EMA, per-position norm)
  experiments/protocol.py   rotations, uniform label budget, last-20% window, RUL
  experiments/baselines.py  B1/B2/B3 assembly
  experiments/residual_bank.py calibration-variant bank, 4-d cond, regime gating
scripts/
  download_phm2010.sh   Kaggle download (QIT: figshare 27323346 -> data/raw/scidata)
  r001_parse_features.py  R001 gate: 315 cuts/tool
  r002_calibrate_twin.py  R002 gate: early R2 >= 0.8
  r003_overfit_sanity.py  R003 gate: overfit MAE < 3um (--synthetic works w/o data)
  w2_baselines.py         LIT/B1/B2/B3 (gates A: 10-45um, B: monotone within SEM)
  w3_mdrdc_train.py       diffusion corrector (archived: gate never passed)
  w5_downstream.py        pre-registered arbitration (archived: NO SEPARATION)
  w6_hb_matrix.py         v3 matrix H1/H3/H4/H5/H6 (pooled-twin results)
  w6b_flag_cp.py          H5b/H6b rescue attempts (archived: both failed)
  figures.py              paper figures fig2-fig5
tests/                    31 unit tests, synthetic-data only
```

## Reproduction order (after data download)
```bash
python3 -m pytest tests/ -q
python3 scripts/r001_parse_features.py --root <phm2010_dir>
python3 scripts/r002_calibrate_twin.py
python3 scripts/w2_baselines.py
python3 scripts/w6_hb_matrix.py          # main-method tables
python3 scripts/w6b_flag_cp.py
python3 scripts/w5_downstream.py         # negative-result table (GPU ~1.5h)
python3 scripts/figures.py
```

## Archived key numbers (for regression checking after re-runs)
- R002 early-15% R2: c1 0.882 / c4 0.940 / c6 0.990
- W2: B1 21.0±3.3, LIT 38.3±13.1; B2@{5,10,20}% = 49.9/50.0/20.4
- W6 H1 pooled@{5,10}% = 22.65/25.01, handbook = 31.82/32.41
- W3 twin crossing baseline 24.3 cuts (censored; per-tool 21.7/42.0/9.2)
- W5: NO SEPARATION (OURS 53.7/48.4 vs B3 53.5/48.5 vs B2 51.4/52.4)
- Detectors (H5/H5b): misfit 4.17/5.03/6.56; divergence 7.5/6.9/7.8 (flat = boundary finding)
