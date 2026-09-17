import numpy as np
import pytest

from mdrdc.experiments.protocol import label_indices
from mdrdc.metrics import mae
from mdrdc.twin.calibrate import PRIOR_TAU
from mdrdc.twin.hierarchical import (Hyperprior, calibrate_pooled,
                                     fit_hyperprior, misfit_flag)
from mdrdc.twin.wear_ode import TwinParams, simulate


def make_population(seed: int = 0, n_tools: int = 3):
    """Synthetic tool population: params jittered around a shared center."""
    rng = np.random.default_rng(seed)
    center = TwinParams(log_k1=np.log(0.5), log_beta_brk=np.log(3.0),
                        log_beta_acc=np.log(1.5))
    tools = {}
    for i in range(n_tools):
        p = center.with_updates(
            log_k1=center.log_k1 + rng.normal(0, 0.15),
            log_beta_acc=center.log_beta_acc + rng.normal(0, 0.2))
        tools[f"t{i}"] = simulate(p, 315)
    return tools


class TestHyperprior:
    def test_pools_and_floors(self):
        tools = make_population()
        hp = fit_hyperprior(tools, n_starts=4)
        assert hp.mu.shape == PRIOR_TAU.shape
        assert np.all(hp.tau >= 0.05 - 1e-12)
        assert np.all(hp.tau <= PRIOR_TAU + 1e-12)  # never wider than handbook
        assert hp.source_names == ("t0", "t1", "t2")

    def test_scaled_robustness_helper(self):
        hp = fit_hyperprior(make_population(), n_starts=2)
        assert np.allclose(hp.scaled(2.0).tau, hp.tau * 2.0)
        with pytest.raises(ValueError):
            hp.scaled(0.0)

    def test_requires_two_sources(self):
        tools = make_population(n_tools=1)
        with pytest.raises(ValueError, match=">=2"):
            fit_hyperprior(tools)


class TestPooledCalibration:
    def test_pooling_beats_handbook_prior_on_noisy_scarce_labels(self):
        """Pooling pays off under NOISE + scarcity (shrinkage prevents chasing
        measurement error); on clean data a wide prior already fits perfectly."""
        rng = np.random.default_rng(7)
        tools = make_population(seed=1, n_tools=4)
        names = list(tools)
        target_name, source_names = names[0], names[1:]
        target = tools[target_name]
        hp = fit_hyperprior({n: tools[n] for n in source_names}, n_starts=4)

        idx = label_indices(len(target), 0.05)  # scarce: ~15 labels
        from mdrdc.twin.calibrate import calibrate
        tail = slice(int(0.8 * len(target)), None)
        wins = 0
        for trial in range(3):
            labeled = np.maximum.accumulate(target[idx] + rng.normal(0, 3.0, len(idx)))
            res_handbook = calibrate(labeled, indices=idx, n_starts=4, seed=trial)
            res_pooled = calibrate_pooled(labeled, idx, hp, n_starts=4, seed=trial)
            m_h = mae(target[tail], res_handbook.extrapolate(len(target), vb0=labeled[0])[tail])
            m_p = mae(target[tail], res_pooled.extrapolate(len(target), vb0=labeled[0])[tail])
            wins += m_p <= m_h * 1.10  # pooled must not lose materially
        assert wins >= 2  # majority of noisy trials

    def test_misfit_flag_fires_on_alien_tool(self):
        tools = make_population(seed=2)
        hp = fit_hyperprior(tools, n_starts=4)
        # in-population target: no flag
        target = simulate(TwinParams(log_k1=np.log(0.5), log_beta_brk=np.log(3.0),
                                     log_beta_acc=np.log(1.5)), 315)
        idx = label_indices(len(target), 0.10)
        res = calibrate_pooled(target[idx], idx, hp, n_starts=4)
        flagged, m = misfit_flag(res, target[idx], idx)
        assert not flagged and m < 8.0
        # alien regime: a mid-life step + slope break the population ODE shape
        # cannot express under the tight pooled prior -> flag must fire.
        # (A smooth late ramp is fittable by the acc term — and genuinely
        # unflaggable, which is exactly the c4 unobservability finding.)
        alien = target.copy()
        alien[120:] = alien[120:] + 40.0 + 0.9 * np.arange(len(alien) - 120)
        alien = np.maximum.accumulate(alien)
        idx_a = label_indices(len(alien), 0.10)
        res_a = calibrate_pooled(alien[idx_a], idx_a, hp, n_starts=4)
        flagged_a, m_a = misfit_flag(res_a, alien[idx_a], idx_a)
        assert flagged_a and m_a > 8.0
