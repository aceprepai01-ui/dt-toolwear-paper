import numpy as np
import pytest

from mdrdc import metrics
from mdrdc.twin import calibrate as cal
from mdrdc.twin.force_model import fit_force_map
from mdrdc.twin.wear_ode import TwinParams, simulate
from tests.synthetic import synth_wear


class TestMetrics:
    def test_mae_rmse_hand_values(self):
        assert metrics.mae([0, 0], [1, -1]) == pytest.approx(1.0)
        assert metrics.rmse([0, 0], [3, -4]) == pytest.approx(np.sqrt(12.5))

    def test_phm_score_asymmetry(self):
        late = metrics.phm_score([100.0], [110.0])   # predicted 10 late (dangerous)
        early = metrics.phm_score([100.0], [90.0])   # predicted 10 early (safe)
        assert late > early > 0

    def test_phm_score_perfect_is_zero(self):
        assert metrics.phm_score([50, 100], [50, 100]) == pytest.approx(0.0)

    def test_picp_nmpiw(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo, hi = y - 1, y + 1
        assert metrics.picp(y, lo, hi) == 1.0
        assert metrics.nmpiw(y, lo, hi) == pytest.approx(2.0 / 3.0)

    def test_crps_sharp_beats_diffuse(self):
        rng = np.random.default_rng(0)
        sharp = rng.normal(0, 0.1, 4000)
        diffuse = rng.normal(0, 5.0, 4000)
        assert metrics.crps_empirical(sharp, 0.0) < metrics.crps_empirical(diffuse, 0.0)

    def test_crossing_time_interpolates(self):
        path = np.array([0.0, 10.0, 30.0])
        assert metrics.crossing_time(path, 20.0) == pytest.approx(1.5)
        assert metrics.crossing_time(path, 100.0) == float("inf")


class TestWearODE:
    def test_simulate_monotone_and_three_phase(self):
        path = simulate(TwinParams(), 315)
        d = np.diff(path)
        assert np.all(d >= 0)
        # break-in decelerates, acceleration phase re-accelerates
        assert d[0] > d[50]
        assert d[-1] > d[100]

    def test_condition_scaling(self):
        p = TwinParams()
        fast = simulate(p, 100, vc=1.5 * 10400.0)
        slow = simulate(p, 100)
        assert fast[-1] > slow[-1]

    def test_params_immutable_roundtrip(self):
        p = TwinParams()
        v = p.to_vector()
        assert TwinParams.from_vector(v) == p
        p2 = p.with_updates(a=2.0)
        assert p.a == 1.0 and p2.a == 2.0


class TestCalibration:
    def test_recovers_synthetic_twin(self):
        true_params = TwinParams(log_k1=np.log(0.4), log_beta_brk=np.log(4.0))
        wear = simulate(true_params, 315)
        res = cal.calibrate_early_fraction(wear, fraction=0.15, n_starts=4)
        assert res.r2_fit > 0.95  # clean synthetic: near-perfect early fit
        extrap = res.extrapolate(315, vb0=float(wear[0]))
        # extrapolation should track the true curve within measurement scale
        assert metrics.mae(wear, extrap) < 20.0

    def test_gate_on_noisy_data(self):
        wear = synth_wear(315, noise=2.0, seed=3)
        res = cal.calibrate_early_fraction(np.maximum.accumulate(wear), fraction=0.15)
        assert res.r2_fit > 0.8  # the R002 gate on realistic noise

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            cal.calibrate(np.array([1.0, 2.0]))  # too short


class TestForceMap:
    def test_fit_and_predict_recovers_linear_wear_trend(self):
        rng = np.random.default_rng(0)
        vb = np.linspace(0, 200, 100)
        base = np.array([10.0, 20.0])
        kappa = np.array([0.5, 0.2])
        X = base[None, :] * (1 + kappa[None, :] * (vb[:, None] / 200.0))
        X = X + rng.normal(0, 0.05, X.shape)
        fm = fit_force_map(vb, X)
        pred = fm.predict(vb)
        assert metrics.r2(X[:, 0], pred[:, 0]) > 0.99
        assert fm.ridge_w.size <= 7 * 2  # low-capacity guarantee (K4 defense)
