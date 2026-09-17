import numpy as np
import pytest

from mdrdc.data import phm2010, resample
from mdrdc.data.features import FEATURE_NAMES, extract_cut_features, feature_matrix
from tests.synthetic import make_fake_phm2010, synth_wear


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("phm2010")
    make_fake_phm2010(str(root), n_cuts=12)
    return str(root)


class TestPHM2010Parsing:
    def test_load_tool_counts_match(self, fake_root):
        rec = phm2010.load_tool(fake_root, "c1")
        assert rec.n_cuts == 12
        assert len(rec.signal_files) == 12
        assert rec.wear_per_flute.shape == (12, 3)

    def test_wear_target_max_vs_mean(self, fake_root):
        w_max, _ = phm2010.load_wear(fake_root, "c1", target="max")
        w_mean, _ = phm2010.load_wear(fake_root, "c1", target="mean")
        assert np.all(w_max >= w_mean)  # max flute >= mean by construction

    def test_signal_files_sorted_numerically(self, fake_root):
        rec = phm2010.load_tool(fake_root, "c4")
        nums = [int(f.rsplit("_", 1)[1].split(".")[0]) for f in rec.signal_files]
        assert nums == sorted(nums) == list(range(1, 13))

    def test_unlabeled_tool_rejected(self, fake_root):
        with pytest.raises(ValueError, match="not a labeled tool"):
            phm2010.load_tool(fake_root, "c2")

    def test_missing_dataset_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download"):
            phm2010.load_tool(str(tmp_path), "c1")

    def test_read_cut_signals_validates_columns(self, fake_root):
        rec = phm2010.load_tool(fake_root, "c1")
        df = phm2010.read_cut_signals(rec.signal_files[0])
        assert list(df.columns) == list(phm2010.SIGNAL_COLUMNS)


class TestFeatures:
    def test_feature_vector_shape_and_names(self, fake_root):
        rec = phm2010.load_tool(fake_root, "c1")
        df = phm2010.read_cut_signals(rec.signal_files[0])
        v = extract_cut_features(df)
        assert v.shape == (len(FEATURE_NAMES),) == (12,)
        assert np.all(np.isfinite(v))

    def test_features_track_wear_amplitude(self, fake_root):
        rec = phm2010.load_tool(fake_root, "c1")
        frames = [phm2010.read_cut_signals(f) for f in rec.signal_files]
        X = feature_matrix(frames)
        # synthetic force amplitude grows with wear -> rms of last cut > first cut
        assert X[-1, 0] > X[0, 0]


class TestResample:
    def test_isotonize_makes_monotone(self):
        w = synth_wear(100, noise=3.0, seed=1)
        iso = resample.isotonize(w)
        assert np.all(np.diff(iso) >= 0)

    def test_resample_preserves_endpoints_and_monotonicity(self):
        w = synth_wear(315)
        p = resample.resample_path(w, 64)
        assert p.shape == (64,)
        assert np.all(np.diff(p) >= -1e-9)
        assert p[0] == pytest.approx(w[0], abs=1e-6)
        assert p[-1] == pytest.approx(w[-1], rel=1e-3)

    def test_u_roundtrip(self):
        w = synth_wear(315)
        w0, u = resample.path_to_u(w, 64)
        back = resample.u_to_path(w0, u)
        target = resample.resample_path(w, 64)
        np.testing.assert_allclose(back, target, atol=1e-4)

    def test_softplus_inverse(self):
        x = np.linspace(-5, 5, 50)
        np.testing.assert_allclose(resample.softplus_inv(resample.softplus(x)), x, atol=1e-6)

    def test_decoded_path_always_monotone(self):
        rng = np.random.default_rng(0)
        u = rng.normal(0, 3, 63)  # arbitrary, even wild, latent
        path = resample.u_to_path(10.0, u)
        assert np.all(np.diff(path) >= 0)  # monotone by construction
