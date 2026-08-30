"""The synthetic-data generator.

The generator is the yardstick everything downstream is measured against, so
these tests are mostly about it being exactly as advertised: the noise-free
case is the model itself, the seed makes it reproducible, and the noise really
does have the requested scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import DRC_2020, INITIAL_STATE, Observations, generate_observations, simulate


def test_shape_and_names():
    obs = generate_observations(n_points=17, t_span=(0.0, 20.0))
    assert obs.n_points == 17
    assert obs.names == ("A", "T")
    assert obs.t[0] == 0.0
    assert obs.t[-1] == 20.0
    for name in obs.names:
        assert obs.values[name].shape == (17,)


def test_names_follow_compartment_order():
    """Not alphabetical, not insertion order - the order of the state vector."""
    obs = generate_observations(observed=("T", "I2", "A"), noise=0.0)
    assert obs.names == ("I2", "A", "T")


def test_noise_free_data_is_the_model():
    obs = generate_observations(noise=0.0, t_span=(0.0, 30.0), n_points=31)
    solution = simulate(DRC_2020, INITIAL_STATE, (0.0, 30.0), t_eval=obs.t)
    for name in obs.names:
        assert obs.values[name] == pytest.approx(solution[name], rel=1e-9)
        assert obs.truth[name] == pytest.approx(solution[name], rel=1e-9)


def test_seed_makes_it_reproducible():
    a = generate_observations(seed=99)
    b = generate_observations(seed=99)
    c = generate_observations(seed=100)
    assert a.values["A"] == pytest.approx(b.values["A"])
    assert not np.allclose(a.values["A"], c.values["A"])


def test_truth_does_not_depend_on_the_seed():
    """Only the measurement error is random; the trajectory is not."""
    a = generate_observations(seed=1)
    b = generate_observations(seed=2)
    assert a.truth["T"] == pytest.approx(b.truth["T"])


@pytest.mark.parametrize("eta", [0.02, 0.05, 0.10])
def test_constant_noise_has_the_requested_scale(eta):
    """The observed spread should match the requested sigma to sampling error.

    With n points the sample standard deviation is itself uncertain by roughly
    1/sqrt(2n), so a 25% tolerance on 400 points is loose but not vacuous.
    """
    obs = generate_observations(
        noise=eta, noise_model="constant", n_points=400, seed=4, clip=False
    )
    for name in obs.names:
        residual = obs.values[name] - obs.truth[name]
        expected = eta * np.mean(np.abs(obs.truth[name]))
        assert np.std(residual) == pytest.approx(expected, rel=0.25)
        assert abs(np.mean(residual)) < 0.5 * expected


def test_proportional_noise_scales_with_the_signal():
    obs = generate_observations(
        noise=0.05, noise_model="proportional", n_points=41, seed=5, clip=False
    )
    ratio = obs.sigma["A"] / np.abs(obs.truth["A"])
    assert ratio == pytest.approx(np.full_like(ratio, 0.05))


def test_clipping_keeps_observations_non_negative():
    obs = generate_observations(noise=0.8, n_points=200, seed=6, clip=True)
    for name in obs.names:
        assert np.all(obs.values[name] >= 0.0)


def test_zero_noise_leaves_sigma_at_zero():
    obs = generate_observations(noise=0.0)
    assert np.all(obs.sigma["A"] == 0.0)


def test_stack_matches_the_series_order():
    obs = generate_observations(n_points=5)
    stacked = obs.stack("values")
    assert stacked.shape == (10,)
    assert stacked[:5] == pytest.approx(obs.values["A"])
    assert stacked[5:] == pytest.approx(obs.values["T"])


def test_stack_rejects_missing_sources():
    obs = Observations(t=np.linspace(0, 1, 3), values={"A": np.zeros(3)})
    with pytest.raises(ValueError, match="not available"):
        obs.stack("truth")


def test_scale_tracks_the_size_of_each_series():
    """The scale is a series average, so it separates S from the small ones."""
    obs = generate_observations(observed=("S", "A", "T"), noise=0.0)
    scale = obs.scale()
    assert scale["S"] > 1000 * scale["A"]
    assert scale["A"] > 0.0 and scale["T"] > 0.0
    assert scale["A"] == pytest.approx(np.mean(obs.values["A"]))


def test_csv_round_trip(tmp_path):
    obs = generate_observations(noise=0.05, seed=8, n_points=12)
    path = obs.to_csv(tmp_path / "nested" / "observations.csv")
    assert path.exists()

    reloaded = Observations.from_csv(path)
    assert reloaded.names == obs.names
    assert reloaded.t == pytest.approx(obs.t, rel=1e-9)
    for name in obs.names:
        assert reloaded.values[name] == pytest.approx(obs.values[name], rel=1e-9)
        assert reloaded.truth[name] == pytest.approx(obs.truth[name], rel=1e-9)


def test_reloaded_data_has_no_ground_truth_parameters():
    """A file carries the series, not the parameters that made them."""
    obs = generate_observations(n_points=4)
    assert obs.parameters is not None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"noise": -0.1}, "non-negative"),
        ({"noise_model": "poisson"}, "noise_model"),
        ({"observed": ("A", "Z")}, "unknown compartment"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate_observations(**kwargs)


def test_empty_csv_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("t,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        Observations.from_csv(path)


def test_generating_from_other_parameters():
    """The generator must follow its parameters, not the module default."""
    other = DRC_2020.replace(beta=0.30)
    obs = generate_observations(other, noise=0.0)
    baseline = generate_observations(noise=0.0)
    assert obs.parameters.beta == 0.30
    assert not np.allclose(obs.truth["A"], baseline.truth["A"])
