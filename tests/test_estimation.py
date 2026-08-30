"""Parameter estimation.

The decisive test is recovery: generate from known parameters, hide them, and
check the optimiser finds them again.  On noise-free data that has to succeed
to many digits, because any error there is the estimator's own and not the
data's.  On noisy data the estimate is a random variable, so the tests ask for
the looser things that must still hold - the right order of magnitude, and
confidence intervals that contain the truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import (
    DRC_2020,
    Observations,
    cost_surface,
    estimate_multistart,
    estimate_parameters,
    generate_observations,
    predict,
    residuals,
    weight_vector,
)


@pytest.fixture(scope="module")
def clean() -> Observations:
    """Noise-free observations - the estimator has no excuse on these."""
    return generate_observations(noise=0.0)


@pytest.fixture(scope="module")
def noisy() -> Observations:
    """5% proportional noise, the default scenario."""
    return generate_observations(noise=0.05, seed=20260830)


# -- the forward map ------------------------------------------------------


def test_predict_returns_the_observed_series_at_the_observation_times(clean):
    modelled = predict(DRC_2020, clean)
    assert set(modelled) == set(clean.names)
    for name in clean.names:
        assert modelled[name].shape == clean.t.shape
        assert modelled[name] == pytest.approx(clean.truth[name], rel=1e-8)


def test_residuals_vanish_at_the_generating_parameters(clean):
    r = residuals((DRC_2020.beta, DRC_2020.alpha), ("beta", "alpha"), clean)
    assert np.max(np.abs(r)) < 1e-9


def test_residuals_are_ordered_like_the_stacked_observations(noisy):
    r = residuals((0.2, 0.04), ("beta", "alpha"), noisy, w=None)
    modelled = predict(DRC_2020.replace(beta=0.2, alpha=0.04), noisy)
    expected = np.concatenate(
        [modelled[name] - noisy.values[name] for name in noisy.names]
    )
    assert r == pytest.approx(expected)


# -- weighting ------------------------------------------------------------


def test_scale_weighting_rescues_a_small_series_from_a_large_one():
    """The case weighting exists for: one compartment dwarfing another.

    Observing S (about 88 million) alongside A (about 0.05 million), the raw
    residuals differ by nearly three orders of magnitude and an unweighted fit
    is effectively a fit to S alone.  Dividing by the series scale brings the
    two contributions within a factor of a few.
    """
    obs = generate_observations(observed=("S", "A"), noise=0.05, seed=1)
    n = obs.n_points

    def parts(w):
        r = residuals((0.25, 0.05), ("beta", "alpha"), obs, w=w)
        return np.sqrt(np.mean(r[:n] ** 2)), np.sqrt(np.mean(r[n:] ** 2))

    s_raw, a_raw = parts(None)
    assert s_raw / a_raw > 100.0

    s_weighted, a_weighted = parts(weight_vector(obs, "scale"))
    assert 0.1 < s_weighted / a_weighted < 10.0


def test_weighting_leaves_the_default_pair_broadly_balanced(noisy):
    """A and T average within a factor of two, so neither can swamp the other."""
    n = noisy.n_points
    r = residuals((0.25, 0.05), ("beta", "alpha"), noisy, w=weight_vector(noisy, "scale"))
    a_part, t_part = np.sqrt(np.mean(r[:n] ** 2)), np.sqrt(np.mean(r[n:] ** 2))
    assert 0.1 < a_part / t_part < 10.0


def test_sigma_weighting_needs_recorded_sigma():
    obs = Observations(t=np.linspace(0, 1, 3), values={"A": np.ones(3)})
    with pytest.raises(ValueError, match="recorded sigma"):
        weight_vector(obs, "sigma")


def test_unknown_weighting_is_rejected(noisy):
    with pytest.raises(ValueError, match="unknown weighting"):
        weight_vector(noisy, "inverse-variance-ish")


# -- recovery -------------------------------------------------------------


def test_noise_free_recovery_is_essentially_exact(clean):
    fit = estimate_parameters(clean, fit=("beta", "alpha"))
    assert fit.success
    for name, value in fit.relative_errors().items():
        assert abs(value) < 1e-4, f"{name} off by {value}%"
    assert fit.cost < 1e-16


def test_recovery_of_a_single_parameter(clean):
    fit = estimate_parameters(clean, fit=("beta",))
    assert fit.names == ("beta",)
    assert fit.estimates["beta"] == pytest.approx(DRC_2020.beta, rel=1e-5)


def test_recovery_of_three_parameters(clean):
    fit = estimate_parameters(clean, fit=("beta", "alpha", "sigma2"))
    assert fit.estimates["sigma2"] == pytest.approx(DRC_2020.sigma2, rel=1e-4)


def test_recovery_does_not_start_from_the_answer(clean):
    """The default guess must come from the bounds, not from the baseline."""
    far = estimate_parameters(clean, guess={"beta": 1.5, "alpha": 0.6})
    assert far.estimates["beta"] == pytest.approx(DRC_2020.beta, rel=1e-3)
    assert far.estimates["alpha"] == pytest.approx(DRC_2020.alpha, rel=1e-3)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_noisy_recovery_lands_near_the_truth(seed):
    obs = generate_observations(noise=0.05, seed=seed)
    fit = estimate_parameters(obs)
    for name, value in fit.relative_errors().items():
        assert abs(value) < 30.0, f"{name} off by {value:.1f}%"
    assert all(fit.covers_truth().values())


def test_noise_free_intervals_are_tight_and_noisy_ones_are_not(clean, noisy):
    tight = estimate_parameters(clean)
    loose = estimate_parameters(noisy)
    for name in ("beta", "alpha"):
        assert tight.stderr[name] < loose.stderr[name]


def test_the_fit_beats_the_truth_on_its_own_data(noisy):
    """A least-squares optimum must fit the sample at least as well as truth.

    If the cost at the estimate were above the cost at the generating values,
    the optimiser stopped early - a real failure, and one that is invisible if
    only the parameter error is checked.
    """
    fit = estimate_parameters(noisy)
    w = weight_vector(noisy, "scale")
    r_truth = residuals(
        (DRC_2020.beta, DRC_2020.alpha), ("beta", "alpha"), noisy, w=w
    )
    assert fit.cost <= 0.5 * float(r_truth @ r_truth)


# -- bounds ---------------------------------------------------------------


def test_bounds_are_respected(clean):
    fit = estimate_parameters(clean, fit=("beta",), bounds={"beta": (0.30, 0.40)})
    assert 0.30 <= fit.estimates["beta"] <= 0.40
    assert fit.estimates["beta"] == pytest.approx(0.30, abs=1e-6)


def test_estimates_stay_inside_the_default_box(noisy):
    from hiv_drc import PARAMETER_BOUNDS

    fit = estimate_parameters(noisy)
    for name in fit.names:
        low, high = PARAMETER_BOUNDS[name]
        assert low <= fit.estimates[name] <= high


def test_a_guess_outside_the_bounds_is_rejected(clean):
    with pytest.raises(ValueError, match="outside the bounds"):
        estimate_parameters(clean, fit=("beta",), guess={"beta": 9.0})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fit": ()}, "nothing to fit"),
        ({"fit": ("gamma",)}, "unknown parameter"),
        ({"fit": ("beta", "alpha"), "guess": {"beta": 0.2}}, "no starting value"),
    ],
)
def test_invalid_requests_are_rejected(clean, kwargs, match):
    with pytest.raises(ValueError, match=match):
        estimate_parameters(clean, **kwargs)


# -- what the result carries ---------------------------------------------


def test_result_reports_uncertainty_consistently(noisy):
    fit = estimate_parameters(noisy)
    for j, name in enumerate(fit.names):
        low, high = fit.ci95[name]
        assert low < fit.estimates[name] < high
        assert fit.stderr[name] == pytest.approx(np.sqrt(fit.covariance[j, j]))
    assert fit.correlation.shape == (2, 2)
    assert fit.correlation[0, 0] == pytest.approx(1.0)
    assert abs(fit.correlation[0, 1]) <= 1.0
    assert fit.correlation[0, 1] == pytest.approx(fit.correlation[1, 0])


def test_result_carries_a_usable_parameter_set_and_trajectory(noisy):
    fit = estimate_parameters(noisy)
    assert fit.parameters.beta == fit.estimates["beta"]
    assert fit.parameters.alpha == fit.estimates["alpha"]
    # Everything not fitted is untouched.
    assert fit.parameters.sigma1 == DRC_2020.sigma1
    assert fit.solution.t[0] == noisy.t[0]
    assert fit.solution.t[-1] == noisy.t[-1]
    assert fit.R0 > 0.0
    assert fit.theta == pytest.approx([fit.estimates[n] for n in fit.names])


def test_summary_mentions_every_fitted_parameter(noisy):
    text = estimate_parameters(noisy).summary()
    for name in ("beta", "alpha"):
        assert name in text
    assert "95% CI" in text


def test_scoring_needs_ground_truth(tmp_path):
    """Data read back from a file has no parameters, so scoring must refuse."""
    path = generate_observations(n_points=8).to_csv(tmp_path / "obs.csv")
    reloaded = Observations.from_csv(path)
    fit = estimate_parameters(reloaded, fit=("alpha",))
    assert fit.truth is None
    for method in (fit.errors, fit.relative_errors, fit.covers_truth):
        with pytest.raises(ValueError, match="no ground truth"):
            method()
    assert "95% CI" in fit.summary()


# -- global search and the landscape --------------------------------------


def test_multistart_agrees_with_itself(noisy):
    fit = estimate_multistart(noisy, n_starts=5, seed=3)
    assert len(fit.starts) == 5
    for name in fit.names:
        values = [start[name] for start in fit.starts]
        assert max(values) - min(values) < 1e-3, f"{name} basins disagree"


def test_multistart_is_at_least_as_good_as_one_start(noisy):
    single = estimate_parameters(noisy)
    best = estimate_multistart(noisy, n_starts=4, seed=11)
    assert best.cost <= single.cost * (1.0 + 1e-9)


def test_multistart_needs_a_start():
    with pytest.raises(ValueError, match="at least 1"):
        estimate_multistart(generate_observations(n_points=4), n_starts=0)


def test_cost_surface_is_minimised_near_the_estimate(noisy):
    fit = estimate_parameters(noisy)
    x, y, cost = cost_surface(noisy, n=15)
    assert cost.shape == (y.size, x.size)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    assert abs(x[j] - fit.estimates["beta"]) < (x[1] - x[0])
    assert abs(y[i] - fit.estimates["alpha"]) < (y[1] - y[0])
    assert cost.min() >= fit.cost * (1.0 - 1e-9)


def test_cost_surface_needs_exactly_two_parameters(noisy):
    with pytest.raises(ValueError, match="exactly two"):
        cost_surface(noisy, names=("beta",), n=3)


def test_cost_surface_accepts_explicit_grids(noisy):
    grids = [np.linspace(0.10, 0.20, 4), np.linspace(0.03, 0.04, 3)]
    x, y, cost = cost_surface(noisy, grids=grids)
    assert x.size == 4
    assert cost.shape == (3, 4)


# -- robustness -----------------------------------------------------------


def test_a_hopeless_parameter_gives_finite_residuals(noisy):
    """The optimiser must be able to step into a bad region and come back."""
    r = residuals((2.0, 1.0), ("beta", "alpha"), noisy)
    assert np.all(np.isfinite(r))


def test_holding_the_wrong_baseline_biases_the_fit(clean):
    """Fixed parameters are assumptions, and wrong ones show up as bias.

    Not a defect to be fixed - a property worth pinning down, because it is the
    reason a fitted set should stay small and the rest should be defensible.
    """
    wrong = DRC_2020.replace(sigma2=0.12)
    fit = estimate_parameters(clean, fit=("beta", "alpha"), baseline=wrong)
    assert abs(fit.relative_errors()["alpha"]) > 1.0
