"""Bayesian parameter estimation via MCMC.

An MCMC run is expensive - tens of thousands of forward integrations - so
this file runs it as few times as the coverage below allows, and structures
each run as a module-scoped fixture that many tests then interrogate.  The
diagnostics themselves (log_prior, log_likelihood, split_rhat) are cheap and
tested directly and independently of any sampler run, the same way the
frequentist suite validates `weight_vector` and the covariance formula
without needing a fit around them.

The most important test is the cheapest one of all: whether
`log_posterior == log_prior + log_likelihood` as plain arithmetic.  If that
identity fails, nothing downstream can be trusted, no matter how converged
the chain looks.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import (
    DRC_2020,
    ETA_BOUNDS,
    MCMC_BOUNDS,
    PARAMETER_BOUNDS,
    BayesianFitResult,
    Observations,
    estimate_parameters,
    generate_observations,
    log_likelihood,
    log_posterior,
    log_prior,
    reproduction_number,
    run_mcmc,
    split_rhat,
)


@pytest.fixture(scope="module")
def noisy() -> Observations:
    """5% proportional noise, the scenario the frequentist suite also uses."""
    return generate_observations(noise=0.05, seed=20260830)


@pytest.fixture(scope="module")
def tight() -> Observations:
    """Low noise, for a recovery check that should land close to the truth."""
    return generate_observations(noise=0.02, seed=7)


@pytest.fixture(scope="module")
def posterior(noisy: Observations) -> BayesianFitResult:
    """One modest MCMC run, reused by most of the tests below."""
    return run_mcmc(noisy, n_walkers=16, n_steps=400, burn=100, seed=1)


@pytest.fixture(scope="module")
def posterior_tight(tight: Observations) -> BayesianFitResult:
    """A second run on low-noise data, for a tighter recovery check."""
    return run_mcmc(tight, n_walkers=16, n_steps=300, burn=80, seed=2)


# -- log_prior --------------------------------------------------------------


def test_log_prior_finite_inside_the_box(noisy: Observations):
    theta = [0.15, 0.035, np.log(0.05), np.log(0.05)]
    assert np.isfinite(log_prior(theta, ("beta", "alpha"), noisy))


def test_log_prior_neg_inf_outside_a_fit_bound(noisy: Observations):
    theta = [10.0, 0.035, np.log(0.05), np.log(0.05)]
    assert log_prior(theta, ("beta", "alpha"), noisy) == -np.inf


def test_log_prior_neg_inf_outside_eta_bounds(noisy: Observations):
    theta = [0.15, 0.035, np.log(10.0), np.log(0.05)]
    assert log_prior(theta, ("beta", "alpha"), noisy) == -np.inf


def test_log_prior_is_flat_inside_the_box(noisy: Observations):
    """A uniform prior has the same density everywhere inside its support."""
    names = ("beta", "alpha")
    a = log_prior([0.1, 0.02, np.log(0.03), np.log(0.03)], names, noisy)
    b = log_prior([0.4, 0.25, np.log(0.2), np.log(0.2)], names, noisy)
    assert a == pytest.approx(b)


def test_log_prior_rejects_wrong_eta_count(noisy: Observations):
    with pytest.raises(ValueError, match="expected 2 noise parameters"):
        log_prior([0.15, 0.035, np.log(0.05)], ("beta", "alpha"), noisy)


def test_mcmc_bounds_are_tighter_than_the_safety_box():
    """MCMC_BOUNDS exists to narrow beta/alpha, not to widen them."""
    for name in MCMC_BOUNDS:
        lo, hi = MCMC_BOUNDS[name]
        safety_lo, safety_hi = PARAMETER_BOUNDS[name]
        assert safety_lo <= lo
        assert hi <= safety_hi
        assert hi < safety_hi, f"{name} bound was not actually tightened"


# -- log_likelihood -----------------------------------------------------


def test_log_likelihood_favours_the_truth_over_a_bad_guess(noisy: Observations):
    good = [DRC_2020.beta, DRC_2020.alpha, np.log(0.05), np.log(0.05)]
    bad = [0.4, 0.2, np.log(0.05), np.log(0.05)]
    names = ("beta", "alpha")
    assert log_likelihood(good, names, noisy) > log_likelihood(bad, names, noisy)


def test_log_likelihood_peaks_near_the_true_noise_level(noisy: Observations):
    """The hand-derived Gaussian likelihood should itself prefer eta ~ 0.05.

    ``noisy`` was generated with 5% proportional noise; scanning the
    likelihood over a few noise levels at the true rates should score 0.05
    (or its neighbour on the grid) above levels several times too small or
    too large - an independent check that the likelihood formula is doing
    what the docstring claims, not just that it runs.
    """
    names = ("beta", "alpha")
    rates = [DRC_2020.beta, DRC_2020.alpha]
    grid = [0.01, 0.03, 0.05, 0.07, 0.15, 0.40]
    scores = {
        eta: log_likelihood([*rates, np.log(eta), np.log(eta)], names, noisy) for eta in grid
    }
    best = max(scores, key=scores.get)
    assert best in (0.03, 0.05, 0.07)


def test_log_likelihood_is_finite_across_the_prior_support(noisy: Observations):
    rng = np.random.default_rng(0)
    names = ("beta", "alpha")
    for _ in range(20):
        beta = rng.uniform(*MCMC_BOUNDS["beta"])
        alpha = rng.uniform(*MCMC_BOUNDS["alpha"])
        log_eta = rng.uniform(np.log(ETA_BOUNDS[0]), np.log(ETA_BOUNDS[1]), size=2)
        value = log_likelihood([beta, alpha, *log_eta], names, noisy)
        assert np.isfinite(value)


# -- log_posterior: the arithmetic identity ------------------------------


def test_log_posterior_equals_prior_plus_likelihood(noisy: Observations):
    theta = [0.15, 0.035, np.log(0.05), np.log(0.05)]
    names = ("beta", "alpha")
    lp = log_prior(theta, names, noisy)
    ll = log_likelihood(theta, names, noisy)
    assert log_posterior(theta, names, noisy) == pytest.approx(lp + ll)


def test_log_posterior_short_circuits_on_a_forbidden_prior(noisy: Observations):
    """Outside the box, the posterior must be -inf without needing the likelihood."""
    theta = [10.0, 0.035, np.log(0.05), np.log(0.05)]
    assert log_posterior(theta, ("beta", "alpha"), noisy) == -np.inf


# -- split_rhat -----------------------------------------------------------


def test_split_rhat_near_one_for_a_well_mixed_chain():
    rng = np.random.default_rng(0)
    mixed = rng.standard_normal((2000, 16))
    assert split_rhat(mixed) == pytest.approx(1.0, abs=0.02)


def test_split_rhat_large_for_stuck_walkers():
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-5, 5, size=16)
    stuck = offsets[None, :] + 0.1 * rng.standard_normal((2000, 16))
    assert split_rhat(stuck) > 5.0


def test_split_rhat_nan_for_too_short_a_chain():
    chain = np.zeros((2, 8))
    assert np.isnan(split_rhat(chain))


# -- run_mcmc: input validation -------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fit": ()}, "nothing to fit"),
        ({"fit": ("gamma",)}, "unknown parameter"),
        ({"n_steps": 100, "burn": 100}, "burn"),
        ({"n_walkers": 2}, "n_walkers"),
    ],
)
def test_run_mcmc_rejects_bad_input(noisy: Observations, kwargs, match):
    call_kwargs = {"n_steps": 100, "burn": 20, "n_walkers": 16, **kwargs}
    with pytest.raises(ValueError, match=match):
        run_mcmc(noisy, **call_kwargs)


# -- run_mcmc: structural properties of a real run ------------------------


def test_samples_respect_the_prior_box(posterior: BayesianFitResult):
    for j, name in enumerate(posterior.names):
        lo, hi = MCMC_BOUNDS[name]
        assert np.all(posterior.samples[:, j] >= lo)
        assert np.all(posterior.samples[:, j] <= hi)
    for k, _name in enumerate(posterior.noise_names):
        j = len(posterior.names) + k
        assert np.all(posterior.samples[:, j] >= ETA_BOUNDS[0])
        assert np.all(posterior.samples[:, j] <= ETA_BOUNDS[1])


def test_chain_shape_matches_the_run_configuration(posterior: BayesianFitResult):
    kept = posterior.n_steps - posterior.burn
    ndim = len(posterior.all_names)
    assert posterior.chain.shape == (kept, posterior.n_walkers, ndim)
    assert posterior.samples.shape == (kept * posterior.n_walkers, ndim)


def test_credible_interval_brackets_the_median(posterior: BayesianFitResult):
    for name in posterior.all_names:
        lo, hi = posterior.credible_interval[name]
        assert lo <= posterior.median[name] <= hi


def test_rhat_reported_for_every_parameter(posterior: BayesianFitResult):
    for name in posterior.all_names:
        assert np.isfinite(posterior.rhat[name])
        assert posterior.rhat[name] > 0.9


def test_ess_is_none_or_a_positive_number(posterior: BayesianFitResult):
    for name in posterior.all_names:
        ess = posterior.ess[name]
        assert ess is None or ess > 0


def test_acceptance_fraction_is_healthy(posterior: BayesianFitResult):
    """emcee's own rule of thumb: roughly 0.2-0.5 signals a well-tuned ensemble."""
    assert 0.1 < posterior.acceptance_fraction < 0.8


def test_run_is_reproducible_given_the_same_seed(noisy: Observations):
    a = run_mcmc(noisy, n_walkers=16, n_steps=60, burn=15, seed=42)
    b = run_mcmc(noisy, n_walkers=16, n_steps=60, burn=15, seed=42)
    assert a.samples == pytest.approx(b.samples)


def test_different_seeds_give_different_chains(noisy: Observations):
    a = run_mcmc(noisy, n_walkers=16, n_steps=60, burn=15, seed=1)
    b = run_mcmc(noisy, n_walkers=16, n_steps=60, burn=15, seed=2)
    assert not np.allclose(a.samples, b.samples)


# -- run_mcmc: does it recover the right answer? --------------------------


def test_posterior_median_agrees_with_the_frequentist_estimate(
    noisy: Observations, posterior: BayesianFitResult
):
    """Two independent methods, one cross-check.

    With a weakly informative prior, the posterior median and the
    least-squares point estimate are answering close to the same question on
    the same data, so they should land near each other even though nothing
    in `run_mcmc` reuses `estimate_parameters`'s answer beyond centring the
    walkers.
    """
    ls = estimate_parameters(noisy, fit=("beta", "alpha"))
    for name in ("beta", "alpha"):
        assert posterior.median[name] == pytest.approx(ls.estimates[name], rel=0.25)


def test_tight_noise_recovery_lands_close_to_the_truth(posterior_tight: BayesianFitResult):
    """A relative-error bound, not a coverage claim.

    A 95% credible interval is, by construction, expected to miss the truth
    on roughly one run in twenty - asserting it always covers on one fixed
    seed would be asserting a coin never comes up tails. Coverage is instead
    checked in aggregate across many replicates for the README's comparison
    against the frequentist Wald interval; here it is enough that the point
    estimate itself is in the right neighbourhood.
    """
    for name, value in posterior_tight.relative_errors().items():
        assert abs(value) < 25.0, f"{name} off by {value:.1f}%"


def test_noisy_recovery_is_in_the_right_ballpark(posterior: BayesianFitResult):
    for name, value in posterior.relative_errors().items():
        assert abs(value) < 40.0, f"{name} off by {value:.1f}%"


# -- what the result carries -----------------------------------------------


def test_parameters_and_solution_use_the_posterior_median(posterior: BayesianFitResult):
    assert posterior.parameters.beta == posterior.median["beta"]
    assert posterior.parameters.alpha == posterior.median["alpha"]
    assert posterior.parameters.sigma1 == DRC_2020.sigma1  # untouched
    assert posterior.solution.t[0] == pytest.approx(0.0)
    assert posterior.solution.t[-1] == pytest.approx(30.0)


def test_R0_matches_the_reproduction_module(posterior: BayesianFitResult):
    expected = reproduction_number(posterior.parameters).R0
    assert posterior.R0 == pytest.approx(expected)


def test_least_squares_field_is_a_real_fit(posterior: BayesianFitResult):
    assert posterior.least_squares.names == posterior.names
    assert posterior.least_squares.success


def test_summary_mentions_diagnostics(posterior: BayesianFitResult):
    text = posterior.summary()
    assert "R-hat" in text
    assert "95% CI" in text
    for name in posterior.all_names:
        assert name in text


def test_posterior_predictive_shape_and_rough_coverage(
    noisy: Observations, posterior: BayesianFitResult
):
    predictive = posterior.posterior_predictive(noisy, n_draws=80, seed=0)
    assert set(predictive) == set(noisy.names)
    for name in noisy.names:
        draws = predictive[name]
        assert draws.shape == (80, noisy.n_points)
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        inside = np.mean((noisy.values[name] >= lo) & (noisy.values[name] <= hi))
        assert inside > 0.5


def test_scoring_needs_ground_truth():
    """Constructed directly, without a real fit, to test the dataclass logic alone."""
    stub = BayesianFitResult(
        names=("beta",),
        noise_names=("eta_A",),
        samples=np.zeros((10, 2)),
        chain=np.zeros((5, 2, 2)),
        median={"beta": 0.1, "eta_A": 0.05},
        mean={"beta": 0.1, "eta_A": 0.05},
        std={"beta": 0.01, "eta_A": 0.01},
        credible_interval={"beta": (0.08, 0.12), "eta_A": (0.03, 0.07)},
        rhat={"beta": 1.0, "eta_A": 1.0},
        ess={"beta": 100.0, "eta_A": 100.0},
        acceptance_fraction=0.5,
        parameters=DRC_2020,
        solution=None,  # type: ignore[arg-type]
        least_squares=None,  # type: ignore[arg-type]
        n_walkers=4,
        n_steps=10,
        burn=5,
        truth=None,
    )
    for method in (stub.errors, stub.relative_errors, stub.covers_truth):
        with pytest.raises(ValueError, match="no ground truth"):
            method()


# -- generalises beyond the two-parameter default --------------------------


def test_fitting_three_parameters(tight: Observations):
    result = run_mcmc(
        tight, fit=("beta", "alpha", "sigma2"), n_walkers=12, n_steps=120, burn=30, seed=3
    )
    assert result.names == ("beta", "alpha", "sigma2")
    assert result.samples.shape[1] == 5  # 3 rates + 2 noise levels
    assert result.credible_interval["sigma2"][0] <= result.median["sigma2"]
    assert result.credible_interval["sigma2"][1] >= result.median["sigma2"]
