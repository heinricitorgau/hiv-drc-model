"""Informative priors and the contraction diagnostic.

The distributions are checked against properties that must hold rather than
against a re-implementation of their formulae: a density integrates to one,
a log-normal's median is its median, samples reproduce the parameters they
were drawn from.

The contraction diagnostic gets the most attention, because it is the thing
standing between an informative-prior fit and a self-fulfilling one. It is
validated on two constructed cases with a known answer - a posterior that is
literally the prior (contraction 0) and one ten times sharper (contraction
0.9) - before being trusted on a real chain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from hiv_drc.priors import (
    SCALEUP_PRIORS,
    LogNormal,
    Normal,
    Prior,
    Uniform,
    contraction,
    log_density,
    sample,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20260902)


# -- the densities integrate to one ----------------------------------------


@pytest.mark.parametrize(
    "prior",
    [Normal(0.15, 0.05), LogNormal(0.35, 0.7), Uniform(0.0, 2.0)],
)
def test_the_density_integrates_to_one(prior):
    lo, hi = (-5.0, 5.0) if isinstance(prior, Normal) else (1e-9, 50.0)
    total, _ = quad(lambda x: math.exp(prior.logpdf(x)), lo, hi, limit=400)
    assert total == pytest.approx(1.0, abs=1e-3)


# -- and behave the way their parameters say -------------------------------


def test_normal_peaks_at_its_mean():
    prior = Normal(0.15, 0.05)
    assert prior.logpdf(0.15) > prior.logpdf(0.20) > prior.logpdf(0.30)


def test_lognormal_is_zero_below_zero():
    prior = LogNormal(0.35, 0.7)
    assert prior.logpdf(-1.0) == -math.inf
    assert prior.logpdf(0.0) == -math.inf


def test_lognormal_median_is_its_median(rng):
    prior = LogNormal(0.35, 0.7)
    draws = prior.rvs(rng, 200_000)
    assert float(np.median(draws)) == pytest.approx(0.35, rel=0.02)


def test_lognormal_sigma_is_the_spread_of_the_log(rng):
    prior = LogNormal(0.35, 0.7)
    draws = prior.rvs(rng, 200_000)
    assert float(np.std(np.log(draws))) == pytest.approx(0.7, rel=0.02)


def test_uniform_is_flat_inside_and_forbids_outside():
    prior = Uniform(0.0, 2.0)
    assert prior.logpdf(0.3) == pytest.approx(prior.logpdf(1.7))
    assert prior.logpdf(2.5) == -math.inf


def test_samples_respect_bounds_when_truncated(rng):
    draws = sample(LogNormal(4.0, 0.8), rng, size=20_000, bounds=(1.0, 5.0))
    assert draws.size > 0
    assert np.all(draws >= 1.0) and np.all(draws <= 5.0)


# -- combining them --------------------------------------------------------


def test_log_density_sums_over_named_parameters():
    priors = {"beta": Normal(0.15, 0.05), "alpha": Normal(0.035, 0.01)}
    values = {"beta": 0.15, "alpha": 0.035}
    expected = priors["beta"].logpdf(0.15) + priors["alpha"].logpdf(0.035)
    assert log_density(values, priors) == pytest.approx(expected)


def test_parameters_without_a_prior_contribute_nothing():
    priors = {"beta": Normal(0.15, 0.05)}
    with_extra = log_density({"beta": 0.15, "kappa1": 0.2}, priors)
    alone = log_density({"beta": 0.15}, priors)
    assert with_extra == pytest.approx(alone)


def test_no_priors_at_all_is_a_flat_contribution():
    assert log_density({"beta": 0.15}, None) == 0.0
    assert log_density({"beta": 0.15}, {}) == 0.0


def test_an_impossible_value_forbids_the_whole_set():
    priors = {"beta": LogNormal(0.15, 0.7)}
    assert log_density({"beta": -1.0}, priors) == -math.inf


# -- the contraction diagnostic, on cases with a known answer --------------


def test_a_posterior_that_is_the_prior_contracts_by_nothing(rng):
    prior = Normal(0.0, 1.0)
    echo = prior.rvs(rng, 50_000)
    assert contraction(echo, prior, rng) == pytest.approx(0.0, abs=0.03)


def test_a_ten_times_sharper_posterior_contracts_by_nine_tenths(rng):
    prior = Normal(0.0, 1.0)
    sharp = rng.normal(0.0, 0.1, 50_000)
    assert contraction(sharp, prior, rng) == pytest.approx(0.9, abs=0.03)


def test_a_posterior_wider_than_its_prior_contracts_negatively(rng):
    """Data pulling against the prior widens the answer, and that shows."""
    prior = Normal(0.0, 1.0)
    wide = rng.normal(0.0, 2.0, 50_000)
    assert contraction(wide, prior, rng) < -0.5


def test_contraction_uses_the_truncated_prior(rng):
    """Bounds narrow the prior, so they must narrow the yardstick too.

    A prior clipped hard by its box is already narrow; measuring a posterior
    against the untruncated spread would credit the data for the clipping.
    """
    prior = Normal(0.0, 1.0)
    posterior = rng.normal(0.0, 0.3, 50_000)
    unbounded = contraction(posterior, prior, rng)
    truncated = contraction(posterior, prior, rng, bounds=(-0.5, 0.5))
    assert truncated < unbounded


# -- the shipped scale-up priors -------------------------------------------


def test_every_scaleup_prior_is_a_prior_and_says_why():
    for name, prior in SCALEUP_PRIORS.items():
        assert isinstance(prior, Prior), name
        assert prior.why, f"{name} has no stated provenance"


def test_the_scaleup_priors_admit_the_published_values():
    """A prior that ruled out the paper's own numbers would be too strong."""
    assert math.isfinite(SCALEUP_PRIORS["beta"].logpdf(0.15))


def test_the_ceiling_priors_sit_above_the_published_rates():
    """They encode a scale-up, so their mass belongs above the starting rates."""
    assert SCALEUP_PRIORS["lam_ceiling"].median > 0.0015 * 10
    assert SCALEUP_PRIORS["alpha_ceiling"].median > 0.035 * 10


def test_the_midpoint_priors_land_inside_a_twenty_year_window():
    for name in ("alpha_midpoint", "lam_midpoint"):
        prior = SCALEUP_PRIORS[name]
        assert 0.0 < prior.mu < 20.0
