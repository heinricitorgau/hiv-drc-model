"""The closed form for R0 against an independently assembled next-generation matrix."""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import next_generation_matrices, r0_from_ngm, reproduction_number
from hiv_drc.reproduction import susceptible_fraction

from .conftest import perturbed


def test_matches_the_published_value(baseline):
    """Table 3 of the paper reports R0 = 1.2145."""
    assert reproduction_number(baseline).R0 == pytest.approx(1.2145, abs=1e-4)


def test_closed_form_matches_spectral_radius(baseline, rng):
    """Equations (3.9)-(3.12) against rho(F V^-1), over random parameters.

    The closed form is the only place in the package where the paper's algebra
    is trusted rather than derived, so it gets checked against the definition
    of R0 rather than against itself.
    """
    for _ in range(300):
        p = perturbed(baseline, rng, spread=0.6)
        assert r0_from_ngm(p) == pytest.approx(reproduction_number(p).R0, rel=1e-10)


def test_decomposition_sums_to_R0(baseline):
    result = reproduction_number(baseline)
    assert result.R0 == pytest.approx(
        baseline.beta * result.theta * (result.R1 + result.R2 + result.R3), rel=1e-14
    )


def test_R1_is_the_mean_time_in_I1(baseline):
    """R1 is 1/a1: the average duration of the unaware-infective stage."""
    assert reproduction_number(baseline).R1 == pytest.approx(1.0 / baseline.a1, rel=1e-14)


def test_determinant_of_V_matches_the_papers_D(baseline, rng):
    """The denominator D in (3.10)-(3.11) is exactly det(V)."""
    for _ in range(100):
        p = perturbed(baseline, rng)
        _, V = next_generation_matrices(p)
        D = p.a1 * (p.a2 * p.a3 * p.a4 - p.kappa1 * p.kappa2 * p.sigma2 - p.alpha * p.kappa2 * p.a3)
        assert np.linalg.det(V) == pytest.approx(D, rel=1e-12)


def test_R0_is_linear_in_beta(baseline):
    """Doubling the contact rate has to double R0."""
    base = reproduction_number(baseline).R0
    doubled = reproduction_number(baseline.replace(beta=2 * baseline.beta)).R0
    assert doubled == pytest.approx(2 * base, rel=1e-14)


def test_susceptible_fraction(baseline):
    assert susceptible_fraction(baseline) == pytest.approx(
        baseline.mu / (baseline.mu + baseline.phi), rel=1e-14
    )


def test_F_has_a_single_nonzero_row(baseline):
    """Only susceptibles get infected, and they all land in I1."""
    F, _ = next_generation_matrices(baseline)
    assert np.count_nonzero(F[1:]) == 0
    assert np.count_nonzero(F[0]) == 3  # I1, I2 and A all transmit
