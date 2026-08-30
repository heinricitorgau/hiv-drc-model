"""Equilibria, and the threshold theorem relating stability to R0."""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import (
    disease_free_equilibrium,
    endemic_equilibrium,
    reproduction_number,
    rhs,
)

from .conftest import perturbed


def test_disease_free_equilibrium_is_a_root(baseline, rng):
    for _ in range(50):
        p = perturbed(baseline, rng)
        dfe = disease_free_equilibrium(p)
        assert dfe.residual < 1e-11
        assert np.max(np.abs(rhs(0.0, dfe.state, p))) < 1e-11


def test_disease_free_population_is_the_carrying_capacity(baseline):
    """With no disease deaths, N settles at exactly Lambda/mu."""
    dfe = disease_free_equilibrium(baseline)
    assert dfe.total_population == pytest.approx(baseline.carrying_capacity, rel=1e-12)
    assert dfe.infected == 0.0


def test_threshold_theorem(baseline):
    """R0 < 1 iff the disease-free equilibrium is locally asymptotically stable.

    This is the paper's main local-stability result.  Rather than trusting a
    single parameter set, sweep beta across the threshold and require the sign
    of the dominant eigenvalue to track the sign of R0 - 1 at every point.
    """
    for beta in np.linspace(0.04, 0.30, 40):
        p = baseline.replace(beta=beta)
        R0 = reproduction_number(p).R0
        if abs(R0 - 1.0) < 1e-6:  # the threshold itself needs a centre-manifold argument
            continue
        dominant = disease_free_equilibrium(p).dominant_eigenvalue.real
        assert np.sign(dominant) == np.sign(R0 - 1.0), f"beta={beta:.4f}, R0={R0:.6f}"


def test_endemic_equilibrium_is_positive_and_stable(baseline):
    """With the DRC baseline (R0 = 1.2145) the endemic state exists and attracts."""
    assert reproduction_number(baseline).R0 > 1.0

    ee = endemic_equilibrium(baseline)
    assert ee.endemic
    assert ee.residual < 1e-10
    assert ee.state.min() > 0.0
    assert ee.is_stable
    assert ee.total_population < baseline.carrying_capacity


def test_subcritical_flow_reaches_the_disease_free_state(baseline):
    """Below the threshold the infection dies out rather than settling somewhere else.

    This also rules out a backward bifurcation at these parameters: if a stable
    endemic branch existed for R0 < 1, the trajectory could land on it instead.
    """
    p = baseline.replace(beta=0.05)
    assert reproduction_number(p).R0 < 1.0

    attractor = endemic_equilibrium(p)
    assert not attractor.endemic
    np.testing.assert_allclose(attractor.state, disease_free_equilibrium(p).state, atol=1e-6)


def test_eigenvalues_are_sorted(baseline):
    eigenvalues = disease_free_equilibrium(baseline).eigenvalues
    assert np.all(np.diff(eigenvalues.real) <= 1e-15)
