"""Sensitivity indices, including the two that have exact closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import (
    R0_PARAMETERS,
    global_sensitivity,
    local_sensitivity_index,
    local_sensitivity_indices,
    prcc,
)
from hiv_drc.sensitivity import latin_hypercube

from .conftest import perturbed


def test_beta_index_is_exactly_one(baseline, rng):
    """R0 is proportional to beta, so its normalised index is +1 everywhere."""
    for _ in range(30):
        p = perturbed(baseline, rng)
        assert local_sensitivity_index(p, "beta") == pytest.approx(1.0, abs=1e-6)


def test_phi_index_has_a_closed_form(baseline, rng):
    """phi enters only through mu/(mu+phi), giving an index of -phi/(mu+phi)."""
    for _ in range(30):
        p = perturbed(baseline, rng)
        expected = -p.phi / (p.mu + p.phi)
        assert local_sensitivity_index(p, "phi") == pytest.approx(expected, abs=1e-6)


def test_indices_cover_every_r0_parameter(baseline):
    indices = local_sensitivity_indices(baseline)
    assert set(indices) == set(R0_PARAMETERS)
    assert all(np.isfinite(v) for v in indices.values())


def test_indices_are_sorted_by_magnitude(baseline):
    magnitudes = [abs(v) for v in local_sensitivity_indices(baseline).values()]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_latin_hypercube_respects_its_bounds(baseline):
    spread = 0.25
    samples = latin_hypercube(baseline, R0_PARAMETERS, n_samples=200, spread=spread)

    assert samples.shape == (200, len(R0_PARAMETERS))
    for j, name in enumerate(R0_PARAMETERS):
        centre = getattr(baseline, name)
        assert samples[:, j].min() >= centre * (1 - spread) - 1e-12
        assert samples[:, j].max() <= centre * (1 + spread) + 1e-12


def test_latin_hypercube_is_reproducible(baseline):
    first = latin_hypercube(baseline, R0_PARAMETERS, n_samples=50, seed=1)
    second = latin_hypercube(baseline, R0_PARAMETERS, n_samples=50, seed=1)
    np.testing.assert_array_equal(first, second)


def test_prcc_recovers_a_known_monotone_relationship():
    """A synthetic case where the answer is known by construction.

    y depends positively on x0, negatively on x1, and not at all on x2.  The
    nonlinearity is monotone, which is exactly the regime rank correlation is
    meant for.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 2.0, size=(600, 3))
    y = X[:, 0] ** 3 - 2.0 * np.exp(X[:, 1])

    coefficients, p_values = prcc(X, y)

    assert coefficients[0] > 0.9
    assert coefficients[1] < -0.9
    assert abs(coefficients[2]) < 0.2
    assert p_values[0] < 1e-6
    assert p_values[2] > 0.01


def test_global_screen_ranks_beta_and_phi_first(baseline):
    """The global screen should agree with the paper on which rates dominate."""
    result = global_sensitivity(baseline, n_samples=300, spread=0.25, seed=7)
    ranked = [name for name, *_ in result.table("R0")]

    assert set(ranked[:2]) == {"beta", "phi"}

    coefficients = {name: value for name, value, *_ in result.table("R0")}
    assert coefficients["beta"] > 0    # more contact raises R0
    assert coefficients["phi"] < 0     # behaviour change lowers it
    assert result.outputs["R0"].shape == (300,)
