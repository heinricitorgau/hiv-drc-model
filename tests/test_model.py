"""Structural properties of the vector field.

Nothing here compares one implementation against another; each test checks the
model against a property that has to hold on mathematical grounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import INITIAL_STATE, jacobian, rhs, simulate, total_population
from hiv_drc.model import force_of_infection

from .conftest import perturbed


def test_population_balance(baseline, rng):
    """Summing the six equations must give dN/dt = Lambda - mu N - d1 A - d2 T.

    This catches the most likely kind of typo in the vector field: a term that
    leaves one compartment without arriving in another.
    """
    for _ in range(200):
        state = rng.uniform(1e-3, 100.0, size=6)
        derivative = rhs(0.0, state, baseline)
        expected = (
            baseline.Lambda
            - baseline.mu * state.sum()
            - baseline.delta1 * state[3]
            - baseline.delta2 * state[4]
        )
        assert derivative.sum() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_infection_term_is_shared_between_S_and_I1(baseline, rng):
    """Everyone who leaves S through infection has to arrive in I1."""
    for _ in range(50):
        state = rng.uniform(1e-3, 100.0, size=6)
        derivative = rhs(0.0, state, baseline)
        infection = force_of_infection(state, baseline)

        leaving_S = baseline.Lambda - (baseline.mu + baseline.phi) * state[0] - derivative[0]
        entering_I1 = derivative[1] + baseline.a1 * state[1]

        assert leaving_S == pytest.approx(infection, rel=1e-12)
        assert entering_I1 == pytest.approx(infection, rel=1e-12)


def test_no_infection_without_infectives(baseline):
    """With every infected class empty, no new infections can occur."""
    state = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 20.0])
    assert force_of_infection(state, baseline) == 0.0
    assert rhs(0.0, state, baseline)[1] == 0.0


def test_jacobian_matches_finite_differences(baseline):
    """Complex-step derivatives against a plain central difference."""
    state = np.array([80.0, 0.5, 0.4, 0.3, 0.2, 30.0])
    analytic = jacobian(state, baseline)

    numerical = np.empty((6, 6))
    for j in range(6):
        h = 1e-6 * max(1.0, abs(state[j]))
        up, down = state.copy(), state.copy()
        up[j] += h
        down[j] -= h
        numerical[:, j] = (rhs(0.0, up, baseline) - rhs(0.0, down, baseline)) / (2 * h)

    np.testing.assert_allclose(analytic, numerical, atol=1e-7)


def test_jacobian_is_finite_over_random_parameters(baseline, rng):
    for _ in range(30):
        p = perturbed(baseline, rng)
        state = rng.uniform(1e-3, 100.0, size=6)
        assert np.all(np.isfinite(jacobian(state, p)))


def test_solution_stays_non_negative_and_bounded(baseline):
    """The biologically meaningful region has to be forward invariant.

    N is bounded above by Lambda/mu because dN/dt = Lambda - mu N - (deaths),
    and no compartment can go negative because every outflow carries a factor
    of the compartment it leaves.
    """
    solution = simulate(baseline, INITIAL_STATE, (0.0, 3000.0), n_points=3001)

    assert solution.y.min() > -1e-8
    assert solution.total_population.max() < baseline.carrying_capacity + 1e-6


def test_total_population_helper(baseline):
    assert total_population(INITIAL_STATE) == pytest.approx(INITIAL_STATE.sum())
