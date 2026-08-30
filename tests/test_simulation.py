"""The integration wrapper and the Solution convenience API."""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import DRC_2020, INITIAL_STATE, simulate
from hiv_drc.analysis import critical_beta, r0_grid
from hiv_drc.reproduction import reproduction_number


def test_shapes_and_initial_condition(baseline):
    solution = simulate(baseline, INITIAL_STATE, (0.0, 50.0), n_points=201)

    assert solution.t.shape == (201,)
    assert solution.y.shape == (6, 201)
    np.testing.assert_allclose(solution.y[:, 0], INITIAL_STATE, atol=1e-10)


def test_compartments_are_addressable_by_name(baseline):
    solution = simulate(baseline, INITIAL_STATE, (0.0, 10.0), n_points=11)

    np.testing.assert_array_equal(solution["S"], solution.y[0])
    np.testing.assert_array_equal(solution["R"], solution.y[5])
    np.testing.assert_allclose(solution.infected, solution.y[1:5].sum(axis=0))

    with pytest.raises(KeyError):
        solution["nope"]


def test_solvers_agree(baseline):
    """LSODA, RK45 and Radau should land in the same place.

    Disagreement here would mean the tolerances are too loose for the stiffness
    of the system, which would quietly corrupt every downstream analysis.
    """
    reference = simulate(baseline, INITIAL_STATE, (0.0, 200.0), n_points=2, method="LSODA")
    for method in ("RK45", "Radau"):
        other = simulate(baseline, INITIAL_STATE, (0.0, 200.0), n_points=2, method=method)
        np.testing.assert_allclose(other.y[:, -1], reference.y[:, -1], rtol=1e-6)


def test_interpolation_at_a_time_point(baseline):
    solution = simulate(baseline, INITIAL_STATE, (0.0, 50.0), n_points=501)
    np.testing.assert_allclose(solution.at(0.0), INITIAL_STATE, atol=1e-9)
    assert solution.at(25.0).shape == (6,)


def test_bad_method_is_reported(baseline):
    with pytest.raises(ValueError):
        simulate(baseline, INITIAL_STATE, (0.0, 1.0), method="not-a-solver")


def test_critical_beta_puts_R0_at_one(baseline):
    beta_c = critical_beta(baseline)
    assert reproduction_number(baseline.replace(beta=beta_c)).R0 == pytest.approx(1.0, abs=1e-10)
    assert beta_c < baseline.beta  # the DRC baseline sits above the threshold


def test_r0_grid_is_monotone(baseline):
    """R0 rises with beta and falls with sigma1, everywhere on the grid."""
    beta, sigma1, grid = r0_grid(
        baseline, beta=np.linspace(0.05, 0.35, 12), sigma1=np.linspace(0.001, 0.05, 12)
    )

    assert grid.shape == (sigma1.size, beta.size)
    assert np.all(np.diff(grid, axis=1) > 0)   # increasing in beta
    assert np.all(np.diff(grid, axis=0) < 0)   # decreasing in sigma1


def test_module_level_defaults_are_the_paper_values():
    assert DRC_2020.beta == 0.15
    assert DRC_2020.Lambda == 8.8261
    np.testing.assert_allclose(INITIAL_STATE, [88.516, 0.014, 0.0846, 0.30832, 0.07708, 0.0])
