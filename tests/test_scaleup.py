"""Time-varying rates.

The point of these tests is mostly negative: a scale-up must change nothing
at all unless it is switched on, and it must *stop* the constant-coefficient
theory from returning numbers that would look fine and mean nothing.

The one positive result worth pinning down is the structural ceiling: with
the published diagnosis rate, no treatment-uptake rate however large can push
ART coverage past about 56%, because alpha drains a queue that only lambda
fills. That is the finding behind the README's scale-up section, and it would
be easy to lose in a future refactor.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import DRC_2020, INITIAL_STATE, reproduction_number, rhs, simulate
from hiv_drc.reproduction import reproduction_number_at


@pytest.fixture
def ramp():
    """A treatment scale-up from 0.01 to 0.5 per year, half way at t = 10."""
    return DRC_2020.replace(
        alpha=0.01, alpha_ceiling=0.5, alpha_midpoint=10.0, alpha_rate=0.4
    )


# -- switched off, nothing changes -----------------------------------------


def test_the_baseline_has_no_scaleup():
    assert not DRC_2020.is_time_varying
    assert DRC_2020.alpha_ceiling is None
    assert DRC_2020.lam_ceiling is None


def test_alpha_at_is_the_constant_when_no_scaleup_is_set():
    for t in (0.0, 5.0, 1000.0):
        assert DRC_2020.alpha_at(t) == DRC_2020.alpha
        assert DRC_2020.lam_at(t) == DRC_2020.lam


def test_the_system_stays_autonomous_without_a_scaleup():
    """The published model must not acquire a time dependence by accident."""
    early = rhs(0.0, INITIAL_STATE, DRC_2020)
    late = rhs(250.0, INITIAL_STATE, DRC_2020)
    assert np.array_equal(early, late)


def test_the_published_r0_is_unchanged():
    assert reproduction_number(DRC_2020).R0 == pytest.approx(1.2145, abs=5e-5)


# -- switched on, it ramps --------------------------------------------------


def test_the_ramp_runs_from_floor_to_ceiling(ramp):
    assert ramp.alpha_at(-1e3) == pytest.approx(ramp.alpha, abs=1e-9)
    assert ramp.alpha_at(1e3) == pytest.approx(ramp.alpha_ceiling, abs=1e-9)


def test_the_midpoint_is_half_way(ramp):
    midpoint_value = ramp.alpha_at(ramp.alpha_midpoint)
    assert midpoint_value == pytest.approx(0.5 * (ramp.alpha + ramp.alpha_ceiling))


def test_the_ramp_is_monotonic(ramp):
    values = [ramp.alpha_at(t) for t in np.linspace(-5.0, 30.0, 60)]
    assert np.all(np.diff(values) > 0.0)


def test_a_steeper_rate_ramps_faster(ramp):
    steep = ramp.replace(alpha_rate=2.0)
    just_after = ramp.alpha_midpoint + 1.0
    assert steep.alpha_at(just_after) > ramp.alpha_at(just_after)


def test_a_large_negative_exponent_does_not_overflow(ramp):
    """exp() overflows long before the limit stops being well defined."""
    value = ramp.alpha_at(-1e6)
    assert np.isfinite(value)
    assert value == pytest.approx(ramp.alpha)


def test_the_system_becomes_non_autonomous(ramp):
    early = rhs(0.0, INITIAL_STATE, ramp)
    late = rhs(30.0, INITIAL_STATE, ramp)
    assert not np.array_equal(early, late)


def test_a_scaleup_actually_moves_people_onto_treatment(ramp):
    flat = simulate(DRC_2020.replace(alpha=0.01), INITIAL_STATE, (0.0, 30.0), n_points=31)
    rising = simulate(ramp, INITIAL_STATE, (0.0, 30.0), n_points=31)
    assert rising["T"][-1] > flat["T"][-1]


def test_lambda_scales_up_the_same_way():
    ramp = DRC_2020.replace(lam=0.001, lam_ceiling=0.4, lam_midpoint=8.0, lam_rate=0.5)
    assert ramp.is_time_varying
    assert ramp.lam_at(-1e3) == pytest.approx(0.001, abs=1e-9)
    assert ramp.lam_at(1e3) == pytest.approx(0.4, abs=1e-9)
    assert ramp.lam_at(8.0) == pytest.approx(0.5 * (0.001 + 0.4))


# -- the constant-coefficient theory refuses rather than misleads -----------


def test_r0_refuses_a_time_varying_parameter_set(ramp):
    with pytest.raises(ValueError, match="not defined for a time-varying"):
        reproduction_number(ramp)


def test_the_derived_exit_rates_refuse_too(ramp):
    with pytest.raises(ValueError, match="a2 is not a constant"):
        _ = ramp.a2
    lam_ramp = DRC_2020.replace(lam_ceiling=0.4)
    with pytest.raises(ValueError, match="a1 is not a constant"):
        _ = lam_ramp.a1


def test_the_instantaneous_r0_freezes_the_rates(ramp):
    """R0 at time t must equal R0 of the constant model with alpha = alpha(t)."""
    for t in (0.0, 10.0, 25.0):
        frozen = DRC_2020.replace(alpha=ramp.alpha_at(t))
        assert reproduction_number_at(ramp, t).R0 == pytest.approx(
            reproduction_number(frozen).R0
        )


def test_the_instantaneous_r0_agrees_with_the_constant_case():
    for t in (0.0, 40.0):
        assert reproduction_number_at(DRC_2020, t).R0 == pytest.approx(
            reproduction_number(DRC_2020).R0
        )


# -- the structural ceiling -------------------------------------------------


def test_alpha_saturates_because_lambda_starves_it():
    """No treatment rate can beat the queue that feeds it.

    With the published diagnosis rate, ART coverage stops responding to alpha
    well before it reaches what the DRC actually achieved (71% by 2024). This
    is why a time-varying alpha alone does not rescue the real-data fit.
    """
    observed_2024 = 0.71  # DRC ART coverage, UNAIDS
    coverage = {}
    for alpha in (5.0, 50.0, 500.0):
        solution = simulate(
            DRC_2020.replace(alpha=alpha), INITIAL_STATE, (0.0, 19.0), n_points=40
        )
        coverage[alpha] = float((solution["T"] / solution.infected)[-1])

    # a hundredfold increase in alpha buys almost nothing
    assert coverage[500.0] - coverage[5.0] < 0.02
    # and the ceiling still falls short of what the DRC actually reached.
    # Where exactly it sits depends on the initial state - about 61% from
    # Table 2, about 56% from the real 2005 numbers - but never 71%.
    assert coverage[500.0] < observed_2024 - 0.05


def test_raising_lambda_lifts_the_ceiling():
    """The complement: the bottleneck really is diagnosis, not treatment."""
    starved = simulate(
        DRC_2020.replace(alpha=0.5), INITIAL_STATE, (0.0, 19.0), n_points=40
    )
    fed = simulate(
        DRC_2020.replace(alpha=0.5, lam=0.2), INITIAL_STATE, (0.0, 19.0), n_points=40
    )
    starved_coverage = float((starved["T"] / starved.infected)[-1])
    fed_coverage = float((fed["T"] / fed.infected)[-1])
    assert fed_coverage > starved_coverage + 0.2
