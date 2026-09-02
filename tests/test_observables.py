"""Observation operators.

The operators are small enough that testing them against a re-implementation
would be circular. Instead they are checked against **identities that must
hold between them** - `plhiv` has to equal `untreated + T`, `art_coverage` has
to equal `on_art / plhiv` - so an error in any one operator breaks a relation
with the others rather than having to be spotted directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import DRC_2020, generate_observations, simulate
from hiv_drc.observables import OBSERVABLES, apply, resolve


@pytest.fixture(scope="module")
def solution():
    return simulate(DRC_2020, t_span=(0.0, 30.0), n_points=31)


# -- the registry -----------------------------------------------------------


def test_every_compartment_is_registered():
    for name in ("S", "I1", "I2", "A", "T", "R"):
        assert name in OBSERVABLES


def test_surveillance_aggregates_are_registered():
    for name in ("plhiv", "diagnosed", "on_art", "prevalence", "art_coverage"):
        assert name in OBSERVABLES


def test_resolve_rejects_unknown_names_and_says_what_exists():
    with pytest.raises(ValueError, match="unknown observable"):
        resolve(("plhiv", "nonsense"))
    try:
        resolve(("nonsense",))
    except ValueError as error:
        assert "plhiv" in str(error), "the error should list what is available"


def test_resolve_accepts_a_custom_operator():
    extra = {"double_T": lambda sol: 2.0 * sol["T"]}
    table = resolve(("double_T",), extra)
    assert "double_T" in table


def test_a_custom_operator_can_shadow_a_standard_one(solution):
    shadowed = apply(solution, ("on_art",), {"on_art": lambda sol: sol["T"] * 0.0})
    assert np.all(shadowed["on_art"] == 0.0)


# -- the operators, checked against each other ------------------------------


def test_compartment_operators_are_the_identity(solution):
    series = apply(solution, ("A", "T"))
    assert series["A"] == pytest.approx(solution["A"])
    assert series["T"] == pytest.approx(solution["T"])


def test_plhiv_is_every_infected_compartment(solution):
    series = apply(solution, ("plhiv", "I1", "I2", "A", "T"))
    expected = series["I1"] + series["I2"] + series["A"] + series["T"]
    assert series["plhiv"] == pytest.approx(expected)


def test_plhiv_splits_into_treated_and_untreated(solution):
    series = apply(solution, ("plhiv", "untreated", "on_art"))
    assert series["plhiv"] == pytest.approx(series["untreated"] + series["on_art"])


def test_plhiv_splits_into_diagnosed_and_undiagnosed(solution):
    series = apply(solution, ("plhiv", "diagnosed", "I1"))
    assert series["plhiv"] == pytest.approx(series["diagnosed"] + series["I1"])


def test_art_coverage_is_the_treated_share(solution):
    series = apply(solution, ("art_coverage", "on_art", "plhiv"))
    assert series["art_coverage"] == pytest.approx(series["on_art"] / series["plhiv"])
    assert np.all(series["art_coverage"] >= 0.0)
    assert np.all(series["art_coverage"] <= 1.0)


def test_prevalence_is_the_infected_share(solution):
    series = apply(solution, ("prevalence", "plhiv", "population"))
    assert series["prevalence"] == pytest.approx(series["plhiv"] / series["population"])


def test_population_is_the_sum_of_all_six(solution):
    series = apply(solution, ("population", "S", "I1", "I2", "A", "T", "R"))
    total = sum(series[name] for name in ("S", "I1", "I2", "A", "T", "R"))
    assert series["population"] == pytest.approx(total)


# -- generating and fitting through an operator -----------------------------


def test_observations_can_be_generated_from_aggregates():
    obs = generate_observations(observed=("plhiv", "on_art"), noise=0.0)
    assert set(obs.names) == {"plhiv", "on_art"}
    solution = simulate(DRC_2020, t_span=(0.0, 30.0), t_eval=obs.t)
    assert obs.truth["plhiv"] == pytest.approx(solution.infected, rel=1e-8)


def test_compartment_observations_are_unchanged_by_the_operator_layer():
    """The pre-existing default must behave exactly as it did before."""
    obs = generate_observations(noise=0.0)
    solution = simulate(DRC_2020, t_span=(0.0, 30.0), t_eval=obs.t)
    assert obs.names == ("A", "T")
    for name in ("A", "T"):
        assert obs.truth[name] == pytest.approx(solution[name], rel=1e-9)


def test_names_order_is_deterministic_across_mixed_kinds():
    obs = generate_observations(observed=("plhiv", "T", "art_coverage", "A"), noise=0.0)
    # compartments first in model order, then the aggregates alphabetically
    assert obs.names == ("A", "T", "art_coverage", "plhiv")


def test_a_custom_observable_survives_into_the_fit():
    """An operator supplied at generation time must still resolve at predict time."""
    from hiv_drc import estimate_parameters

    extra = {"half_plhiv": lambda sol: 0.5 * sol.infected}
    obs = generate_observations(observed=("half_plhiv", "on_art"), noise=0.0, observables=extra)
    fit = estimate_parameters(obs, fit=("alpha",))
    assert fit.estimates["alpha"] == pytest.approx(DRC_2020.alpha, rel=1e-3)


def test_fitting_through_published_aggregates_recovers_the_rates():
    """The decisive test for real data: UNAIDS-shaped inputs must identify the rates."""
    from hiv_drc import estimate_parameters

    obs = generate_observations(observed=("plhiv", "on_art"), noise=0.0)
    fit = estimate_parameters(obs, fit=("beta", "alpha"))
    for name, value in fit.relative_errors().items():
        assert abs(value) < 1e-3, f"{name} off by {value}%"
