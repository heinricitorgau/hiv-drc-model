"""Loading real surveillance data.

No test here touches the network - downloading lives in
``scripts/fetch_worldbank.py`` and its output is committed, so these run
offline and against a fixed snapshot rather than against whatever the API
returns today.

The unit conversion is the thing most worth pinning down: the published
counts are people and the model counts millions, and getting that wrong does
not raise, it silently rescales the epidemic by a factor of a million.
"""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import COMPARTMENTS, estimate_parameters
from hiv_drc.realdata import (
    PAPER_UNTREATED_SPLIT,
    initial_state_from_data,
    load_worldbank,
)

DATA = "data/real/drc_worldbank.csv"


@pytest.fixture
def messy_csv(tmp_path):
    """A file with the gaps a real download has: missing coverage in early years."""
    path = tmp_path / "messy.csv"
    path.write_text(
        "year,plhiv_adults,art_coverage_pct,population\n"
        "1998,600000,,49000000\n"      # no coverage reported
        "1999,610000,,50000000\n"      # no coverage reported
        "2000,620000,0,51000000\n"
        "2001,620000,1,52000000\n"
        "2002,610000,2,53000000\n",
        encoding="utf-8",
    )
    return path


# -- loading ----------------------------------------------------------------


def test_counts_are_converted_to_millions(messy_csv):
    obs, context = load_worldbank(messy_csv)
    assert obs.values["plhiv"][0] == pytest.approx(0.62)       # 620,000 people
    assert context["population"][0] == pytest.approx(51.0)     # 51,000,000 people


def test_percentages_are_converted_to_fractions(messy_csv):
    obs, _ = load_worldbank(messy_csv)
    assert obs.values["art_coverage"][-1] == pytest.approx(0.02)  # 2%


def test_rows_missing_a_requested_series_are_dropped(messy_csv):
    obs, context = load_worldbank(messy_csv)
    assert obs.n_points == 3                     # 1998 and 1999 have no coverage
    assert list(context["year"]) == [2000, 2001, 2002]


def test_time_is_rebased_to_the_first_kept_year(messy_csv):
    obs, _ = load_worldbank(messy_csv)
    assert obs.t[0] == 0.0
    assert obs.t == pytest.approx([0.0, 1.0, 2.0])


def test_a_series_that_needs_no_coverage_keeps_the_early_years(messy_csv):
    obs, context = load_worldbank(messy_csv, observed=("plhiv",))
    assert obs.n_points == 5
    assert context["year"][0] == 1998


def test_on_art_is_derived_from_coverage(messy_csv):
    obs, _ = load_worldbank(messy_csv, observed=("plhiv", "on_art"))
    expected = obs.values["plhiv"] * np.array([0.0, 0.01, 0.02])
    assert obs.values["on_art"] == pytest.approx(expected)


def test_an_empty_window_raises_rather_than_fitting_nothing(messy_csv):
    with pytest.raises(ValueError, match="no rows"):
        load_worldbank(messy_csv, first_year=2050)


def test_unbuildable_series_are_rejected(messy_csv):
    with pytest.raises(ValueError, match="cannot build"):
        load_worldbank(messy_csv, observed=("plhiv", "new_infections"))


def test_real_data_carries_no_ground_truth(messy_csv):
    """Real data has nothing to score against, and must say so."""
    obs, _ = load_worldbank(messy_csv)
    assert obs.truth is None
    assert obs.parameters is None
    fit = estimate_parameters(obs, fit=("alpha",))
    assert fit.truth is None
    with pytest.raises(ValueError, match="no ground truth"):
        fit.relative_errors()


def test_the_committed_snapshot_loads():
    obs, context = load_worldbank(DATA, first_year=2005)
    assert obs.names == ("art_coverage", "plhiv")
    assert obs.n_points == 20
    assert context["year"][0] == 2005
    # sanity: DRC prevalence is around half a million, not half a person
    assert 0.3 < float(np.mean(obs.values["plhiv"])) < 1.0
    assert np.all(obs.values["art_coverage"] <= 1.0)


# -- the initial state ------------------------------------------------------


def test_initial_state_reproduces_the_observed_quantities():
    state = initial_state_from_data(plhiv=0.510, art_coverage=0.53, population=95.99)
    assert state.shape == (len(COMPARTMENTS),)
    assert float(state.sum()) == pytest.approx(95.99)
    assert float(state[4]) == pytest.approx(0.510 * 0.53)          # T
    assert float(state[1:5].sum()) == pytest.approx(0.510)         # PLHIV
    assert float(state[0]) == pytest.approx(95.99 - 0.510)         # S


def test_the_untreated_split_is_honoured():
    split = {"I1": 0.5, "I2": 0.3, "A": 0.2}
    state = initial_state_from_data(1.0, 0.4, 100.0, untreated_split=split)
    untreated = 1.0 - 0.4
    assert float(state[1]) == pytest.approx(untreated * 0.5)
    assert float(state[2]) == pytest.approx(untreated * 0.3)
    assert float(state[3]) == pytest.approx(untreated * 0.2)


def test_the_paper_split_sums_to_one():
    assert sum(PAPER_UNTREATED_SPLIT.values()) == pytest.approx(1.0)


def test_behaviour_changed_comes_out_of_the_susceptible_pool():
    state = initial_state_from_data(1.0, 0.5, 100.0, behaviour_changed=10.0)
    assert float(state[5]) == pytest.approx(10.0)
    assert float(state[0]) == pytest.approx(100.0 - 1.0 - 10.0)
    assert float(state.sum()) == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"plhiv": 1.0, "art_coverage": 1.5, "population": 10.0}, "fraction"),
        ({"plhiv": -1.0, "art_coverage": 0.5, "population": 10.0}, "non-negative"),
        ({"plhiv": 20.0, "art_coverage": 0.5, "population": 10.0}, "exceeds"),
    ],
)
def test_invalid_initial_states_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        initial_state_from_data(**kwargs)


def test_an_untreated_split_that_does_not_sum_to_one_is_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        initial_state_from_data(1.0, 0.5, 10.0, untreated_split={"I1": 0.5, "I2": 0.2, "A": 0.2})


def test_an_incomplete_untreated_split_is_rejected():
    with pytest.raises(ValueError, match="needs entries"):
        initial_state_from_data(1.0, 0.5, 10.0, untreated_split={"I1": 1.0})


# -- the finding the README reports -----------------------------------------


def test_table_2_disagrees_with_the_2020_data_on_treatment():
    """Pins down the discrepancy the README and realdata's docstring describe.

    Not a defect in this package - a property of the published initial
    conditions - but one that silently biases any real-data fit, so it is
    worth failing loudly if a future edit quietly 'fixes' Table 2.
    """
    from hiv_drc import INITIAL_STATE

    obs, context = load_worldbank(DATA, first_year=2020, last_year=2020)
    real_coverage = float(obs.values["art_coverage"][0])

    table2_plhiv = float(INITIAL_STATE[1:5].sum())
    table2_coverage = float(INITIAL_STATE[4]) / table2_plhiv

    assert real_coverage == pytest.approx(0.53)
    assert table2_coverage == pytest.approx(0.159, abs=0.002)
    assert real_coverage / table2_coverage > 3.0
