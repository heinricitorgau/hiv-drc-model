"""Turning published surveillance data into something the estimator can fit.

:mod:`hiv_drc.synthetic` makes data whose answer is known.  This module reads
data whose answer is not, from the World Bank's republication of the UNAIDS
country estimates (fetch it with ``scripts/fetch_worldbank.py``).

Nothing here touches the network.  Downloading is a separate script, so that
loading, unit conversion and the initial-state construction stay pure and
testable offline - and so that a fit is reproducible from a committed CSV
rather than from whatever the API returned that day.

Three things stand between a downloaded CSV and a fittable
:class:`~hiv_drc.synthetic.Observations`, and each is a decision rather than a
formality:

**Units.** The published counts are people; the model counts millions. Getting
this wrong does not raise, it silently rescales the epidemic by :math:`10^6`.

**What is actually measured.** The published series are ``plhiv`` and
``art_coverage``, not compartments - see :mod:`hiv_drc.observables`. The
observability study in the README found that pairing is close to the best
available for identifying both rates, which is fortunate, since it is also all
there is.

**The initial state.** This is the hard one. The model needs all six
compartments at :math:`t_0`; the data constrain three numbers (population,
PLHIV, ART coverage). The remaining three degrees of freedom - how the
untreated infected split between undiagnosed :math:`I_1`, diagnosed
:math:`I_2` and symptomatic :math:`A`, and how many people sit in the
behaviour-changed class :math:`R` - are **assumptions, and are not in the
data**.  :func:`initial_state_from_data` makes them explicit arguments rather
than burying them.

A warning about the published initial conditions
------------------------------------------------
The paper's Table 2 is labelled DRC 2020, but it does not reconcile with the
UNAIDS estimates for that year:

=====================  ===============  ==========================
Quantity               Table 2          UNAIDS / World Bank
=====================  ===============  ==========================
Total population       89.00 million    95.99 million (2020)
People living w/ HIV   0.484 million    0.510 million (2020)
On treatment           0.077 million    0.270 million (2020)
ART coverage           15.9%            53% (2020)
=====================  ===============  ==========================

Population and PLHIV are within a few percent, but the treatment compartment
is off by a factor of 3.5. Table 2's ART coverage of 15.9% matches DRC in
**2013** (14%), not 2020. Since :math:`T` is the series that identifies the
treatment-uptake rate :math:`\\alpha`, starting a real-data fit from Table 2
pushes that error straight into the estimate of the parameter the fit exists
to recover. Use :func:`initial_state_from_data` instead.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .synthetic import Observations

__all__ = [
    "PAPER_UNTREATED_SPLIT",
    "load_worldbank",
    "initial_state_from_data",
]

#: How Table 2 splits the untreated infected between :math:`I_1`, :math:`I_2`
#: and :math:`A`, as fractions summing to one.  Used as the default only
#: because the paper offers nothing better - it is not an estimate from data,
#: and its implication that three quarters of untreated infections are already
#: symptomatic is on its face implausible for a country with an active
#: treatment programme.  Override it.
PAPER_UNTREATED_SPLIT: dict[str, float] = {
    "I1": 0.014 / 0.40692,
    "I2": 0.0846 / 0.40692,
    "A": 0.30832 / 0.40692,
}


def load_worldbank(
    path: str | Path = "data/real/drc_worldbank.csv",
    observed: tuple[str, ...] = ("plhiv", "art_coverage"),
    first_year: int | None = None,
    last_year: int | None = None,
) -> tuple[Observations, dict[str, NDArray]]:
    """Read the downloaded indicators into observations the estimator can fit.

    Rows missing any requested series are dropped, and time is re-based so
    ``t = 0`` is the first year kept.

    Parameters
    ----------
    path:
        CSV written by ``scripts/fetch_worldbank.py``.
    observed:
        Which series to fit. ``"plhiv"`` (millions) and ``"art_coverage"``
        (a fraction) are both registered observables, so they need no special
        handling downstream.
    first_year, last_year:
        Restrict the window. ART coverage is only published from 2000, and is
        near zero until the mid-2000s, so the default window starts wherever
        the data does.

    Returns
    -------
    observations:
        With ``truth=None`` and ``parameters=None`` - this is real data, so
        there is nothing to score against, and the scoring methods on
        :class:`~hiv_drc.estimation.FitResult` will correctly refuse.
    context:
        The columns not being fitted but needed to set up the problem, keyed
        by name: ``"year"`` and ``"population"`` in millions.

    Raises
    ------
    ValueError
        If no rows survive the filtering, rather than returning an empty fit.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no data rows")

    # `on_art` is derived as plhiv * coverage, so it needs the coverage column
    # just as much as `art_coverage` does - a row missing coverage would
    # otherwise survive the filter and produce a silent nan.
    needs_coverage = bool({"art_coverage", "on_art"} & set(observed))

    kept = []
    for row in rows:
        year = int(row["year"])
        if first_year is not None and year < first_year:
            continue
        if last_year is not None and year > last_year:
            continue
        if not row["plhiv_adults"] or not row["population"]:
            continue
        if needs_coverage and not row["art_coverage_pct"]:
            continue
        kept.append(row)

    if not kept:
        raise ValueError(
            f"no rows in {path} have every requested series "
            f"{observed} within {first_year}-{last_year}"
        )

    years = np.array([int(row["year"]) for row in kept], dtype=float)
    plhiv = np.array([float(row["plhiv_adults"]) for row in kept]) / 1e6
    population = np.array([float(row["population"]) for row in kept]) / 1e6
    coverage = np.array(
        [float(row["art_coverage_pct"]) / 100.0 if row["art_coverage_pct"] else np.nan
         for row in kept]
    )

    available = {
        "plhiv": plhiv,
        "art_coverage": coverage,
        "on_art": plhiv * coverage,
        "population": population,
    }
    unknown = set(observed) - set(available)
    if unknown:
        raise ValueError(
            f"cannot build {sorted(unknown)} from this file; "
            f"available: {sorted(available)}"
        )

    observations = Observations(
        t=years - years[0],
        values={name: available[name] for name in observed},
    )
    context = {"year": years, "population": population}
    return observations, context


def initial_state_from_data(
    plhiv: float,
    art_coverage: float,
    population: float,
    untreated_split: dict[str, float] | None = None,
    behaviour_changed: float = 0.0,
) -> NDArray:
    """Build a six-compartment initial state consistent with observed data.

    The data pin down three numbers; the model needs six. What the data give:

    * ``T = plhiv * art_coverage``
    * the untreated infected, ``plhiv - T``, whose total is known
    * ``S = population - plhiv - R``

    What they do not give, and what therefore has to be assumed:

    * how the untreated infected divide between :math:`I_1` (undiagnosed),
      :math:`I_2` (diagnosed) and :math:`A` (symptomatic) - nobody counts the
      undiagnosed, which is the whole reason this is an inverse problem
    * how many people are in :math:`R`, which is not a measurable category at
      all

    Both are arguments here rather than hidden defaults, because a reader
    needs to see that they were chosen rather than measured.

    Parameters
    ----------
    plhiv, art_coverage, population:
        Observed at :math:`t_0`; counts in millions, coverage as a fraction.
    untreated_split:
        Fractions for ``I1``, ``I2`` and ``A``, summing to 1. Defaults to
        :data:`PAPER_UNTREATED_SPLIT` - see the warning there.
    behaviour_changed:
        :math:`R(0)` in millions. Defaults to zero, matching the paper.

    Returns
    -------
    NDArray
        ``[S, I1, I2, A, T, R]`` in millions, ordered as ``COMPARTMENTS``.

    Examples
    --------
    DRC in 2020, from the UNAIDS figures rather than from Table 2:

    >>> state = initial_state_from_data(plhiv=0.510, art_coverage=0.53,
    ...                                 population=95.99)
    >>> round(float(state[4]), 4)          # T, on treatment
    0.2703
    >>> bool(abs(state.sum() - 95.99) < 1e-9)   # the six add up to N
    True
    """
    if not 0.0 <= art_coverage <= 1.0:
        raise ValueError(f"art_coverage must be a fraction in [0, 1], got {art_coverage}")
    if plhiv < 0.0 or population <= 0.0:
        raise ValueError("plhiv must be non-negative and population positive")
    if plhiv + behaviour_changed > population:
        raise ValueError(
            f"plhiv ({plhiv}) plus R ({behaviour_changed}) exceeds the "
            f"population ({population})"
        )

    split = dict(PAPER_UNTREATED_SPLIT if untreated_split is None else untreated_split)
    missing = {"I1", "I2", "A"} - set(split)
    if missing:
        raise ValueError(f"untreated_split needs entries for {sorted(missing)}")
    total = sum(split[name] for name in ("I1", "I2", "A"))
    if not np.isclose(total, 1.0):
        raise ValueError(f"untreated_split must sum to 1, got {total}")

    treated = plhiv * art_coverage
    untreated = plhiv - treated
    susceptible = population - plhiv - behaviour_changed
    return np.array(
        [
            susceptible,
            untreated * split["I1"],
            untreated * split["I2"],
            untreated * split["A"],
            treated,
            behaviour_changed,
        ]
    )
