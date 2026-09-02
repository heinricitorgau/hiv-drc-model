"""What surveillance actually reports, as functions of the model state.

Everything upstream of this module assumes an observation *is* a compartment:
:func:`~hiv_drc.estimation.predict` looked up ``solution[name]``, so the only
fittable data was a direct measurement of ``A`` or ``T``. Real surveillance
data does not work that way. UNAIDS, WHO and national programmes publish
aggregates over a *different* partition of the same population:

* "people living with HIV" is :math:`I_1 + I_2 + A + T` — it does not
  distinguish undiagnosed from diagnosed from symptomatic,
* "people on antiretroviral therapy" is :math:`T`, which happens to coincide
  with a compartment,
* "ART coverage" is :math:`T / (I_1 + I_2 + A + T)`, a *ratio* and therefore
  not even a linear function of the state,
* "known status" is :math:`I_2 + A + T` — everyone diagnosed, whether or not
  they are being treated.

The model's own decomposition (undiagnosed / diagnosed / symptomatic /
treated) is a modelling choice; the published decomposition is an artefact of
what a health system can count. Fitting real data means mapping between them,
and that map is what an *observation operator* is.

An operator here is any callable taking a :class:`~hiv_drc.simulation.Solution`
and returning one series. That is deliberately more general than a matrix:
``art_coverage`` is a ratio, and a linear-operator formulation could not
express it.

The six raw compartments are registered as operators too, so observing a
compartment directly is the identity case rather than a special case, and
every existing caller keeps working unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from numpy.typing import NDArray

from .parameters import COMPARTMENTS
from .simulation import Solution

__all__ = [
    "Observable",
    "OBSERVABLES",
    "OBSERVABLE_LABELS",
    "resolve",
    "apply",
]

#: An observation operator: model state in, one observed series out.
Observable = Callable[[Solution], NDArray]


def _compartment(name: str) -> Observable:
    """The identity operator for a single compartment."""

    def observe(solution: Solution) -> NDArray:
        return solution[name]

    observe.__name__ = f"observe_{name}"
    observe.__doc__ = f"The {name} compartment, observed directly."
    return observe


#: Named observation operators.
#:
#: The six compartments are the model's own variables; the rest are the
#: aggregates a surveillance system actually publishes. Extend this mapping
#: (or pass ``observables=`` explicitly) for a data source that reports
#: something else.
OBSERVABLES: dict[str, Observable] = {
    **{name: _compartment(name) for name in COMPARTMENTS},
    "plhiv": lambda solution: solution.infected,
    "diagnosed": lambda solution: solution["I2"] + solution["A"] + solution["T"],
    "on_art": lambda solution: solution["T"],
    "untreated": lambda solution: solution["I1"] + solution["I2"] + solution["A"],
    "population": lambda solution: solution.total_population,
    "prevalence": lambda solution: solution.prevalence,
    "art_coverage": lambda solution: solution["T"] / solution.infected,
}

#: Human-readable names, for plot titles and axis labels.
OBSERVABLE_LABELS: dict[str, str] = {
    "plhiv": "people living with HIV (millions)",
    "diagnosed": "diagnosed, any stage (millions)",
    "on_art": "on antiretroviral therapy (millions)",
    "untreated": "living with HIV, not on treatment (millions)",
    "population": "total population (millions)",
    "prevalence": "HIV prevalence (fraction of N)",
    "art_coverage": "ART coverage (fraction of PLHIV)",
}


def resolve(
    names: tuple[str, ...],
    observables: Mapping[str, Observable] | None = None,
) -> dict[str, Observable]:
    """Look ``names`` up in the operator registry.

    ``observables`` overrides or extends :data:`OBSERVABLES` for one call,
    which is how a data source reporting something unusual gets fitted
    without editing this module.

    Raises
    ------
    ValueError
        If a name has no operator - listing what is available, since the most
        common cause is a typo or a compartment name in the wrong case.

    Examples
    --------
    >>> sorted(resolve(("A", "plhiv")))
    ['A', 'plhiv']
    >>> resolve(("nonsense",))
    Traceback (most recent call last):
    ValueError: unknown observable(s): ['nonsense']; available: ...
    """
    table = dict(OBSERVABLES)
    if observables is not None:
        table.update(observables)
    unknown = [name for name in names if name not in table]
    if unknown:
        raise ValueError(
            f"unknown observable(s): {sorted(unknown)}; "
            f"available: {sorted(table)}"
        )
    return {name: table[name] for name in names}


def apply(
    solution: Solution,
    names: tuple[str, ...],
    observables: Mapping[str, Observable] | None = None,
) -> dict[str, NDArray]:
    """Evaluate the named operators against ``solution``.

    Examples
    --------
    >>> from hiv_drc import simulate
    >>> sol = simulate(t_span=(0.0, 5.0), n_points=6)
    >>> series = apply(sol, ("plhiv", "on_art"))
    >>> bool((series["plhiv"] >= series["on_art"]).all())
    True
    """
    return {name: operator(solution) for name, operator in resolve(names, observables).items()}
