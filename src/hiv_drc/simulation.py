"""Numerical integration of the model.

``solve_ivp`` is used rather than the older ``odeint`` interface: it exposes
the choice of method, supports dense output, and returns a result object that
carries the solver's own diagnostics.

The system is mildly stiff — the fastest rate is κ₁ = 0.2/yr and the slowest
is μ = 0.0166/yr, roughly a factor of 12, and the interesting behaviour plays
out over several multiples of 1/μ ≈ 60 years.  ``LSODA`` handles that without
fuss by switching between non-stiff and stiff steppers on its own, so it is
the default here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .model import rhs
from .parameters import COMPARTMENTS, DRC_2020, INITIAL_STATE, Parameters

__all__ = ["Solution", "simulate"]


@dataclass(frozen=True)
class Solution:
    """A solved trajectory, with the compartments addressable by name."""

    t: NDArray
    y: NDArray
    """Shape ``(6, len(t))``, rows ordered as :data:`hiv_drc.parameters.COMPARTMENTS`."""
    parameters: Parameters

    def __getitem__(self, name: str) -> NDArray:
        """Return one compartment's time series, e.g. ``sol["I1"]``."""
        try:
            return self.y[COMPARTMENTS.index(name)]
        except ValueError:
            raise KeyError(
                f"unknown compartment {name!r}; expected one of {COMPARTMENTS}"
            ) from None

    @property
    def total_population(self) -> NDArray:
        """``N(t)``, the sum over all six compartments."""
        return self.y.sum(axis=0)

    @property
    def infected(self) -> NDArray:
        """``I₁ + I₂ + A + T``, everyone living with HIV."""
        return self.y[1:5].sum(axis=0)

    @property
    def prevalence(self) -> NDArray:
        """Infected fraction of the population."""
        return self.infected / self.total_population

    def at(self, time: float) -> NDArray:
        """State at ``time``, linearly interpolated between stored points."""
        return np.array([np.interp(time, self.t, row) for row in self.y])


def simulate(
    p: Parameters = DRC_2020,
    y0: ArrayLike = INITIAL_STATE,
    t_span: tuple[float, float] = (0.0, 50.0),
    n_points: int = 1001,
    method: str = "LSODA",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Solution:
    """Integrate the model and return a :class:`Solution`.

    Parameters
    ----------
    p:
        Model parameters; defaults to the DRC 2020 baseline.
    y0:
        Initial state in millions, ordered as ``COMPARTMENTS``.
    t_span:
        ``(t_start, t_end)`` in years.
    n_points:
        How many evenly spaced output times to record.
    method, rtol, atol:
        Passed straight through to ``scipy.integrate.solve_ivp``.  The default
        tolerances are tight because several downstream analyses (equilibria,
        stability) rely on the trajectory actually having settled.

    Raises
    ------
    RuntimeError
        If the solver reports failure.

    Examples
    --------
    >>> sol = simulate(t_span=(0.0, 10.0), n_points=11)
    >>> sol["S"].shape
    (11,)
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    result = solve_ivp(
        rhs,
        t_span,
        np.asarray(y0, dtype=float),
        args=(p,),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"integration failed: {result.message}")
    return Solution(t=result.t, y=result.y, parameters=p)
