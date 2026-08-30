r"""Equilibria and their local stability.

The disease-free equilibrium is available in closed form,

.. math:: S^* = \frac{\Lambda}{\mu + \phi}, \quad R^* = \frac{\phi S^*}{\mu},
          \quad N^* = \frac{\Lambda}{\mu},

with all infected classes empty.  The endemic equilibrium has no useful closed
form, so it is found numerically: integrate far enough for the trajectory to
settle onto the attractor, then polish with Newton's method using the exact
Jacobian from :func:`hiv_drc.model.jacobian`.

Seeding Newton from the attractor rather than from a guess matters.  The
system has a second, biologically meaningless equilibrium with negative
components, and a cold-started root finder is perfectly happy to converge to
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .model import jacobian, rhs
from .parameters import INITIAL_STATE, Parameters
from .simulation import simulate

__all__ = [
    "Equilibrium",
    "disease_free_equilibrium",
    "endemic_equilibrium",
    "dominant_eigenvalue",
    "is_locally_stable",
]


@dataclass(frozen=True)
class Equilibrium:
    """A steady state, with enough diagnostics to judge whether to trust it."""

    state: NDArray
    parameters: Parameters
    residual: float
    """``max |f(y*)|`` — how close the right-hand side really is to zero."""
    iterations: int = 0
    endemic: bool = False
    """True when the infected classes are non-negligible."""

    @property
    def infected(self) -> float:
        """``I₁ + I₂ + A + T`` at the equilibrium (millions)."""
        return float(self.state[1:5].sum())

    @property
    def total_population(self) -> float:
        return float(self.state.sum())

    @property
    def prevalence(self) -> float:
        return self.infected / self.total_population

    @property
    def eigenvalues(self) -> NDArray:
        """Eigenvalues of the Jacobian, sorted by decreasing real part."""
        ev = np.linalg.eigvals(jacobian(self.state, self.parameters))
        return ev[np.argsort(-ev.real)]

    @property
    def dominant_eigenvalue(self) -> complex:
        return self.eigenvalues[0]

    @property
    def is_stable(self) -> bool:
        """Locally asymptotically stable, i.e. every eigenvalue has Re < 0."""
        return bool(self.dominant_eigenvalue.real < 0.0)


def disease_free_equilibrium(p: Parameters) -> Equilibrium:
    """The disease-free equilibrium, in closed form."""
    S = p.Lambda / (p.mu + p.phi)
    state = np.array([S, 0.0, 0.0, 0.0, 0.0, p.phi * S / p.mu])
    return Equilibrium(
        state=state,
        parameters=p,
        residual=float(np.max(np.abs(rhs(0.0, state, p)))),
        endemic=False,
    )


def endemic_equilibrium(
    p: Parameters,
    settle_years: float = 20_000.0,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> Equilibrium:
    """Locate the attractor and refine it to a true root of the vector field.

    When ``R₀ < 1`` the trajectory settles on the disease-free equilibrium, and
    that is what comes back — with ``endemic=False``.  Callers that need to
    distinguish the two cases should check that flag rather than assume.

    Parameters
    ----------
    settle_years:
        Length of the preliminary integration.  The slowest timescale is
        ``1/μ ≈ 60`` years, so the default is a few hundred time constants.
    tol:
        Stop once ``max |f(y)|`` falls below this.
    max_iter:
        Cap on Newton iterations.
    """
    sol = simulate(p, INITIAL_STATE, (0.0, settle_years), n_points=2, method="LSODA")
    y = sol.y[:, -1].copy()

    # Counts Newton steps actually taken, so a settled trajectory that needed
    # no polishing reports 0 rather than a misleading 1.
    iterations = 0
    while iterations < max_iter:
        f = rhs(0.0, y, p)
        if np.max(np.abs(f)) < tol:
            break
        iterations += 1
        step = np.linalg.solve(jacobian(y, p), -f)

        # Damp the step so Newton cannot walk the state negative, which would
        # be both unphysical and a good way to land on the spurious root.
        scale = 1.0
        while np.any(y + scale * step < 0.0) and scale > 1e-10:
            scale *= 0.5
        y = y + scale * step

    residual = float(np.max(np.abs(rhs(0.0, y, p))))
    return Equilibrium(
        state=y,
        parameters=p,
        residual=residual,
        iterations=iterations,
        endemic=bool(y[1:5].sum() > 1e-6),
    )


def dominant_eigenvalue(state: NDArray, p: Parameters) -> complex:
    """Eigenvalue of the Jacobian with the largest real part."""
    ev = np.linalg.eigvals(jacobian(state, p))
    return ev[np.argmax(ev.real)]


def is_locally_stable(state: NDArray, p: Parameters) -> bool:
    """Whether every Jacobian eigenvalue at ``state`` has negative real part."""
    return bool(dominant_eigenvalue(state, p).real < 0.0)
