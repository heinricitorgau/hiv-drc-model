"""Parameter sweeps built on top of the core model.

These are the computations behind the non-trivial figures: the threshold
surface, the bifurcation diagram, and the critical contact rate.  They are
kept separate from :mod:`hiv_drc.plotting` so that a figure can be redrawn
without recomputing, and separate from the core modules so that importing the
model stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .equilibria import disease_free_equilibrium, endemic_equilibrium
from .parameters import DRC_2020, Parameters
from .reproduction import reproduction_number

__all__ = ["r0_grid", "critical_beta", "BifurcationSweep", "bifurcation_sweep"]


def r0_grid(
    p: Parameters = DRC_2020,
    beta: NDArray | None = None,
    sigma1: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Evaluate R0 on a (beta, sigma1) grid.

    Returns ``(beta, sigma1, R0)`` where ``R0`` has shape
    ``(len(sigma1), len(beta))``, matching ``np.meshgrid`` defaults so it can
    go straight into a contour plot.
    """
    beta = np.linspace(0.05, 0.35, 120) if beta is None else np.asarray(beta)
    sigma1 = np.linspace(0.001, 0.05, 120) if sigma1 is None else np.asarray(sigma1)

    values = np.empty((sigma1.size, beta.size))
    for i, s1 in enumerate(sigma1):
        for j, b in enumerate(beta):
            values[i, j] = reproduction_number(p.replace(beta=b, sigma1=s1)).R0
    return beta, sigma1, values


def critical_beta(p: Parameters = DRC_2020, bracket: tuple[float, float] = (1e-4, 5.0)) -> float:
    """The contact rate at which R0 crosses 1, holding everything else fixed.

    R0 is linear in beta, so this could be done by division; it is solved as a
    root instead because that stays correct if the model is later changed in a
    way that breaks the linearity.
    """
    return float(brentq(lambda b: reproduction_number(p.replace(beta=b)).R0 - 1.0, *bracket))


@dataclass(frozen=True)
class BifurcationSweep:
    """R0 against the endemic level and the dominant eigenvalues."""

    beta: NDArray
    R0: NDArray
    infected: NDArray
    eig_dfe: NDArray
    """max Re(lambda) at the disease-free equilibrium."""
    eig_attractor: NDArray
    """max Re(lambda) at whichever equilibrium the flow actually settles on."""

    @property
    def threshold_mismatches(self) -> int:
        """Points where sign(max Re lambda at DFE) disagrees with sign(R0 - 1).

        The paper's local-stability theorem says this should be zero away from
        the threshold itself.
        """
        away = np.abs(self.R0 - 1.0) > 1e-6
        return int(np.sum(np.sign(self.eig_dfe[away]) != np.sign(self.R0[away] - 1.0)))


def bifurcation_sweep(
    p: Parameters = DRC_2020,
    beta: NDArray | None = None,
    settle_years: float = 20_000.0,
) -> BifurcationSweep:
    """Sweep beta and record the equilibrium structure at each value.

    This is the expensive one: every point runs a long integration followed by
    Newton polishing.  About 30 seconds for the default 41 points.
    """
    beta = np.linspace(0.04, 0.30, 41) if beta is None else np.asarray(beta)

    R0 = np.empty(beta.size)
    infected = np.empty(beta.size)
    eig_dfe = np.empty(beta.size)
    eig_attractor = np.empty(beta.size)

    for i, b in enumerate(beta):
        q = p.replace(beta=b)
        R0[i] = reproduction_number(q).R0
        eig_dfe[i] = disease_free_equilibrium(q).dominant_eigenvalue.real
        attractor = endemic_equilibrium(q, settle_years=settle_years)
        infected[i] = attractor.infected
        eig_attractor[i] = attractor.dominant_eigenvalue.real

    return BifurcationSweep(
        beta=beta,
        R0=R0,
        infected=infected,
        eig_dfe=eig_dfe,
        eig_attractor=eig_attractor,
    )
