"""Model parameters and initial conditions.

Everything the model needs to be pinned down lives here, so that changing a
scenario never means touching the dynamics.  The baseline values are the DRC
2020 figures from Table 3 of the paper; the initial state is Table 2.

The parameter container is a frozen dataclass, so a :class:`Parameters` value
can be used as a dictionary key, cached, and shared between threads without
anybody quietly mutating it.  Build variants with :meth:`Parameters.replace`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields, replace

import numpy as np

__all__ = [
    "Parameters",
    "DRC_2020",
    "INITIAL_STATE",
    "COMPARTMENTS",
    "COMPARTMENT_LABELS",
    "R0_PARAMETERS",
]


#: Order of the state vector used everywhere in this package.
COMPARTMENTS = ("S", "I1", "I2", "A", "T", "R")

#: Human-readable names, for plot legends.
COMPARTMENT_LABELS = {
    "S": "S — susceptible",
    "I1": "I₁ — infected, unaware",
    "I2": "I₂ — infected, aware",
    "A": "A — symptomatic (AIDS)",
    "T": "T — on treatment",
    "R": "R — changed behaviour",
}


@dataclass(frozen=True)
class Parameters:
    """Epidemiological and demographic rates.

    Populations are counted in **millions of people** and time in **years**,
    so every rate below has units of 1/year (the dimensionless ones are noted).

    Attributes
    ----------
    Lambda:
        Recruitment rate into the susceptible class (millions/year).
    beta:
        Effective contact rate.
    mu:
        Natural (non-disease) death rate.
    phi:
        Rate at which susceptibles adopt protective behaviour and leave to R.
    c:
        Relative infectiousness of aware infectives I₂ (dimensionless).
    d:
        Relative infectiousness of symptomatic individuals A (dimensionless).
    sigma1:
        Progression I₁ → A.
    sigma2:
        Progression I₂ → A.
    lam:
        Rate at which unaware infectives learn their status, I₁ → I₂.
    delta1:
        Disease-induced death rate in A.
    delta2:
        Disease-induced death rate on treatment.
    alpha:
        Treatment uptake I₂ → T.
    kappa1:
        Treatment uptake A → T.
    kappa2:
        Return from treatment to the aware asymptomatic class, T → I₂.
    """

    Lambda: float = 8.8261
    beta: float = 0.15
    mu: float = 0.0166
    phi: float = 0.083
    c: float = 0.03
    d: float = 0.001
    sigma1: float = 0.0025
    sigma2: float = 0.06
    lam: float = 0.0015
    delta1: float = 0.0909
    delta2: float = 0.0667
    alpha: float = 0.035
    kappa1: float = 0.2
    kappa2: float = 0.04

    # -- derived exit rates -------------------------------------------------
    # a1..a4 are the total per-capita rates out of I1, I2, A and T.  They show
    # up in every formula in the paper, so they are computed once here.

    @property
    def a1(self) -> float:
        """Total exit rate from I₁: σ₁ + λ + μ."""
        return self.sigma1 + self.lam + self.mu

    @property
    def a2(self) -> float:
        """Total exit rate from I₂: σ₂ + μ + α."""
        return self.sigma2 + self.mu + self.alpha

    @property
    def a3(self) -> float:
        """Total exit rate from A: μ + δ₁ + κ₁."""
        return self.mu + self.delta1 + self.kappa1

    @property
    def a4(self) -> float:
        """Total exit rate from T: κ₂ + μ + δ₂."""
        return self.kappa2 + self.mu + self.delta2

    @property
    def carrying_capacity(self) -> float:
        """Upper bound Λ/μ on the total population (millions)."""
        return self.Lambda / self.mu

    # -- convenience --------------------------------------------------------

    def replace(self, **changes: float) -> Parameters:
        """Return a copy with the named rates overridden.

        >>> DRC_2020.replace(beta=0.2).beta
        0.2
        """
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
        return replace(self, **changes)

    def as_dict(self) -> dict[str, float]:
        """Return the stored rates as a plain dict (no derived quantities)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return iter(self.as_dict().items())


#: Baseline parameters — DRC, 2020 (Table 3 of the paper).
DRC_2020 = Parameters()

#: Initial state in millions of people (Table 2), ordered as ``COMPARTMENTS``.
INITIAL_STATE = np.array([88.516, 0.014, 0.0846, 0.30832, 0.07708, 0.0])

#: The 13 parameters R₀ actually depends on.  Λ is excluded: it sets the size
#: of the disease-free population but cancels out of R₀.
R0_PARAMETERS = (
    "beta",
    "mu",
    "phi",
    "c",
    "d",
    "sigma1",
    "sigma2",
    "lam",
    "delta1",
    "delta2",
    "alpha",
    "kappa1",
    "kappa2",
)
