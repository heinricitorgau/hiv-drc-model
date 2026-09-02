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


def _logistic(t: float, floor: float, ceiling: float, midpoint: float, rate: float) -> float:
    """Smooth ramp from ``floor`` to ``ceiling``, half way at ``midpoint``.

    A logistic is the shape a public-health programme actually scales up in:
    slow while it is being stood up, fastest in the middle, flattening as it
    approaches the population it can reach.  It is also smooth in ``t``, which
    the integrator much prefers to a piecewise-constant policy step.
    """
    # A strongly negative exponent overflows; np.exp gives inf, the quotient
    # goes to zero, and the limit is correct.
    with np.errstate(over="ignore"):
        growth = 1.0 + np.exp(-rate * (t - midpoint))
    return float(floor + (ceiling - floor) / growth)


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
    alpha_ceiling:
        Asymptotic treatment-uptake rate of a logistic scale-up.  ``None``
        (the default) means α is constant at :attr:`alpha`, which is the
        published model and the behaviour of every result reproduced here.
        Setting it turns α into a function of time; see :meth:`alpha_at` and
        the note below on what that costs.
    alpha_midpoint:
        Time in years at which the scale-up passes its half-way point.
    alpha_rate:
        Steepness of the scale-up, in 1/year.  Larger is more abrupt.

    Notes
    -----
    **A time-varying α makes the system non-autonomous, and that invalidates
    most of the theory in this package.**  :math:`R_0`, the equilibria and the
    stability theorem are all statements about a system with constant
    coefficients; with a scale-up running there is no fixed point to be stable
    about.  :func:`~hiv_drc.reproduction.reproduction_number` and the
    equilibrium solvers therefore refuse a time-varying parameter set rather
    than returning a number that would look fine and mean nothing.  Use
    :func:`~hiv_drc.reproduction.reproduction_number_at` for the instantaneous
    value at one moment.
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

    # -- optional logistic scale-up of the treatment-uptake rate ------------
    # None keeps alpha constant, so every pre-existing result is untouched.
    alpha_ceiling: float | None = None
    alpha_midpoint: float = 0.0
    alpha_rate: float = 0.0

    # ...and of the diagnosis rate.  Treatment cannot outrun diagnosis: alpha
    # moves people out of I2, which only lambda can fill.  See lam_at.
    lam_ceiling: float | None = None
    lam_midpoint: float = 0.0
    lam_rate: float = 0.0

    # -- derived exit rates -------------------------------------------------
    # a1..a4 are the total per-capita rates out of I1, I2, A and T.  They show
    # up in every formula in the paper, so they are computed once here.

    @property
    def a1(self) -> float:
        """Total exit rate from I₁: σ₁ + λ + μ.

        Only meaningful while λ is constant; under a scale-up use ``a1_at(t)``.
        """
        if self.lam_ceiling is not None:
            raise ValueError(
                "a1 is not a constant while lambda is scaling up; use a1_at(t)"
            )
        return self.sigma1 + self.lam + self.mu

    def a1_at(self, t: float) -> float:
        """Total exit rate from I₁ at time ``t``: σ₁ + λ(t) + μ."""
        return self.sigma1 + self.lam_at(t) + self.mu

    @property
    def a2(self) -> float:
        """Total exit rate from I₂: σ₂ + μ + α.

        Only meaningful while α is constant.  Under a scale-up the exit rate
        is a function of time and the right quantity is ``a2_at(t)``.
        """
        if self.is_time_varying:
            raise ValueError(
                "a2 is not a constant while alpha is scaling up; use a2_at(t)"
            )
        return self.sigma2 + self.mu + self.alpha

    def a2_at(self, t: float) -> float:
        """Total exit rate from I₂ at time ``t``: σ₂ + μ + α(t)."""
        return self.sigma2 + self.mu + self.alpha_at(t)

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

    # -- treatment scale-up -------------------------------------------------

    @property
    def is_time_varying(self) -> bool:
        """Whether any rate depends on time, i.e. whether a scale-up is configured."""
        return self.alpha_ceiling is not None or self.lam_ceiling is not None

    def alpha_at(self, t: float) -> float:
        r"""Treatment-uptake rate at time ``t``.

        With no scale-up configured this is just :attr:`alpha`, evaluated
        without touching ``t`` - the constant-α model, unchanged.

        With one, α follows a logistic curve from :attr:`alpha` (the rate
        before the scale-up) to :attr:`alpha_ceiling` (the rate it settles
        at), passing the half-way point at :attr:`alpha_midpoint`:

        .. math::

            lpha(t) = lpha
                + rac{lpha_{	ext{ceiling}} - lpha}
                       {1 + e^{-r (t - t_{1/2})}}

        A logistic is the right shape because a treatment programme scales up
        the way it does: slowly while it is being set up, fastest in the
        middle, then flattening as it approaches the population it can reach.
        It is also smooth in ``t``, which the integrator prefers to a
        piecewise-constant policy step.

        Examples
        --------
        >>> DRC_2020.alpha_at(0.0) == DRC_2020.alpha    # no scale-up
        True
        >>> ramp = DRC_2020.replace(alpha=0.01, alpha_ceiling=0.5,
        ...                         alpha_midpoint=10.0, alpha_rate=0.4)
        >>> round(ramp.alpha_at(10.0), 4)               # half way, at midpoint
        0.255
        >>> ramp.alpha_at(0.0) < ramp.alpha_at(20.0)
        True
        """
        if self.alpha_ceiling is None:
            return self.alpha
        return _logistic(
            t, self.alpha, self.alpha_ceiling, self.alpha_midpoint, self.alpha_rate
        )

    def lam_at(self, t: float) -> float:
        r"""Diagnosis rate λ at time ``t``, same logistic form as :meth:`alpha_at`.

        This one matters more than it looks.  Treatment uptake α moves people
        out of :math:`I_2`, but only λ puts them in, so λ is a hard ceiling on
        how much of the infected population can ever be on treatment.  At the
        published λ = 0.0015/yr - a mean time to diagnosis of 667 years -
        driving α to 500 still caps ART coverage near 56%, against the 71%
        the DRC actually reached.  No α, constant or scaled up, can clear
        that; see the README.
        """
        if self.lam_ceiling is None:
            return self.lam
        return _logistic(t, self.lam, self.lam_ceiling, self.lam_midpoint, self.lam_rate)

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
