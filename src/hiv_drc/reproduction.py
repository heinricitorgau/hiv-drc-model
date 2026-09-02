r"""The basic reproduction number :math:`R_0`.

Two independent routes to the same number are provided:

``reproduction_number``
    The closed form derived in the paper, equations (3.9)–(3.12).

``r0_from_ngm``
    The spectral radius of the next-generation matrix :math:`FV^{-1}`,
    assembled numerically.

They agree to machine precision, and the test suite checks that they keep
agreeing over randomly perturbed parameters.  Having the second one is what
makes the first trustworthy: a typo in the algebra would show up immediately
instead of silently propagating into every figure.

The closed form is

.. math:: R_0 = \frac{\beta\mu}{\mu + \phi}\,(R_1 + R_2 + R_3),

where :math:`\mu/(\mu+\phi)` is the susceptible fraction :math:`S^*/N^*` at the
disease-free equilibrium, and, with :math:`D = a_1(a_2a_3a_4 - \kappa_1\kappa_2\sigma_2
- \alpha\kappa_2 a_3)`,

.. math::

    R_1 &= \frac{1}{a_1} \
    R_2 &= \frac{c\,(\lambda a_3 a_4 + \kappa_1\kappa_2\sigma_1)}{D} \
    R_3 &= \frac{d\,\bigl(a_4(\sigma_1\sigma_2 + \mu\sigma_1 + \lambda\sigma_2)
           + \alpha\sigma_1(\mu + \delta_2)\bigr)}{D}.

Each :math:`R_i` is the contribution of one infectious class: :math:`R_1` from
the unaware infectives, :math:`R_2` from the aware ones, :math:`R_3` from the
symptomatic.  :math:`D` is exactly :math:`\det V`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .parameters import Parameters

__all__ = [
    "ReproductionNumber",
    "reproduction_number",
    "next_generation_matrices",
    "r0_from_ngm",
    "susceptible_fraction",
    "reproduction_number_at",
]


class ReproductionNumber(NamedTuple):
    """:math:`R_0` together with its per-class decomposition."""

    R0: float
    R1: float
    R2: float
    R3: float
    theta: float
    """Susceptible fraction ``S*/N* = μ/(μ+φ)`` at the disease-free state."""

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"R0 = {self.R0:.6f}  "
            f"(R1 = {self.R1:.6f}, R2 = {self.R2:.6f}, R3 = {self.R3:.6f})"
        )


def susceptible_fraction(p: Parameters) -> float:
    """Susceptible fraction at the disease-free equilibrium, ``μ/(μ+φ)``.

    Behaviour change is the only thing that dilutes the susceptible pool in
    the absence of disease, so this factor is where φ enters ``R₀``.
    """
    return p.mu / (p.mu + p.phi)


def reproduction_number(p: Parameters) -> ReproductionNumber:
    """Evaluate the closed form of equations (3.9)–(3.12).

    Raises
    ------
    ValueError
        If ``p`` configures a treatment scale-up.  :math:`R_0` is a property
        of a system with constant coefficients: it counts secondary
        infections from one case over a whole infectious lifetime, assuming
        the rates that govern that lifetime do not move.  While α is still
        scaling up they do, and a single number would be quietly meaningless.
        :func:`reproduction_number_at` gives the instantaneous value instead.

    >>> from hiv_drc.parameters import DRC_2020
    >>> round(reproduction_number(DRC_2020).R0, 4)
    1.2145
    """
    if p.is_time_varying:
        raise ValueError(
            "R0 is not defined for a time-varying alpha; "
            "use reproduction_number_at(p, t) for the instantaneous value"
        )
    a1, a2, a3, a4 = p.a1, p.a2, p.a3, p.a4
    D = a1 * (a2 * a3 * a4 - p.kappa1 * p.kappa2 * p.sigma2 - p.alpha * p.kappa2 * a3)

    R1 = 1.0 / a1
    R2 = p.c * (p.lam * a3 * a4 + p.kappa1 * p.kappa2 * p.sigma1) / D
    R3 = (
        p.d
        * (
            a4 * (p.sigma1 * p.sigma2 + p.mu * p.sigma1 + p.lam * p.sigma2)
            + p.alpha * p.sigma1 * (p.mu + p.delta2)
        )
        / D
    )

    theta = susceptible_fraction(p)
    return ReproductionNumber(
        R0=p.beta * theta * (R1 + R2 + R3), R1=R1, R2=R2, R3=R3, theta=theta
    )


def next_generation_matrices(p: Parameters) -> tuple[NDArray, NDArray]:
    """Return ``(F, V)`` for the infected subsystem ``(I₁, I₂, A, T)``.

    ``F`` collects new infections and ``V`` the remaining transfers, both
    linearised at the disease-free equilibrium.  Only susceptibles can be
    newly infected and they all enter through I₁, so ``F`` has a single
    non-zero row.
    """
    theta = susceptible_fraction(p)

    F = np.zeros((4, 4))
    F[0, :] = p.beta * theta * np.array([1.0, p.c, p.d, 0.0])

    V = np.array(
        [
            [p.a1, 0.0, 0.0, 0.0],
            [-p.lam, p.a2, 0.0, -p.kappa2],
            [-p.sigma1, -p.sigma2, p.a3, 0.0],
            [0.0, -p.alpha, -p.kappa1, p.a4],
        ]
    )
    return F, V


def r0_from_ngm(p: Parameters) -> float:
    """Spectral radius of :math:`FV^{-1}` — an independent check on the algebra."""
    F, V = next_generation_matrices(p)
    return float(np.max(np.abs(np.linalg.eigvals(F @ np.linalg.inv(V)))))


def reproduction_number_at(p: Parameters, t: float) -> ReproductionNumber:
    """:math:`R_0` of the system frozen at time ``t``.

    Under a scale-up, this evaluates the closed form with α held at
    :math:`lpha(t)` - the reproduction number the epidemic *would* settle
    into if the programme stopped changing at that moment.

    It is a diagnostic, not a threshold theorem.  Crossing
    :math:`R_0(t) = 1` says nothing on its own about whether the epidemic
    dies out, because the system never sits at any of these frozen states
    long enough for the corresponding equilibrium to assert itself.  Read it
    as "how hard is the programme pushing right now", and read the trajectory
    itself for what actually happens.

    Examples
    --------
    >>> from hiv_drc.parameters import DRC_2020
    >>> ramp = DRC_2020.replace(alpha=0.01, alpha_ceiling=0.6,
    ...                         alpha_midpoint=10.0, alpha_rate=0.5)
    >>> early, late = reproduction_number_at(ramp, 0.0), reproduction_number_at(ramp, 25.0)
    >>> early.R0 > late.R0    # more treatment, less transmission
    True
    """
    frozen = p.replace(
        alpha=p.alpha_at(t), alpha_ceiling=None,
        lam=p.lam_at(t), lam_ceiling=None,
    )
    return reproduction_number(frozen)
