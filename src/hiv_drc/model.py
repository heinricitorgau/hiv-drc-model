r"""The six-compartment ODE system.

The state vector is ``y = [S, I1, I2, A, T, R]`` in millions of people, and
time is measured in years.  With ``N = S + I1 + I2 + A + T + R``:

.. math::

    \frac{dS}{dt}  &= \Lambda - \frac{\beta S}{N}(I_1 + cI_2 + dA)
                      - \mu S - \phi S \
    \frac{dI_1}{dt} &= \frac{\beta S}{N}(I_1 + cI_2 + dA)
                      - (\sigma_1 + \lambda + \mu) I_1 \
    \frac{dI_2}{dt} &= \lambda I_1 - (\sigma_2 + \mu + \alpha) I_2
                      + \kappa_2 T \
    \frac{dA}{dt}   &= \sigma_1 I_1 + \sigma_2 I_2
                      - (\mu + \delta_1 + \kappa_1) A \
    \frac{dT}{dt}   &= \alpha I_2 + \kappa_1 A
                      - (\kappa_2 + \mu + \delta_2) T \
    \frac{dR}{dt}   &= \phi S - \mu R

Summing the six equations gives the population balance

.. math:: \frac{dN}{dt} = \Lambda - \mu N - \delta_1 A - \delta_2 T,

so :math:`N(t)` never exceeds :math:`\Lambda/\mu`.  That identity is checked in
the test suite and is a good first thing to look at if a change here goes wrong.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameters import Parameters

__all__ = ["force_of_infection", "rhs", "total_population", "jacobian"]


def total_population(state: ArrayLike) -> float:
    """Total population ``N`` for a single state vector."""
    return np.sum(state, axis=-1)


def force_of_infection(state: ArrayLike, p: Parameters):
    """Per-capita rate at which susceptibles acquire infection, times S.

    This is the shared term ``βS(I₁ + cI₂ + dA)/N`` that moves people from S
    into I₁.  Aware infectives and symptomatic individuals transmit at reduced
    relative infectiousness ``c`` and ``d``.
    """
    state = np.asarray(state)
    S, I1, I2, A = state[0], state[1], state[2], state[3]
    N = total_population(state)
    return p.beta * S / N * (I1 + p.c * I2 + p.d * A)


def rhs(t: float, y: ArrayLike, p: Parameters) -> NDArray:
    """Right-hand side of the ODE system, in ``solve_ivp`` argument order.

    Parameters
    ----------
    t:
        Time in years.  The system is autonomous, so this is unused; it is
        present because ``solve_ivp`` requires the ``f(t, y)`` signature.
    y:
        State vector ``[S, I1, I2, A, T, R]``.
    p:
        Model parameters.

    Notes
    -----
    Only ``+ - * /`` appear below, with no branching, absolute values or
    comparisons.  That is deliberate: it makes the function analytic in each
    coordinate, which is what lets :func:`jacobian` use complex-step
    differentiation to get derivatives to machine precision.
    """
    y = np.asarray(y)
    S, I1, I2, A, T, R = y
    N = S + I1 + I2 + A + T + R

    infection = p.beta * S / N * (I1 + p.c * I2 + p.d * A)

    return np.array(
        [
            p.Lambda - infection - p.mu * S - p.phi * S,
            infection - p.a1 * I1,
            p.lam * I1 - p.a2 * I2 + p.kappa2 * T,
            p.sigma1 * I1 + p.sigma2 * I2 - p.a3 * A,
            p.alpha * I2 + p.kappa1 * A - p.a4 * T,
            p.phi * S - p.mu * R,
        ]
    )


def jacobian(state: ArrayLike, p: Parameters) -> NDArray:
    """Jacobian ∂f/∂y evaluated at ``state``, by complex-step differentiation.

    Perturbing coordinate *j* into the imaginary direction and reading off
    ``imag(f(y + ih·e_j))/h`` gives the derivative with no subtractive
    cancellation, so ``h`` can be made absurdly small (1e-200 here) and the
    result is accurate to machine precision — unlike a finite difference,
    which trades truncation error against round-off.

    This works only because :func:`rhs` is analytic; see its notes.
    """
    y = np.asarray(state, dtype=complex)
    n = y.size
    h = 1e-200
    J = np.empty((n, n), dtype=float)
    for j in range(n):
        perturbed = y.copy()
        perturbed[j] += 1j * h
        J[:, j] = np.imag(rhs(0.0, perturbed, p)) / h
    return J
