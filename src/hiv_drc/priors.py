"""Informative priors, and a way to tell whether they are doing all the work.

The Bayesian layer's default priors are deliberately weak: uniform on the
rates inside a wide box.  That is the right default when the question is "what
does this data say", because the answer then comes from the data alone.

It stops being enough when the model has more parameters than the data can
separate.  Fitting a nine-parameter scale-up to twenty annual points of two
series leaves runs with indistinguishable cost landing on parameters that
differ by two orders of magnitude (see the README).  Informative priors are
the standard remedy: bring in what is known from outside this dataset, and
the posterior concentrates.

**They are also the standard way to fool yourself.**  A tight enough prior
will always produce a tight posterior, whether or not the data agreed with
it, and the result looks exactly like a well-identified fit.  The difference
is measurable: compare how much narrower the posterior is than the prior it
started from.

.. math::

    \\text{contraction} = 1 - \\frac{\\operatorname{sd}(\\text{posterior})}
                                    {\\operatorname{sd}(\\text{prior})}

Near 1, the data did the work.  Near 0, the posterior is an echo of the prior
and the parameter is still unidentified - it just looks pinned down.
:func:`contraction` computes it, and the scale-up study reports it per
parameter rather than presenting credible intervals on their own.

Provenance matters as much as shape.  Every prior offered here is justified
from something outside the series being fitted - a published rate, a WHO
guideline date, the typical duration of a programme roll-out - and
:data:`SCALEUP_PRIORS` documents which is which.  A prior tuned until the fit
looks good is not a prior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Prior",
    "Normal",
    "LogNormal",
    "Uniform",
    "SCALEUP_PRIORS",
    "log_density",
    "sample",
    "contraction",
]


@runtime_checkable
class Prior(Protocol):
    """Anything that can score a value and be drawn from."""

    def logpdf(self, x: float) -> float:
        """Log density at ``x``; ``-inf`` outside the support."""

    def rvs(self, rng: np.random.Generator, size: int) -> NDArray:
        """Draw ``size`` samples."""


@dataclass(frozen=True)
class Normal:
    """Gaussian prior, for a quantity known to sit near a value on a linear scale."""

    mu: float
    sigma: float
    why: str = ""

    def logpdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return -0.5 * z * z - math.log(self.sigma * math.sqrt(2.0 * math.pi))

    def rvs(self, rng: np.random.Generator, size: int) -> NDArray:
        return rng.normal(self.mu, self.sigma, size=size)


@dataclass(frozen=True)
class LogNormal:
    """Log-normal prior, for a positive rate known only to within a factor.

    Parameterised by the **median** and by ``sigma``, the standard deviation
    of ``log(x)`` - so ``sigma = 0.7`` means "within about a factor of two,
    one standard deviation either way", which is usually the honest strength
    of belief about an epidemiological rate.
    """

    median: float
    sigma: float
    why: str = ""

    def logpdf(self, x: float) -> float:
        if x <= 0.0:
            return -math.inf
        z = (math.log(x) - math.log(self.median)) / self.sigma
        return -0.5 * z * z - math.log(x * self.sigma * math.sqrt(2.0 * math.pi))

    def rvs(self, rng: np.random.Generator, size: int) -> NDArray:
        return np.exp(rng.normal(math.log(self.median), self.sigma, size=size))


@dataclass(frozen=True)
class Uniform:
    """Flat prior, stated explicitly rather than left implicit."""

    lo: float
    hi: float
    why: str = ""

    def logpdf(self, x: float) -> float:
        if not self.lo <= x <= self.hi:
            return -math.inf
        return -math.log(self.hi - self.lo)

    def rvs(self, rng: np.random.Generator, size: int) -> NDArray:
        return rng.uniform(self.lo, self.hi, size=size)


#: Priors for the logistic scale-up parameters, on a window whose ``t = 0`` is
#: **2005** - the midpoints are dates, so they only mean anything relative to
#: the start of the observation window.
#:
#: Each is justified from outside the DRC series being fitted.  None of them
#: was adjusted after seeing a fit.
SCALEUP_PRIORS: dict[str, Prior] = {
    "beta": LogNormal(
        median=0.15,
        sigma=0.7,
        why="the published value, to within about a factor of two either way",
    ),
    "lam_ceiling": LogNormal(
        median=0.35,
        sigma=0.7,
        why="a mature testing programme diagnoses in a few years, not decades; "
        "0.35/yr is roughly three years to diagnosis",
    ),
    "alpha_ceiling": LogNormal(
        median=4.0,
        sigma=0.8,
        why="under WHO treat-all, initiation follows diagnosis in months; "
        "4/yr is roughly three months",
    ),
    "lam_midpoint": Normal(
        mu=11.0,
        sigma=3.0,
        why="WHO's 2016 treat-all guidance is 11 years into a window starting "
        "in 2005; testing scaled up alongside it",
    ),
    "alpha_midpoint": Normal(
        mu=11.0,
        sigma=3.0,
        why="same guideline anchor as the testing scale-up",
    ),
    "lam_rate": LogNormal(
        median=0.9,
        sigma=0.6,
        why="a national roll-out goes from a tenth to nine tenths of its "
        "ceiling in roughly five years, which is a logistic rate near 0.9/yr",
    ),
    "alpha_rate": LogNormal(
        median=0.9,
        sigma=0.6,
        why="same roll-out duration as the testing scale-up",
    ),
}


def log_density(
    values: Mapping[str, float], priors: Mapping[str, Prior] | None
) -> float:
    """Total log prior density over the named values.

    Parameters with no prior contribute nothing, which leaves them on whatever
    box constraint the caller already applies - the existing uniform default.
    """
    if not priors:
        return 0.0
    total = 0.0
    for name, value in values.items():
        prior = priors.get(name)
        if prior is None:
            continue
        density = prior.logpdf(float(value))
        if not math.isfinite(density):
            return -math.inf
        total += density
    return total


def sample(
    prior: Prior,
    rng: np.random.Generator,
    size: int = 20000,
    bounds: tuple[float, float] | None = None,
) -> NDArray:
    """Draw from ``prior``, discarding anything outside ``bounds``.

    The sampler's box truncates every prior, so the spread that matters for
    :func:`contraction` is the spread of the *truncated* prior, not the
    textbook formula for the untruncated one.
    """
    draws = prior.rvs(rng, size)
    if bounds is not None:
        lo, hi = bounds
        draws = draws[(draws >= lo) & (draws <= hi)]
    return draws


def contraction(
    posterior: NDArray,
    prior: Prior,
    rng: np.random.Generator | None = None,
    bounds: tuple[float, float] | None = None,
) -> float:
    """How much narrower the posterior is than the prior it started from.

    Returns ``1 - sd(posterior) / sd(prior)``: near 1 the data determined the
    parameter, near 0 the posterior is just the prior repeated back, and
    negative means the data pulled *against* the prior hard enough to widen
    the answer.

    This is the number to read before any credible interval from an
    informative-prior fit.  A tight interval with contraction near zero is not
    a result.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> flat = Normal(0.0, 1.0)
    >>> echo = rng.normal(0.0, 1.0, 20000)      # posterior == prior
    >>> bool(abs(contraction(echo, flat, rng)) < 0.05)
    True
    >>> sharp = rng.normal(0.0, 0.1, 20000)     # data was ten times sharper
    >>> bool(contraction(sharp, flat, rng) > 0.85)
    True
    """
    rng = np.random.default_rng(0) if rng is None else rng
    prior_draws = sample(prior, rng, bounds=bounds)
    prior_sd = float(np.std(prior_draws, ddof=1))
    if prior_sd <= 0.0:
        return float("nan")
    return 1.0 - float(np.std(np.asarray(posterior), ddof=1)) / prior_sd
