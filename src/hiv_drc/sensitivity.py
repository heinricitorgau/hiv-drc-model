r"""Sensitivity of :math:`R_0` — local indices and a global LHS/PRCC screen.

The paper reports *local* normalised sensitivity indices (its equations
(4.2)–(4.3)),

.. math:: \Upsilon_x^{R_0} = \frac{\partial R_0}{\partial x}\cdot\frac{x}{R_0},

which measure the fractional change in :math:`R_0` per fractional change in
:math:`x`, holding everything else at baseline.  Two of them have exact values
worth knowing: :math:`R_0` is linear in :math:`\beta`, so its index is exactly
:math:`+1`, and the index for :math:`\phi` is exactly :math:`-\phi/(\mu+\phi)`.
Both are asserted in the test suite.

Local indices only describe one point in parameter space.  This module also
provides a *global* screen: Latin hypercube sampling over all 13 parameters at
once, scored with partial rank correlation coefficients.  The two views can
disagree, and where they do the disagreement is the interesting part — see the
project README.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import qmc

from .parameters import DRC_2020, R0_PARAMETERS, Parameters
from .reproduction import reproduction_number

__all__ = [
    "local_sensitivity_index",
    "local_sensitivity_indices",
    "latin_hypercube",
    "prcc",
    "GlobalSensitivity",
    "global_sensitivity",
]


# ---------------------------------------------------------------- local ----


def local_sensitivity_index(p: Parameters, name: str, rel_step: float = 1e-6) -> float:
    """Normalised sensitivity index of :math:`R_0` to one parameter.

    Evaluated by central difference.  ``rel_step`` is relative to the parameter
    value, which keeps the step meaningful across rates that differ by two
    orders of magnitude.
    """
    x = getattr(p, name)
    h = max(abs(x), 1e-8) * rel_step
    up = reproduction_number(p.replace(**{name: x + h})).R0
    down = reproduction_number(p.replace(**{name: x - h})).R0
    return (up - down) / (2.0 * h) * x / reproduction_number(p).R0


def local_sensitivity_indices(
    p: Parameters = DRC_2020, names: Sequence[str] = R0_PARAMETERS
) -> dict[str, float]:
    """Local indices for every parameter in ``names``, sorted by magnitude."""
    values = {name: local_sensitivity_index(p, name) for name in names}
    return dict(sorted(values.items(), key=lambda kv: -abs(kv[1])))


# --------------------------------------------------------------- global ----


def latin_hypercube(
    p: Parameters,
    names: Sequence[str] = R0_PARAMETERS,
    n_samples: int = 1000,
    spread: float = 0.25,
    seed: int | None = 42,
) -> NDArray:
    """Latin hypercube sample of shape ``(n_samples, len(names))``.

    Each parameter is drawn uniformly from ``[(1-spread)·x₀, (1+spread)·x₀]``.
    A uniform relative band is a deliberately neutral choice: the paper does
    not report confidence intervals for its rates, so inventing distributions
    would dress up an assumption as data.
    """
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n_samples)
    base = np.array([getattr(p, name) for name in names])
    return base * (1.0 - spread) + unit * (2.0 * spread * base)


def prcc(X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Partial rank correlation coefficients and their two-sided p-values.

    For each column *j*, the ranks of that column and of ``y`` are both
    regressed on the ranks of the remaining columns, and the correlation of the
    two residuals is reported.  Removing the influence of the other parameters
    is what makes this readable when several of them move together; the rank
    transform is what makes it robust to the monotone-but-nonlinear response
    that ``R₀`` has to most of these rates.
    """
    Xr = np.apply_along_axis(stats.rankdata, 0, X)
    yr = stats.rankdata(y)
    n, k = Xr.shape

    coefficients = np.empty(k)
    p_values = np.empty(k)
    for j in range(k):
        others = np.delete(Xr, j, axis=1)
        design = np.column_stack([np.ones(n), others])

        x_residual = Xr[:, j] - design @ np.linalg.lstsq(design, Xr[:, j], rcond=None)[0]
        y_residual = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]

        r, pv = stats.pearsonr(x_residual, y_residual)
        coefficients[j] = r

        # pearsonr's p-value assumes n-2 degrees of freedom; k-1 were spent on
        # the regressions above, so recompute the tail with the right df.
        df = n - 2 - (k - 1)
        if df > 0 and abs(r) < 1.0:
            t = r * np.sqrt(df / (1.0 - r * r))
            p_values[j] = 2.0 * stats.t.sf(abs(t), df)
        else:  # pragma: no cover - degenerate sample
            p_values[j] = pv

    return coefficients, p_values


@dataclass(frozen=True)
class GlobalSensitivity:
    """Result of a Latin hypercube / PRCC screen."""

    names: tuple[str, ...]
    samples: NDArray
    outputs: dict[str, NDArray]
    coefficients: dict[str, NDArray]
    p_values: dict[str, NDArray]
    local: dict[str, float]
    spread: float

    def table(self, output: str) -> list[tuple[str, float, float, float]]:
        """Rows of ``(name, prcc, p_value, local_index)``, strongest first."""
        rows = [
            (
                name,
                float(self.coefficients[output][i]),
                float(self.p_values[output][i]),
                float(self.local[name]),
            )
            for i, name in enumerate(self.names)
        ]
        return sorted(rows, key=lambda row: -abs(row[1]))


def global_sensitivity(
    p: Parameters = DRC_2020,
    names: Sequence[str] = R0_PARAMETERS,
    n_samples: int = 1000,
    spread: float = 0.25,
    seed: int | None = 42,
    extra_outputs: dict[str, Callable[[Parameters], float]] | None = None,
) -> GlobalSensitivity:
    """Run the LHS/PRCC screen on :math:`R_0` and any extra scalar outputs.

    Parameters
    ----------
    extra_outputs:
        Optional mapping from a label to a function of the sampled parameters.
        The command-line report uses this to add the 50-year infected total,
        which is where the local and global pictures part company.
    """
    names = tuple(names)
    X = latin_hypercube(p, names, n_samples, spread, seed)

    outputs: dict[str, NDArray] = {"R0": np.empty(n_samples)}
    extras = extra_outputs or {}
    for label in extras:
        outputs[label] = np.empty(n_samples)

    for i, row in enumerate(X):
        sample = p.replace(**dict(zip(names, row, strict=True)))
        outputs["R0"][i] = reproduction_number(sample).R0
        for label, fn in extras.items():
            outputs[label][i] = fn(sample)

    coefficients: dict[str, NDArray] = {}
    p_values: dict[str, NDArray] = {}
    for label, y in outputs.items():
        coefficients[label], p_values[label] = prcc(X, y)

    return GlobalSensitivity(
        names=names,
        samples=X,
        outputs=outputs,
        coefficients=coefficients,
        p_values=p_values,
        local=local_sensitivity_indices(p, names),
        spread=spread,
    )
