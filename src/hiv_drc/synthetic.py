"""Synthetic observations: the forward model plus measurement error.

Fitting code should not be debugged against real data.  Real data brings
missing years, revised denominators and reporting artefacts, and when a fit
comes out wrong there is no way to tell whether the optimiser, the model or
the data is at fault.  Here the answer is known: a trajectory is generated
from :data:`~hiv_drc.parameters.DRC_2020`, corrupted with Gaussian noise, and
handed to the estimator, which is then judged on whether it recovers the
parameters it was never told.

Two compartments are treated as observable, matching what a surveillance
system actually reports:

* ``A`` - people with symptomatic disease (AIDS case notifications),
* ``T`` - people on antiretroviral therapy (programme registers).

``S``, ``I1``, ``I2`` and ``R`` are latent.  Nobody counts undiagnosed
infections directly, which is what makes this an inverse problem rather than
a curve fit.

Noise models
------------
``"proportional"`` (the default) draws each point from ``N(y, (eta*y)**2)`` -
larger compartments carry larger absolute error, the usual situation for count
data.  ``"constant"`` uses ``sigma = eta * mean(y)`` for the whole series,
which is the right choice when the error is dominated by a fixed survey design
rather than by the size of the thing being measured.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameters import COMPARTMENTS, DRC_2020, INITIAL_STATE, Parameters
from .simulation import simulate

__all__ = [
    "Observations",
    "generate_observations",
    "OBSERVABLE",
    "NOISE_MODELS",
]

#: The compartments a surveillance system is assumed to report.
OBSERVABLE = ("A", "T")

#: Supported measurement-error models.
NOISE_MODELS = ("proportional", "constant")


@dataclass(frozen=True)
class Observations:
    """A noisy time series, with the truth kept alongside for scoring.

    Attributes
    ----------
    t:
        Observation times in years, shape ``(n,)``.
    values:
        Observed (noisy) series, keyed by compartment name.
    truth:
        The noise-free trajectory the observations were drawn from, same keys.
        ``None`` for data loaded from a file that carried no truth columns -
        that is, for anything real.
    sigma:
        The standard deviation actually used to draw each point.  This is the
        generator's private knowledge; see the note in
        :func:`~hiv_drc.estimation.estimate_parameters` on why the estimator
        does not reach for it by default.
    parameters:
        The parameter set that produced ``truth``, or ``None`` if unknown.
    noise:
        The relative noise level ``eta`` passed to the generator.
    noise_model:
        Which entry of :data:`NOISE_MODELS` was used.
    """

    t: NDArray
    values: dict[str, NDArray]
    truth: dict[str, NDArray] | None = None
    sigma: dict[str, NDArray] | None = None
    parameters: Parameters | None = None
    noise: float = 0.0
    noise_model: str = "proportional"

    @property
    def names(self) -> tuple[str, ...]:
        """Observed compartment names, in the canonical compartment order."""
        return tuple(sorted(self.values, key=COMPARTMENTS.index))

    @property
    def n_points(self) -> int:
        """Number of observation times."""
        return int(self.t.size)

    def stack(self, source: str = "values") -> NDArray:
        """Concatenate the series into one vector, ordered by :attr:`names`.

        ``source`` selects ``"values"``, ``"truth"`` or ``"sigma"``.  This is
        the layout the residual vector uses.
        """
        table = getattr(self, source)
        if table is None:
            raise ValueError(f"{source!r} is not available on these observations")
        return np.concatenate([np.asarray(table[name]) for name in self.names])

    def scale(self) -> dict[str, float]:
        """Per-series magnitude ``mean|y_obs|``, computed from the data alone.

        Used to put compartments of different size on a common footing when
        weighting residuals.  Over the default window ``A`` and ``T`` average
        0.047 and 0.096 million, so for that pair the correction is mild; it
        becomes essential the moment a large compartment is observed, since
        ``S`` averages 88 million and would otherwise absorb the entire fit.
        """
        return {
            name: float(np.mean(np.abs(values))) or 1.0
            for name, values in self.values.items()
        }

    def to_csv(self, path) -> Path:
        """Write to CSV: ``t``, one column per series, plus ``<name>_true``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = self.names
        header = ["t", *names]
        if self.truth is not None:
            header += [f"{name}_true" for name in names]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for i in range(self.n_points):
                row = [self.t[i], *(self.values[name][i] for name in names)]
                if self.truth is not None:
                    row += [self.truth[name][i] for name in names]
                writer.writerow([f"{value:.10g}" for value in row])
        return path

    @classmethod
    def from_csv(cls, path) -> Observations:
        """Read back a file written by :meth:`to_csv`.

        Truth columns are picked up when present, so a round trip preserves
        everything the estimator is scored against.
        """
        path = Path(path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{path} contains no data rows")
        columns = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
        t = columns.pop("t")
        observed = {k: v for k, v in columns.items() if not k.endswith("_true")}
        truth = {k[:-5]: v for k, v in columns.items() if k.endswith("_true")}
        return cls(t=t, values=observed, truth=truth or None)


def generate_observations(
    p: Parameters = DRC_2020,
    y0: ArrayLike = INITIAL_STATE,
    t_span: tuple[float, float] = (0.0, 30.0),
    n_points: int = 31,
    observed: tuple[str, ...] = OBSERVABLE,
    noise: float = 0.05,
    noise_model: str = "proportional",
    seed: int | np.random.Generator | None = 20260830,
    clip: bool = True,
) -> Observations:
    """Simulate the model and add Gaussian measurement error.

    Parameters
    ----------
    p:
        Parameters to generate from - the values the estimator has to recover.
    y0:
        Initial state in millions, ordered as ``COMPARTMENTS``.
    t_span, n_points:
        Observation window in years, and how many evenly spaced samples to
        take.  The default is 31 annual observations over 30 years, roughly
        what a national programme accumulates.
    observed:
        Which compartments are reported.
    noise:
        Relative noise level ``eta``; ``0.05`` is 5%.  Zero gives clean data,
        which is the first case to fit when the estimator misbehaves.
    noise_model:
        ``"proportional"`` or ``"constant"``; see the module docstring.
    seed:
        Seed or ``Generator``.  Fixed by default, so a run is reproducible.
    clip:
        Clip observations at zero.  A negative count is not a measurement, and
        on a small declining series the noise can reach below zero.

    Returns
    -------
    Observations

    Examples
    --------
    >>> obs = generate_observations(noise=0.05, seed=0)
    >>> obs.n_points, obs.names
    (31, ('A', 'T'))
    >>> bool(np.all(obs.values["A"] >= 0.0))
    True
    >>> round(float(generate_observations(noise=0.0).values["T"][0]), 5)
    0.07708
    """
    if noise < 0.0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    if noise_model not in NOISE_MODELS:
        raise ValueError(
            f"noise_model must be one of {NOISE_MODELS}, got {noise_model!r}"
        )
    unknown = set(observed) - set(COMPARTMENTS)
    if unknown:
        raise ValueError(f"unknown compartment(s): {sorted(unknown)}")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    t = np.linspace(t_span[0], t_span[1], n_points)
    solution = simulate(p, y0, t_span, t_eval=t)

    truth: dict[str, NDArray] = {name: solution[name].copy() for name in observed}
    sigma: dict[str, NDArray] = {}
    values: dict[str, NDArray] = {}
    for name, clean in truth.items():
        if noise_model == "proportional":
            s = noise * np.abs(clean)
        else:
            s = np.full_like(clean, noise * float(np.mean(np.abs(clean))))
        noisy = clean + s * rng.standard_normal(clean.shape)
        values[name] = np.clip(noisy, 0.0, None) if clip else noisy
        sigma[name] = s

    return Observations(
        t=t,
        values=values,
        truth=truth,
        sigma=sigma,
        parameters=p,
        noise=noise,
        noise_model=noise_model,
    )
