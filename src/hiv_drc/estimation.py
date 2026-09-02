r"""The inverse problem: recovering parameters from noisy observations.

The forward problem is "given the rates, what does the epidemic do".  This
module solves the backward one - given a handful of noisy series, which rates
could have produced them.  It is a nonlinear weighted least-squares fit,

.. math::

    \hat{\theta} = \arg\min_{\theta \in [\ell, u]}
        \sum_{k} \sum_{i} w_k^2 \left(y_{k}(t_i;\theta) - \hat{y}_{k,i}\right)^2,

where every evaluation of the objective costs one full integration of the
six-compartment system.  ``scipy.optimize.least_squares`` is used rather than
a general minimiser because it sees the individual residuals rather than only
their sum, which lets the trust-region solver build a Gauss-Newton model of
the curvature and gives the Jacobian needed for the parameter covariance.

Four things matter for getting a trustworthy answer out of this:

**Bounds.**  Every rate is non-negative and physically bounded, and the
optimiser must be told so.  Without bounds a trust-region step happily tries
:math:`\beta < 0`, which makes the "infection" term a source of susceptibles
and the integration diverges.  See :data:`PARAMETER_BOUNDS`.

**Weighting.**  A sum of squares in raw units is dominated by whichever
compartment happens to be largest.  Observing ``S`` (about 88 million) and
``A`` (about 0.05 million) together, the ``S`` residuals outweigh the ``A``
residuals by roughly 800 to 1, and the fit simply ignores ``A``.  Dividing
each series by its own magnitude removes that; for the default ``A`` and ``T``
pair, whose means are within a factor of two, it is only a mild correction.
See :func:`weight_vector`.

**Finite-difference step.**  The residual is only accurate to the integrator's
tolerance, so differencing it with the default step of :math:`\sqrt{\epsilon}
\approx 1.5\times10^{-8}` amplifies solver noise by :math:`10^{8}` and the fit
stalls short of the minimum.  ``diff_step`` defaults to :math:`10^{-6}` here,
which keeps the derivative noise near :math:`10^{-4}` relative.

**Identifiability.**  A small residual does not mean the parameters are
pinned down: two rates that only ever appear as a product will trade off
against each other along a flat valley.  :attr:`FitResult.correlation` and
:func:`cost_surface` are what to look at before believing an estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from .observables import apply
from .parameters import DRC_2020, INITIAL_STATE, Parameters
from .reproduction import reproduction_number
from .simulation import Solution, simulate
from .synthetic import Observations

__all__ = [
    "PARAMETER_BOUNDS",
    "FitResult",
    "predict",
    "weight_vector",
    "residuals",
    "estimate_parameters",
    "estimate_multistart",
    "cost_surface",
]


#: Physically admissible range for each rate, used as the default box.
#:
#: These are deliberately generous - wide enough that the box is not doing the
#: estimating, tight enough to keep the solver inside the region where the
#: model means anything.  All rates are per year; ``c`` and ``d`` are relative
#: infectiousness and cannot exceed that of an unaware infective.
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "Lambda": (0.0, 50.0),
    "beta": (0.0, 2.0),
    "mu": (0.0, 0.2),
    "phi": (0.0, 1.0),
    "c": (0.0, 1.0),
    "d": (0.0, 1.0),
    "sigma1": (0.0, 1.0),
    "sigma2": (0.0, 1.0),
    "lam": (0.0, 1.0),
    "delta1": (0.0, 1.0),
    "delta2": (0.0, 1.0),
    "alpha": (0.0, 1.0),
    "kappa1": (0.0, 1.0),
    "kappa2": (0.0, 1.0),
}

#: Value returned per residual when the integration fails outright.  Large
#: enough to push the trust region back, finite so the solver keeps working.
_FAILED_RESIDUAL = 1e6


def _as_parameters(
    theta: ArrayLike, names: Sequence[str], baseline: Parameters
) -> Parameters:
    """``baseline`` with the named entries replaced by ``theta``."""
    return baseline.replace(**dict(zip(names, np.asarray(theta, dtype=float), strict=True)))


def predict(
    p: Parameters,
    obs: Observations,
    y0: ArrayLike = INITIAL_STATE,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict[str, NDArray]:
    """Model values for the observed series at the observation times.

    The solver is asked for exactly ``obs.t``, so no interpolation error
    enters the residual.

    Each name is mapped through its observation operator (see
    :mod:`hiv_drc.observables`), so this returns whatever the data actually
    measures - a raw compartment, or an aggregate like ``"plhiv"`` that no
    single compartment corresponds to.
    """
    solution = simulate(
        p,
        y0,
        (float(obs.t[0]), float(obs.t[-1])),
        t_eval=obs.t,
        rtol=rtol,
        atol=atol,
    )
    return apply(solution, obs.names, obs.observables)


def weight_vector(obs: Observations, weights: str | None = "scale") -> NDArray:
    """Residual weights, laid out to match :meth:`Observations.stack`.

    ``"scale"``
        ``1 / mean|y_obs|`` per series.  Computed from the observations alone,
        so it works unchanged on real data.  This is the default.
    ``"sigma"``
        ``1 / sigma`` point by point, the maximum-likelihood weighting for
        known Gaussian error.  Only available for synthetic data, where the
        generator recorded the standard deviations it used - on real data the
        error model is exactly what you do not know.
    ``None`` or ``"none"``
        Unweighted.  Included so the effect of weighting can be measured
        rather than asserted.

    Examples
    --------
    >>> from hiv_drc import generate_observations
    >>> obs = generate_observations(n_points=3, noise=0.0)
    >>> weight_vector(obs, None)
    array([1., 1., 1., 1., 1., 1.])
    """
    if weights is None or weights == "none":
        return np.ones(obs.n_points * len(obs.names))
    if weights == "scale":
        scale = obs.scale()
        return np.concatenate(
            [np.full(obs.n_points, 1.0 / scale[name]) for name in obs.names]
        )
    if weights == "sigma":
        if obs.sigma is None:
            raise ValueError("sigma weighting needs observations that recorded sigma")
        sigma = obs.stack("sigma")
        floor = np.max(sigma) * 1e-6 if np.max(sigma) > 0 else 1.0
        return 1.0 / np.maximum(sigma, floor)
    raise ValueError(f"unknown weighting {weights!r}")


def residuals(
    theta: ArrayLike,
    names: Sequence[str],
    obs: Observations,
    baseline: Parameters = DRC_2020,
    y0: ArrayLike = INITIAL_STATE,
    w: NDArray | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> NDArray:
    """Weighted residual vector ``w * (model - observed)``.

    This is the objective function handed to ``least_squares``.  A failed
    integration returns :data:`_FAILED_RESIDUAL` rather than raising: the
    optimiser can recover from an expensive step into a bad region, but not
    from an exception.
    """
    if w is None:
        w = np.ones(obs.n_points * len(obs.names))
    p = _as_parameters(theta, names, baseline)
    try:
        modelled = predict(p, obs, y0, rtol=rtol, atol=atol)
    except (RuntimeError, ValueError, FloatingPointError):
        return np.full(obs.n_points * len(obs.names), _FAILED_RESIDUAL)
    stacked = np.concatenate([modelled[name] for name in obs.names])
    if not np.all(np.isfinite(stacked)):
        return np.full_like(stacked, _FAILED_RESIDUAL)
    return w * (stacked - obs.stack("values"))


@dataclass(frozen=True)
class FitResult:
    """Everything the fit produced, including how much to believe it.

    Attributes
    ----------
    names:
        The fitted parameter names, in the order of the state vector ``theta``.
    estimates, stderr, ci95:
        Point estimates, asymptotic standard errors and 95% confidence
        intervals, keyed by name.
    truth:
        The generating values when they are known, else ``None``.
    covariance, correlation:
        Asymptotic covariance ``s^2 (J^T J)^-1`` and the correlation matrix
        derived from it.  Off-diagonal correlations near +/-1 are the warning
        sign that the data cannot separate two parameters.
    parameters:
        ``baseline`` with the estimates substituted in - ready to simulate.
    solution:
        The best-fit trajectory, on a dense grid for plotting.
    cost:
        Final value of ``0.5 * sum(weighted residuals**2)``.
    rmse:
        Unweighted root-mean-square error per series, in millions of people.
    nfev, success, message:
        Solver diagnostics, passed through from ``least_squares``.
    """

    names: tuple[str, ...]
    estimates: dict[str, float]
    stderr: dict[str, float]
    ci95: dict[str, tuple[float, float]]
    covariance: NDArray
    correlation: NDArray
    parameters: Parameters
    solution: Solution
    cost: float
    rmse: dict[str, float]
    nfev: int
    success: bool
    message: str
    truth: dict[str, float] | None = None
    starts: tuple[dict[str, float], ...] = field(default=(), repr=False)
    """Estimates from every multi-start run, when :func:`estimate_multistart` was used."""

    @property
    def theta(self) -> NDArray:
        """Estimates as a plain vector, ordered as :attr:`names`."""
        return np.array([self.estimates[name] for name in self.names])

    @property
    def R0(self) -> float:
        """Basic reproduction number implied by the fitted parameters."""
        return float(reproduction_number(self.parameters).R0)

    def errors(self) -> dict[str, float]:
        """Signed absolute error against the truth, ``estimate - truth``."""
        if self.truth is None:
            raise ValueError("no ground truth available for these observations")
        return {name: self.estimates[name] - self.truth[name] for name in self.names}

    def relative_errors(self) -> dict[str, float]:
        """Signed relative error in percent."""
        if self.truth is None:
            raise ValueError("no ground truth available for these observations")
        return {
            name: 100.0 * (self.estimates[name] - self.truth[name]) / self.truth[name]
            for name in self.names
        }

    def covers_truth(self) -> dict[str, bool]:
        """Whether each 95% interval contains the generating value.

        Over many synthetic replicates this should come out true about 95% of
        the time; if it is far lower, the asymptotic errors are optimistic.
        """
        if self.truth is None:
            raise ValueError("no ground truth available for these observations")
        return {
            name: self.ci95[name][0] <= self.truth[name] <= self.ci95[name][1]
            for name in self.names
        }

    def summary(self) -> str:
        """A printable table of estimates, intervals and (if known) errors."""
        known = self.truth is not None
        lines = []
        header = f"  {'param':8s} {'estimate':>12s} {'std err':>11s} {'95% CI':>26s}"
        if known:
            header += f" {'truth':>10s} {'error':>10s} {'rel err':>9s}"
        lines.append(header)
        for name in self.names:
            lo, hi = self.ci95[name]
            row = (
                f"  {name:8s} {self.estimates[name]:12.6f} {self.stderr[name]:11.6f} "
                f"  [{lo:10.6f}, {hi:10.6f}]"
            )
            if known:
                row += (
                    f" {self.truth[name]:10.6f} {self.errors()[name]:+10.6f} "
                    f"{self.relative_errors()[name]:+8.2f}%"
                )
            lines.append(row)
        rmse = "   ".join(f"{name}: {value:.6f}" for name, value in self.rmse.items())
        lines.append(f"\n  cost = {self.cost:.6e}   RMSE (millions)   {rmse}")
        lines.append(f"  R0 at the estimate = {self.R0:.6f}")
        if len(self.names) > 1:
            lines.append(
                f"  corr({self.names[0]}, {self.names[1]}) = "
                f"{self.correlation[0, 1]:+.4f}"
            )
        lines.append(f"  {self.nfev} residual evaluations - {self.message}")
        return "\n".join(lines)


def _covariance(jac: NDArray, residual: NDArray) -> NDArray:
    """Asymptotic covariance ``s^2 (J^T J)^-1`` from the solver's Jacobian.

    ``J`` is rank-deficient exactly when a parameter is unidentifiable, so the
    inverse is taken through an SVD pseudo-inverse with a relative cutoff: an
    unidentifiable direction reports infinite variance rather than raising,
    which is the honest answer - the data constrain that direction not at all.

    Two consequences worth knowing before reading a standard error from this:

    * An infinite variance makes the Wald interval ``(-inf, +inf)``, which
      "contains" the true value trivially. A coverage study must count such a
      run as a failure, not a success; see ``scripts/coverage_study.py``.
    * ``inf`` propagates through the off-diagonal arithmetic, so a parameter
      *correlated* with an unidentifiable one comes back with a ``nan``
      standard error even when it is itself perfectly well determined.

    This is not hypothetical. At 10% measurement noise the optimiser drives
    ``beta`` onto its lower bound of zero on roughly one run in twenty - at
    ``beta = 0`` the infection term vanishes identically, so the model has no
    sensitivity to ``beta`` at all - and both effects above follow.
    """
    m, n = jac.shape
    _, s, Vt = np.linalg.svd(jac, full_matrices=False)
    cutoff = np.finfo(float).eps * max(m, n) * (s[0] if s.size else 0.0)
    s_inv = np.array([1.0 / value if value > cutoff else np.inf for value in s])
    with np.errstate(over="ignore", invalid="ignore"):
        cov_unit = (Vt.T * s_inv**2) @ Vt
    dof = max(m - n, 1)
    variance = float(residual @ residual) / dof
    return cov_unit * variance


def estimate_parameters(
    obs: Observations,
    fit: Sequence[str] = ("beta", "alpha"),
    guess: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    baseline: Parameters = DRC_2020,
    y0: ArrayLike = INITIAL_STATE,
    weights: str | None = "scale",
    loss: str = "linear",
    f_scale: float = 1.0,
    diff_step: float = 1e-6,
    x_scale: str | ArrayLike = "jac",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_nfev: int | None = 200,
    verbose: int = 0,
) -> FitResult:
    """Recover ``fit`` from ``obs`` by bounded nonlinear least squares.

    Parameters
    ----------
    obs:
        The observations to fit, from :func:`~hiv_drc.synthetic.generate_observations`
        or :meth:`~hiv_drc.synthetic.Observations.from_csv`.
    fit:
        Which parameters to estimate.  Everything else is held at ``baseline``.
        The default recovers the contact rate ``beta`` and the treatment-uptake
        rate ``alpha``: one drives the epidemic and one drives the response, so
        between them they set both the shape and the level of the observed
        series.
    guess:
        Starting point.  Defaults to a quarter of the way up each parameter's
        box, which is deliberately derived from the bounds and not from
        ``baseline`` - starting the optimiser at the answer would make a
        recovery test meaningless.
    bounds:
        Per-parameter ``(lower, upper)``, defaulting to :data:`PARAMETER_BOUNDS`.
    baseline:
        Values for the parameters that are not being fitted.  In a real study
        these come from the literature, and being wrong about them biases the
        estimates - the reason the fitted set should stay small.
    weights:
        Residual weighting; see :func:`weight_vector`.  The default ``"scale"``
        uses only the observations, so nothing about the truth leaks in.
    loss, f_scale:
        Passed to ``least_squares``.  ``loss="soft_l1"`` with ``f_scale`` near
        the typical residual size is the one to reach for when the data may
        contain outliers, which a plain quadratic loss would chase.
    diff_step:
        Relative step for the finite-difference Jacobian; see the module
        docstring on why the default is not the ``least_squares`` default.
    x_scale:
        ``"jac"`` rescales the variables by the Jacobian, which matters here
        because ``beta`` is around 0.15 and ``alpha`` around 0.035 and an
        isotropic trust region would step badly in both.
    rtol, atol, max_nfev, verbose:
        Integrator tolerances and solver controls.

    Returns
    -------
    FitResult

    Examples
    --------
    >>> from hiv_drc import generate_observations
    >>> obs = generate_observations(noise=0.02, seed=7)
    >>> fit = estimate_parameters(obs)
    >>> bool(abs(fit.relative_errors()["beta"]) < 15.0)
    True
    """
    names = tuple(fit)
    if not names:
        raise ValueError("nothing to fit")
    unknown = set(names) - set(PARAMETER_BOUNDS)
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")

    box = dict(PARAMETER_BOUNDS)
    if bounds is not None:
        box.update(bounds)
    lower = np.array([box[name][0] for name in names], dtype=float)
    upper = np.array([box[name][1] for name in names], dtype=float)

    if guess is None:
        x0 = lower + 0.25 * (upper - lower)
    else:
        missing = set(names) - set(guess)
        if missing:
            raise ValueError(f"no starting value for {sorted(missing)}")
        x0 = np.array([float(guess[name]) for name in names])
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError(f"starting point {x0} lies outside the bounds")

    w = weight_vector(obs, weights)

    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        args=(names, obs, baseline, y0, w, rtol, atol),
        method="trf",
        loss=loss,
        f_scale=f_scale,
        diff_step=diff_step,
        x_scale=x_scale,
        max_nfev=max_nfev,
        verbose=verbose,
    )

    estimates = {name: float(value) for name, value in zip(names, result.x, strict=True)}
    fitted = _as_parameters(result.x, names, baseline)

    covariance = _covariance(np.asarray(result.jac), np.asarray(result.fun))
    variances = np.diag(covariance).copy()
    stderr = np.sqrt(np.where(variances > 0, variances, np.nan))
    scale = np.outer(stderr, stderr)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.where(scale > 0, covariance / scale, np.nan)

    dof = max(result.fun.size - len(names), 1)
    critical = float(student_t.ppf(0.975, dof))
    ci95 = {
        name: (
            float(estimates[name] - critical * stderr[j]),
            float(estimates[name] + critical * stderr[j]),
        )
        for j, name in enumerate(names)
    }

    modelled = predict(fitted, obs, y0, rtol=rtol, atol=atol)
    rmse = {
        name: float(np.sqrt(np.mean((modelled[name] - obs.values[name]) ** 2)))
        for name in obs.names
    }

    truth = None
    if obs.parameters is not None:
        truth = {name: float(getattr(obs.parameters, name)) for name in names}

    dense = simulate(fitted, y0, (float(obs.t[0]), float(obs.t[-1])), n_points=601)

    return FitResult(
        names=names,
        estimates=estimates,
        stderr={name: float(stderr[j]) for j, name in enumerate(names)},
        ci95=ci95,
        covariance=covariance,
        correlation=correlation,
        parameters=fitted,
        solution=dense,
        cost=float(result.cost),
        rmse=rmse,
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        truth=truth,
    )


def estimate_multistart(
    obs: Observations,
    fit: Sequence[str] = ("beta", "alpha"),
    n_starts: int = 8,
    seed: int | np.random.Generator | None = 20260830,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    **kwargs,
) -> FitResult:
    """Run the fit from several random starting points and keep the best.

    A single local solve only ever finds the basin it was dropped into.
    Scattering the starts over the box and comparing the final costs is the
    cheapest available evidence that the reported optimum is the global one:
    if every start lands on the same estimate, the surface has one basin; if
    they scatter, :attr:`FitResult.starts` shows how far apart.

    The best fit is returned, with every start's estimates in
    :attr:`FitResult.starts`.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    names = tuple(fit)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    box = dict(PARAMETER_BOUNDS)
    if bounds is not None:
        box.update(bounds)
    lower = np.array([box[name][0] for name in names])
    upper = np.array([box[name][1] for name in names])

    best: FitResult | None = None
    collected: list[dict[str, float]] = []
    for k in range(n_starts):
        # The first start is the deterministic default, so a one-start run
        # reproduces estimate_parameters exactly.
        if k == 0:
            start = lower + 0.25 * (upper - lower)
        else:
            start = lower + rng.uniform(size=len(names)) * (upper - lower)
        candidate = estimate_parameters(
            obs, fit=names, guess=dict(zip(names, start, strict=True)), bounds=bounds, **kwargs
        )
        collected.append(candidate.estimates)
        if best is None or candidate.cost < best.cost:
            best = candidate

    assert best is not None
    return replace(best, starts=tuple(collected))


def cost_surface(
    obs: Observations,
    names: Sequence[str] = ("beta", "alpha"),
    grids: Sequence[ArrayLike] | None = None,
    baseline: Parameters = DRC_2020,
    y0: ArrayLike = INITIAL_STATE,
    weights: str | None = "scale",
    n: int = 41,
    span: float = 3.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[NDArray, NDArray, NDArray]:
    """Evaluate the objective over a 2-D grid, for an identifiability picture.

    Returns ``(x, y, cost)`` with ``cost`` of shape ``(len(y), len(x))``, ready
    for ``contourf``.  A round basin means both parameters are pinned down; a
    long diagonal valley means only some combination of them is, and the
    reported standard errors will be large and strongly correlated.

    ``span`` sets the default grid half-width as a multiple of the baseline
    value.  Tolerances are looser than in the fit itself because a contour
    plot does not need ten digits, and this is 1681 integrations by default.
    """
    if len(names) != 2:
        raise ValueError("cost_surface needs exactly two parameter names")
    if grids is None:
        grids = [
            np.linspace(
                max(PARAMETER_BOUNDS[name][0], getattr(baseline, name) / span),
                min(PARAMETER_BOUNDS[name][1], getattr(baseline, name) * span),
                n,
            )
            for name in names
        ]
    x, y = (np.asarray(grid, dtype=float) for grid in grids)

    w = weight_vector(obs, weights)
    cost = np.empty((y.size, x.size))
    for i, y_value in enumerate(y):
        for j, x_value in enumerate(x):
            r = residuals(
                (x_value, y_value), names, obs, baseline, y0, w, rtol, atol
            )
            cost[i, j] = 0.5 * float(r @ r)
    return x, y, cost
