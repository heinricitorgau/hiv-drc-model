r"""Bayesian parameter estimation via MCMC.

:mod:`hiv_drc.estimation` answers "what is the single best-fitting parameter
value, and how uncertain is it" by finding the least-squares optimum and
reading curvature off the Jacobian there (the Wald interval).  That answer is
only as good as the assumption behind it: that the log-likelihood surface is
locally quadratic at the optimum.  The frequentist module's own coverage study
(see the README) shows that assumption under strain at 10% measurement noise,
where the nominal 95% interval covers the true parameter only 88% of the time.

This module makes no such assumption.  It samples the actual posterior with
an affine-invariant ensemble sampler (`emcee <https://emcee.readthedocs.io/>`_),
so a skewed or curved credible region comes out skewed or curved, at the cost
of tens of thousands of forward integrations instead of a handful.

What that buys, measured rather than assumed (``scripts/coverage_study.py``,
20 replicates at 10% noise): **not better coverage, but far tighter intervals
at the same coverage** - 3.7x tighter on ``beta`` and 2.2x on ``alpha``.  The
Wald interval for ``beta`` reaches 100% coverage only by being four times as
wide as the true value, which constrains nothing.  And on one replicate in
twenty it fails outright: the optimiser drives ``beta`` onto its lower bound
of zero, where the model has no sensitivity to ``beta`` at all, and the
reported standard error becomes ``inf`` for ``beta`` and ``nan`` for whatever
it is correlated with.  Sampling a bounded posterior cannot produce that
failure mode.

It also treats the measurement noise level itself as unknown.  Least-squares
fitting sidesteps that question with an ad hoc residual weighting (see
:func:`~hiv_drc.estimation.weight_vector`) chosen to make two series
comparable, not because it is anyone's best guess at the true noise.  Here
each observed series gets its own relative noise level ``eta`` as an extra
parameter, given a Jeffreys (log-uniform) prior - the standard weakly
informative choice for a scale parameter - and inferred jointly with the
epidemiological rates.  On real data, where nobody knows the reporting error
in advance, that is the honest question to ask.

The cost of an MCMC run is real: at the default settings this integrates the
ODE system roughly 16,000 times (16 walkers x 1,000 steps), which is why
`run_mcmc` takes tens of seconds to a couple of minutes rather than the
fraction of a second `estimate_parameters` needs. The defaults get the split-
:math:`\hat R` diagnostic comfortably under the conventional 1.1 threshold on
the two-parameter problem this module ships with; the effective-sample-size
estimate typically still flags itself as unreliable at that budget (it wants
50 times the autocorrelation time), which `run_mcmc` reports honestly as
`None` rather than a falsely precise number - pass larger `n_steps` for a
publication-grade run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import emcee
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimation import PARAMETER_BOUNDS, FitResult, estimate_parameters, predict
from .parameters import DRC_2020, INITIAL_STATE, Parameters
from .priors import Prior, log_density
from .reproduction import reproduction_number, reproduction_number_at
from .simulation import Solution, simulate
from .synthetic import Observations

__all__ = [
    "ETA_BOUNDS",
    "MCMC_BOUNDS",
    "BayesianFitResult",
    "log_prior",
    "log_likelihood",
    "log_posterior",
    "split_rhat",
    "run_mcmc",
]


#: Default prior range for each series' relative noise level ``eta`` (0.1% to
#: 200% of the signal). Wide enough to be uninformative for any realistic
#: surveillance data; the log-uniform (Jeffreys) prior over this range makes
#: no claim about the *scale* within it, only that it is a scale parameter.
ETA_BOUNDS: tuple[float, float] = (1e-3, 2.0)

#: Tighter default priors for the two most commonly fitted rates, overriding
#: :data:`~hiv_drc.estimation.PARAMETER_BOUNDS` for MCMC specifically.  That
#: box is a safety rail for a local gradient-based search that never strays
#: far from its starting point; an ensemble sampler explores its *entire*
#: prior support, so reusing it verbatim - beta up to 2, alpha up to 1 -
#: would spend most of the walk on contact and treatment-uptake rates with no
#: epidemiological plausibility, for no benefit.  These ranges still contain
#: the published values with several times the margin needed; override with
#: `bounds` for a specific study.
MCMC_BOUNDS: dict[str, tuple[float, float]] = {
    "beta": (0.0, 0.6),
    "alpha": (0.0, 0.3),
    "alpha_ceiling": (0.0, 1.0),
    "alpha_rate": (0.0, 2.0),
}


def _split(theta: NDArray, n_fit: int) -> tuple[NDArray, NDArray]:
    """``theta`` as ``(fitted rates, log relative-noise levels)``."""
    return theta[:n_fit], theta[n_fit:]


def log_prior(
    theta: ArrayLike,
    names: Sequence[str],
    obs: Observations,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    eta_bounds: tuple[float, float] = ETA_BOUNDS,
    priors: Mapping[str, Prior] | None = None,
) -> float:
    """Log prior density: uniform on the rates, log-uniform on the noise levels.

    ``priors`` adds an informative density on top of the box for any named
    parameter; anything without one keeps the uniform default.  See
    :mod:`hiv_drc.priors`, and read
    :func:`~hiv_drc.priors.contraction` before trusting an interval that came
    out of one.

    Returns ``-inf`` outside the box.  Inside it, the density is a constant -
    included explicitly (rather than dropped as irrelevant to sampling) so
    that ``log_posterior`` is a real log-density and
    ``log_prior + log_likelihood == log_posterior`` holds as an identity that
    can be tested independently of the sampler.

    Examples
    --------
    >>> from hiv_drc import generate_observations
    >>> obs = generate_observations(n_points=3)
    >>> log_prior([10.0, 0.035], ("beta", "alpha"), obs)  # beta outside [0, 2]
    -inf
    """
    theta = np.asarray(theta, dtype=float)
    fit_part, log_eta = _split(theta, len(names))

    box = dict(PARAMETER_BOUNDS)
    if bounds is not None:
        box.update(bounds)
    for name, value in zip(names, fit_part, strict=True):
        lo, hi = box[name]
        if not (lo <= value <= hi):
            return -np.inf

    lo_eta, hi_eta = np.log(eta_bounds[0]), np.log(eta_bounds[1])
    if log_eta.size != len(obs.names):
        raise ValueError(
            f"expected {len(obs.names)} noise parameters, got {log_eta.size}"
        )
    if np.any(log_eta < lo_eta) or np.any(log_eta > hi_eta):
        return -np.inf

    fit_density = -sum(np.log(box[name][1] - box[name][0]) for name in names)
    eta_density = -log_eta.size * np.log(hi_eta - lo_eta)
    informative = log_density(dict(zip(names, fit_part, strict=True)), priors)
    if not np.isfinite(informative):
        return -np.inf
    return float(fit_density + eta_density + informative)


def log_likelihood(
    theta: ArrayLike,
    names: Sequence[str],
    obs: Observations,
    baseline: Parameters = DRC_2020,
    y0: ArrayLike = INITIAL_STATE,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> float:
    r"""Log likelihood under independent Gaussian noise, ``sigma = eta * |model|``.

    Each observed series contributes
    :math:`\sum_i -\tfrac12\left(\frac{y_i - \hat y_i}{\eta\lvert\hat y_i\rvert}\right)^2
    - \log(\eta\lvert\hat y_i\rvert) - \tfrac12\log(2\pi)`, matching the
    proportional noise model :func:`~hiv_drc.synthetic.generate_observations`
    uses to build synthetic data - so on synthetic data this likelihood is the
    correctly specified one, not an approximation to it.

    Returns ``-inf`` if the integration fails or produces a non-finite state,
    exactly as :func:`~hiv_drc.estimation.residuals` does, so a bad proposal
    steers the sampler away rather than raising.
    """
    theta = np.asarray(theta, dtype=float)
    fit_part, log_eta = _split(theta, len(names))
    p = baseline.replace(**dict(zip(names, fit_part, strict=True)))
    try:
        modelled = predict(p, obs, y0, rtol=rtol, atol=atol)
    except (RuntimeError, ValueError, FloatingPointError):
        return -np.inf

    total = 0.0
    for name, log_eta_i in zip(obs.names, log_eta, strict=True):
        model = modelled[name]
        if not np.all(np.isfinite(model)):
            return -np.inf
        sigma = np.maximum(np.exp(log_eta_i) * np.abs(model), 1e-12)
        resid = (obs.values[name] - model) / sigma
        total += (
            -0.5 * float(np.sum(resid**2))
            - float(np.sum(np.log(sigma)))
            - 0.5 * model.size * np.log(2.0 * np.pi)
        )
    return total


def log_posterior(
    theta: ArrayLike,
    names: Sequence[str],
    obs: Observations,
    baseline: Parameters = DRC_2020,
    y0: ArrayLike = INITIAL_STATE,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    eta_bounds: tuple[float, float] = ETA_BOUNDS,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    priors: Mapping[str, Prior] | None = None,
) -> float:
    """``log_prior + log_likelihood``, skipping the integration when the prior forbids it.

    This is the callable handed to ``emcee.EnsembleSampler`` - its positional
    signature after ``theta`` is passed straight through via ``args=``.
    """
    lp = log_prior(theta, names, obs, bounds, eta_bounds, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, names, obs, baseline, y0, rtol, atol)


def split_rhat(chain: NDArray) -> float:
    r"""Gelman-Rubin split-:math:`\hat R` for one parameter's walker chain.

    ``chain`` has shape ``(n_steps, n_walkers)``.  Each walker is itself split
    in half and the two halves treated as separate chains, which is what
    catches a walker that is still drifting within a single run rather than
    merely disagreeing with the others (Gelman et al., *Bayesian Data
    Analysis*, 3rd ed., section 11.4).  Values near 1.0 indicate the chains
    are exploring the same distribution; values above roughly 1.1 are the
    conventional signal that they are not, and the chain needs to run longer.

    Returns ``nan`` if there are too few post-split samples to estimate a
    variance from.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> mixed = rng.standard_normal((2000, 16))
    >>> round(split_rhat(mixed), 2)
    1.0
    >>> stuck = rng.uniform(-5, 5, size=16) + 0.1 * rng.standard_normal((2000, 16))
    >>> split_rhat(stuck) > 5.0
    True
    """
    n_steps, _n_walkers = chain.shape
    half = n_steps // 2
    if half < 2:
        return float("nan")
    split = np.concatenate([chain[:half], chain[-half:]], axis=1)
    chain_means = split.mean(axis=0)
    chain_vars = split.var(axis=0, ddof=1)
    within = float(chain_vars.mean())
    if within <= 0:
        return float("nan")
    between = half * float(chain_means.var(ddof=1))
    var_hat = ((half - 1) / half) * within + between / half
    return float(np.sqrt(var_hat / within))


@dataclass(frozen=True)
class BayesianFitResult:
    """A posterior sample, plus the diagnostics needed to judge whether to trust it.

    Attributes
    ----------
    names:
        Fitted epidemiological rate names.
    noise_names:
        One relative-noise name per observed series, e.g. ``"eta_A"``.
    samples:
        Post-burn-in draws, shape ``(n_draws, len(names) + len(noise_names))``,
        columns ordered as ``names`` then ``noise_names``.  Noise columns are
        already exponentiated out of the sampler's log-space.
    chain:
        The same post-burn-in draws before flattening, shape
        ``(kept_steps, n_walkers, len(names) + len(noise_names))``.  Kept
        alongside ``samples`` because a trace plot needs each walker's own
        path, which flattening destroys.
    median, mean, std:
        Marginal posterior summaries, over both rates and noise levels.
    credible_interval:
        Central 95% credible interval (2.5th/97.5th percentiles) per parameter.
    rhat:
        Split-:math:`\\hat R` per parameter from :func:`split_rhat`; above
        about 1.1 means the chains have not mixed and the run should be longer.
    ess:
        Effective sample size per parameter from emcee's autocorrelation-time
        estimate, or ``None`` where the chain was too short relative to that
        time for the estimate to be trustworthy (emcee's own 50-tau rule).
    acceptance_fraction:
        Mean fraction of proposed moves accepted, averaged over walkers.
        Healthy runs are typically in the 0.2-0.5 range.
    parameters:
        ``baseline`` with the posterior median rates substituted in.
    solution:
        Dense trajectory at the posterior median, for plotting.
    least_squares:
        The frequentist fit used to centre the walkers, kept for direct
        comparison against the posterior.
    n_walkers, n_steps, burn:
        The run's own configuration.
    truth:
        Generating values for the fitted rates, when known.
    """

    names: tuple[str, ...]
    noise_names: tuple[str, ...]
    samples: NDArray
    chain: NDArray
    median: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]
    credible_interval: dict[str, tuple[float, float]]
    rhat: dict[str, float]
    ess: dict[str, float | None]
    acceptance_fraction: float
    parameters: Parameters
    solution: Solution
    least_squares: FitResult
    n_walkers: int
    n_steps: int
    burn: int
    truth: dict[str, float] | None = None
    priors: Mapping[str, Prior] | None = field(default=None, repr=False)
    """The informative priors used, kept so contraction can be computed after the fact."""

    @property
    def all_names(self) -> tuple[str, ...]:
        """Fitted rates followed by noise levels - the column order of :attr:`samples`."""
        return self.names + self.noise_names

    @property
    def R0(self) -> float:
        """Basic reproduction number at the posterior median."""
        return float(reproduction_number(self.parameters).R0)

    def errors(self) -> dict[str, float]:
        """Signed absolute error of the posterior median against the truth."""
        if self.truth is None:
            raise ValueError("no ground truth available for these observations")
        return {name: self.median[name] - self.truth[name] for name in self.names}

    def relative_errors(self) -> dict[str, float]:
        """Signed relative error of the posterior median, in percent."""
        if self.truth is None:
            raise ValueError("no ground truth available for these observations")
        return {
            name: 100.0 * (self.median[name] - self.truth[name]) / self.truth[name]
            for name in self.names
        }

    def covers_truth(self) -> dict[str, bool]:
        """Whether each 95% credible interval contains the generating value."""
        if self.truth is None:
            raise ValueError("no ground truth available for these observations")
        return {
            name: self.credible_interval[name][0] <= self.truth[name]
            <= self.credible_interval[name][1]
            for name in self.names
        }

    def posterior_predictive(
        self,
        obs: Observations,
        n_draws: int = 200,
        seed: int | np.random.Generator | None = None,
        baseline: Parameters = DRC_2020,
        y0: ArrayLike = INITIAL_STATE,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> dict[str, NDArray]:
        """Simulate at random posterior draws, including the fitted noise.

        Returns each observed series as an array of shape
        ``(n_draws, len(obs.t))``: not the model trajectory alone, but the
        model *plus* a draw of its own fitted measurement noise.  This is
        what a posterior predictive check needs - the predictive distribution
        for a new observation, which is wider than the trajectory's own
        uncertainty band by exactly the fitted noise level.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n_fit = len(self.names)
        idx = rng.integers(0, self.samples.shape[0], size=n_draws)
        draws = self.samples[idx]

        predictions: dict[str, list[NDArray]] = {name: [] for name in obs.names}
        for row in draws:
            fit_part, etas = row[:n_fit], row[n_fit:]
            p = baseline.replace(**dict(zip(self.names, fit_part, strict=True)))
            solution = simulate(
                p, y0, (float(obs.t[0]), float(obs.t[-1])), t_eval=obs.t, rtol=rtol, atol=atol
            )
            for name, eta in zip(obs.names, etas, strict=True):
                model = solution[name]
                sigma = np.maximum(eta * np.abs(model), 1e-12)
                predictions[name].append(model + sigma * rng.standard_normal(model.shape))
        return {name: np.array(rows) for name, rows in predictions.items()}

    def summary(self) -> str:
        """A printable table of posterior summaries, diagnostics and (if known) errors."""
        known = self.truth is not None
        lines = [
            f"  {'param':10s} {'median':>12s} {'std':>11s} "
            f"{'95% CI':>26s} {'R-hat':>7s} {'ESS':>8s}"
        ]
        for name in self.all_names:
            lo, hi = self.credible_interval[name]
            ess = self.ess[name]
            ess_text = f"{ess:8.0f}" if ess is not None else "     n/a"
            row = (
                f"  {name:10s} {self.median[name]:12.6f} {self.std[name]:11.6f} "
                f"  [{lo:10.6f}, {hi:10.6f}] {self.rhat[name]:7.3f} {ess_text}"
            )
            if known and name in self.truth:
                relative = 100.0 * (self.median[name] - self.truth[name]) / self.truth[name]
                row += f"   truth {self.truth[name]:.6f}  ({relative:+.2f}%)"
            lines.append(row)
        lines.append(
            f"\n  acceptance fraction {self.acceptance_fraction:.3f}   "
            f"{self.n_walkers} walkers x {self.n_steps} steps "
            f"({self.burn} discarded as burn-in)"
        )
        if self.parameters.is_time_varying:
            # R0 is undefined while alpha is still moving; report the
            # instantaneous value at each end of the window instead.
            first, last = float(self.solution.t[0]), float(self.solution.t[-1])
            lines.append(
                f"  R0(t) at the estimate: "
                f"{reproduction_number_at(self.parameters, first).R0:.6f} at t={first:g}"
                f" -> {reproduction_number_at(self.parameters, last).R0:.6f} at t={last:g}"
            )
        else:
            lines.append(f"  R0 at the posterior median = {self.R0:.6f}")
        return "\n".join(lines)


def run_mcmc(
    obs: Observations,
    fit: Sequence[str] = ("beta", "alpha"),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    eta_bounds: tuple[float, float] = ETA_BOUNDS,
    baseline: Parameters = DRC_2020,
    y0: ArrayLike = INITIAL_STATE,
    n_walkers: int = 16,
    n_steps: int = 1000,
    burn: int = 250,
    thin: int = 1,
    seed: int | np.random.Generator | None = 20260830,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    progress: bool = False,
    priors: Mapping[str, Prior] | None = None,
) -> BayesianFitResult:
    """Sample the posterior of ``fit`` given ``obs`` with an ensemble MCMC sampler.

    Parameters
    ----------
    obs:
        Observations to fit; see :func:`~hiv_drc.synthetic.generate_observations`.
    fit:
        Which epidemiological rates to estimate. Everything else is held at
        ``baseline`` - matching :func:`~hiv_drc.estimation.estimate_parameters`,
        since the reasons to keep this set small are the same ones.
    bounds, baseline, y0:
        As in :func:`~hiv_drc.estimation.estimate_parameters`.
    eta_bounds:
        Prior range for each series' relative noise level; see :data:`ETA_BOUNDS`.
    n_walkers, n_steps, burn, thin:
        Ensemble size, chain length, burn-in to discard, and thinning applied
        when flattening the kept chain. ``n_walkers`` must be at least twice
        the number of dimensions (emcee's own requirement for the stretch
        move to work); the default of 24 covers up to nine fitted rates
        against the default two-series observation set.
    seed:
        Seed or ``Generator``, controlling both the starting positions and the
        sampler's internal randomness - the whole run is reproducible.
    rtol, atol:
        Integrator tolerances. Looser than :func:`~hiv_drc.estimation.predict`'s
        default because a run needs tens of thousands of evaluations rather
        than a few dozen; see the module docstring for the cost this buys back.
    progress:
        Show emcee's progress bar (needs ``tqdm``).
    priors:
        Informative priors per parameter; see :mod:`hiv_drc.priors`.  Anything
        unnamed keeps the uniform default.  Whenever these are used, report
        :func:`~hiv_drc.priors.contraction` alongside the intervals - a tight
        posterior under a tight prior is not evidence of anything.

    Returns
    -------
    BayesianFitResult

    Examples
    --------
    >>> from hiv_drc import generate_observations
    >>> obs = generate_observations(noise=0.02, seed=3)
    >>> result = run_mcmc(obs, n_walkers=10, n_steps=200, burn=60, seed=1)
    >>> bool(abs(result.relative_errors()["alpha"]) < 30.0)
    True
    """
    names = tuple(fit)
    if not names:
        raise ValueError("nothing to fit")
    unknown = set(names) - set(PARAMETER_BOUNDS)
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    if burn >= n_steps:
        raise ValueError(f"burn ({burn}) must be less than n_steps ({n_steps})")

    noise_names = tuple(f"eta_{name}" for name in obs.names)
    ndim = len(names) + len(noise_names)
    if n_walkers < 2 * ndim:
        raise ValueError(f"n_walkers ({n_walkers}) must be at least 2 * ndim ({2 * ndim})")

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # Resolve the effective prior box once, so the centring fit below and the
    # sampler itself agree on exactly the same support.
    box = dict(PARAMETER_BOUNDS)
    box.update(MCMC_BOUNDS)
    if bounds is not None:
        box.update(bounds)

    # Centre the ensemble on the frequentist optimum: it is nearly free to
    # compute and it means the sampler's budget goes to exploring the
    # posterior rather than to finding it in the first place.
    least_squares = estimate_parameters(obs, fit=names, bounds=box, baseline=baseline, y0=y0)
    scale = obs.scale()
    center_eta = np.array(
        [max(least_squares.rmse[name] / scale[name], 2.0 * ETA_BOUNDS[0]) for name in obs.names]
    )
    center = np.concatenate([least_squares.theta, np.log(center_eta)])

    lower = np.concatenate(
        [[box[name][0] for name in names], np.full(len(noise_names), np.log(eta_bounds[0]))]
    )
    upper = np.concatenate(
        [[box[name][1] for name in names], np.full(len(noise_names), np.log(eta_bounds[1]))]
    )

    jitter = 0.02 * np.maximum(np.abs(center), 1e-6)
    p0 = center + jitter * rng.standard_normal((n_walkers, ndim))
    margin = 1e-9 * (upper - lower)
    p0 = np.clip(p0, lower + margin, upper - margin)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_posterior,
        args=(names, obs, baseline, y0, box, eta_bounds, rtol, atol, priors),
    )
    seed_state = np.random.RandomState(int(rng.integers(0, 2**31 - 1)))
    sampler.random_state = seed_state.get_state()
    sampler.run_mcmc(p0, n_steps, progress=progress)

    chain = sampler.get_chain(discard=burn, thin=thin)  # (steps, walkers, ndim), eta in log-space
    kept_steps = chain.shape[0]
    flat = chain.reshape(-1, ndim).copy()
    flat[:, len(names) :] = np.exp(flat[:, len(names) :])
    # R-hat is computed on `chain` (log-space) below, since that is the space
    # the sampler actually explores; `chain_display` matches `samples`'
    # exponentiated units for anything user-facing, such as a trace plot.
    chain_display = chain.copy()
    chain_display[:, :, len(names) :] = np.exp(chain_display[:, :, len(names) :])

    all_names = names + noise_names
    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))

    tau = sampler.get_autocorr_time(discard=burn, thin=thin, quiet=True)
    reliable = kept_steps >= 50 * tau
    n_total = kept_steps * n_walkers
    ess = {
        name: (float(n_total / tau[j]) if reliable[j] else None)
        for j, name in enumerate(all_names)
    }

    median = {name: float(np.median(flat[:, j])) for j, name in enumerate(all_names)}
    mean = {name: float(np.mean(flat[:, j])) for j, name in enumerate(all_names)}
    std = {name: float(np.std(flat[:, j], ddof=1)) for j, name in enumerate(all_names)}
    credible = {
        name: (
            float(np.percentile(flat[:, j], 2.5)),
            float(np.percentile(flat[:, j], 97.5)),
        )
        for j, name in enumerate(all_names)
    }
    rhat = {name: split_rhat(chain[:, :, j]) for j, name in enumerate(all_names)}

    fitted = baseline.replace(**{name: median[name] for name in names})
    dense = simulate(fitted, y0, (float(obs.t[0]), float(obs.t[-1])), n_points=601)

    truth = None
    if obs.parameters is not None:
        truth = {name: float(getattr(obs.parameters, name)) for name in names}

    return BayesianFitResult(
        names=names,
        noise_names=noise_names,
        samples=flat,
        chain=chain_display,
        median=median,
        mean=mean,
        std=std,
        credible_interval=credible,
        rhat=rhat,
        ess=ess,
        acceptance_fraction=acceptance_fraction,
        parameters=fitted,
        solution=dense,
        least_squares=least_squares,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn=burn,
        truth=truth,
        priors=priors,
    )
