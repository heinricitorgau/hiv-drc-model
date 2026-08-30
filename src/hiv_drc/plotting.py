"""Figures.

Every function takes already-computed results and returns a
``matplotlib.figure.Figure``, so nothing here integrates, solves or samples.
Keeping the plotting free of computation means a figure can be restyled
without re-running a twenty-thousand-year integration, and the analysis
modules stay importable in environments with no display.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .equilibria import Equilibrium
from .estimation import FitResult
from .parameters import COMPARTMENT_LABELS, Parameters
from .reproduction import reproduction_number
from .sensitivity import GlobalSensitivity
from .simulation import Solution
from .synthetic import Observations

__all__ = [
    "plot_population_dynamics",
    "plot_scenarios",
    "plot_r0_surface",
    "plot_local_sensitivity",
    "plot_global_sensitivity",
    "plot_bifurcation",
    "plot_fit",
    "plot_cost_surface",
    "save_all",
]

_INFECTED = ("I1", "I2", "A", "T")


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_population_dynamics(solution: Solution) -> Figure:
    """The main dashboard: all six compartments, plus totals and prevalence.

    S and R are plotted apart from the infected classes because they differ by
    three orders of magnitude; on shared axes the epidemic becomes a flat line
    along the bottom.
    """
    p = solution.parameters
    R0 = reproduction_number(p).R0
    t = solution.t

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    fig.suptitle(f"HIV/AIDS dynamics in the DRC  -  R0 = {R0:.4f}", fontsize=14)

    ax = axes[0, 0]
    for name in ("S", "R"):
        ax.plot(t, solution[name], lw=2, label=COMPARTMENT_LABELS[name])
    _style(ax, "Uninfected compartments", "time (years)", "population (millions)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for name in _INFECTED:
        ax.plot(t, solution[name], lw=2, label=COMPARTMENT_LABELS[name])
    _style(ax, "Infected compartments", "time (years)", "population (millions)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(t, solution.total_population, lw=2, color="0.25", label="N(t)")
    ax.axhline(
        p.carrying_capacity,
        ls="--",
        color="crimson",
        lw=1.5,
        label=f"carrying capacity {p.carrying_capacity:.1f}",
    )
    _style(ax, "Total population", "time (years)", "population (millions)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(t, 100.0 * solution.prevalence, lw=2, color="darkorange")
    _style(ax, "HIV prevalence", "time (years)", "infected share of N (%)")

    fig.tight_layout()
    return fig


def plot_scenarios(solutions: Sequence[tuple[str, Solution]]) -> Figure:
    """Total infected population under several parameter scenarios."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for label, solution in solutions:
        R0 = reproduction_number(solution.parameters).R0
        ax.plot(solution.t, solution.infected, lw=2, label=f"{label}  (R0 = {R0:.4f})")
    _style(
        ax,
        "Scenario comparison",
        "time (years)",
        "infected I1 + I2 + A + T (millions)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_r0_surface(
    beta: np.ndarray, sigma1: np.ndarray, R0: np.ndarray, baseline: Parameters
) -> Figure:
    """R0 over the (beta, sigma1) plane, with the R0 = 1 threshold drawn on.

    Left panel is a filled contour, right panel the same surface in 3-D cut by
    a translucent R0 = 1 plane.  The 2-D view is the one to read numbers off;
    the 3-D view shows that the threshold is a single clean crossing rather
    than a fold.
    """
    B, S1 = np.meshgrid(beta, sigma1)

    fig = plt.figure(figsize=(14.0, 5.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.15)

    ax = fig.add_subplot(grid[0, 0])
    filled = ax.contourf(B, S1, R0, levels=24, cmap="viridis")
    fig.colorbar(filled, ax=ax, label="R0")
    ax.contour(B, S1, R0, levels=[1.0], colors="black", linewidths=2.5)
    ax.plot(baseline.beta, baseline.sigma1, "*", ms=16, color="red", zorder=5)
    ax.annotate(
        f"DRC baseline\nR0 = {reproduction_number(baseline).R0:.4f}",
        xy=(baseline.beta, baseline.sigma1),
        xytext=(8, 10),
        textcoords="offset points",
        color="red",
        fontweight="bold",
    )
    ax.text(0.085, 0.044, "R0 < 1", color="white", fontweight="bold")
    ax.text(0.275, 0.006, "R0 > 1", color="white", fontweight="bold")
    _style(ax, "R0 over (beta, sigma1)", "beta", "sigma1")

    ax3 = fig.add_subplot(grid[0, 1], projection="3d", computed_zorder=False)
    ax3.plot_surface(B, S1, R0, cmap="viridis", alpha=0.85, linewidth=0, rstride=2, cstride=2)
    ax3.plot_surface(
        np.array([[beta[0], beta[-1]], [beta[0], beta[-1]]]),
        np.array([[sigma1[0], sigma1[0]], [sigma1[-1], sigma1[-1]]]),
        np.ones((2, 2)),
        color="0.45",
        alpha=0.18,
        linewidth=0,
    )
    # The intersection itself, drawn on top so it survives the transparency.
    ax3.contour(B, S1, R0, levels=[1.0], colors="black", linewidths=3.5, zorder=10)

    ax3.set_xlabel("beta", labelpad=8)
    ax3.set_ylabel("sigma1", labelpad=8)
    ax3.set_zlabel("R0", labelpad=4)
    ax3.set_title("cut by the R0 = 1 plane")
    ax3.view_init(elev=24, azim=-58)
    ax3.set_box_aspect((1.35, 1.0, 0.85), zoom=1.05)

    return fig


def plot_local_sensitivity(indices: dict[str, float]) -> Figure:
    """Tornado plot of the normalised sensitivity indices from the paper."""
    names = list(indices)
    values = [indices[n] for n in names]
    colors = ["tab:red" if v > 0 else "tab:blue" for v in values]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=1)
    _style(ax, "Local sensitivity indices of R0", "normalised index", "")
    ax.text(
        0.98,
        0.04,
        "red raises R0, blue lowers it",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="0.4",
    )
    fig.tight_layout()
    return fig


def plot_global_sensitivity(result: GlobalSensitivity) -> Figure:
    """PRCC bars for each output, plus the R0 distribution over the sample."""
    order = [name for name, *_ in result.table("R0")]
    index = [result.names.index(name) for name in order]
    outputs = list(result.outputs)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    fig.suptitle(
        f"Global sensitivity - Latin hypercube, n = {len(result.samples)}, "
        f"+/-{100 * result.spread:.0f}% on every parameter",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.bar(order, result.coefficients["R0"][index], color="tab:blue")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylim(-1, 1)
    _style(ax, "PRCC on R0", "", "PRCC")

    ax = axes[0, 1]
    width = 0.4
    positions = np.arange(len(order))
    ax.bar(
        positions - width / 2,
        result.coefficients["R0"][index],
        width,
        label="PRCC (global)",
    )
    ax.bar(
        positions + width / 2,
        [result.local[name] for name in order],
        width,
        label="local index (paper)",
    )
    ax.set_xticks(positions, order)
    ax.axhline(0, color="black", lw=1)
    _style(ax, "Global versus local", "", "")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    secondary = outputs[1] if len(outputs) > 1 else "R0"
    ax.bar(order, result.coefficients[secondary][index], color="tab:orange")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylim(-1, 1)
    _style(ax, f"PRCC on {secondary}", "", "PRCC")

    ax = axes[1, 1]
    R0 = result.outputs["R0"]
    ax.hist(R0, bins=40, color="tab:blue", alpha=0.75)
    ax.axvline(1.0, color="crimson", ls="--", lw=2, label="R0 = 1")
    _style(
        ax,
        f"R0 over the sample - P(R0 < 1) = {100 * np.mean(R0 < 1):.1f}%",
        "R0",
        "count",
    )
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_bifurcation(
    R0: np.ndarray,
    infected: np.ndarray,
    eig_dfe: np.ndarray,
    eig_attractor: np.ndarray,
    baseline: Equilibrium | None = None,
) -> Figure:
    """Endemic level and dominant eigenvalues against R0.

    A forward (supercritical) transcritical bifurcation looks like this: the
    endemic branch leaves zero exactly at R0 = 1, and the two eigenvalue curves
    meet there and swap sign.  A backward bifurcation would instead show an
    endemic branch surviving below 1, which would mean pushing R0 under one is
    not by itself enough to clear the infection.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    ax = axes[0]
    ax.plot(R0, infected, lw=2.5, color="tab:blue")
    ax.axvline(1.0, color="black", ls="--", lw=1.2)
    if baseline is not None:
        R0_base = reproduction_number(baseline.parameters).R0
        ax.plot(R0_base, baseline.infected, "*", ms=16, color="red", zorder=5)
        ax.annotate(
            "DRC baseline",
            xy=(R0_base, baseline.infected),
            xytext=(10, -4),
            textcoords="offset points",
            color="red",
            fontweight="bold",
        )
    _style(
        ax,
        "Forward transcritical bifurcation at R0 = 1",
        "R0",
        "infected at equilibrium (millions)",
    )

    ax = axes[1]
    ax.plot(R0, eig_dfe, lw=2.5, label="at the disease-free equilibrium")
    ax.plot(R0, eig_attractor, lw=2.5, label="at the attractor")
    ax.axhline(0, color="black", lw=1)
    ax.axvline(1.0, color="black", ls="--", lw=1.2)
    _style(ax, "Dominant eigenvalue", "R0", "max Re(lambda)")
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_fit(obs: Observations, result: FitResult) -> Figure:
    """The inverse-problem figure: noisy data, recovered curve, and the error.

    Top row is one panel per observed compartment - the scatter is what the
    estimator was given, the solid curve is the model at the recovered
    parameters, and the dashed curve is the trajectory the data were generated
    from.  The two curves lying on top of each other while the scatter jumps
    around them is the whole point: the fit recovers the signal, not the noise.

    Bottom left is the residuals, which are the real diagnostic.  Noise-shaped
    scatter around zero means the model has extracted what there was to
    extract; visible curvature or drift means it has not, and no amount of
    optimiser tuning will fix a structural mismatch.

    Bottom right compares each estimate against its true value on a common
    axis, scaled by the truth so that parameters of different magnitude are
    legible together.  Bars are 95% confidence intervals.
    """
    names = obs.names
    fitted = result.names
    k = len(names)

    fig = plt.figure(figsize=(6.6 * k, 9.0))
    grid = fig.add_gridspec(2, 2 * k, hspace=0.32, wspace=0.55)

    noise_note = (
        f"{100 * obs.noise:.0f}% {obs.noise_model} noise" if obs.noise else "noise-free"
    )
    fig.suptitle(
        f"Parameter recovery from noisy observations - fitting "
        f"{', '.join(fitted)} to {k * obs.n_points} points ({noise_note})",
        fontsize=14,
    )

    # -- top row: one observed compartment per panel ------------------------
    for j, name in enumerate(names):
        ax = fig.add_subplot(grid[0, 2 * j : 2 * j + 2])
        ax.plot(
            obs.t,
            obs.values[name],
            "o",
            ms=5.5,
            color="0.25",
            alpha=0.7,
            label="observed (noisy)",
            zorder=3,
        )
        ax.plot(
            result.solution.t,
            result.solution[name],
            lw=2.4,
            color="crimson",
            label="fitted model",
            zorder=2,
        )
        if obs.truth is not None:
            ax.plot(
                obs.t,
                obs.truth[name],
                "--",
                lw=1.8,
                color="tab:blue",
                label="ground truth",
                zorder=1,
            )
        _style(
            ax,
            COMPARTMENT_LABELS.get(name, name),
            "time (years)",
            "population (millions)",
        )
        ax.legend(frameon=False, fontsize=9)

    # -- bottom left: residuals --------------------------------------------
    ax = fig.add_subplot(grid[1, 0:k])
    modelled = {
        name: np.interp(obs.t, result.solution.t, result.solution[name])
        for name in names
    }
    for name in names:
        ax.plot(
            obs.t,
            modelled[name] - obs.values[name],
            "o-",
            ms=4,
            lw=1.0,
            alpha=0.8,
            label=f"{name}  (RMSE {result.rmse[name]:.5f})",
        )
    ax.axhline(0, color="black", lw=1)
    _style(ax, "Residuals: fitted - observed", "time (years)", "millions of people")
    ax.legend(frameon=False, fontsize=9)

    # -- bottom right: estimate against truth ------------------------------
    ax = fig.add_subplot(grid[1, k : 2 * k])
    known = result.truth is not None
    positions = np.arange(len(fitted))
    centres, errors, labels = [], [], []
    for name in fitted:
        reference = result.truth[name] if known else result.estimates[name]
        lo, hi = result.ci95[name]
        centres.append(result.estimates[name] / reference)
        errors.append(
            [
                (result.estimates[name] - lo) / reference,
                (hi - result.estimates[name]) / reference,
            ]
        )
        labels.append(name)

    ax.errorbar(
        centres,
        positions,
        xerr=np.array(errors).T,
        fmt="o",
        ms=10,
        color="crimson",
        ecolor="crimson",
        elinewidth=2,
        capsize=6,
        label="estimate with 95% CI",
        zorder=3,
    )
    ax.axvline(
        1.0,
        color="tab:blue",
        ls="--",
        lw=2,
        label="true value" if known else "point estimate",
    )
    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.6, len(fitted) - 0.15)
    _style(ax, "Recovered against true parameters", "estimate / true value", "")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    if known:
        relative = result.relative_errors()
        for position, name in zip(positions, fitted, strict=True):
            ax.annotate(
                f"{name}: {result.estimates[name]:.5f} vs {result.truth[name]:.5f}"
                f"   ({relative[name]:+.2f}%)",
                xy=(1.0, position),
                xytext=(0, 20),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none",
                      "alpha": 0.85},
            )
        summary = "   ".join(
            f"{name}: {relative[name]:+.2f}%" for name in fitted
        )
        fig.text(
            0.5,
            0.945,
            f"relative error   {summary}        R0 recovered: {result.R0:.4f}",
            ha="center",
            fontsize=11,
            color="crimson",
        )

    return fig


def plot_cost_surface(
    x: np.ndarray,
    y: np.ndarray,
    cost: np.ndarray,
    result: FitResult,
    names: tuple[str, str] = ("beta", "alpha"),
) -> Figure:
    """The objective over a 2-D slice, with the truth and the estimate marked.

    Plotted on a log colour scale, because the cost varies by orders of
    magnitude across the box while the region that matters - the floor of the
    valley - is nearly flat.  The shape of the contours near the minimum is
    the identifiability diagnostic: concentric circles mean both parameters
    are determined, a long thin valley means only a combination of them is,
    and the tilt of that valley is the correlation reported by the fit.
    """
    fig, ax = plt.subplots(figsize=(8.8, 6.4))

    levels = np.linspace(np.log10(cost.min()), np.log10(cost.max()), 30)
    filled = ax.contourf(x, y, np.log10(cost), levels=levels, cmap="viridis")
    fig.colorbar(filled, ax=ax, label="log10 weighted sum of squares")
    ax.contour(x, y, np.log10(cost), levels=levels[:12], colors="white",
               linewidths=0.6, alpha=0.5)

    if result.truth is not None:
        ax.plot(
            result.truth[names[0]],
            result.truth[names[1]],
            "*",
            ms=20,
            color="red",
            mec="white",
            mew=1.2,
            label="true parameters",
            zorder=5,
        )
    ax.plot(
        result.estimates[names[0]],
        result.estimates[names[1]],
        "X",
        ms=14,
        color="white",
        mec="black",
        mew=1.4,
        label="least-squares estimate",
        zorder=5,
    )

    lo_x, hi_x = result.ci95[names[0]]
    lo_y, hi_y = result.ci95[names[1]]
    ax.add_patch(
        plt.Rectangle(
            (lo_x, lo_y),
            hi_x - lo_x,
            hi_y - lo_y,
            fill=False,
            ec="white",
            ls="--",
            lw=1.6,
            label="95% confidence box",
            zorder=4,
        )
    )

    _style(
        ax,
        f"Objective landscape over ({names[0]}, {names[1]})   "
        f"corr = {result.correlation[0, 1]:+.3f}",
        names[0],
        names[1],
    )
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.legend(frameon=False, labelcolor="white", loc="upper right")
    fig.tight_layout()
    return fig


def save_all(figures: Iterable[tuple[str, Figure]], outdir, dpi: int = 150) -> list[Path]:
    """Write ``(name, figure)`` pairs into ``outdir`` as PNGs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fig in figures:
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written
