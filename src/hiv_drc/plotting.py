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
from .parameters import COMPARTMENT_LABELS, Parameters
from .reproduction import reproduction_number
from .sensitivity import GlobalSensitivity
from .simulation import Solution

__all__ = [
    "plot_population_dynamics",
    "plot_scenarios",
    "plot_r0_surface",
    "plot_local_sensitivity",
    "plot_global_sensitivity",
    "plot_bifurcation",
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
