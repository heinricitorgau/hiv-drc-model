"""Command-line entry point.

``python -m hiv_drc`` runs the full analysis and writes every figure to
``figures/``.  Individual stages can be selected by name, which is useful
because the bifurcation sweep takes far longer than everything else combined.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import analysis, plotting
from .equilibria import disease_free_equilibrium, endemic_equilibrium
from .parameters import DRC_2020, INITIAL_STATE, Parameters
from .reproduction import next_generation_matrices, r0_from_ngm, reproduction_number
from .sensitivity import global_sensitivity, local_sensitivity_indices
from .simulation import simulate

STAGES = ("dynamics", "r0", "stability", "sensitivity")


def _rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def _scenarios() -> list[tuple[str, Parameters]]:
    """Baseline plus the two comparison scenarios discussed in the paper."""
    return [
        ("baseline (beta = 0.15)", DRC_2020),
        ("higher contact rate (beta = 0.2)", DRC_2020.replace(beta=0.2)),
        (
            "enhanced intervention",
            DRC_2020.replace(phi=0.09, alpha=0.05, sigma1=0.0125, sigma2=0.16, lam=0.0025),
        ),
    ]


def run_dynamics(args) -> list[tuple[str, object]]:
    _rule("BASELINE DYNAMICS")
    solution = simulate(DRC_2020, INITIAL_STATE, (0.0, args.years))

    print(f"initial state (millions): {np.array2string(INITIAL_STATE, precision=5)}")
    print(f"final state   (millions): {np.array2string(solution.y[:, -1], precision=5)}")
    print(f"total population: {solution.total_population[0]:.3f} "
          f"-> {solution.total_population[-1]:.3f} "
          f"(bound {DRC_2020.carrying_capacity:.3f})")
    print(f"infected total:   {solution.infected[0]:.5f} -> {solution.infected[-1]:.5f}")
    print(f"prevalence:       {100 * solution.prevalence[0]:.4f}% "
          f"-> {100 * solution.prevalence[-1]:.4f}%")

    _rule("SCENARIOS")
    solutions = []
    for label, p in _scenarios():
        sol = simulate(p, INITIAL_STATE, (0.0, args.years))
        R0 = reproduction_number(p).R0
        print(f"  {label:36s} R0 = {R0:.4f}   infected at t = {args.years:g}: "
              f"{sol.infected[-1]:.5f}")
        solutions.append((label, sol))

    return [
        ("01_population_dynamics", plotting.plot_population_dynamics(solution)),
        ("02_scenarios", plotting.plot_scenarios(solutions)),
    ]


def run_r0(args) -> list[tuple[str, object]]:
    _rule("BASIC REPRODUCTION NUMBER")
    result = reproduction_number(DRC_2020)
    print(f"  R1 = {result.R1:.6f}   (unaware infectives I1)")
    print(f"  R2 = {result.R2:.6f}   (aware infectives I2)")
    print(f"  R3 = {result.R3:.6f}   (symptomatic A)")
    print(f"  susceptible fraction at the DFE, mu/(mu+phi) = {result.theta:.6f}")
    print(f"\n  closed form (paper eq. 3.9-3.12) : R0 = {result.R0:.12f}")
    print(f"  spectral radius of F V^-1        : R0 = {r0_from_ngm(DRC_2020):.12f}")
    print("  paper                            : R0 = 1.2145")

    F, V = next_generation_matrices(DRC_2020)
    print(f"\n  det(V) = {np.linalg.det(V):.6e}")

    beta_c = analysis.critical_beta(DRC_2020)
    print(f"\n  critical contact rate: beta = {beta_c:.6f}")
    print(f"  that is {100 * (DRC_2020.beta - beta_c) / DRC_2020.beta:.1f}% below the "
          f"baseline beta = {DRC_2020.beta}")

    beta, sigma1, grid = analysis.r0_grid(DRC_2020)
    return [("03_r0_surface", plotting.plot_r0_surface(beta, sigma1, grid, DRC_2020))]


def run_stability(args) -> list[tuple[str, object]]:
    _rule("EQUILIBRIA AND LOCAL STABILITY")

    dfe = disease_free_equilibrium(DRC_2020)
    print("disease-free equilibrium")
    print(f"  state    : {np.array2string(dfe.state, precision=4)}")
    print(f"  residual : {dfe.residual:.3e}")
    print(f"  max Re(lambda) = {dfe.dominant_eigenvalue.real:+.6f}  "
          f"-> {'stable' if dfe.is_stable else 'unstable'}")

    ee = endemic_equilibrium(DRC_2020)
    print("\nendemic equilibrium")
    print(f"  state    : {np.array2string(ee.state, precision=4)}")
    print(f"  residual : {ee.residual:.3e}  ({ee.iterations} Newton iterations)")
    print(f"  infected : {ee.infected:.4f} million  "
          f"(prevalence {100 * ee.prevalence:.2f}%)")
    print(f"  max Re(lambda) = {ee.dominant_eigenvalue.real:+.6f}  "
          f"-> {'stable' if ee.is_stable else 'unstable'}")

    print(f"\nsweeping beta over {args.sweep_points} points ...")
    sweep = analysis.bifurcation_sweep(
        DRC_2020, beta=np.linspace(0.04, 0.30, args.sweep_points)
    )
    print(f"  sign(max Re lambda at DFE) disagrees with sign(R0 - 1) at "
          f"{sweep.threshold_mismatches} / {sweep.beta.size} points")
    print(f"  largest max Re(lambda) at the attractor: "
          f"{sweep.eig_attractor.max():+.6e}  (should be <= 0)")

    return [
        (
            "04_bifurcation",
            plotting.plot_bifurcation(
                sweep.R0, sweep.infected, sweep.eig_dfe, sweep.eig_attractor, ee
            ),
        )
    ]


def run_sensitivity(args) -> list[tuple[str, object]]:
    _rule("SENSITIVITY")

    local = local_sensitivity_indices(DRC_2020)
    print("local normalised indices (paper eq. 4.2-4.3)")
    for name, value in local.items():
        print(f"  {name:8s} {value:+.6g}")

    def infected_at_50(p: Parameters) -> float:
        return float(simulate(p, INITIAL_STATE, (0.0, 50.0), n_points=2).infected[-1])

    print(f"\nrunning LHS/PRCC with n = {args.samples} ...")
    result = global_sensitivity(
        DRC_2020,
        n_samples=args.samples,
        spread=args.spread,
        extra_outputs={"infected at t = 50": infected_at_50},
    )

    R0 = result.outputs["R0"]
    print(f"  R0 over the sample: median {np.median(R0):.4f}, "
          f"range [{R0.min():.4f}, {R0.max():.4f}], "
          f"P(R0 < 1) = {100 * np.mean(R0 < 1):.1f}%")

    print(f"\n  {'param':8s} {'PRCC(R0)':>10s} {'p':>10s} {'local':>12s} "
          f"{'PRCC(I@50)':>12s}")
    prcc_secondary = result.coefficients["infected at t = 50"]
    for name, coefficient, p_value, local_value in result.table("R0"):
        j = result.names.index(name)
        print(f"  {name:8s} {coefficient:+10.4f} {p_value:10.2g} "
              f"{local_value:+12.5f} {prcc_secondary[j]:+12.4f}")

    return [
        ("05_local_sensitivity", plotting.plot_local_sensitivity(local)),
        ("06_global_sensitivity", plotting.plot_global_sensitivity(result)),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hiv_drc",
        description="Reproduce the HIV/AIDS DRC treatment model "
        "(Tamkang J. Math. 57(2), 149-169, 2026).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "stages",
        nargs="*",
        default=list(STAGES),
        metavar="STAGE",
        help=f"which stages to run; any of {', '.join(STAGES)} (default: all)",
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures"),
                        help="where to write PNGs")
    parser.add_argument("--years", type=float, default=50.0,
                        help="time horizon for the dynamics")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Latin hypercube sample size")
    parser.add_argument("--spread", type=float, default=0.25,
                        help="relative half-width of the LHS sampling band")
    parser.add_argument("--sweep-points", type=int, default=41,
                        help="points in the bifurcation sweep")
    parser.add_argument("--show", action="store_true",
                        help="open the figures instead of only saving them")
    parser.add_argument("--no-save", action="store_true",
                        help="skip writing PNGs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    unknown = [stage for stage in args.stages if stage not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; choose from {', '.join(STAGES)}")

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    runners = {
        "dynamics": run_dynamics,
        "r0": run_r0,
        "stability": run_stability,
        "sensitivity": run_sensitivity,
    }

    started = time.perf_counter()
    figures: list[tuple[str, object]] = []
    for stage in args.stages:
        figures.extend(runners[stage](args))

    if not args.no_save:
        _rule(f"FIGURES -> {args.outdir}")
        for path in plotting.save_all(figures, args.outdir):
            print(f"  {path}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()

    print(f"\ndone in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
