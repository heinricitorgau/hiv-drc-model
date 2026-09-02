"""One-off measurement: which rates have to vary in time to fit the real data?

The published model holds every rate constant. Real DRC ART coverage went
from 1% in 2005 to 71% in 2024 under a scale-up of both testing and
treatment. This compares three model variants on the same series:

1. everything constant - the published model
2. a logistic scale-up of the treatment-uptake rate alpha
3. scale-ups of both alpha and the diagnosis rate lambda

Two findings come out, and the second matters more than the first. (2) barely
helps and (3) fits far better, because alpha moves people out of I2 while
only lambda puts them in - lambda is a hard ceiling on how much of the
infected population can ever be treated. But (3) is not *identified*: nine
parameters against twenty annual points of two series leaves runs with
indistinguishable cost landing on parameters that differ by two orders of
magnitude.

Multi-start is essential here: the scale-up parameters make the objective
multi-modal, and a single local solve lands wherever it happens to start.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from hiv_drc import (  # noqa: E402
    DRC_2020,
    PARAMETER_BOUNDS,
    estimate_multistart,
    estimate_parameters,
    simulate,
)
from hiv_drc.realdata import initial_state_from_data, load_worldbank  # noqa: E402

obs, context = load_worldbank(first_year=2005)
y0 = initial_state_from_data(
    plhiv=float(obs.values["plhiv"][0]),
    art_coverage=float(obs.values["art_coverage"][0]),
    population=float(context["population"][0]),
)
target = float(obs.values["art_coverage"][-1])
window = float(obs.t[-1])

VARIANTS = {
    "constant (published)": (
        DRC_2020,
        ("beta", "alpha"),
    ),
    "alpha(t)": (
        DRC_2020.replace(alpha_ceiling=0.2, alpha_midpoint=12.0, alpha_rate=0.3),
        ("beta", "alpha", "alpha_ceiling", "alpha_midpoint", "alpha_rate"),
    ),
    "alpha(t) + lambda(t)": (
        DRC_2020.replace(
            alpha_ceiling=0.2, alpha_midpoint=12.0, alpha_rate=0.3,
            lam_ceiling=0.2, lam_midpoint=12.0, lam_rate=0.3,
        ),
        ("beta", "alpha", "alpha_ceiling", "alpha_midpoint", "alpha_rate",
         "lam", "lam_ceiling", "lam_midpoint", "lam_rate"),
    ),
}

print(f"Fitting {obs.n_points} annual observations, "
      f"{int(context['year'][0])}-{int(context['year'][-1])}, "
      f"observing {' and '.join(obs.names)}")
print()
header = f"{'model':24s} {'params':>7s} {'cost':>11s} {'cov RMSE':>9s} {'plhiv RMSE':>11s}"
print(header)
print("-" * len(header))

fits = {}
for label, (baseline, names) in VARIANTS.items():
    fit = estimate_multistart(
        obs, fit=names, baseline=baseline, y0=y0, n_starts=12, seed=20260902
    )
    fits[label] = fit
    print(f"{label:24s} {len(names):7d} {fit.cost:11.4e} "
          f"{fit.rmse['art_coverage']:9.4f} {fit.rmse['plhiv']:11.4f}")

# -- why alpha alone cannot work -------------------------------------------

print()
print()
print("Why alpha alone cannot work: lambda caps the treatable fraction.")
print(f"Target ART coverage at year {window:g}: {100 * target:.0f}%")
print()
print(f"{'alpha':>10s} {'coverage':>10s}   (lambda held at the published 0.0015/yr)")
for alpha in (0.035, 0.5, 5.0, 50.0, 500.0):
    solution = simulate(DRC_2020.replace(alpha=alpha), y0, (0.0, window), n_points=40)
    coverage = solution["T"] / solution.infected
    print(f"{alpha:10.3f} {100 * float(coverage[-1]):9.1f}%")

print()
print(f"{'lambda':>10s} {'coverage':>10s}   (alpha held at 0.5/yr)")
for lam in (0.0015, 0.05, 0.2, 1.0):
    solution = simulate(DRC_2020.replace(alpha=0.5, lam=lam), y0, (0.0, window), n_points=40)
    coverage = solution["T"] / solution.infected
    print(f"{lam:10.4f} {100 * float(coverage[-1]):9.1f}%")

print()
print("alpha saturates: past roughly alpha = 5 the curve stops moving, because")
print("the queue it drains (I2) is refilled only at rate lambda. The published")
print("lambda = 0.0015/yr is a mean time to diagnosis of 667 years.")

best = fits["alpha(t) + lambda(t)"]
print()
print()
print("Best model, parameters recovered:")
print(best.summary())
print()
print("fitted scale-ups, start of window -> end:")
for name, at in (("alpha", best.parameters.alpha_at), ("lambda", best.parameters.lam_at)):
    print(f"  {name:6s} {at(0.0):.4f} -> {at(window):.4f} /yr")

# -- but are those nine parameters actually identified? ---------------------

print()
print()
print("Are the nine parameters identified? Ten random starts:")
print()
names = VARIANTS["alpha(t) + lambda(t)"][1]
baseline = VARIANTS["alpha(t) + lambda(t)"][0]
rng = np.random.default_rng(7)
rows = []
for _ in range(10):
    guess = {name: float(rng.uniform(*PARAMETER_BOUNDS[name])) for name in names}
    fit = estimate_parameters(obs, fit=names, baseline=baseline, y0=y0, guess=guess)
    rows.append((fit.cost, fit.estimates["beta"],
                 fit.estimates["alpha_ceiling"], fit.estimates["lam_ceiling"]))
rows.sort()

print(f"{'cost':>10s} {'beta':>10s} {'alpha_ceiling':>14s} {'lam_ceiling':>12s}")
for cost, beta, ceiling, lam_ceiling in rows:
    print(f"{cost:10.4f} {beta:10.4f} {ceiling:14.4f} {lam_ceiling:12.4f}")

print()
print(f"beta across starts: {min(r[1] for r in rows):.3f} - {max(r[1] for r in rows):.3f}")
print("Runs with indistinguishable cost land on wildly different parameters -")
print("look for rows agreeing on cost to three decimals whose lam_ceiling")
print("differs by two orders of magnitude. Every standard error above is inf")
print("for the same reason: nine parameters against twenty annual points of")
print("two series is not an identified problem, however well it fits.")
