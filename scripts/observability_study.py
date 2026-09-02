"""One-off measurement: which observable set actually identifies the rates?

The package's default observation set - the compartments ``A`` and ``T`` - is
an idealisation. No surveillance system publishes "people currently in the
symptomatic compartment"; what UNAIDS, WHO and national programmes publish are
aggregates over a different partition: people living with HIV, people on
treatment, ART coverage.

So the question that decides whether this package can fit real data is not
"can it fit" but "does the *published* decomposition contain enough
information to pin down beta and alpha at all". This script measures that by
fitting the same underlying epidemic through several observation sets and
comparing the resulting uncertainty.

Run once, read the numbers off, put them in the README by hand.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from hiv_drc import (  # noqa: E402
    DRC_2020,
    estimate_parameters,
    generate_observations,
)

NOISE = 0.05
SEEDS = range(20)

SETS: dict[str, tuple[str, ...]] = {
    "A + T (package default)": ("A", "T"),
    "PLHIV + on ART (UNAIDS)": ("plhiv", "on_art"),
    "PLHIV only": ("plhiv",),
    "on ART only": ("on_art",),
    "prevalence + ART coverage": ("prevalence", "art_coverage"),
    "diagnosed + on ART": ("diagnosed", "on_art"),
}

print(f"{len(SEEDS)} noise realisations at {100 * NOISE:.0f}% noise, "
      f"fitting beta and alpha\n")
header = (f"{'observation set':28s} {'beta err':>10s} {'beta sd':>9s} "
          f"{'alpha err':>10s} {'alpha sd':>9s} {'cover':>7s} {'degen':>6s}")
print(header)
print("-" * len(header))

for label, observed in SETS.items():
    beta_err, alpha_err, covers, degenerate = [], [], [], 0
    for seed in SEEDS:
        obs = generate_observations(observed=observed, noise=NOISE, seed=4000 + seed)
        fit = estimate_parameters(obs)
        errors = fit.relative_errors()
        beta_err.append(errors["beta"])
        alpha_err.append(errors["alpha"])
        widths = [fit.ci95[n][1] - fit.ci95[n][0] for n in fit.names]
        if not all(np.isfinite(w) for w in widths):
            degenerate += 1
        else:
            covers.append(all(fit.covers_truth().values()))
    coverage = 100.0 * float(np.mean(covers)) if covers else float("nan")
    print(
        f"{label:28s} {np.mean(beta_err):9.2f}% {np.std(beta_err):8.2f}% "
        f"{np.mean(alpha_err):9.2f}% {np.std(alpha_err):8.2f}% "
        f"{coverage:6.0f}% {degenerate:5d}"
    )

print("\nbeta err / alpha err are mean signed relative errors; sd is their")
print("spread across realisations - the practical identifiability measure.")
print(f"truth: beta = {DRC_2020.beta}, alpha = {DRC_2020.alpha}")
