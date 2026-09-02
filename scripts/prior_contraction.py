"""One-off measurement: do informative priors identify the scale-up model,
or do they just hand the answer back?

The nine-parameter scale-up fits the real DRC series well and is not
identified: runs with indistinguishable cost land two orders of magnitude
apart, and every standard error is inf. Informative priors are the standard
remedy. They are also the standard way to manufacture a result, because a
tight prior produces a tight posterior whether or not the data agreed.

The number that tells the two apart is the prior-to-posterior contraction,
1 - sd(posterior)/sd(prior): near 1 the data determined the parameter, near 0
the posterior is the prior repeated back. This runs the fit under
SCALEUP_PRIORS and reports it per parameter.

Requires data/real/drc_worldbank.csv - run scripts/fetch_worldbank.py first.
Takes a few minutes.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from hiv_drc import DRC_2020, PARAMETER_BOUNDS, run_mcmc  # noqa: E402
from hiv_drc.priors import SCALEUP_PRIORS, contraction  # noqa: E402
from hiv_drc.realdata import initial_state_from_data, load_worldbank  # noqa: E402

obs, context = load_worldbank(first_year=2005)
y0 = initial_state_from_data(
    plhiv=float(obs.values["plhiv"][0]),
    art_coverage=float(obs.values["art_coverage"][0]),
    population=float(context["population"][0]),
)

NAMES = (
    "beta",
    "alpha", "alpha_ceiling", "alpha_midpoint", "alpha_rate",
    "lam", "lam_ceiling", "lam_midpoint", "lam_rate",
)
BASELINE = DRC_2020.replace(
    alpha_ceiling=0.2, alpha_midpoint=11.0, alpha_rate=0.9,
    lam_ceiling=0.2, lam_midpoint=11.0, lam_rate=0.9,
)
# The scale-up ceilings need room the default MCMC box does not give them:
# alpha_ceiling's prior sits at 4/yr, well above the 1.0 default.
BOUNDS = {
    "alpha_ceiling": (0.0, 30.0),
    "lam_ceiling": (0.0, 5.0),
    "alpha_midpoint": (0.0, 25.0),
    "lam_midpoint": (0.0, 25.0),
    "alpha_rate": (0.0, 5.0),
    "lam_rate": (0.0, 5.0),
}

print("Priors in use, and where each comes from:\n")
for name in NAMES:
    prior = SCALEUP_PRIORS.get(name)
    if prior is None:
        print(f"  {name:16s} uniform on its box (no outside information)")
    else:
        print(f"  {name:16s} {type(prior).__name__:10s} {prior.why}")

# Budget is a command-line override because this problem mixes badly: the
# scale-up floors sit against their lower bound of zero, which the stretch
# move handles poorly, and the autocorrelation time runs to a few hundred
# steps. Short runs give a split-R-hat above 3 and contraction numbers that
# are artefacts of a chain still wandering, not of the parameter itself.
n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
burn = int(sys.argv[2]) if len(sys.argv) > 2 else 800
print(f"\n\nSampling with informative priors "
      f"(32 walkers x {n_steps} steps, {burn} burn-in) ...")
informed = run_mcmc(
    obs, fit=NAMES, baseline=BASELINE, y0=y0, bounds=BOUNDS,
    n_walkers=32, n_steps=n_steps, burn=burn, seed=20260902,
    priors=SCALEUP_PRIORS,
)
print(informed.summary())

print("\n\nDid the data actually inform these, or is the posterior an echo?\n")
header = f"{'parameter':16s} {'post. median':>13s} {'post. sd':>10s} {'contraction':>12s}   verdict"
print(header)
print("-" * len(header))
rng = np.random.default_rng(0)
box = dict(PARAMETER_BOUNDS)
box.update(BOUNDS)
for j, name in enumerate(NAMES):
    prior = SCALEUP_PRIORS.get(name)
    if prior is None:
        print(f"{name:16s} {informed.median[name]:13.4f} {informed.std[name]:10.4f} "
              f"{'n/a':>12s}   (uniform prior)")
        continue
    value = contraction(informed.samples[:, j], prior, rng, bounds=box[name])
    if value > 0.5:
        verdict = "data determined it"
    elif value > 0.2:
        verdict = "data helped"
    else:
        verdict = "PRIOR ECHO - not identified"
    print(f"{name:16s} {informed.median[name]:13.4f} {informed.std[name]:10.4f} "
          f"{value:12.2f}   {verdict}")

print("\nA tight credible interval with contraction near zero is not a result:")
print("it is the prior, restated. Read this column before the intervals above.")

print(f"\nworst split-R-hat: {max(informed.rhat.values()):.3f}")
