"""One-off measurement: what happens when the published model meets real data?

Fits the DRC UNAIDS/World Bank series over several windows and reports the
estimate from each. The answer is a negative result, and it is the most
useful thing this package has to say about applying the model to real data,
so it is measured here rather than asserted anywhere.

Requires ``data/real/drc_worldbank.csv`` - run ``scripts/fetch_worldbank.py``
first.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from hiv_drc import INITIAL_STATE, estimate_parameters, reproduction_number  # noqa: E402
from hiv_drc.realdata import initial_state_from_data, load_worldbank  # noqa: E402

WINDOWS = [(2005, 2024), (2010, 2024), (2016, 2024), (2020, 2024)]

# -- how the published initial conditions compare with the published data ----

obs_all, ctx_all = load_worldbank(first_year=2020, last_year=2020)
plhiv_2020 = float(obs_all.values["plhiv"][0])
coverage_2020 = float(obs_all.values["art_coverage"][0])
population_2020 = float(ctx_all["population"][0])

table2_plhiv = float(INITIAL_STATE[1:5].sum())
table2_coverage = float(INITIAL_STATE[4]) / table2_plhiv

print("Table 2 (labelled DRC 2020) against the UNAIDS estimates for 2020\n")
print(f"{'quantity':22s} {'Table 2':>12s} {'UNAIDS 2020':>13s} {'ratio':>8s}")
for label, paper, real in (
    ("total population", float(INITIAL_STATE.sum()), population_2020),
    ("people living w/ HIV", table2_plhiv, plhiv_2020),
    ("on treatment", float(INITIAL_STATE[4]), plhiv_2020 * coverage_2020),
    ("ART coverage", table2_coverage, coverage_2020),
):
    print(f"{label:22s} {paper:12.4f} {real:13.4f} {real / paper:7.2f}x")

# -- fitting the real series over several windows ---------------------------

print("\n\nFitting beta and alpha to the real series, by window\n")
header = (f"{'window':12s} {'n':>3s} {'beta':>9s} {'alpha':>9s} {'R0':>7s} "
          f"{'cov RMSE':>9s} {'usable CI':>10s}")
print(header)
print("-" * len(header))

for first, last in WINDOWS:
    obs, ctx = load_worldbank(first_year=first, last_year=last)
    y0 = initial_state_from_data(
        plhiv=float(obs.values["plhiv"][0]),
        art_coverage=float(obs.values["art_coverage"][0]),
        population=float(ctx["population"][0]),
    )
    fit = estimate_parameters(obs, fit=("beta", "alpha"), y0=y0)
    usable = all(np.isfinite(fit.stderr[name]) for name in fit.names)
    print(
        f"{first}-{last:<7d} {obs.n_points:3d} {fit.estimates['beta']:9.4f} "
        f"{fit.estimates['alpha']:9.4f} {reproduction_number(fit.parameters).R0:7.3f} "
        f"{fit.rmse['art_coverage']:9.4f} {str(usable):>10s}"
    )

print("\nalpha spans its entire admissible range across windows, sitting on a")
print("bound at both extremes; beta moves by an order of magnitude. A constant")
print("treatment-uptake rate cannot reproduce a coverage curve that policy drove")
print("from 1% to 71%, so each window's estimate reflects that window's slope.")
