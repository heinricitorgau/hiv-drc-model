"""One-off measurement: how do the frequentist Wald interval and the MCMC
credible interval compare at 10% noise, where the Wald interval's own coverage
study (see the README) already drops to 88%?

Not part of the test suite - too slow, and coverage is a statement about
long-run frequency rather than a pass/fail property of one run, exactly like
the frequentist 40-replicate table this mirrors. Run once, read the numbers
off, put them in the README by hand.

Per-replicate results are written to `coverage_study.json` so the summary can
be re-derived without paying for the sampling again.

Two details this reports that a naive mean would hide:

* The Wald interval can come back **degenerate**. At 10% noise the optimiser
  occasionally drives beta onto its lower bound of zero, where the model has
  no sensitivity to beta at all, the Jacobian loses rank, and the reported
  standard error is `inf` (or `nan` for whatever it is correlated with). Such
  an interval "covers" the truth trivially, so counting it as a success would
  flatter the frequentist method for failing.
* Widths are summarised by the **median**, not the mean, for the same reason:
  one infinite interval makes a mean width infinite and an alpha width `nan`.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from hiv_drc import estimate_parameters, generate_observations, run_mcmc  # noqa: E402

N_REPLICATES = 20
NOISE = 0.10
NAMES = ("beta", "alpha")

started = time.perf_counter()
records = []

for seed in range(N_REPLICATES):
    obs = generate_observations(noise=NOISE, seed=2000 + seed)
    truth = {name: float(getattr(obs.parameters, name)) for name in NAMES}

    f = estimate_parameters(obs)
    b = run_mcmc(obs, n_walkers=16, n_steps=500, burn=120, seed=seed)

    record = {"seed": 2000 + seed, "truth": truth, "frequentist": {}, "bayesian": {}}
    for name in NAMES:
        lo_f, hi_f = f.ci95[name]
        record["frequentist"][name] = {
            "estimate": float(f.estimates[name]),
            "lo": float(lo_f),
            "hi": float(hi_f),
            "width": float(hi_f - lo_f),
            "covers": bool(lo_f <= truth[name] <= hi_f),
            "finite": bool(np.isfinite(hi_f - lo_f)),
        }
        lo_b, hi_b = b.credible_interval[name]
        record["bayesian"][name] = {
            "median": float(b.median[name]),
            "lo": float(lo_b),
            "hi": float(hi_b),
            "width": float(hi_b - lo_b),
            "covers": bool(lo_b <= truth[name] <= hi_b),
            "finite": bool(np.isfinite(hi_b - lo_b)),
            "rhat": float(b.rhat[name]),
        }
    records.append(record)
    print(f"[{seed + 1}/{N_REPLICATES}] elapsed {time.perf_counter() - started:.1f}s", flush=True)

with open("coverage_study.json", "w", encoding="utf-8") as handle:
    json.dump(records, handle, indent=1)


def summarise(method: str, name: str, only_finite_wald: bool) -> tuple[float, float, int]:
    rows = records
    if only_finite_wald:
        rows = [r for r in rows if all(r["frequentist"][n]["finite"] for n in NAMES)]
    covers = [r[method][name]["covers"] for r in rows]
    widths = [r[method][name]["width"] for r in rows]
    finite_widths = [w for w in widths if np.isfinite(w)]
    return (
        100.0 * float(np.mean(covers)),
        float(np.median(finite_widths)) if finite_widths else float("nan"),
        len(rows),
    )


degenerate = [r["seed"] for r in records if not all(r["frequentist"][n]["finite"] for n in NAMES)]

print(f"\n{N_REPLICATES} replicates at {100 * NOISE:.0f}% noise\n")
print(f"degenerate Wald intervals (non-finite width): {len(degenerate)} / {N_REPLICATES}"
      f"  seeds {degenerate}")
for seed in degenerate:
    row = next(r for r in records if r["seed"] == seed)
    print("   " + "   ".join(
        f"{n}: est {row['frequentist'][n]['estimate']:.5f} width {row['frequentist'][n]['width']}"
        for n in NAMES))

for label, only_finite in (("all replicates", False), ("degenerate Wald runs excluded", True)):
    print(f"\n-- {label} --")
    print(f"{'param':8s} {'freq cover':>11s} {'bayes cover':>12s} "
          f"{'freq width':>12s} {'bayes width':>12s}   (median widths)")
    for name in NAMES:
        fc, fw, n_rows = summarise("frequentist", name, only_finite)
        bc, bw, _ = summarise("bayesian", name, only_finite)
        print(f"{name:8s} {fc:10.0f}% {bc:11.0f}% {fw:12.5f} {bw:12.5f}   (n = {n_rows})")

worst_rhat = max(r["bayesian"][n]["rhat"] for r in records for n in NAMES)
print(f"\nworst split-R-hat across all runs and parameters: {worst_rhat:.3f}")
print(f"total time: {time.perf_counter() - started:.1f}s")
