# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproduction of Mbayi, Mpompi & Munyakazi, *Tamkang Journal of Mathematics* **57**(2), 149–169
(2026) — a six-compartment HIV/AIDS ODE model (S → I₁ → I₂ → A → T → R) for the DRC, plus the
inverse problem: recovering rates from noisy surveillance-shaped data by least squares and MCMC.
Populations are in **millions of people**, time in **years**, so every rate is 1/year.

The README is not a summary — it is the write-up of the results, including the ones that came out
negative. Read the relevant section before changing anything it describes.

## Commands

```bash
pip install -e ".[dev]"          # numpy, scipy, matplotlib, emcee + pytest, ruff, streamlit
pytest                           # 238 tests, ~3-5 min
pytest tests/test_estimation.py -q                       # one file
pytest tests/test_app.py -q -k "glyph"                   # one test
ruff check src tests             # exactly what CI lints; app.py is outside both
python -m hiv_drc                # all 12 figures into figures/
python -m hiv_drc r0 estimate --outdir /tmp/figures      # selected stages
streamlit run app.py             # the dashboard
```

Stages are `dynamics`, `r0`, `stability`, `sensitivity`, `estimate`, `bayes`. The bifurcation
sweep inside `stability` dominates the runtime; `--sweep-points` and `--samples` cut it down.

Two things make the suite slower than it looks: `--doctest-modules` is on, so **every docstring
example in `src/` is executed** and a stale one is a test failure; and the Bayesian tests run real
(small) MCMC chains, because convergence diagnostics cannot be tested without sampling.

`scripts/` holds one-off studies behind specific README claims. They are not part of the suite and
are expected to take minutes to hours. `scripts/fetch_worldbank.py` is the only code that touches
the network.

## Architecture

**`parameters.py` is the configuration block.** `Parameters` is a frozen dataclass — build variants
with `.replace(...)`, never mutate. `DRC_2020` is Table 3 of the paper and `INITIAL_STATE` is
Table 2. `COMPARTMENTS` fixes the state-vector order that everything else depends on.

**Layering is deliberate and load-bearing:**

- `plotting.py` never computes; the analysis modules never plot. A figure can be restyled without
  re-running a twenty-thousand-year integration, and every analysis module imports on a machine
  with no display.
- `synthetic.py` knows how data are made and nothing about fitting. `estimation.py` knows how to
  fit and reaches for the generating parameters only to *score* a result, never to produce one —
  that boundary is what makes a recovery test mean anything, and it is why
  `Observations.from_csv` returns `truth=None` so real and synthetic data take the same path.
- `bayesian.py` sits beside `estimation.py`, not on top: they share `Observations` and
  `PARAMETER_BOUNDS`, but neither imports the other's result type, and either could be deleted
  without breaking the other.
- `app.py` is presentation only. It imports the public API and computes nothing; keep it that way,
  or the dashboard becomes a second implementation that can disagree with the tested one.

**Observation operators** (`observables.py`) are what fits are defined over — a name resolves to a
function of a `Solution`, so `"plhiv"` or `"art_coverage"` can be fitted as directly as a raw
compartment.

## Invariants that will bite

**A time-varying rate makes the system non-autonomous, and most of the theory then does not
apply.** `reproduction_number()`, the derived exit rates `a1`/`a2`, and the equilibrium solvers
*raise* on a parameter set with a scale-up configured. That is intentional: R₀ and the stability
theorem are statements about constant coefficients. Use `reproduction_number_at(p, t)` for the
instantaneous value, and do not "fix" those exceptions by returning a number.

**`rhs` must stay analytic** — arithmetic only, no branching and no absolute values. `jacobian`
uses complex-step differentiation, which gives machine-precision derivatives *because* there is no
subtraction; a `max`, an `abs` or an `if` on the state silently degrades it instead of failing.

**`hiv_drc/__init__` imports `bayesian`, which imports `emcee`.** Even purely ODE work needs it
installed; a missing `emcee` looks like the package itself is broken.

**Estimation defaults are not arbitrary.** `diff_step=1e-6` (not `sqrt(eps)`) because the residual
is only as accurate as the integrator; `weights="scale"` so a large compartment cannot absorb the
fit; `x_scale="jac"` because β ≈ 0.15 and α ≈ 0.035. `weights="sigma"` only works on synthetic
data, where the generator recorded the σ it used.

**Non-identifiability is a documented finding, not a bug to hide.** Wide or infinite intervals,
`inf` standard errors and correlations near ±1 are reported deliberately (see the scale-up and
prior sections of the README). Do not narrow bounds or drop parameters to make output look tidier.

**`app.py` picks its figure language from the fonts present** — Chinese where a CJK font exists,
English where none does. matplotlib does not fail on a missing glyph; it draws a blank box, which
is invisible on a developer machine that has the font and wrong everywhere else.

## Testing conventions

The stated principle is that **no test compares the code to itself** — each checks against an
independent derivation (closed-form R₀ against the spectral radius of *FV*⁻¹, the complex-step
Jacobian against finite differences, PRCC against a problem with a known monotone answer) or a
property that must hold mathematically. Follow it when adding tests.

For a bug fix, confirm the new test **fails against the defect** before keeping it; several tests
here were verified that way and the commit messages say so. A test never observed failing is a
guess about what it covers.

`tests/test_app.py` drives the real `app.py` through Streamlit's `AppTest`; it skips when streamlit
is absent, and both the `dev` and `app` extras install it.

## Conventions

Commit messages here are narrative and report what was measured, including negative results and
what was ruled out — match that shape rather than writing one-line summaries. When a measured
number changes, the README's corresponding claim is expected to change with it.
