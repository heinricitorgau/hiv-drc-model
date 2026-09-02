# HIV/AIDS Transmission Dynamics with Treatment — The DRC Case

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/heinricitorgau/hiv-drc-model/actions/workflows/ci.yml/badge.svg)](https://github.com/heinricitorgau/hiv-drc-model/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/DOI-10.5556%2Fj.tkjm.57.5817.2026-orange)](https://doi.org/10.5556/j.tkjm.57.5817.2026)

A complete, tested reproduction of the six-compartment HIV/AIDS treatment model of

> C. K. Mbayi, J.-M. N. Mpompi and J. B. Munyakazi,
> *A model of HIV/AIDS transmission dynamics with treatment: The case of the DRC*,
> **Tamkang Journal of Mathematics** 57(2), 149–169 (2026).
> [doi:10.5556/j.tkjm.57.5817.2026](https://doi.org/10.5556/j.tkjm.57.5817.2026)

The package reproduces the published basic reproduction number to six digits, verifies
the paper's local-stability theorem numerically across the threshold, and adds a global
sensitivity analysis that the original does not include.

**It then runs the model backwards.** The paper solves the forward problem — rates in,
trajectories out. But $\beta$ and $\alpha$ are not quantities anyone measures; nobody counts
contacts. So this package also solves the **inverse problem**: given only noisy observations
of the two compartments a surveillance system can actually see, recover the hidden rates that
produced them.

- **Six-compartment ODE system** integrated with `solve_ivp`, verified against the
  population-balance identity and cross-checked across three solvers.
- **Synthetic data generator** with two Gaussian measurement-error models, so the estimator is
  developed against data whose true answer is known.
- **Bounded nonlinear least squares** (`scipy.optimize.least_squares`, trust-region
  reflective) with per-series residual weighting, multi-start global search, and asymptotic
  confidence intervals from the solver's Jacobian.
- **Honest uncertainty.** The fit reports what it cannot determine as well as what it can:
  95% intervals, a parameter correlation matrix, and a mapped objective landscape. Over 40
  replicates at 5% noise the intervals achieve 95% coverage.
- **Bayesian uncertainty quantification.** An ensemble MCMC sampler (`emcee`) explores the
  actual posterior instead of a local quadratic approximation to it, and infers each series'
  own measurement-noise level jointly with the epidemiological rates rather than guessing at
  it. Measured against the least-squares interval over 20 replicates at 10% noise: the same
  coverage in 2–4× tighter intervals, and no degenerate output on the run where the
  frequentist standard error blows up to `inf`.
- **Observation operators** mapping model state to what surveillance actually publishes
  (people living with HIV, ART coverage), because no health system reports "people currently
  in the symptomatic compartment" — with a measured answer to which data is worth collecting.
- **157 tests**, no test comparing the code to itself.

---

## Table of contents

- [The model](#the-model)
- [The basic reproduction number](#the-basic-reproduction-number)
- [Installation](#installation)
- [Usage](#usage)
- [Output figures](#output-figures)
- [Results](#results)
- [The inverse problem](#the-inverse-problem)
- [What can actually be observed](#what-can-actually-be-observed)
- [Bayesian uncertainty](#bayesian-uncertainty)
- [Testing and verification](#testing-and-verification)
- [Project structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## The model

Six compartments, all counted in **millions of people**, with time in **years**.

| Symbol | Compartment | Description |
| --- | --- | --- |
| $S$ | Susceptible | At risk, not infected |
| $I_1$ | Infected, unaware | Asymptomatic, status unknown to them |
| $I_2$ | Infected, aware | Asymptomatic, diagnosed |
| $A$ | Symptomatic | Has progressed to AIDS |
| $T$ | On treatment | Receiving antiretroviral therapy |
| $R$ | Behaviour-changed | Left the susceptible pool through protective behaviour |

Writing $N = S + I_1 + I_2 + A + T + R$ for the total population, the system is

$$
\begin{aligned}
\frac{dS}{dt}   &= \Lambda - \frac{\beta S}{N}\left(I_1 + cI_2 + dA\right) - \mu S - \phi S \\[4pt]
\frac{dI_1}{dt} &= \frac{\beta S}{N}\left(I_1 + cI_2 + dA\right) - \left(\sigma_1 + \lambda + \mu\right) I_1 \\[4pt]
\frac{dI_2}{dt} &= \lambda I_1 - \left(\sigma_2 + \mu + \alpha\right) I_2 + \kappa_2 T \\[4pt]
\frac{dA}{dt}   &= \sigma_1 I_1 + \sigma_2 I_2 - \left(\mu + \delta_1 + \kappa_1\right) A \\[4pt]
\frac{dT}{dt}   &= \alpha I_2 + \kappa_1 A - \left(\kappa_2 + \mu + \delta_2\right) T \\[4pt]
\frac{dR}{dt}   &= \phi S - \mu R
\end{aligned}
$$

Three features are worth pointing out before reading the results.

**Transmission is driven mainly by the undiagnosed.** All three infected classes transmit,
but $I_2$ and $A$ do so at reduced relative infectiousness $c = 0.03$ and $d = 0.001$.
Diagnosis is therefore modelled as an intervention in itself, independent of treatment.

**Treatment is not an absorbing state.** The $\kappa_2 T$ term returns people from $T$ to
$I_2$, representing interruption or failure of therapy. That feedback loop is what makes
the endemic equilibrium impossible to write down in closed form.

**The population is not constant.** Summing the six equations gives

$$\frac{dN}{dt} = \Lambda - \mu N - \delta_1 A - \delta_2 T,$$

so $N(t)$ is bounded above by the carrying capacity $\Lambda/\mu = 531.69$ million but is
free to grow toward it. This identity is checked in the test suite; it is the quickest way
to catch a term that leaves one compartment without arriving in another.

### Baseline parameters (DRC, 2020)

Reproduced from Table 3 of the paper. All rates are per year; $c$ and $d$ are dimensionless.

| Parameter | Symbol | Value | Meaning |
| --- | --- | --- | --- |
| `Lambda` | $\Lambda$ | 8.8261 | Recruitment into $S$ (millions/year) |
| `beta` | $\beta$ | 0.15 | Effective contact rate |
| `mu` | $\mu$ | 0.0166 | Natural death rate |
| `phi` | $\phi$ | 0.083 | Rate of adopting protective behaviour |
| `c` | $c$ | 0.03 | Relative infectiousness of $I_2$ |
| `d` | $d$ | 0.001 | Relative infectiousness of $A$ |
| `sigma1` | $\sigma_1$ | 0.0025 | Progression $I_1 \to A$ |
| `sigma2` | $\sigma_2$ | 0.06 | Progression $I_2 \to A$ |
| `lam` | $\lambda$ | 0.0015 | Diagnosis $I_1 \to I_2$ |
| `delta1` | $\delta_1$ | 0.0909 | Disease-induced death in $A$ |
| `delta2` | $\delta_2$ | 0.0667 | Disease-induced death on treatment |
| `alpha` | $\alpha$ | 0.035 | Treatment uptake $I_2 \to T$ |
| `kappa1` | $\kappa_1$ | 0.2 | Treatment uptake $A \to T$ |
| `kappa2` | $\kappa_2$ | 0.04 | Return $T \to I_2$ |

Initial conditions from Table 2, in millions:
$S(0) = 88.516$, $I_1(0) = 0.014$, $I_2(0) = 0.0846$, $A(0) = 0.30832$, $T(0) = 0.07708$, $R(0) = 0$.

---

## The basic reproduction number

With $a_1 = \sigma_1 + \lambda + \mu$, $a_2 = \sigma_2 + \mu + \alpha$,
$a_3 = \mu + \delta_1 + \kappa_1$, $a_4 = \kappa_2 + \mu + \delta_2$, and
$D = a_1\left(a_2a_3a_4 - \kappa_1\kappa_2\sigma_2 - \alpha\kappa_2 a_3\right)$, the paper's
next-generation calculation gives

$$R_0 = \frac{\beta\mu}{\mu + \phi}\left(R_1 + R_2 + R_3\right),$$

$$
R_1 = \frac{1}{a_1}, \qquad
R_2 = \frac{c\left(\lambda a_3 a_4 + \kappa_1\kappa_2\sigma_1\right)}{D}, \qquad
R_3 = \frac{d\left(a_4\left(\sigma_1\sigma_2 + \mu\sigma_1 + \lambda\sigma_2\right) + \alpha\sigma_1\left(\mu + \delta_2\right)\right)}{D}.
$$

Each term is the contribution of one infectious class, and the prefactor $\mu/(\mu+\phi)$ is
the susceptible fraction $S^*/N^*$ at the disease-free equilibrium — which is how behaviour
change $\phi$ enters.

This package implements the closed form **and**, independently, assembles the
next-generation matrices $F$ and $V$ and takes the spectral radius of $FV^{-1}$. The two
agree to twelve significant figures, and the test suite requires them to keep agreeing over
hundreds of randomly perturbed parameter sets. Without that second route, a typo in the
algebra would propagate silently into every figure in the repository.

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/heinricitorgau/hiv-drc-model.git
cd hiv-drc-model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Or, without installing the package, just the dependencies:

```bash
pip install -r requirements.txt
```

For development (adds pytest and ruff):

```bash
pip install -e ".[dev]"
```

Dependencies are only `numpy`, `scipy` and `matplotlib`.

---

## Usage

### Command line

Run everything and write all twelve figures to `figures/`:

```bash
python -m hiv_drc
```

The six stages are `dynamics`, `r0`, `stability`, `sensitivity`, `estimate` and `bayes`.
Individual stages can be selected by name:

```bash
python -m hiv_drc dynamics
```

```bash
python -m hiv_drc r0 stability
```

Generate noisy data and recover the parameters from it:

```bash
python -m hiv_drc estimate --noise 0.05 --starts 8 --data data/synthetic/observations.csv
```

Sample the posterior instead of finding a point estimate:

```bash
python -m hiv_drc bayes --noise 0.05 --mcmc-walkers 16 --mcmc-steps 1000
```

Useful options:

| Flag | Default | Effect |
| --- | --- | --- |
| `--outdir DIR` | `figures` | Where PNGs are written |
| `--years N` | `50` | Time horizon for the dynamics |
| `--samples N` | `1000` | Latin hypercube sample size |
| `--spread F` | `0.25` | Relative half-width of the sampling band |
| `--sweep-points N` | `41` | Points in the bifurcation sweep |
| `--show` | off | Open figures interactively instead of only saving |
| `--no-save` | off | Skip writing PNGs |

Options for the `estimate` stage:

| Flag | Default | Effect |
| --- | --- | --- |
| `--fit PARAM...` | `beta alpha` | Which parameters to recover |
| `--noise F` | `0.05` | Relative measurement noise on the synthetic data |
| `--noise-model M` | `proportional` | `proportional` or `constant` measurement error |
| `--obs-years N` | `30` | Length of the observation window |
| `--obs-points N` | `31` | Number of observations |
| `--seed N` | `20260830` | Seed for the noise and the multi-start draws |
| `--starts N` | `8` | Multi-start restarts for the optimiser |
| `--weights W` | `scale` | Residual weighting: `scale`, `sigma` or `none` |
| `--grid N` | `41` | Resolution of the cost-surface grid |
| `--data PATH` | none | Write the synthetic observations to this CSV |

Options for the `bayes` stage (in addition to `--fit`, `--noise`, `--noise-model`,
`--obs-years`, `--obs-points` and `--seed` above, which it shares with `estimate`):

| Flag | Default | Effect |
| --- | --- | --- |
| `--mcmc-walkers N` | `16` | Ensemble size for the MCMC sampler |
| `--mcmc-steps N` | `1000` | Steps per walker |
| `--mcmc-burn N` | `250` | Steps discarded as burn-in |

At the defaults this evaluates the forward model roughly 16,000 times; expect anywhere from
tens of seconds to a few minutes depending on the machine.

After `pip install -e .` the same thing is available as a console script:

```bash
hiv-drc --samples 2000 --outdir results
```

### As a library

```python
from hiv_drc import DRC_2020, reproduction_number, simulate

result = reproduction_number(DRC_2020)
print(result)                      # R0 = 1.214450 (R1 = 48.543689, ...)

solution = simulate(DRC_2020, t_span=(0.0, 50.0))
print(solution["I1"][-1])          # unaware infectives at t = 50
print(solution.prevalence[-1])     # infected share of the population
```

Parameters are an immutable dataclass, so scenarios are built by copying rather than
mutating:

```python
from hiv_drc import critical_beta, endemic_equilibrium

intervention = DRC_2020.replace(phi=0.09, alpha=0.05, sigma1=0.0125)
print(reproduction_number(intervention).R0)     # 0.7644 — below threshold

print(critical_beta(DRC_2020))                  # 0.123513

equilibrium = endemic_equilibrium(DRC_2020)
print(equilibrium.infected, equilibrium.is_stable)
```

Recovering parameters from data takes three lines:

```python
from hiv_drc import generate_observations, estimate_parameters

observations = generate_observations(noise=0.05, seed=20260830)   # or Observations.from_csv
fit = estimate_parameters(observations, fit=("beta", "alpha"))
print(fit.summary())
```

```
  param        estimate     std err                     95% CI      truth      error   rel err
  beta         0.165803    0.074724   [  0.016333,   0.315274]   0.150000  +0.015803   +10.54%
  alpha        0.037321    0.005075   [  0.027170,   0.047473]   0.035000  +0.002321    +6.63%

  cost = 1.648974e-01   RMSE (millions)   A: 0.004153   T: 0.005151
  R0 at the estimate = 1.342384
  corr(beta, alpha) = +0.1184
```

Sampling the posterior instead takes one more line, and answers a different question: not
"what is the best fit" but "what does the data actually rule out":

```python
from hiv_drc import run_mcmc

posterior = run_mcmc(observations, fit=("beta", "alpha"), n_walkers=16, n_steps=600, burn=150)
print(posterior.summary())
```

```
  param            median         std                     95% CI   R-hat      ESS
  beta           0.145171    0.021042   [  0.095625,   0.175486]   1.162      n/a   truth 0.150000  (-3.22%)
  alpha          0.034575    0.002350   [  0.030047,   0.039226]   1.146      n/a   truth 0.035000  (-1.22%)
  eta_A          0.046998    0.006543   [  0.037134,   0.062503]   1.109      n/a
  eta_T          0.056340    0.007368   [  0.044648,   0.074339]   1.157      n/a

  acceptance fraction 0.587   16 walkers x 600 steps (150 discarded as burn-in)
  R0 at the posterior median = 1.175355
```

`eta_A` and `eta_T` are not epidemiological parameters — they are each series' own inferred
relative measurement-noise level, fitted jointly rather than assumed.

---

## Output figures

All twelve are written to `figures/` (git-ignored — they are regenerated by one command).

| File | Contents |
| --- | --- |
| `01_population_dynamics.png` | Four panels: uninfected compartments ($S$, $R$), infected compartments ($I_1$, $I_2$, $A$, $T$), total population against the carrying capacity, and prevalence. $S$ and $R$ are drawn apart from the infected classes because they differ by three orders of magnitude. |
| `02_scenarios.png` | Total infected population under the baseline, a raised contact rate ($\beta = 0.2$), and the paper's enhanced-intervention scenario, each annotated with its $R_0$. |
| `03_r0_surface.png` | $R_0$ over the $(\beta, \sigma_1)$ plane. Left: filled contour with the $R_0 = 1$ threshold and the DRC operating point. Right: the same surface in 3-D cut by a translucent $R_0 = 1$ plane, the intersection drawn in black. |
| `04_bifurcation.png` | Endemic infected level against $R_0$, and the dominant Jacobian eigenvalue at both the disease-free equilibrium and the attractor. This is the picture of a forward transcritical bifurcation. |
| `05_local_sensitivity.png` | Tornado plot of the paper's normalised sensitivity indices. |
| `06_global_sensitivity.png` | Latin-hypercube/PRCC screen: PRCC on $R_0$, global versus local comparison, PRCC on the 50-year infected total, and the sampled $R_0$ distribution. |
| `07_parameter_fit.png` | The inverse problem: noisy observations of $A$ and $T$ as scatter, the recovered model as a curve, the generating trajectory dashed underneath, the residuals, and each estimate against its true value with 95% intervals. |
| `08_cost_surface.png` | The objective over the $(\beta, \alpha)$ plane on a log colour scale, with the true parameters, the least-squares estimate and the 95% confidence box marked. The shape of the basin is the identifiability diagnostic. |
| `09_posterior.png` | Pairwise marginal posteriors over $\beta$, $\alpha$ and the two fitted noise levels — the MCMC "corner plot". A round cloud means a pair is separately identified; an elongated one means only some combination of them is. |
| `10_trace.png` | Per-parameter walker traces: every walker's path over the kept steps, with its split-$\hat R$ annotated. A fuzzy horizontal band is a well-mixed chain; a visibly separated walker is not. |
| `11_posterior_predictive.png` | Observed $A$ and $T$ against the posterior *predictive* band (model plus each draw's own fitted noise), with the fraction of points actually inside the band reported per panel. |
| `12_bayes_vs_frequentist.png` | The least-squares Wald interval and the MCMC credible interval for the same fit, drawn on the same axis. Where they disagree, the quadratic approximation behind the Wald interval was already starting to break down. |

---

## Results

### The published value is reproduced

```
R1 = 48.543689   (unaware infectives I1)
R2 =  0.033712   (aware infectives I2)
R3 =  0.000614   (symptomatic A)

closed form (paper eq. 3.9-3.12) : R0 = 1.214450376356
spectral radius of F V^-1        : R0 = 1.214450376356
paper                            : R0 = 1.2145
```

$R_1$ dominates by three orders of magnitude. Because $c$ and $d$ are so small, essentially
all onward transmission comes from people who do not know they are infected — so in this
model diagnosis rate matters more than treatment capacity.

### The stability theorem holds numerically

The paper proves that the disease-free equilibrium is locally asymptotically stable when
$R_0 < 1$ and that the endemic equilibrium is stable when $R_0 > 1$. Sweeping $\beta$ over
41 points spanning the threshold:

```
sign(max Re lambda at DFE) disagrees with sign(R0 - 1) at 0 / 41 points
largest max Re(lambda) at the attractor: -1.63e-04   (should be <= 0)
```

At the DRC baseline:

| | $S$ | $I_1$ | $I_2$ | $A$ | $T$ | $R$ | max Re $\lambda$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Disease-free | 88.6155 | 0 | 0 | 0 | 0 | 443.0773 | **+0.004416** (unstable) |
| Endemic | 70.8708 | 85.7947 | 1.9860 | 1.0850 | 2.3238 | 354.3538 | **−0.003593** (stable) |

The bifurcation is **forward** (supercritical): the endemic branch leaves zero exactly at
$R_0 = 1$, with no backward branch below it. That is a substantive epidemiological result —
it means driving $R_0$ below one is by itself sufficient to clear the infection, with no
bistable region requiring a stricter target.

**The critical contact rate is $\beta = 0.1235$**, which is **17.7% below** the DRC baseline
of 0.15.

### Local and global sensitivity disagree in a useful way

The paper's local indices are reproduced exactly, including two with exact closed forms:
$R_0$ is linear in $\beta$, so its index is exactly $+1$, and the index for $\phi$ is exactly
$-\phi/(\mu+\phi) = -0.8333$.

A Latin-hypercube screen (n = 1000, every parameter varied by ±25% simultaneously) gives the
same top ranking, but two differences stand out:

| Parameter | PRCC on $R_0$ | Local index | PRCC on infected at $t = 50$ |
| --- | --- | --- | --- |
| `beta` | +0.979 | +1.000 | +0.987 |
| `phi` | −0.970 | −0.833 | −0.957 |
| `sigma1` | −0.469 | −0.121 | −0.128 |
| `lam` | −0.362 | −0.072 | −0.175 |
| `mu` | +0.133 | +0.027 | **−0.767** |
| `delta2` | −0.003 | −0.0002 | −0.377 |
| `c` | −0.011 | +0.0007 | +0.263 |

First, $\sigma_1$ is understated by the local analysis — its influence roughly quadruples once
the other parameters are allowed to move.

Second, and more importantly: **$\mu$ barely enters $R_0$ (local index +0.027) yet dominates
the 50-year infected burden (PRCC −0.767)**. The same is true to a lesser degree of
$\delta_2$, $c$ and $\kappa_2$. An analysis that ranks interventions by their effect on $R_0$
alone would miss all of them. $R_0$ is a statement about the long-run threshold, not about
what the epidemic does over a planning horizon.

Across the sampled band, **17.4% of parameter draws put $R_0$ below 1** — the DRC is not far
from the threshold.

### A caveat about the 50-year window

The endemic equilibrium carries 91.2 million infected at a prevalence of 17.7%. Over the
paper's 50-year simulation window the infected total falls from 0.484 million to a minimum of
0.160 and ends at 0.178 — three orders of magnitude away. This is not a contradiction: the
slowest timescale
in the system is $1/\mu \approx 60$ years, so **the published trajectories describe a transient,
not the long-run state**. The $R$ compartment is likewise still filling at $t = 50$ (it has no
outflow but natural death), and $N$ is still climbing toward $\Lambda/\mu$. Anyone comparing
these curves to observed DRC data should keep the distinction in view.

---

## The inverse problem

Everything above runs the model forward. This section runs it backwards.

### Why it is a different problem

The paper's parameters come from the literature, but the two that matter most for policy are
the two nobody measures directly. $\beta$ is an effective contact rate, a product of contact
frequency and per-contact transmission probability that has no instrument. $\alpha$ is the
rate at which diagnosed people start treatment, which programme registers only see
indirectly. Both have to be **inferred from their consequences**.

That inference is an inverse problem, and it is harder than the forward one in a specific
way: the forward map is a well-posed integration, while its inverse need not be unique or
stable. Two different parameter pairs can produce trajectories that differ by less than the
measurement noise, and no algorithm can separate them.

### The set-up

A surveillance system observes two of the six compartments — symptomatic cases $A$ and people
on treatment $T$. The other four, including *both* infected classes, are latent; undiagnosed
infection is precisely what is not counted. `synthetic.py` generates 31 annual observations
over 30 years from the published parameters and corrupts them with Gaussian error.
`estimation.py` is then handed the noisy series alone, with the generating values withheld,
and asked to recover them by solving

$$
\hat{\theta} \;=\; \arg\min_{\theta \in [\ell,\, u]}
    \sum_{k} \sum_{i} w_k^2 \left(y_k(t_i;\theta) - \hat{y}_{k,i}\right)^2
$$

with `scipy.optimize.least_squares` (trust-region reflective). Every evaluation of the
objective is a full integration of the six-compartment system.

### Four decisions that determine whether the answer is trustworthy

| Decision | Why it matters here |
| --- | --- |
| **Bounds** | Every rate is non-negative. Unbounded, a trust-region step tries $\beta < 0$, which turns the infection term into a *source* of susceptibles and the integration diverges. The box in `PARAMETER_BOUNDS` is wide enough not to be doing the estimating itself. |
| **Weighting** | A sum of squares in raw units is dominated by the largest compartment. Observing $S$ (≈ 88 million) beside $A$ (≈ 0.05 million), the $S$ residuals outweigh the $A$ residuals by roughly 800 : 1 and the fit ignores $A$ entirely. Each series is divided by its own magnitude, computed from the observations alone so nothing about the truth leaks in. |
| **Finite-difference step** | The residual is only accurate to the integrator's tolerance. Differencing it with the default $\sqrt{\epsilon} \approx 1.5 \times 10^{-8}$ amplifies solver noise by $10^{8}$ and the fit stalls short of the minimum. `diff_step` defaults to $10^{-6}$, keeping derivative noise near $10^{-4}$ relative. |
| **Identifiability** | A small residual does not mean the parameters are pinned down. `FitResult.correlation` and `cost_surface` are what to check before believing an estimate — and here they show that $\beta$ is only weakly determined by this data. |

### Does it work?

On **noise-free** data, recovery is exact to the limits of the arithmetic — relative errors of
$2.5\times10^{-8}$ for $\beta$ and $3.2\times10^{-10}$ for $\alpha$, from a starting point of
$\beta = 0.5$, $\alpha = 0.25$. Any error there would be the estimator's own, so this is the
first thing the test suite pins down.

On **noisy** data the estimate is a random variable, so a single run says little. Over 40
independent noise realisations, fitting $\beta$ and $\alpha$ from $A$ and $T$:

| Noise | $\beta$ bias | $\beta$ sd | $\alpha$ bias | $\alpha$ sd | 95% CI coverage |
| --- | --- | --- | --- | --- | --- |
| 2% | −0.5% | 6.9% | +0.5% | 4.3% | 98% |
| 5% | −3.7% | 21.8% | +1.3% | 10.9% | 95% |
| 10% | −11.2% | 41.3% | +3.3% | 21.9% | 88% |

Three things are worth reading off this table, and two of them are limitations:

1. **$\alpha$ is well determined; $\beta$ is not.** At 5% noise, $\alpha$ is recovered with a
   standard deviation of 11% while $\beta$ scatters by 22% — and the single default run
   overshoots $\beta$ by 10.5%. This is not an optimiser failure. $T$ responds to $\alpha$
   directly, whereas $\beta$ reaches the observed compartments only through the unobserved
   $I_1 \to I_2$ chain, so the data constrain it far more loosely. The mapped objective in
   `08_cost_surface.png` shows the same thing geometrically: a broad flat floor along $\beta$.
2. **The confidence intervals are calibrated where the linearisation holds.** At 5% noise
   they cover the truth 95% of the time, exactly nominal. At 10% they cover only 88% — the
   asymptotic Wald intervals become optimistic once the model's nonlinearity matters over the
   width of the interval. Reported intervals should be read with that in mind.
3. **The remaining parameters are assumptions, not results.** Only $\beta$ and $\alpha$ are
   fitted; the other twelve are held at published values. Being wrong about those biases the
   estimates, which is why the fitted set is kept small — `test_holding_the_wrong_baseline_biases_the_fit`
   pins that behaviour down rather than papering over it.

Eight multi-start runs from random points across the box all converge to the same estimate to
within $2\times10^{-6}$, so the optimum found is the global one for this objective.

---

## What can actually be observed

Everything above fits the compartments $A$ and $T$ directly. That is an
idealisation, and it is the thing standing between this package and real data: **no
surveillance system publishes "people currently in the symptomatic compartment."** UNAIDS, WHO
and national programmes report aggregates over a *different* partition of the same population:

| Published quantity | In model terms | Registry name |
| --- | --- | --- |
| People living with HIV | $I_1 + I_2 + A + T$ | `plhiv` |
| People on antiretroviral therapy | $T$ | `on_art` |
| Known HIV status | $I_2 + A + T$ | `diagnosed` |
| ART coverage | $T / (I_1 + I_2 + A + T)$ | `art_coverage` |
| HIV prevalence | $(I_1 + I_2 + A + T) / N$ | `prevalence` |

The model's decomposition (undiagnosed / diagnosed / symptomatic / treated) is a *modelling*
choice; the published decomposition is an artefact of what a health system can count. Fitting
real data means mapping between them, and `hiv_drc.observables` is that map — each name is an
**observation operator**, a function from model state to one observed series. Note that
`art_coverage` is a ratio, so the operators are callables rather than a matrix; a linear
formulation could not express it. The six compartments are registered as identity operators,
so observing a compartment is the ordinary case rather than a special one, and every existing
call keeps working.

```python
observations = generate_observations(observed=("plhiv", "on_art"), noise=0.05)
fit = estimate_parameters(observations, fit=("beta", "alpha"))
```

### Which data is worth collecting?

This turns an abstract question into a measurable one: does the decomposition that actually
gets published contain enough information to pin down $\beta$ and $\alpha$?
[`scripts/observability_study.py`](scripts/observability_study.py) fits the same underlying
epidemic through six observation sets, 20 noise realisations each at 5% noise. The **spread**
of the estimate across realisations is the practical identifiability measure — the mean error
is near zero everywhere, so it is the scatter that matters:

| Observation set | $\beta$ spread | $\alpha$ spread | Coverage | Degenerate fits |
| --- | --- | --- | --- | --- |
| `A` + `T` (the idealisation) | 14.5% | 9.6% | 95% | 0 |
| **`plhiv` + `on_art`** (what UNAIDS publishes) | **1.6%** | 11.8% | 95% | 0 |
| `prevalence` + `art_coverage` | 2.2% | 13.3% | 100% | 0 |
| `diagnosed` + `on_art` | 23.4% | 10.4% | 95% | 0 |
| `plhiv` alone | 5.0% | **117.6%** | 84% | 1 |
| `on_art` alone | **55.9%** | 14.1% | 89% | 2 |

Three things fall out of this, and the first is good news:

1. **The data that actually exists is better than the idealisation, by a lot.** Fitting
   `plhiv` + `on_art` pins $\beta$ down about **nine times more tightly** than the package's own
   `A` + `T` default (1.6% spread against 14.5%). $A$ is a small, fast-decaying compartment
   carrying little information about transmission; PLHIV aggregates the whole infected pool,
   including the undiagnosed $I_1$ where the transmission signal lives.
2. **Each series pins down a different rate, and one alone pins down neither.** PLHIV on its
   own leaves $\alpha$ essentially unidentified (118% spread, and one fit degenerates entirely);
   ART counts alone leave $\beta$ unidentified (56%). The transmission signal and the
   treatment-uptake signal live in different series, so a real study needs both.
3. **Excluding the undiagnosed costs you most of the transmission signal.** `diagnosed` +
   `on_art` differs from `plhiv` + `on_art` only by dropping $I_1$, and $\beta$'s spread goes
   from 1.6% to 23.4% — a fifteen-fold loss from omitting the one compartment nobody counts
   directly.

For anyone pointing this at real data: collect PLHIV and ART numbers, and do not substitute
"diagnosed" for "living with HIV".

---

## Bayesian uncertainty

Least squares answers "what is the single best fit, and how uncertain is it" by reading
curvature off the Jacobian at the optimum — the Wald interval. That is only as good as its
assumption that the log-likelihood is locally quadratic, and the coverage table above shows
that assumption under strain at 10% noise, where the nominal 95% interval covers the truth
only 88% of the time. `hiv_drc.bayesian` answers the same question a different way — by
sampling the actual posterior with an affine-invariant ensemble MCMC sampler
([`emcee`](https://emcee.readthedocs.io/)) — so a skewed or curved credible region comes out
skewed or curved, at the cost of tens of thousands of forward integrations instead of a
handful.

Measured head-to-head, the gain turns out **not** to be better coverage but **much tighter
intervals at the same coverage**, plus immunity to a failure mode the Wald interval has and
the posterior does not. The numbers are [below](#does-it-calibrate-better).

### The noise level is inferred, not assumed

Least-squares fitting sidesteps the question of how noisy the data actually are with an ad hoc
residual weighting (`estimation.weight_vector`), chosen to make two series of different
magnitude comparable — not because it is anyone's best guess at the true measurement error. The
Bayesian model asks the honest question instead: each observed series gets its own relative
noise level `eta`, given a log-uniform (Jeffreys) prior and inferred jointly with the
epidemiological rates. `test_log_likelihood_peaks_near_the_true_noise_level` checks that this
actually works — scanning the likelihood over a grid of noise levels at the true rates picks
out something close to the 5% level the synthetic data were generated with, without ever being
told what it was.

### Getting an expensive forward model to sample fast enough

Three engineering decisions matter here, each learned by measuring rather than guessing:

1. **Tighter priors than the least-squares safety box.** `estimation.PARAMETER_BOUNDS` is wide
   enough that a local gradient search never has reason to test its edges — $\beta$ up to 2,
   $\alpha$ up to 1. An ensemble sampler explores its *entire* prior support, including during
   burn-in, so reusing that box verbatim would waste most of the walk on contact and
   treatment-uptake rates with no epidemiological plausibility. `bayesian.MCMC_BOUNDS`
   overrides just those two with a still-generous but defensible range.
2. **Centring the ensemble on the least-squares estimate.** `run_mcmc` calls
   `estimate_parameters` first — a few hundredths of a second — and starts every walker in a
   tight jitter around that point rather than scattered across the prior. The sampler's budget
   then goes to characterising the posterior, not to finding it.
3. **A looser integrator tolerance than the frequentist fit uses**, since a single MCMC run
   needs on the order of $10^4$ evaluations rather than a few dozen, and each one only has to
   be smooth and stable, not accurate to the frequentist fit's ten digits.

At the CLI's defaults (16 walkers, 1,000 steps) that is roughly 16,000 integrations, which is
why `run_mcmc` takes tens of seconds to a few minutes rather than the fraction of a second
`estimate_parameters` needs — a real cost, disclosed rather than hidden behind a spinner.

### Reading the diagnostics honestly

A converged-looking posterior mean is worth nothing if the chain has not actually mixed, so
`BayesianFitResult` reports the numbers that would say so rather than only the numbers that
look good:

- **Split-$\hat R$** (`split_rhat`), the standard Gelman–Rubin diagnostic applied to each
  walker split in half. Validated against two synthetic cases with a known answer — chains
  drawn from the same distribution give $\hat R \approx 1.00$, chains stuck at unrelated offsets
  give $\hat R \gg 1$ — before ever being trusted on a real chain (`test_split_rhat_*`).
- **Effective sample size**, from emcee's own autocorrelation-time estimate. At the CLI's
  default budget this consistently reports itself as `None` rather than a falsely precise
  number: emcee's own rule of thumb wants the chain to run 50 times longer than the
  autocorrelation time it is measuring, which a two-fitted-parameter problem does not reach
  until well past what a demo run budgets for. Reporting `None` instead of guessing is the
  point.
- **Acceptance fraction**, which should sit roughly in emcee's own 0.2–0.5 comfort band; the
  defaults here land around 0.55–0.60.

### Does it calibrate better?

Twenty independent noise realisations at 10% noise, each fitted both ways
([`scripts/coverage_study.py`](scripts/coverage_study.py), about 20 minutes). Widths are
medians, because one frequentist run returns an infinite interval and would make a mean
meaningless:

| Parameter | Wald coverage | MCMC coverage | Wald width | MCMC width |
| --- | --- | --- | --- | --- |
| $\beta$ | 100% | 90% | 0.614 | 0.165 |
| $\alpha$ | 90% | 90% | 0.0361 | 0.0161 |

**The result is not the one this module was built expecting.** The hypothesis was that the
Wald interval would under-cover and MCMC would fix it. What actually happens is more
interesting:

1. **The Wald interval for $\beta$ hits 100% coverage by being nearly vacuous.** Its median
   width is 0.614 — four times the true value of $\beta = 0.15$, spanning essentially the whole
   plausible range of a contact rate. Over-covering a nominal 95% interval is not a success;
   it means the interval is too wide to constrain anything. The MCMC interval is **3.7× tighter**
   and still covers 90% of the time, far closer to its nominal rate.
2. **For $\alpha$ the coverage is identical at 90%, but the MCMC interval is 2.2× tighter.**
   Same calibration, less than half the width — the honest gain.
3. **One frequentist run in twenty degenerates completely.** On seed 2009 the optimiser drove
   $\beta$ onto its lower bound of zero, where the infection term vanishes identically and the
   model has no sensitivity to $\beta$ at all. The Jacobian loses rank, $\beta$'s standard error
   becomes `inf`, and — because `inf` propagates through the off-diagonal arithmetic — $\alpha$'s
   standard error comes back `nan` even though $\alpha$ itself was estimated fine (0.0401). MCMC
   on that same data returned a perfectly usable finite interval for both. Excluding that run,
   the frequentist numbers become 100% / 95% coverage and the picture above is unchanged.

**A caveat on these numbers.** Each replicate ran a deliberately short chain (16 walkers ×
500 steps) to keep the study to 20 minutes, and the worst split-$\hat R$ across all 20 runs and
both parameters was **1.449** — above the conventional 1.1 threshold. Some of those chains had
not fully converged, so the MCMC widths above should be read as indicative rather than
definitive; a longer run would likely widen them somewhat. The qualitative conclusion (much
tighter intervals at comparable coverage, and no degenerate output) is robust to that, but the
exact ratios are not.

---

## Testing and verification

```bash
pytest
```

157 tests. The frequentist ones finish in a few seconds; the Bayesian ones run real (small)
MCMC chains and add roughly one to three minutes depending on the machine — there is no way to
test a sampler's convergence diagnostics without actually sampling. The design principle
throughout is that **no test compares the code to itself** — each one checks against an
independent derivation or a property that must hold mathematically:

| What is checked | Against what |
| --- | --- |
| Closed-form $R_0$ | Spectral radius of $FV^{-1}$, over 300 random parameter sets |
| $D$ in the denominator | $\det V$, computed numerically |
| Complex-step Jacobian | Central finite differences |
| Equilibria | The residual $\max\lvert f(y^*)\rvert$ of the vector field |
| Threshold theorem | Sign of the dominant eigenvalue against sign of $R_0 - 1$, swept across the threshold |
| Trajectories | The population balance $dN/dt = \Lambda - \mu N - \delta_1 A - \delta_2 T$, and the invariant region $N \le \Lambda/\mu$ |
| Sensitivity indices | The exact values $+1$ for $\beta$ and $-\phi/(\mu+\phi)$ for $\phi$ |
| PRCC implementation | A synthetic problem with a known monotone answer |
| Integration | Agreement between LSODA, RK45 and Radau |
| Synthetic generator | Noise-free output against the integrator; empirical noise scale against the requested $\sigma$; CSV round trip |
| Parameter recovery | The known generating values, to $10^{-4}$ relative on noise-free data |
| The optimum itself | Cost at the estimate against cost at the truth — an early stop is invisible if only the parameter error is checked |
| Confidence intervals | Whether they contain the generating value across noise realisations |
| Cost surface | Grid minimum against the optimiser's estimate |
| `log_posterior` | The arithmetic identity `log_prior + log_likelihood`, independent of any sampler run |
| The noise-level prior | Whether the likelihood itself peaks near the true generating noise level, on a grid it was never told |
| Split-$\hat R$ | Two synthetic chains with a known answer: well-mixed gives $\approx 1$, stuck walkers give $\gg 1$ |
| MCMC recovery | Reproducibility under a fixed seed; agreement with the independent least-squares estimate; coverage of the truth |

Docstring examples are executed as part of the run (`--doctest-modules`).

A note on the Jacobian: it is computed by **complex-step differentiation**, perturbing each
coordinate into the imaginary direction and reading off
$\operatorname{Im} f(y + ih\mathbf{e}_j)/h$. Because there is no subtraction, there is no
cancellation error, so $h$ can be taken as small as $10^{-200}$ and the result is accurate to
machine precision. This works only because the right-hand side is analytic — it uses nothing
but arithmetic, with no branching or absolute values. That constraint is documented in
`model.py` and should be preserved by anyone extending the model.

---

## Project structure

```
hiv-drc-model/
├── src/hiv_drc/
│   ├── __init__.py          Public API
│   ├── __main__.py          Enables `python -m hiv_drc`
│   ├── parameters.py        Table 2 and Table 3 — the configuration block
│   ├── model.py             The ODE right-hand side and its Jacobian
│   ├── simulation.py        solve_ivp wrapper and the Solution type
│   ├── reproduction.py      R0: closed form and next-generation matrices
│   ├── equilibria.py        Disease-free and endemic equilibria, stability
│   ├── sensitivity.py       Local indices, Latin hypercube sampling, PRCC
│   ├── analysis.py          Parameter sweeps: R0 grid, bifurcation, threshold
│   ├── observables.py       What surveillance reports, as functions of the state
│   ├── synthetic.py         Mock data: forward model + Gaussian measurement error
│   ├── estimation.py        The inverse problem: bounded nonlinear least squares
│   ├── bayesian.py          The same inverse problem, solved by MCMC (emcee)
│   ├── plotting.py          Figures (no computation lives here)
│   └── cli.py               Command-line interface
├── tests/                   145 tests (157 with doctests)
│   ├── conftest.py          Shared fixtures
│   ├── test_model.py        ODE right-hand side and Jacobian
│   ├── test_simulation.py   Integration, conservation, solver agreement
│   ├── test_reproduction.py R0 closed form against the next-generation matrix
│   ├── test_equilibria.py   Equilibria and the stability theorem
│   ├── test_sensitivity.py  Local indices and PRCC
│   ├── test_observables.py  Operators, checked against identities between them
│   ├── test_synthetic.py    The data generator
│   ├── test_estimation.py   Recovery, bounds, uncertainty, identifiability
│   └── test_bayesian.py     Priors, likelihood, split-R-hat, MCMC recovery
├── notebooks/
│   └── 01_inverse_problem.ipynb   Narrated walkthrough of the estimation pipeline
├── scripts/
│   ├── coverage_study.py    One-off measurement behind the Bayesian coverage table
│   └── observability_study.py  Which observation sets identify the rates
├── data/
│   ├── README.md            Provenance and column definitions
│   └── synthetic/           Generated observations (`--data` writes here)
├── figures/                 Generated output (git-ignored)
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

The layering is deliberate: `plotting.py` never computes and the analysis modules never
plot, so a figure can be restyled without repeating a twenty-thousand-year integration, and
every analysis module stays importable on a machine with no display.

The same separation runs through the estimation layer. `synthetic.py` knows how data are made
and nothing about fitting; `estimation.py` knows how to fit and reaches for the generating
parameters only to *score* a result, never to produce one. That boundary is what makes a
recovery test meaningful — and it is why `Observations.from_csv` returns data with
`truth=None`, so real data and synthetic data take exactly the same path through the fitter.

`bayesian.py` sits beside `estimation.py` rather than on top of it: the two modules share
`synthetic.Observations` and `estimation.PARAMETER_BOUNDS`, and `run_mcmc` calls
`estimate_parameters` once to centre its walkers, but neither module imports the other's
result type. Either can be deleted without breaking the other.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{Mbayi2026HIVDRC,
  author  = {Mbayi, Charles K. and Mpompi, Jean-Marie N. and Munyakazi, Justin B.},
  title   = {A model of {HIV/AIDS} transmission dynamics with treatment:
             The case of the {DRC}},
  journal = {Tamkang Journal of Mathematics},
  volume  = {57},
  number  = {2},
  pages   = {149--169},
  year    = {2026},
  doi     = {10.5556/j.tkjm.57.5817.2026}
}
```

This repository is an independent reproduction. It is not affiliated with the authors, and
any error in the code is the fault of this implementation rather than of the paper.

---

## License

[MIT](LICENSE). The paper itself remains under its own publisher's terms.
