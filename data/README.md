# Data

## Provenance

Everything in `synthetic/` is **generated, not observed**. It is produced by
`hiv_drc.synthetic.generate_observations`, which integrates the six-compartment model at the
published DRC 2020 parameters and adds Gaussian measurement error. No file here contains real
surveillance data, and none should be cited as if it did.

The point of generating data rather than collecting it is that the answer is known. When a fit
comes out wrong on real data there is no way to tell whether the optimiser, the model or the
data is at fault; here there is.

Regenerate with:

```bash
python -m hiv_drc estimate --data data/synthetic/observations.csv
```

## `synthetic/observations.csv`

31 annual observations over 30 years, 5% proportional noise, seed `20260830`.

| Column | Units | Meaning |
| --- | --- | --- |
| `t` | years | Time since the start of the observation window |
| `A` | millions | **Observed** symptomatic (AIDS) population, with noise |
| `T` | millions | **Observed** population on antiretroviral therapy, with noise |
| `A_true` | millions | The noise-free trajectory `A` was drawn from |
| `T_true` | millions | The noise-free trajectory `T` was drawn from |

The `_true` columns exist only to score the estimator. They are never read by the fitting
code: `Observations.from_csv` loads them into `truth`, which `FitResult` uses to report error
against, while the residuals are computed from the observed columns alone. A file with no
`_true` columns — real data — takes exactly the same path through the fitter and simply
reports no error.

## Adding real data

To fit real observations, write a CSV with a `t` column and one column per observed
compartment named as in `hiv_drc.parameters.COMPARTMENTS` (`S`, `I1`, `I2`, `A`, `T`, `R`),
then:

```python
from hiv_drc import Observations, estimate_parameters

observations = Observations.from_csv("data/real/drc_notifications.csv")
fit = estimate_parameters(observations, fit=("beta", "alpha"))
```

Two things to settle before trusting the output. The initial state passed as `y0` must match
the first row of the data, since the fit conditions on it rather than estimating it. And the
parameters *not* being fitted are assumptions — being wrong about them biases the estimates,
so keep the fitted set small and be able to defend the rest.
