"""HIV/AIDS transmission dynamics with treatment - the DRC case.

A reproduction of

    C. K. Mbayi, J.-M. N. Mpompi and J. B. Munyakazi,
    "A model of HIV/AIDS transmission dynamics with treatment:
    The case of the DRC",
    Tamkang Journal of Mathematics 57(2), 149-169 (2026).
    doi:10.5556/j.tkjm.57.5817.2026

Quick start
-----------
>>> from hiv_drc import DRC_2020, reproduction_number, simulate
>>> round(reproduction_number(DRC_2020).R0, 4)
1.2145
>>> solution = simulate(t_span=(0.0, 50.0))
>>> solution.infected[-1] < solution.infected[0]
np.True_

Inverse problem
---------------
>>> from hiv_drc import generate_observations, estimate_parameters
>>> observations = generate_observations(noise=0.05, seed=1)
>>> fit = estimate_parameters(observations, fit=("beta", "alpha"))
>>> bool(abs(fit.relative_errors()["alpha"]) < 20.0)
True

Bayesian uncertainty
--------------------
>>> from hiv_drc import run_mcmc
>>> posterior = run_mcmc(observations, n_walkers=10, n_steps=200, burn=60, seed=1)
>>> bool(abs(posterior.relative_errors()["alpha"]) < 30.0)
True
"""

from .analysis import BifurcationSweep, bifurcation_sweep, critical_beta, r0_grid
from .bayesian import (
    ETA_BOUNDS,
    MCMC_BOUNDS,
    BayesianFitResult,
    log_likelihood,
    log_posterior,
    log_prior,
    run_mcmc,
    split_rhat,
)
from .equilibria import (
    Equilibrium,
    disease_free_equilibrium,
    dominant_eigenvalue,
    endemic_equilibrium,
    is_locally_stable,
)
from .estimation import (
    PARAMETER_BOUNDS,
    FitResult,
    cost_surface,
    estimate_multistart,
    estimate_parameters,
    predict,
    residuals,
    weight_vector,
)
from .model import force_of_infection, jacobian, rhs, total_population
from .parameters import (
    COMPARTMENTS,
    DRC_2020,
    INITIAL_STATE,
    R0_PARAMETERS,
    Parameters,
)
from .reproduction import (
    ReproductionNumber,
    next_generation_matrices,
    r0_from_ngm,
    reproduction_number,
)
from .sensitivity import (
    GlobalSensitivity,
    global_sensitivity,
    local_sensitivity_index,
    local_sensitivity_indices,
    prcc,
)
from .simulation import Solution, simulate
from .synthetic import (
    NOISE_MODELS,
    OBSERVABLE,
    Observations,
    generate_observations,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # parameters
    "Parameters",
    "DRC_2020",
    "INITIAL_STATE",
    "COMPARTMENTS",
    "R0_PARAMETERS",
    # model
    "rhs",
    "jacobian",
    "force_of_infection",
    "total_population",
    # simulation
    "simulate",
    "Solution",
    # reproduction number
    "reproduction_number",
    "ReproductionNumber",
    "next_generation_matrices",
    "r0_from_ngm",
    # equilibria
    "disease_free_equilibrium",
    "endemic_equilibrium",
    "Equilibrium",
    "dominant_eigenvalue",
    "is_locally_stable",
    # sensitivity
    "local_sensitivity_index",
    "local_sensitivity_indices",
    "global_sensitivity",
    "GlobalSensitivity",
    "prcc",
    # synthetic data
    "Observations",
    "generate_observations",
    "OBSERVABLE",
    "NOISE_MODELS",
    # parameter estimation
    "estimate_parameters",
    "estimate_multistart",
    "FitResult",
    "predict",
    "residuals",
    "weight_vector",
    "cost_surface",
    "PARAMETER_BOUNDS",
    # Bayesian uncertainty
    "run_mcmc",
    "BayesianFitResult",
    "log_prior",
    "log_likelihood",
    "log_posterior",
    "split_rhat",
    "ETA_BOUNDS",
    "MCMC_BOUNDS",
    # sweeps
    "r0_grid",
    "critical_beta",
    "bifurcation_sweep",
    "BifurcationSweep",
]
