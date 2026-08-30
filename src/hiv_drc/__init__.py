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
"""

from .analysis import BifurcationSweep, bifurcation_sweep, critical_beta, r0_grid
from .equilibria import (
    Equilibrium,
    disease_free_equilibrium,
    dominant_eigenvalue,
    endemic_equilibrium,
    is_locally_stable,
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
    # sweeps
    "r0_grid",
    "critical_beta",
    "bifurcation_sweep",
    "BifurcationSweep",
]
