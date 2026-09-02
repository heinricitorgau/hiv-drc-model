"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from hiv_drc import DRC_2020, Parameters


@pytest.fixture
def baseline() -> Parameters:
    """The DRC 2020 parameter set."""
    return DRC_2020


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so a failure can always be reproduced."""
    return np.random.default_rng(20260830)


def perturbed(p: Parameters, rng: np.random.Generator, spread: float = 0.5) -> Parameters:
    """A random parameter set within ``+/- spread`` of ``p``, all rates positive.

    Only the rates are perturbed.  The scale-up fields are left alone: the
    ``*_ceiling`` switches are ``None`` when no scale-up is configured, and
    scaling them at random would turn an autonomous system into a
    non-autonomous one, which is not what any caller of this helper is
    testing.
    """
    return p.replace(
        **{
            name: value * (1.0 + spread * rng.uniform(-1.0, 1.0))
            for name, value in p.as_dict().items()
            if isinstance(value, float) and not name.endswith(("_ceiling", "_midpoint", "_rate"))
        }
    )
