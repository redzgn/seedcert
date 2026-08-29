"""seedcert: statistical certification that a GNN re-implementation reproduces a
published result.

Public surface. Everything importable from here is API; submodules are
implementation detail. Stage 1: algorithm bodies raise ``NotImplementedError``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from seedcert.cache.registry import RunRecord, RunRegistry
from seedcert.cache.runs import RecipeRuns
from seedcert.cache.spec import RecipeKey, RunKey
from seedcert.certificate import Certificate, Decision, TestDirection, minimum_seeds
from seedcert.claim import PublishedClaim
from seedcert.recipe import Recipe, resolve_hyperparameters
from seedcert.verifiers.base import BaseVerifier
from seedcert.verifiers.reproduction import ReproductionVerifier

__all__ = [
    "__version__",
    "Certificate",
    "Decision",
    "TestDirection",
    "minimum_seeds",
    "PublishedClaim",
    "Recipe",
    "resolve_hyperparameters",
    "BaseVerifier",
    "ReproductionVerifier",
    "RecipeRuns",
    "RunKey",
    "RecipeKey",
    "RunRegistry",
    "RunRecord",
]
