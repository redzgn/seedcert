"""The ``Recipe`` - a backbone plus a whitelisted set of hyperparameter
overrides, set to match a published paper (DESIGN D8).

``recipe_hash`` is the content address of a run in the cache; it must be stable
under dict key ordering. ``resolve_hyperparameters`` applies the overrides to the
frozen :class:`~seedcert.models.config.Hyperparameters`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, replace
from typing import Any

from seedcert.models.config import HYPERPARAMETERS, Hyperparameters

_BACKBONES: frozenset[str] = frozenset({"gcn", "gat", "sage"})

#: Hyperparameter fields a recipe may override to match a paper. Anything not
#: listed is fixed for every recipe (DESIGN D8).
OVERRIDABLE_FIELDS: frozenset[str] = frozenset(
    {
        "num_layers",
        "hidden_dim",
        "dropout",
        "lr",
        "weight_decay",
        "max_epochs",
        "patience",
        "gat_heads",
        "gat_head_dim",
        "sage_aggregator",
    }
)


@dataclass(frozen=True, slots=True)
class Recipe:
    """A named training recipe: ``backbone`` in ``{gcn, gat, sage}``, a mapping of
    :data:`OVERRIDABLE_FIELDS` to values, and a short human ``label``."""

    backbone: str
    overrides: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def validate(self) -> None:
        """Check ``backbone`` and that every key of ``overrides`` is in
        :data:`OVERRIDABLE_FIELDS` and names a real ``Hyperparameters`` field.

        Raises:
            ValueError: unknown backbone or a non-whitelisted override key.
        """
        if self.backbone.lower() not in _BACKBONES:
            raise ValueError(
                f"unknown backbone {self.backbone!r}; expected one of {sorted(_BACKBONES)}"
            )
        hp_fields = {f.name for f in fields(Hyperparameters)}
        for key in self.overrides:
            if key not in OVERRIDABLE_FIELDS:
                raise ValueError(
                    f"override {key!r} is not permitted; overridable fields are "
                    f"{sorted(OVERRIDABLE_FIELDS)}"
                )
            if key not in hp_fields:  # pragma: no cover - guarded by the frozenset
                raise ValueError(f"override {key!r} is not a Hyperparameters field")

    def canonical_json(self) -> str:
        """Compact ``json.dumps`` of ``{backbone, overrides}`` with sorted keys -
        the pre-image of :meth:`recipe_hash`. ``label`` is excluded (cosmetic)."""
        payload = {
            "backbone": self.backbone.lower(),
            "overrides": {k: self.overrides[k] for k in sorted(self.overrides)},
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def recipe_hash(self) -> str:
        """``blake2b`` (16-byte digest, 32 hex chars) of :meth:`canonical_json`.
        Calls :meth:`validate` first so a malformed recipe cannot be cached."""
        self.validate()
        return hashlib.blake2b(self.canonical_json().encode(), digest_size=16).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        """``{"backbone", "overrides", "label"}`` for the certificate."""
        return {"backbone": self.backbone, "overrides": dict(self.overrides), "label": self.label}


def resolve_hyperparameters(
    recipe: Recipe,
    *,
    base: Hyperparameters = HYPERPARAMETERS,
) -> Hyperparameters:
    """Return ``base`` with ``recipe.overrides`` applied.

    Raises:
        ValueError: via :meth:`Recipe.validate` on a bad override key/backbone.
    """
    recipe.validate()
    if not recipe.overrides:
        return base
    return replace(base, **recipe.overrides)
