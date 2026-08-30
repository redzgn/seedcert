"""Grid description + reproduction-target enumeration (DESIGN Sec 7, WP4)."""

from __future__ import annotations

from seedcert.experiment.grid import (
    DEFAULT_N_SEEDS,
    GRID_DATASETS,
    MIN_N_SEEDS,
    ReproTarget,
    repro_targets,
    run_count,
)
from seedcert.experiment.published_claims import PUBLISHED_CLAIMS, claim_for
from seedcert.recipe import Recipe


def test_dataset_list_is_deduped_and_nonempty() -> None:
    assert len(GRID_DATASETS) == len(set(GRID_DATASETS)) >= 10


def test_min_seeds_matches_fix1() -> None:
    from seedcert.certificate import minimum_seeds

    assert MIN_N_SEEDS == minimum_seeds(0.05) == 40  # single-run two-sided floor
    # the demonstrated claims are 100-run means, so the default matches
    assert DEFAULT_N_SEEDS == 100 >= MIN_N_SEEDS


def test_repro_target_builds_a_recipe_and_protocol() -> None:
    claim = PUBLISHED_CLAIMS[("gcn", "Cora")]
    t = ReproTarget(dataset="Cora", backbone="gcn", claim=claim, overrides={"hidden_dim": 16})
    assert isinstance(t.recipe(), Recipe)
    assert t.recipe().backbone == "gcn"
    assert t.split_protocol() == "planetoid-public"


def test_run_count_sums_seed_budgets() -> None:
    claim = PUBLISHED_CLAIMS[("gcn", "Cora")]
    targets = [
        ReproTarget("Cora", "gcn", claim, n_seeds=20),
        ReproTarget("CiteSeer", "gcn", claim, n_seeds=30),
    ]
    assert run_count(targets) == 50


def test_claim_for_lookup() -> None:
    assert claim_for("gcn", "Cora") is PUBLISHED_CLAIMS[("gcn", "Cora")]
    assert claim_for("sage", "Nonexistent") is None


def test_repro_targets_enumerates() -> None:
    targets = repro_targets()
    assert len(targets) == len(PUBLISHED_CLAIMS) == 6
    # every claim's protocol matches the dataset's canonical protocol
    assert all(t.claim.split_protocol == t.split_protocol() for t in targets)
    assert {t.split_protocol() for t in targets} == {"planetoid-public"}
    # overrides are attached (Kipf GCN -> hidden_dim 16)
    gcn_cora = next(t for t in targets if t.backbone == "gcn" and t.dataset == "Cora")
    assert gcn_cora.overrides == {"hidden_dim": 16}
    assert gcn_cora.recipe().backbone == "gcn"
    # default seed budget carries the two-sided floor
    assert all(t.n_seeds == DEFAULT_N_SEEDS >= 40 for t in targets)


def test_repro_targets_seed_override() -> None:
    assert all(t.n_seeds == 120 for t in repro_targets(n_seeds=120))


def test_repro_target_claims_are_100_run_means() -> None:
    for t in repro_targets():
        assert t.claim.aggregation == "mean"
        assert t.claim.claimed_n_seeds == 100
