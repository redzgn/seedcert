# seedcert — Design

Design of record. Written before any implementation (Session 0). Section
numbering mirrors `certiforget/DESIGN.md` where the two packages are parallel.

`seedcert` is an independent project. It reuses code *by copying*, not by
importing; there is no runtime dependency on `certiforget` and no shared cache or
lock file. `PROVENANCE.md` records which files began as copies.

---

## 0. What the package is

`seedcert` certifies whether a re-implementation of a published graph
neural network result **reproduces the reported number**, to within the
variation you get from re-running the same recipe under different random seeds.

Given a training recipe, a dataset, a split protocol, and a **published claim**
(the reported metric value, its source, and the split protocol it was measured
under), `seedcert` trains `n` seeds of the recipe and returns a
**`Certificate`**: the re-implementation's mean and confidence interval, a
two-sided rank p-value of the published value against the seed distribution, an
effect size, an explicit assumption list, an optional equivalence result, and
one of `REPRODUCED` / `DISCREPANT` / `INCONCLUSIVE`. As in `certiforget`, the
public API cannot return a bare float.

**Motivation.** Re-implementation and reproduction studies are a standard
contribution type for *Neurocomputing* and the Elsevier OSP tracks. In practice
a re-implementation is judged by eyeballing "close enough" to a paper's table
entry, with no reference distribution, no interval, and no statement of what was
held equal. Fair-comparison work (Shchur et al. 2018, *Pitfalls of GNN
Evaluation*; Errica et al. 2020) has shown how sensitive these numbers are to
seed and to split protocol. `seedcert` packages the check a reproduction needs:
is the gap between my number and the paper's number inside seed noise, and was
the comparison run under the same split.

**Scope.** One recipe versus one published value. `seedcert` does **not** compare
two recipes to each other, implement any GNN method itself, or search
hyperparameters — the recipe is supplied to match the paper.

---

## 1. Locked decisions

| ID | Decision |
|----|----------|
| **D1 Metric** | Headline = **test accuracy**; it alone drives `decision`. Macro **precision**, **recall**, **F1** are also computed and reported in `secondary` (with a per-metric p-value *iff* the claim supplies a value for that metric); they carry no verdict. No ROC-AUC. |
| **D2 Split protocol** | The re-implementation is run under the split protocol the claim names. Canonical published splits (Planetoid `public`, Geom-GCN fixed splits, Heterophilous fixed splits, WikiCS standard splits) are the **primary** data path, loaded by `data/datasets.py`. A claim whose `split_protocol` does not match the run's is a construction-time `ValueError` — `seedcert` will not certify across a split mismatch. |
| **D3 Test** | One-sample **rank location test** of the published value against the `n` per-seed metric values. `p_up = (1 + #{seed_i ≥ claim})/(n+1)`, `p_lo = (1 + #{seed_i ≤ claim})/(n+1)`. Default `direction = TWO_SIDED`: `p = min(1, 2·min(p_lo, p_up))` (doubled smaller tail — a **location** test, not the dispersion-style absolute-deviation-from-median form, which the WP2 dry run showed misses a claim that sits offset but inside the seed excursion envelope). One-sided `UPPER` ("re-impl underperforms the claim") / `LOWER` ("re-impl overperforms") available. |
| **D4 p-floor / minimum seeds** | The two-sided doubled-smaller-tail p floors at `2 / (n + 1)`; one-sided at `1 / (n + 1)`. `minimum_seeds(alpha) = floor(2/alpha − 1) + 1` ⇒ **`minimum_seeds(0.05) = 40`** (two-sided default); `minimum_seeds(alpha, two_sided=False) = floor(1/alpha − 1) + 1 = 20`. Default `n_seeds = 50` (headroom: floor `2/51 ≈ 0.039`). If `p_floor ≥ alpha`, `decision` is forced to `INCONCLUSIVE`. |
| **D5 Intervals / effect size** | `reimpl_mean` with a percentile bootstrap CI of the mean; Cliff's δ of `{claim}` vs `{seed values}` with a bootstrap CI (resampling seed values); `standardized_gap = (mean(seeds) − claim) / sd(seeds)`. |
| **D6 Equivalence** | Optional and **encouraged**: TOST with an **absolute** margin in metric points (e.g. `±1.0` accuracy point). `equivalence.equivalent == True` is the affirmative "reproduced within ±margin" statement — the honest positive result, stronger than a bare failure to reject. |
| **D7 Run cache** | Content-addressed by `(dataset, split_protocol, backbone, recipe_hash, seed)`. `recipe_hash` = blake2b of the canonical JSON of the `Recipe`. Per run: `state_dict.pt`, `logits.npy` (`[N, C]`, full-graph eval, best weights), `metrics.json` (four metrics on all splits), `env.json`. Single GPU model per recipe enforced at index build. |
| **D8 Recipe** | A `Recipe` is `(backbone, overrides: dict[str, Any], label: str)`, `backbone ∈ {gcn, gat, sage}`. `overrides` patches the frozen `Hyperparameters` dataclass (copied verbatim from `certiforget`) so the recipe can be set to match a paper — e.g. `hidden_dim=16` for Kipf & Welling. Only whitelisted fields are overridable; the override set is recorded in the certificate. |
| **D9 Independence** | No import of `certiforget`, no shared cache, no shared lock. `seedcert` owns its copied modules. |
| **D10 Estimand identity** (added at the cross-check, 2026-08-29) | `PublishedClaim.aggregation` names the estimand the value targets. `"single_run"` (default): tested by its rank among single re-implementation runs (D3/D4). `"mean"` (with `claimed_n_seeds = m ≥ 2`): the value is a mean over `m` runs, tested against the **sampling distribution of an `m`-run re-implementation mean** — `n_bootstrap` bootstrap resamples, each the mean of `m` seed values drawn with replacement — with the claim's rank in that distribution (two-sided doubled tail, floor `2/(n_bootstrap+1)`). `INCONCLUSIVE` unless `n_seeds ≥ minimum_seeds_aggregate(m) = max(20, m)`. A claim reported only as a multi-split mean, or under a split protocol with no canonical loader, is **out of scope** (an error), not `INCONCLUSIVE`. The certified `standardized_gap` and `effect_size` stay single-run-referenced in both modes for interpretability; the decision uses the mode's reference. |

### 1a. Relationship to `certiforget`'s FIX 1–3

- **FIX 1** — carried in spirit: the `minimum_seeds` / p-floor gate (D4). The
  bound is `2/(n+1)` here (two-sided default) rather than `certiforget`'s
  `1/(n+1)`, so `n ≥ 40` at `α = 0.05`.
- **FIX 2 analog** — the statistic (published value) and the reference (seed
  values) enter one function, `permutation_p_value`. No summary statistic of the
  seed set stands in for the set.
- **FIX 3** — not applicable (no membership-inference verifier).

### 1b. Decision valence (note — opposite of `certiforget`)

`NOT_DISTINGUISHABLE` here is the **good** outcome. The enum values are renamed to
avoid a misread:

| enum | meaning | p vs α |
|------|---------|--------|
| `REPRODUCED` | seed distribution is statistically consistent with the published value | `p ≥ α` |
| `DISCREPANT` | re-implementation differs from the published value at `α` | `p < α` |
| `INCONCLUSIVE` | `n` too small for `α` (FIX 1), or a required assumption check failed | — |

`REPRODUCED` is a failure to reject, **not** proof of equality; `summary()` says
so and points to the equivalence margin (D6).

---

## 2. The `Certificate` dataclass

Adapted from `certiforget/certificate.py` (frozen, slots, validate-on-construct,
`to_json`/`from_json`, `summary()`, `__float__` raises `TypeError`).

### 2.1 Fields

```
# provenance / identity
schema_version: str
verifier_name: str            # "reproduction"
verifier_version: str
created_at: str
env: dict[str, Any]
wall_clock_s: float

# target
dataset: str
split_protocol: str           # e.g. "planetoid-public"
recipe: dict[str, Any]        # {"backbone", "overrides", "label"}
recipe_hash: str
n_seeds: int
seed_list: tuple[int, ...]
metric_name: str              # "test_accuracy"
claim: dict[str, Any]         # {metric, value, source, split_protocol,
                              #  claimed_sd?, claimed_n_seeds?, doi?}

# statistical result
statistic: float              # = claim value under test
null_distribution: tuple[float, ...]   # the n per-seed metric values
p_value: float
p_floor: float                # 2/(n_seeds+1) two-sided, 1/(n_seeds+1) one-sided
test_direction: TestDirection # TWO_SIDED (default) or LOWER
reimpl_mean: float
reimpl_ci: tuple[float, float] # percentile bootstrap CI of the mean
effect_size: float            # Cliff's δ of {claim} vs {seeds}
effect_size_ci: tuple[float, float]
standardized_gap: float
ci_level: float
alpha: float
n_bootstrap: int
decision: Decision

# assumptions
assumptions: tuple[str, ...]
assumptions_checked: dict[str, bool | None]

# optional
equivalence: dict[str, Any] | None = None   # {margin_points, tost_p, equivalent}
secondary: dict[str, Any] = {}              # per macro metric:
                                            #   {reimpl_mean, reimpl_ci,
                                            #    claim_value | None, p_value | None}
```

### 2.2 Validation on construction

* `0 ≤ p_value ≤ 1`; `p_value ≥ p_floor`; `p_floor == 2/(n_seeds+1)` when
  `test_direction is TWO_SIDED`, else `1/(n_seeds+1)`.
* **FIX 1** — `p_floor ≥ alpha` ⇒ `decision is INCONCLUSIVE`.
* `reimpl_ci` ordered; `effect_size_ci` ordered and bracketing `effect_size`.
* `0 < alpha < 1`, `0 < ci_level < 1`, `n_seeds ≥ 1`.
* `len(seed_list) == n_seeds`; `len(null_distribution) == n_seeds`.
* `assumptions` non-empty; `test_direction ∈ {TWO_SIDED, LOWER}`;
  `decision` is a `Decision`.
* `recipe_hash` matches the canonical hash of `recipe`.
* `claim["split_protocol"] == split_protocol` **and**
  `claim["metric"] == metric_name` (else `ValueError`).
* `decision` consistent with `p_value` vs `alpha` unless `INCONCLUSIVE`.
* each `secondary[m]` has an ordered `reimpl_ci`.

### 2.3 Why a bare float is impossible

`__float__` raises `TypeError` directing the reader to `.decision`, `.p_value`,
`.reimpl_mean` / `.reimpl_ci`, `.equivalence`, and `.assumptions` together —
identical stance to `certiforget`.

---

## 3. `BaseVerifier` and `certify()`

### 3.1 Signature

```python
class ReproductionVerifier(BaseVerifier):
    name = "reproduction"
    version = "0.1.0"

    def certify(
        self,
        runs: RecipeRuns,
        claim: PublishedClaim,
        *,
        alpha: float = 0.05,
        ci_level: float = 0.95,
        n_bootstrap: int = 10_000,
        equivalence_margin_points: float | None = None,
        direction: TestDirection = TestDirection.TWO_SIDED,
        rng: int | np.random.Generator = 0,
    ) -> Certificate: ...
```

`runs` is the cached seed ensemble for one recipe on one dataset + split
protocol; it carries the recipe, the seeds, and the per-seed metric values.
`certify` raises `ValueError` when `runs.split_protocol != claim.split_protocol`
or `runs.n_seeds < 1`, and forces `INCONCLUSIVE` when
`runs.n_seeds < minimum_seeds(alpha)`.

### 3.2 `PublishedClaim`

```python
@dataclass(frozen=True, slots=True)
class PublishedClaim:
    metric: str            # "test_accuracy"
    value: float
    source: str            # "Kipf & Welling 2017, Table 2"
    split_protocol: str    # "planetoid-public"
    claimed_sd: float | None = None
    claimed_n_seeds: int | None = None
    doi: str | None = None
```

### 3.3 `BaseVerifier`

ABC with `name`, `version`, abstract `certify`. One concrete subclass in v1.

---

## 4. The reproduction test — statistic and reference

Let `s = [metric(seed) for seed in runs.seed_list]`, length `n`, the headline
metric on the frozen test split read from each run's cached logits.

* **Statistic** = `claim.value`.
* **Reference** = `s` (the seed distribution of the re-implementation).
* **p-value** — `permutation_p_value(claim.value, s, direction)` with
  `p_up = (1 + #{s_i ≥ claim})/(n+1)` and `p_lo = (1 + #{s_i ≤ claim})/(n+1)`:
  * `TWO_SIDED` (default): `min(1, 2·min(p_lo, p_up))` — doubled smaller tail.
  * `UPPER`: `p_up` — small p ⇒ the claim sits above the seeds, i.e. the
    re-implementation **underperforms** the paper (a regression).
  * `LOWER`: `p_lo` — small p ⇒ the re-implementation **overperforms** the paper.
* **p_floor** = `2/(n+1)` for `TWO_SIDED`, `1/(n+1)` for one-sided.
* **Intervals** — `reimpl_mean = mean(s)`; `reimpl_ci` = percentile bootstrap CI
  of `mean` over `n_bootstrap` resamples of `s`; `effect_size` = Cliff's δ of
  `{claim.value}` vs `s` with a bootstrap CI; `standardized_gap =
  (mean(s) − claim.value) / sd(s)`.
* **Equivalence** — when `equivalence_margin_points` is given: two one-sided
  tests that `|claim.value − mean(s)|` is within the absolute margin, on the
  bootstrap distribution of `mean(s)`; `equivalent == True` ⇒ "reproduced within
  ±margin points".

`secondary["precision"|"recall"|"f1"]` report `reimpl_mean` / `reimpl_ci` always,
and a p-value only if the `claim` carries a value for that metric.

---

## 5. Cached logits / metrics

Each run stores full-graph `[N, C]` logits in eval mode with best-epoch weights
(identical to `certiforget`'s trainer). All four metrics are computed from that
cache on the frozen test mask. Macro precision / recall / F1 come from a small
NumPy helper `macro_prf(logits, labels, mask)` — no scikit-learn dependency.

There is no retain graph, forget set, control pool, or query set: every run
trains on the full training split of the full graph.

---

## 6. Run cache / registry

`cache/spec.py`
: `RunKey(dataset, split_protocol, backbone, recipe_hash, seed)` and
  `RecipeKey` (no seed). `key_string =
  {dataset}/{split_protocol}/{backbone}/{recipe_hash}/seed{seed}`.
  `content_hash` binds the key to the dataset sha256 in `seedcert`'s own
  `datasets.lock.json`.

`cache/registry.py`
: `RunRegistry` — `index.parquet` read/write, `is_stale`, `completed_seeds`,
  `rebuild_index` (raises on a recipe whose `env.json` files disagree on
  `gpu_model`). Structure copied from `certiforget`'s `OracleRegistry`.

`cache/trainer.py`
: `train_run(*, key, data, recipe, hp, device, dataset_sha256) -> RunArtifacts`
  — seeds from `key.seed`, trains with `train_node_classifier` on the full
  graph, full-graph inductive forward, four metrics from the cache, env capture.

`cache/build.py`
: `build_recipe(recipe_key, *, registry, n_seeds, device, resume=True)` with a
  resumable `recipe_manifest.json`; `build_target(target, ...)` builds the
  recipe a `ReproTarget` names. CLI: `seedcert-build-runs`.

`cache/runs.py`
: `RecipeRuns(registry, recipe_key)` — lazily loads all seeds;
  `.metric_values(name) -> np.ndarray[n]`, `.logits_stack()`, `.seed_list`,
  `.recipe_descriptor()`, `.dataset`, `.split_protocol`, `.n_seeds`.

---

## 7. Reproduction grid

`experiment/grid.py` + `experiment/published_claims.py` — describe, do not run.

* `GRID_DATASETS` — the node-classification datasets with a usable canonical
  split and at least one published GCN/GAT/GraphSAGE number: Cora, CiteSeer,
  PubMed, Actor, Chameleon, Squirrel, and the Heterophilous set
  (Roman-Empire, Amazon-Ratings, Minesweeper, Tolokers, Questions), WikiCS.
* `PUBLISHED_CLAIMS` — `{(backbone, dataset): PublishedClaim}`, each with a
  `source` and `split_protocol`. Anchor sources (to be filled in Stage 2 with
  exact table references and DOIs):
  * Kipf & Welling 2017 — GCN, Planetoid `public` split (Cora / CiteSeer / PubMed).
  * Veličković et al. 2018 — GAT, Planetoid `public` split.
  * Platonov et al. 2023 — GCN / GraphSAGE / GAT on the Heterophilous set,
    fixed splits.
  * A benchmark source (e.g. Shchur et al. 2018) for GraphSAGE on Planetoid,
    with its random-splits protocol recorded as a distinct `split_protocol`.
* `REPRO_TARGETS` — `ReproTarget(backbone, overrides, dataset, claim)`; the
  `overrides` set the recipe to the paper's (e.g. `hidden_dim=16` for Kipf).
* `n_seeds` default **50**; **40** is the D4 floor for a two-sided decision at
  `α = 0.05`. Fewer than 40 → every certificate is `INCONCLUSIVE` by construction.

### Approximate cost

`≤ 12 datasets × ≤ 3 backbones × 50 seeds ≈ 1800` models, only where a claim
exists. At `certiforget`'s measured `0.1–1.2 s`/model, ≈ 15–25 min on one GPU.

---

## 8. Demonstrative validation (for the paper)

1. **Calibration.** Hold-one-seed-out: treat each of the `n ≥ 41` seed values in
   turn as a stand-in "claim" and certify it against the other `n−1` (`≥ 40`, so
   the two-sided test can resolve). p-values ≈ uniform (KS test), `REPRODUCED`
   rate ≈ `1 − α`. Mirrors `certiforget`'s leave-one-out calibration.
2. **Discrepancy power.** Certify a deliberately wrong claim (published value
   shifted by, say, 5 accuracy points) — must return `DISCREPANT`.
3. **Split-mismatch guard.** A claim whose `split_protocol` differs from the run
   raises at construction — shown as a worked example, not a silent pass.
4. **Reproduction sweep table.** For each `(backbone, dataset)` with a claim:
   `decision`, `reimpl_mean` ± CI, published value, `standardized_gap`,
   equivalence at `±1.0` point. This is the paper's Table 2 — a map of which
   canonical GNN numbers reproduce under a fixed, stated recipe and the matched
   split.

---

## 9. Rejected alternatives

* **Two-recipe comparison certifier** (the earlier "Fork A"). A valid tool, but
  it overlaps existing fair-comparison benchmarks and needs an arbitrary choice
  of comparison pairs. Dropped in favour of the self-contained reproduction
  question.
* **Bare `float` / "reproduction score".** Same objection as `certiforget`.
* **Certifying across a split mismatch with a warning.** A split-protocol
  difference alone moves Planetoid accuracy 1–2 points; permitting it would make
  most `DISCREPANT` verdicts uninterpretable. Made a hard error (D2).
* **scikit-learn for macro P/R/F1.** A heavy dependency for a few lines of NumPy.
* **TOST margin in seed-SD units.** Less interpretable for a reproduction claim
  than an absolute "±1.0 accuracy point"; absolute is primary (D6).
* **Absolute-deviation-from-median two-sided test** (originally D3, from
  `certiforget`). Keeps the two-sided floor at `1/(n+1)` but is a *dispersion*
  test: the WP2 dry run (2026-08-29) showed it returns `REPRODUCED` for a claim
  1.9 seed-SD above the mean and above every one of 22 seeds (`p = 0.13`).
  Replaced by the doubled-smaller-tail *location* test (D3), accepting the
  `2/(n+1)` floor and `n ≥ 40`.

---

## 10. Open items for sign-off

None blocking. Two things to settle during Stage 2 WP4 (they need library
sources, not a design decision):

1. The exact `PUBLISHED_CLAIMS` table — which paper/table each number comes from
   and its `split_protocol` string.
2. Whether GraphSAGE on Planetoid is in scope given its claim comes from a
   benchmark paper under a random-splits protocol (a different `split_protocol`,
   so more seeds and a note).
