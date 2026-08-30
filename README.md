# seedcert

**Statistical certification that a graph neural network re-implementation
reproduces a published result.**

You re-implement a GNN method (or clone a repo) and want to know whether it
actually gets the number the paper reports. `seedcert` trains your recipe for
*n* seeds and returns a **`Certificate`**: the re-implementation's mean and
confidence interval, a rank test of the published value against a *matched*
reference — the seed distribution for a single-run claim, or the bootstrap
sampling distribution of an *m*-run mean when the paper reports an *m*-seed
average — an effect size, an explicit assumption list, an optional equivalence
result, and one of `REPRODUCED` / `DISCREPANT` / `INCONCLUSIVE`.

The comparison is run under the **split protocol the paper used** — `seedcert`
refuses to certify across a split-protocol mismatch — and the public API cannot
return a bare float.

## Install

```bash
pip install -e ".[dev]"
```

Python ≥ 3.10, PyTorch, PyTorch Geometric. CPU works; a GPU is faster. The
bundled 6-target case study is ~600 model trainings, ~25 min on one CPU core.

## Quickstart

```python
from seedcert.recipe import Recipe
from seedcert.claim import PublishedClaim
from seedcert.cache.registry import RunRegistry
from seedcert.cache.build import build_recipe
from seedcert.cache.runs import RecipeRuns
from seedcert.cache.spec import RecipeKey
from seedcert.verifiers.reproduction import ReproductionVerifier

recipe = Recipe("gcn", overrides={"hidden_dim": 16}, label="kipf-gcn")   # match the paper
claim = PublishedClaim(
    metric="test_accuracy", value=0.815,
    source="Kipf & Welling 2017, Table 2", split_protocol="planetoid-public",
    aggregation="mean", claimed_n_seeds=100,   # the paper's 0.815 is a 100-run mean
)

registry = RunRegistry("run_cache")
key = RecipeKey("Cora", "planetoid-public", "gcn", recipe.recipe_hash())
build_recipe(key, recipe=recipe, registry=registry, n_seeds=100, device="cuda")

cert = ReproductionVerifier().certify(
    RecipeRuns(registry, key), claim,
    alpha=0.05, equivalence_margin_points=0.01,
)
print(cert.summary())
cert.to_json()
```

For a single-run claim, `n_seeds` must be ≥ 40 for a two-sided decision at
`α = 0.05` (the doubled-smaller-tail rank test floors at `2/(n+1)`); one-sided
`direction=TestDirection.UPPER` ("re-impl underperforms the claim") resolves
from 20. For an `aggregation="mean"` claim the run count must be ≥ the paper's
`claimed_n_seeds`, otherwise the certificate is `INCONCLUSIVE`. The default is 100.

## Command line

```bash
seedcert-manifest                       # pin the 12 canonical datasets in datasets.lock.json
seedcert-build-runs --dataset Cora --backbone gcn --overrides hidden_dim=16 --seeds 0-99
seedcert-certify --device cpu           # run the bundled case study -> repro_certs/
seedcert-smoke calibrate --dataset Cora # hold-one-seed-out p-value uniformity
seedcert-timing                         # per-run timing + sweep projection
```

## The bundled case study

`seedcert-certify` re-implements Kipf & Welling's GCN and Veličković et al.'s GAT
on Cora / CiteSeer / PubMed under the Planetoid `public` split and one fixed
recipe (val-loss early stopping, 200 epochs), and certifies each reported value
— a 100-run mean in every case — against a matched 100-run-mean reference:

| backbone | dataset | claim | re-impl mean [95% CI] | decision |
|---|---|---|---|---|
| GCN | Cora | 0.815 | 0.797 [0.795, 0.799] | DISCREPANT |
| GCN | CiteSeer | 0.703 | 0.665 [0.662, 0.669] | DISCREPANT |
| GCN | PubMed | 0.790 | 0.788 [0.787, 0.789] | DISCREPANT — equivalent within ±1 pp |
| GAT | Cora | 0.830 | 0.807 [0.806, 0.809] | DISCREPANT |
| GAT | CiteSeer | 0.725 | 0.675 [0.673, 0.677] | DISCREPANT |
| GAT | PubMed | 0.790 | 0.778 [0.776, 0.779] | DISCREPANT |

All six certify `DISCREPANT`: the reported value sits outside the matched
100-run-mean reference (two-sided *p* at the bootstrap floor ≈ 2×10⁻⁴), and
every gap is negative — the re-implementation runs 0.5–4.4 single-run standard
deviations below the paper. GCN / PubMed is the instructive case: `DISCREPANT`
yet `equivalent` within ±1 percentage point, i.e. statistically distinguishable
but practically close. The certificate reports the gap and the equivalence
result separately so the two questions are not conflated.

## Documentation

`DESIGN.md` is the method of record; `PROVENANCE.md` lists which modules began as
copies of the sibling `certiforget` package. There is no runtime dependency on
`certiforget`.

## License

MIT. See `LICENSE`.
# seedcert

**Statistical certification that a graph neural network re-implementation
reproduces a published result.**

You re-implement a GNN method (or clone a repo) and want to know whether it
actually gets the number the paper reports. `seedcert` trains your recipe for
*n* seeds and returns a **`Certificate`**: the re-implementation's mean and
confidence interval, a rank test of the published value against a *matched*
reference — the seed distribution for a single-run claim, or the bootstrap
sampling distribution of an *m*-run mean when the paper reports an *m*-seed
average — an effect size, an explicit assumption list, an optional equivalence
result, and one of `REPRODUCED` / `DISCREPANT` / `INCONCLUSIVE`.

The comparison is run under the **split protocol the paper used** — `seedcert`
refuses to certify across a split-protocol mismatch — and the public API cannot
return a bare float.

## Install

```bash
pip install -e ".[dev]"
```

Python ≥ 3.10, PyTorch, PyTorch Geometric. CPU works; a GPU is faster. The
bundled 6-target case study is ~600 model trainings, ~25 min on one CPU core.

## Quickstart

```python
from seedcert.recipe import Recipe
from seedcert.claim import PublishedClaim
from seedcert.cache.registry import RunRegistry
from seedcert.cache.build import build_recipe
from seedcert.cache.runs import RecipeRuns
from seedcert.cache.spec import RecipeKey
from seedcert.verifiers.reproduction import ReproductionVerifier

recipe = Recipe("gcn", overrides={"hidden_dim": 16}, label="kipf-gcn")   # match the paper
claim = PublishedClaim(
    metric="test_accuracy", value=0.815,
    source="Kipf & Welling 2017, Table 2", split_protocol="planetoid-public",
    aggregation="mean", claimed_n_seeds=100,   # the paper's 0.815 is a 100-run mean
)

registry = RunRegistry("run_cache")
key = RecipeKey("Cora", "planetoid-public", "gcn", recipe.recipe_hash())
build_recipe(key, recipe=recipe, registry=registry, n_seeds=100, device="cuda")

cert = ReproductionVerifier().certify(
    RecipeRuns(registry, key), claim,
    alpha=0.05, equivalence_margin_points=0.01,
)
print(cert.summary())
cert.to_json()
```

For a single-run claim, `n_seeds` must be ≥ 40 for a two-sided decision at
`α = 0.05` (the doubled-smaller-tail rank test floors at `2/(n+1)`); one-sided
`direction=TestDirection.UPPER` ("re-impl underperforms the claim") resolves
from 20. For an `aggregation="mean"` claim the run count must be ≥ the paper's
`claimed_n_seeds`, otherwise the certificate is `INCONCLUSIVE`. The default is 100.

## Command line

```bash
seedcert-manifest                       # pin the 12 canonical datasets in datasets.lock.json
seedcert-build-runs --dataset Cora --backbone gcn --overrides hidden_dim=16 --seeds 0-99
seedcert-certify --device cpu           # run the bundled case study -> repro_certs/
seedcert-smoke calibrate --dataset Cora # hold-one-seed-out p-value uniformity
seedcert-timing                         # per-run timing + sweep projection
```

## The bundled case study

`seedcert-certify` re-implements Kipf & Welling's GCN and Veličković et al.'s GAT
on Cora / CiteSeer / PubMed under the Planetoid `public` split and one fixed
recipe (val-loss early stopping, 200 epochs), and certifies each reported value
— a 100-run mean in every case — against a matched 100-run-mean reference:

| backbone | dataset | claim | re-impl mean [95% CI] | decision |
|---|---|---|---|---|
| GCN | Cora | 0.815 | 0.797 [0.795, 0.799] | DISCREPANT |
| GCN | CiteSeer | 0.703 | 0.665 [0.662, 0.669] | DISCREPANT |
| GCN | PubMed | 0.790 | 0.788 [0.787, 0.789] | DISCREPANT — equivalent within ±1 pp |
| GAT | Cora | 0.830 | 0.807 [0.806, 0.809] | DISCREPANT |
| GAT | CiteSeer | 0.725 | 0.675 [0.673, 0.677] | DISCREPANT |
| GAT | PubMed | 0.790 | 0.778 [0.776, 0.779] | DISCREPANT |

All six certify `DISCREPANT`: the reported value sits outside the matched
100-run-mean reference (two-sided *p* at the bootstrap floor ≈ 2×10⁻⁴), and
every gap is negative — the re-implementation runs 0.5–4.4 single-run standard
deviations below the paper. GCN / PubMed is the instructive case: `DISCREPANT`
yet `equivalent` within ±1 percentage point, i.e. statistically distinguishable
but practically close. The certificate reports the gap and the equivalence
result separately so the two questions are not conflated.

## Documentation

`DESIGN.md` is the method of record.

## License

MIT. See `LICENSE`.
