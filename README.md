# seedcert

**Statistical certification that a graph neural network re-implementation
reproduces a published result.**

You re-implement a GNN method (or clone a repo) and want to know whether it
actually gets the number the paper reports. `seedcert` trains your recipe for
*n* seeds and returns a **`Certificate`**: the re-implementation's mean and
confidence interval, a two-sided rank p-value of the published value against the
seed distribution, an effect size, an explicit assumption list, an optional
equivalence result, and one of `REPRODUCED` / `DISCREPANT` / `INCONCLUSIVE`.

The comparison is run under the **split protocol the paper used** — `seedcert`
refuses to certify across a split-protocol mismatch — and the public API cannot
return a bare float.

## Install

```bash
pip install -e ".[dev]"
```

Python ≥ 3.10, PyTorch, PyTorch Geometric. CPU works; a GPU is faster. The
bundled 6-target reproduction sweep is ~300 model trainings, ~10 min on CPU.

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
)

registry = RunRegistry("run_cache")
key = RecipeKey("Cora", "planetoid-public", "gcn", recipe.recipe_hash())
build_recipe(key, recipe=recipe, registry=registry, n_seeds=50, device="cuda")

cert = ReproductionVerifier().certify(
    RecipeRuns(registry, key), claim,
    alpha=0.05, equivalence_margin_points=0.01,
)
print(cert.summary())
cert.to_json()
```

`n_seeds` must be ≥ 40 for a two-sided decision at `α = 0.05` (the doubled-
smaller-tail rank test floors at `2/(n+1)`); 50 is the default and leaves
headroom. One-sided `direction=TestDirection.UPPER` ("re-impl underperforms the
claim") resolves from 20.

## Command line

```bash
seedcert-manifest                       # pin the 12 canonical datasets in datasets.lock.json
seedcert-build-runs --dataset Cora --backbone gcn --overrides hidden_dim=16 --seeds 0-49
seedcert-certify --device cpu           # run the bundled reproduction sweep -> repro_certs/
seedcert-smoke calibrate --dataset Cora # hold-one-seed-out p-value uniformity
seedcert-timing                         # per-run timing + sweep projection
```

## The bundled reproduction sweep

`seedcert-certify` re-implements Kipf & Welling's GCN and Veličković et al.'s GAT
on Cora / CiteSeer / PubMed under the Planetoid `public` split and a fixed
recipe (val-loss early stopping, 200 epochs), and certifies each against the
reported number:

| backbone | dataset | claim | re-impl [95% CI] | decision |
|---|---|---|---|---|
| GCN | Cora | 0.815 | 0.798 [0.796, 0.801] | DISCREPANT |
| GCN | CiteSeer | 0.703 | 0.664 [0.659, 0.668] | DISCREPANT |
| GCN | PubMed | 0.790 | 0.788 [0.787, 0.789] | REPRODUCED (equivalent ±0.01) |
| GAT | Cora | 0.830 | 0.807 [0.805, 0.809] | DISCREPANT |
| GAT | CiteSeer | 0.725 | 0.676 [0.673, 0.679] | DISCREPANT |
| GAT | PubMed | 0.790 | 0.777 [0.775, 0.780] | REPRODUCED |

Under this standardized recipe, PubMed reproduces for both backbones and
Cora/CiteSeer for neither — the val-loss early-stopping *criterion*, not the
width or the epoch budget, is the driver.

## Documentation

`DESIGN.md` is the method of record; `PROVENANCE.md` lists which modules began as
copies of the sibling `certiforget` package. There is no runtime dependency on
`certiforget`.

## License

MIT. See `LICENSE`.
