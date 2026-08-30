# Provenance

`seedcert` is an independent package. Several modules began as copies of
`certiforget` (sibling directory) and were then adapted. This file records the
origin so a future refactor knows what is shared history and what diverged.
There is **no runtime dependency** on `certiforget`.

## Verbatim copies (package identifier rewritten `certiforget` -> `seedcert`)

| seedcert path | certiforget origin | notes |
|---|---|---|
| `src/seedcert/rng.py` | `src/certiforget/rng.py` | module docstring reworded |
| `src/seedcert/env.py` | `src/certiforget/env.py` | identical |
| `src/seedcert/hashing.py` | `src/certiforget/hashing.py` | identical |
| `src/seedcert/models/__init__.py` | same | identical |
| `src/seedcert/models/config.py` | same | identical |
| `src/seedcert/models/gcn.py` | same | identical |
| `src/seedcert/models/gat.py` | same | identical |
| `src/seedcert/models/sage.py` | same | identical |
| `src/seedcert/models/train.py` | same | docstring reworded; `RetainGraph` union retained (see `data/graph_ops.py`) |
| `src/seedcert/data/bundle.py` | same | optional alternative source only |
| `src/seedcert/data/graph_ops.py` | same | kept for the manifest / future edge work; `induced_retain_graph` unused in v1 |
| `src/seedcert/data/datasets.py` | `src/certiforget/data/reproduction.py` | **de-quarantined**: now the PRIMARY data path; added `SPLIT_PROTOCOLS` + `split_protocol_for` |

## Adapted from certiforget (structure reused, semantics changed)

| seedcert path | certiforget origin | change |
|---|---|---|
| `src/seedcert/certificate.py` | `certificate.py` | reproduction fields; `Decision` = REPRODUCED/DISCREPANT/INCONCLUSIVE; `minimum_seeds` == `minimum_oracles` |
| `src/seedcert/verifiers/base.py` | `verifiers/base.py` | `certify(runs, claim, ...)` signature |
| `src/seedcert/verifiers/nulls.py` | `verifiers/nulls.py` | one-sample rank test; drop the distance-to-set functional |
| `src/seedcert/verifiers/assumptions.py` | `verifiers/assumptions.py` | reproduction assumption strings |
| `src/seedcert/cache/spec.py` | `oracles/spec.py` | key = `(dataset, split_protocol, backbone, recipe_hash, seed)` |
| `src/seedcert/cache/registry.py` | `oracles/registry.py` | `RunRegistry`; single-GPU-per-recipe |
| `src/seedcert/cache/trainer.py` | `oracles/trainer.py` | trains on the full graph; four metrics |
| `src/seedcert/data/manifest.py` | `data/manifest.py` | manifest over canonical graphs; `resolve_num_classes` / `sha256_of` kept verbatim |
| `src/seedcert/experiment/timing_pilot.py` | `experiment/timing_pilot.py` | recipe timings, not oracle cells |
| `src/seedcert/experiment/smoke.py` | `experiment/smoke.py` | reproduction subcommands |
| `pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore`, `LICENSE` | same | renamed |

## New in seedcert

`recipe.py`, `claim.py`, `verifiers/reproduction.py`, `verifiers/metrics.py`,
`cache/runs.py`, `cache/build.py`, `experiment/grid.py`,
`experiment/published_claims.py`, `experiment/run_reproduction.py`,
`experiment/report.py`, and the `tests/` for all of the above.
