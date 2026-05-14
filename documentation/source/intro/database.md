# Calabi-Yau Geometry Database

## Overview

stringforge provides access to large databases of Calabi-Yau threefold geometries through a unified interface. The data is hosted on [HuggingFace](https://huggingface.co/datasets/aschachner/cy-database) and downloaded on demand — only the files you actually need are fetched and cached locally.

Two databases are currently available:

- **Toric Divisor Flux (TDF)** models from the [Kreuzer-Skarke list](https://arxiv.org/abs/hep-th/0002240), identified by `(ks_id, triang_id)`. These are Calabi-Yau hypersurfaces in toric varieties, constructed from reflexive polytopes.
- **Complete Intersection Calabi-Yau (CICY)** models from the [CICY list](https://arxiv.org/abs/hep-th/8602060), identified by `cicy_id`.

Each model in the database carries:
- Topological data: intersection numbers $\kappa_{ijk}$, second Chern class $c_2$, Euler characteristic $\chi$, Hodge numbers $h^{1,1}$ and $h^{2,1}$
- Kähler cone data: generators, hyperplane constraints
- (Optional) Gopakumar-Vafa / Gromov-Witten invariants
- (Optional) Conifold limit data: conifold curves, GV invariants of shrinking cycles, basis change matrices
- (Optional) Extra precomputed properties: D3-tadpole $\chi/24$, etc.

## Two ways to load models

The stringforge ecosystem supports two complementary ways to load Calabi-Yau geometry:

### Local models (`model_ID`)

A small set of pre-computed models is shipped with the jaxvacua package itself. These are loaded via the `model_ID` parameter:

```python
import jaxvacua as jvc
model = jvc.FluxVacuaFinder(h12=2, model_ID=1, model_type="KS", maximum_degree=2)
```

This requires no internet connection and no external dependencies beyond the local package data. It is ideal for quick tests, tutorials, and reproducibility.

### HuggingFace database

For large-scale studies involving thousands or millions of geometries, the database API in `stringforge` provides access to the full Kreuzer-Skarke and CICY lists:

```python
from stringforge.cy_io import TDFDatabase
from stringforge.lcs_database import LCSDatabase

# Pure I/O — returns geometric data
db = TDFDatabase()
tree = db.load(ks_id=12345, triang_id=0, include_gv=True)

# Model construction — returns a ready-to-use FluxVacuaFinder
db = LCSDatabase(dataset="tdf")
model = db.load_model(ks_id=12345, triang_id=0, Q=24)
```

This requires the `huggingface-hub`, `pandas`, and `pyarrow` packages, plus `jaxvacua` (only invoked at the model-construction stage; pure I/O does not need it).

## How the database is organised

The HuggingFace repository is structured into **splits**, each stored as a collection of [Parquet](https://parquet.apache.org/) shards:

```
aschachner/cy-database/
    tdf/
        catalog.parquet           ← lightweight index (downloaded once)
        schema.json               ← version metadata
        lcs_data/h11_{N}/
            data-00000.parquet    ← geometry data, sharded by h11
            data-00001.parquet
            ...
        gv/h11_{N}/
            data-00000.parquet    ← Gopakumar-Vafa invariants, sharded by h11
        conifolds/h11_{N}/
            data-00000.parquet    ← conifold limit data, sharded by h11
        extra/
            data-00000.parquet    ← additional properties
        polytope/
            data-00000.parquet    ← polytope vertex data (tdf only)
    cicy/
        ...                       ← same structure for CICY models
```

The **catalog** is a small file (~tens of KB) that maps every model to its location in the shards via `(shard_id, row_index)` pointers. It is loaded into memory on first access and serves all subsequent queries without network I/O.

## Lazy downloading and caching

The database uses a **lazy download** strategy:

1. **Constructor** (`TDFDatabase()`) — does nothing. No network calls.
2. **Querying** (`db.query(h11=2)`) — downloads only the catalog (once).
3. **Loading** (`db.load(...)`) — downloads only the specific shard(s) containing the requested model.
4. **Batch loading** (`db.load_batch(h11=2)`) — downloads shards as needed.

Downloaded files are cached in `.stringforge_cache/` (in the current working directory) by default. This keeps data visible and project-local. The location can be changed globally via `stringforge.set_data_dir()` or the `STRINGFORGE_DATA_DIR` environment variable, or per-instance via the `cache_dir` constructor argument.

Subsequent loads of models in the same shard are served from disk (or from an in-memory LRU cache for recently accessed shards). A one-time warning is issued if the data directory exceeds 500 MB.

### Cache modes

The `cache_mode` constructor parameter controls how aggressively files are cached:

| `cache_mode` | Behaviour | Use case |
|---|---|---|
| `"persistent"` (default) | Keep shard files on disk and in LRU memory cache | Repeated access, interactive work |
| `"none"` | Download shard, read needed row, delete shard from disk | Scanning millions of models without disk accumulation |

```python
# For large-scale scans
db = TDFDatabase(cache_mode="none")
trees = db.load_batch(h11=3)  # shards deleted after each read
```

To manually clear the persistent cache:

```python
db.clear_cache()                    # delete shards, keep catalog
db.clear_cache(include_vacua=True)  # also delete stored vacuum solutions
```

## Key API methods

| Method | Description |
|--------|-------------|
| `db.query(h11=..., h12=..., ...)` | Filter the catalog, return a DataFrame |
| `db.query_conifolds(ks_id=...)` | Query the conifold sub-catalog |
| `db.load(ks_id=..., triang_id=...)` | Load a single model as `lcs_tree` |
| `db.load_model(ks_id=..., Q=24)` | Load as a ready-to-use `FluxVacuaFinder` (`LCSDatabase` only) |
| `db.load_batch(df)` or `db.load_batch(h11=2)` | Load multiple models |
| `db.sample(n=10, h11=2)` | Random sample of models |
| `db.info()` | Print database summary |
| `db.clear_cache()` | Delete cached shard files |

## Offline mode

For HPC clusters without internet access:

1. Warm the cache on a machine with internet: load the models you need.
2. Copy the local cache directory (default: `.stringforge_cache/`) to the cluster.
3. Use `TDFDatabase(offline=True)` — all data served from cache.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `STRINGFORGE_DATA_DIR` | `{cwd}/.stringforge_cache` | Override the data directory for all database operations |
| `STRINGFORGE_HF_REPO` | `aschachner/cy-database` | Override the HuggingFace repository ID |
| `STRINGFORGE_VAULT` | (auto-detect) | Override the vacua-vault directory (see vault docs) |
| `STRINGFORGE_VAULT_REPO` | `aschachner/vacua_vault` | Override the HuggingFace vault repository ID |

## Further reading

### On this site

- {doc}`Tutorial notebook: Database interface <../tutorials/database_and_infrastructure/database_interface>`
- {doc}`Tutorial notebook: Vacua storage <../tutorials/database_and_infrastructure/vacua_storage>`
- {doc}`Tutorial notebook: Cluster parallelisation <../tutorials/database_and_infrastructure/cluster_parallelisation>`
- {doc}`API reference: stringforge.cy_io <../api/stringforge.cy_io>`
- {doc}`API reference: stringforge.lcs_database <../api/stringforge.lcs_database>`
- {doc}`Migration from jaxvacua <../ecosystem/migration_from_jaxvacua>` — module rename and the 2026-05-01 mirror-convention swap.

### Physics background

The physics intros (Calabi–Yau geometries, periods, flux compactifications,
moduli stabilisation, perturbatively flat vacua, Type-IIB supergravity) live
in the JAXVacua documentation and are not duplicated here:

- {external+jaxvacua:doc}`intro/geometries` — Calabi–Yau threefolds, KS polytopes, CICYs.
- {external+jaxvacua:doc}`intro/periods` — period vector and prepotential.
- {external+jaxvacua:doc}`intro/flux_compactifications` — Type-IIB flux backgrounds.
- {external+jaxvacua:doc}`intro/moduli_stabilisation` — moduli stabilisation patterns.
- {external+jaxvacua:doc}`intro/pfv` — perturbatively flat vacua.
- {external+jaxvacua:doc}`intro/sugra` — 4-D N=1 Type-IIB supergravity.
