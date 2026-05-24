# KKLT-Vacua Database

## Overview

The `kklt_vacua` sub-dataset is a *curated subset* of the TDF database (Kreuzer–Skarke hypersurfaces with a trilayer orientifold), specialised for KKLT-style de-Sitter searches.  It is indexed by **conifold class** rather than by triangulation, and equipped with a run-tracking layer for persistent provenance of cluster work.

The class is provided by {class}`stringforge.kklt_database.KKLTDatabase`, which inherits from `LCSDatabase` and therefore reuses the existing model-construction machinery.  Equivalently, `LCSDatabase(dataset="kklt_vacua")` constructs a `KKLTDatabase` via a factory dispatch.

```python
from stringforge.kklt_database import KKLTDatabase
db = KKLTDatabase()
```

## Subset criterion

Polytopes are retained iff

$$n_{\text{rigids dual}} > h^{1,2}\,,$$

i.e. there are sufficiently many rigid divisors on the dual side.  This is the precondition for the KKLT-style de-Sitter construction.

The historically well-studied subset $Q := h^{1,1} + h^{1,2} + 2 \ge 100$ is **not** filtered out — `Q` is exposed as a column on `catalog.parquet` so users can recover it on demand:

```python
polys_Q100 = db.query_polytopes(Q_min=100)
```

## Catalog layout

Three local parquet catalogs live under `<cache_dir>/kklt_vacua/`:

| File | Grain | Key columns |
|---|---|---|
| `catalog.parquet` | polytope | `(ks_id, h11, h12)` |
| `conifold_class_catalog.parquet` | (polytope, class) | `(ks_id, coni_class_id)` |
| `conifold_catalog.parquet` | (polytope, class, conifold) | `(ks_id, coni_class_id, coni_id)` |

`conifold_catalog.parquet` carries the **logical** TDF link `(triang_id, tdf_conifold_id)` only.  Physical shard coordinates are deliberately not stored — they would go stale on every TDF rebuild.  Instead, KKLTDatabase resolves the current physical address by querying its wrapped `tdf` database's catalog at fetch time.

A conifold class anchors on the **one-face divisor** common to all conifolds in the class.  This grouping cuts across triangulations of the same polytope, so a single class can span many `(triang_id, tdf_conifold_id)` pairs in TDF.

## Cross-database storage

The KKLT database does **not** duplicate Calabi–Yau geometry data.  Every loader call delegates to the wrapped TDF database (`KKLTDatabase.tdf`).  Memory and disk footprint of the KKLT cache is therefore small: only the three catalogs plus the run log (a few megabytes total).

A *TDF-compat fingerprint* (`tdf_schema_version`, `tdf_catalog_sha256`) is stored in `schema.json` at build time and re-checked on every `KKLTDatabase()` instantiation.  A mismatch emits a `UserWarning` pointing the user at `KKLTDatabase.rebuild_links()`.

## Querying

```python
# Polytope-grain filter — uses mirror convention for h11/h12.
db.query_polytopes(h12=4, Q_min=100, n_rigids_dual_min=10)

# All classes on a given polytope
db.query_classes(ks_id=12345)

# All conifolds in a class
db.query_conifolds(ks_id=12345, coni_class_id=0)
```

## Loading models

```python
# Returns a FluxVacuaFinder, with the linked TDF conifold attached.
model = db.load_model(ks_id=12345, coni_class_id=0, coni_id=3,
                      include_gv=True, maximum_degree=2)
```

Internally, KKLTDatabase resolves the row from `conifold_catalog.parquet`, reads `(triang_id, tdf_conifold_id)`, and calls `db.tdf.load_model(ks_id=..., triang_id=..., conifold_id=...)`.  Rows with `tdf_status="orphaned"` (set by `rebuild_links()` after a TDF rebuild that drops the underlying model) raise a clear `ValueError`.

Batch loaders are available for KKLT rows and use the same mirror-convention
identity surface as the single-model loader.  Pass a `DataFrame` returned by
`query_conifolds(...)`, a list of dictionaries, or tuples
`(h11, ks_id, coni_class_id, coni_id[, h12])`:

```python
rows = db.query_conifolds(ks_id=12345, coni_class_id=0).head(5)

# Bare lcs_tree objects.
trees = db.load_batch(rows, include_gv=False)

# Fully initialised FluxVacuaFinder objects.
models = db.load_batch(rows, as_models=True, include_gv=True)

# Lazy iteration, useful for larger scans.
for model in db.iter_batch(rows, as_models=True):
    ...
```

`sample(n=..., as_models=...)` draws random conifold rows from the released
catalogues and then calls the same batch-loading machinery.

## Run tracking

Cluster work is logged in `run_log.parquet`, an append-only schema (uuid `run_id`, scope, timestamps, status, payload pointer, free-form `task` and `notes`):

```python
# Open a run
run_id = db.start_run(
    scope="conifold",
    ks_id=12345, coni_class_id=0, coni_id=3,
    task="flux_scan",
    job_id="slurm-12345-7",
)

# ... do work ...

# Close the run
db.finish_run(run_id, status="done", n_solutions=42,
              payload_uri="hf://aschachner/vacua_vault/kklt_vacua/...")

# Latest status (derived view over the log)
db.run_status(ks_id=12345, coni_class_id=0, coni_id=3)   # → "done"
```

Two scopes are supported:
- `scope="class"` — work that applies to a whole conifold class (e.g. basis fitting, GV recomputation under a class-adapted basis).
- `scope="conifold"` — work tied to a specific conifold (e.g. a flux-vacuum scan).

Concurrent local writers are serialised via advisory `fcntl.flock` on a sibling lock file.  Periodic `db.push_run_log()` uploads the local log to the Hub for cross-worker visibility.

## TDF-link maintenance

When the underlying TDF database is rebuilt (re-sharded, new conifolds added, models removed), refresh the KKLT links with

```python
report = db.rebuild_links(pick_up_new_conifolds=False)
print(report)
# {"orphaned": [...], "added": [...], "denorm_changed": [...]}
```

Orphaned rows are marked `tdf_status="orphaned"` rather than deleted, so any vacua and run history associated with them survive.

## Vacua storage

Vacua found through `KKLTDatabase.load_model(...)` are written under
`<vault>/kklt_vacua/ks_{ks_id}_tri_{triang_id}/` — i.e. in a sub-directory separate from regular TDF vacua, but inside the same shared `aschachner/vacua_vault` HF repository.  KKLT-specific provenance (`coni_class_id`, `coni_id`) is recorded inside each vacuum row's metadata, not in the path.

## Build provenance

The public dataset is built by the maintainers from the current TDF catalogue
and a conifold-class mapping table.  End users normally do not rebuild it:
they instantiate `KKLTDatabase()`, query the released catalogues, and use
`rebuild_links()` only when a local TDF cache has moved to a newer schema.

Maintainer builds should record the TDF schema version and catalogue checksum
in `schema.json`, then validate the logical TDF links before publishing the
updated parquets.
