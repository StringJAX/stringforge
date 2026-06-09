# KKLT Database

```{admonition} Advanced workflow
:class: warning
This page documents specialised infrastructure for curated KKLT-style searches.
Most users should first read the general TDF/CICY database page and the vault
workflow.  The KKLT layer is public API, but it is not the default first-user
path and should not be read as a complete physics pipeline by itself.
```

The `kklt` sub-dataset is a curated search index inside the general
`aschachner/cy-database` repository.  It is indexed by conifold class, carries
tagging and curation metadata, and links back to the underlying TDF geometry
data through logical keys.  It does not duplicate the full Calabi-Yau geometry
database and it is not itself the store of final vacuum solutions.

```{eval-rst}
.. raw:: html
   :file: ../_static/figures/f4_kklt_database.html
```

## What the KKLT layer stores

- `catalog.parquet`: one row per polytope-level KKLT candidate.
- `conifold_class_catalog.parquet`: one row per `(ks_id, coni_class_id)`.
- `conifold_catalog.parquet`: one row per `(ks_id, coni_class_id, coni_id)`,
  carrying the logical TDF link `(triang_id, tdf_conifold_id)`.
- scalar tag metadata columns such as `tags`, `review_status`, `tag_source`,
  `tag_notes`, `tag_updated_at`, and `tag_schema_version`.

A conifold class anchors on the one-face divisor common to all conifolds in the
class.  This grouping can cut across triangulations of the same polytope.

## Relationship to TDF and the vault

The KKLT database delegates model construction to a wrapped TDF database.  The
TDF layer remains the source of geometry shards; KKLT adds curated indices,
logical links, and curation tags.

Vacua found through KKLT workflows are written to the single shared
`aschachner/vacua_vault` repository, under a `kklt_vacua` subset for actual
vacuum records.  KKLT-specific fields such as `coni_class_id`, `coni_id`, and
the relevant tags are recorded as metadata there.

## Basic usage

```python
from stringforge import KKLTDatabase

db = KKLTDatabase()
polys = db.query_polytopes(Q_min=100, tags_include="kklt_candidate")
classes = db.query_classes(ks_id=int(polys.iloc[0]["ks_id"]))
conifolds = db.query_conifolds(
    ks_id=int(polys.iloc[0]["ks_id"]),
    coni_class_id=int(classes.iloc[0]["coni_class_id"]),
    tags_exclude="orphaned",
)

row = conifolds.iloc[0]
model = db.load_model(
    h11=int(row["h11"]),
    ks_id=int(row["ks_id"]),
    coni_class_id=int(row["coni_class_id"]),
    coni_id=int(row["coni_id"]),
)
```

## TDF-link maintenance

A TDF-compat fingerprint records the TDF schema version and catalogue checksum
used when the KKLT data were built.  If the TDF database is rebuilt, use
`rebuild_links()` to refresh logical links against the current TDF catalogue.
Rows that no longer resolve should be marked orphaned and rejected clearly by
loaders.

## Deferred release material

The older de Sitter pipeline kit is intentionally not part of the first public
StringForge release.  It should be added only after the historical model
identification has been reconciled with the current TDF database.
