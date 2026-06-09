# Calabi-Yau Geometry Database

StringForge provides access to large Calabi-Yau geometry datasets through a
unified interface.  Data are hosted on HuggingFace and downloaded lazily: the
constructor performs no network access, catalogue files are fetched on first
query, and geometry shards are fetched only when a model is loaded.

```{important}
There are three related but distinct layers:

1. **TDF/CICY geometry databases** store Calabi-Yau data.
2. **KKLT curated subset** stores specialised conifold-class provenance and
   logical links back to TDF rows.
3. **`vacua_vault`** is one shared repository for designated vacuum solutions,
   with dataset-specific metadata in each record.
```

## Available database interfaces

- **TDF models** from the Kreuzer-Skarke list, addressed by `(ks_id,
  triang_id)`, through `TDFDatabase` and `LCSDatabase(dataset="tdf")`.
- **CICY models** from the complete-intersection Calabi-Yau list, addressed by
  `cicy_id`, through `CICYDatabase` and `LCSDatabase(dataset="cicy")`.
- **KKLT index**, an advanced curated TDF subset indexed by conifold class
  with curation tags, through `KKLTDatabase`.  See [KKLT Database](./kklt_database.md)
  after reading this page.

Each model may carry topological data, Kähler-cone data, optional GV/GW
invariants, optional conifold-limit data, and extra precomputed properties.

## Two ways to load models

### Pure I/O

Use `CYDatabase`, `TDFDatabase`, or `CICYDatabase` when you want catalogue rows
and raw data without constructing a physics model.

```python
from stringforge import TDFDatabase

db = TDFDatabase()
df = db.query(h11=2)
```

### JAXVacua bridge

Use `LCSDatabase` when you want the public surface in mirror convention and want
to construct JAXVacua-compatible objects.

```python
from stringforge import LCSDatabase

db = LCSDatabase(dataset="tdf")
rows = db.query(h12=2).head()
row = rows.iloc[0]

tree = db.load(
    h11=int(row["h11"]),
    h12=int(row["h12"]),
    ks_id=int(row["ks_id"]),
    triang_id=int(row["triang_id"]),
)
model = db.load_model(
    h11=int(row["h11"]),
    h12=int(row["h12"]),
    ks_id=int(row["ks_id"]),
    triang_id=int(row["triang_id"]),
)
```

```{eval-rst}
.. raw:: html
   :file: ../_static/figures/f2_database_flow.html
```

## Lazy downloading and caching

The cache defaults to `.stringforge_cache/` in the current working directory.
Change it globally with `stringforge.set_data_dir()` or the
`STRINGFORGE_DATA_DIR` environment variable, or per instance with `cache_dir=`.

| Step | Network behaviour |
| --- | --- |
| Constructor | No network access. |
| `query(...)` | Downloads the lightweight catalogue once. |
| `load(...)` | Downloads only the shard containing the requested row. |
| `load_batch(...)` | Downloads only shards needed by the batch. |

Use `cache_mode="none"` when scanning many models without keeping downloaded
shards on disk.  Use `offline=True` after warming a cache for cluster jobs.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `STRINGFORGE_DATA_DIR` | `{cwd}/.stringforge_cache` | Global data/cache directory. |
| `STRINGFORGE_HF_REPO` | `aschachner/cy-database` | Geometry database repository. |
| `STRINGFORGE_VAULT` | explicit or repo-local | Local designated-vacua directory. |
| `STRINGFORGE_VAULT_REPO` | `aschachner/vacua_vault` | Shared HuggingFace vault repository. |

## Further reading

- [Database interface tutorial](../tutorials/database_and_infrastructure/database_interface)
- [Vacua storage tutorial](../tutorials/database_and_infrastructure/vacua_storage)
- [Cluster parallelisation tutorial](../tutorials/database_and_infrastructure/cluster_parallelisation)
- [KKLT Database](./kklt_database.md)
- [API reference](../api/index)
