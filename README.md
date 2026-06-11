# StringForge

**Shared database, model-loading, and vacua-vault infrastructure for string-compactification workflows.**

StringForge is the infrastructure layer for the [StringJAX](https://github.com/AndreasSchachner/stringjax) ecosystem of JAX-based string-compactification packages. It provides reproducible access to Calabi-Yau geometry databases, bridges those data into physics engines such as [JAXVacua](https://github.com/AndreasSchachner/jaxvacua), and manages persistent vacuum-solution storage with provenance.

The package is intentionally solver-light. It does not replace JAXVacua, KahlerJAX, JAXiverse, or CYTools. Instead, it standardises the shared conventions that those packages and downstream scans need: catalogue queries, lazy downloads, cache/offline workflows, model loading, vault layout, validation, and curation.

## What StringForge owns

- **Geometry databases.** Unified access to hosted TDF/Kreuzer-Skarke and CICY datasets through `CYDatabase`, `TDFDatabase`, `CICYDatabase`, and `LCSDatabase`.
- **Lazy local caching.** Catalogues and parquet shards are downloaded on demand and cached under a configurable data directory, with explicit offline mode for HPC jobs.
- **Model-loading bridges.** `LCSDatabase` loads database rows as `jaxvacua.lcs.lcs_tree` objects or fully initialised JAXVacua `FluxVacuaFinder` models when JAXVacua is installed.
- **Vacua vault.** `VacuaWriter` designates, validates, queries, uploads, fetches, retracts, and purges vacuum-solution parquet files in a shared vault layout.
- **Vault validation tools.** `stringforge.vacuavault` validates parquet submissions, rebuilds catalogues, and supports curation workflows without importing physics solvers.
- **Advanced curated indices.** `KKLTDatabase` exposes a specialised conifold-class indexed `kklt` subset used for KKLT-style searches, tags, and TDF hand-off.
- **Production vacuum forging.** `Vulcan` is the cluster-side, append-only counterpart to `VacuaWriter`: workers stage validated parquet shards locally, a head node batches them into one `HfApi.create_commit` call per `max_batch`-sized chunk (default 500 files per commit), the rolling-window budget respects HuggingFace's 100-commit-per-hour cap, and `VulcanReader` / `VulcanMLView` give downstream consumers queryable rows and deterministic, geometry-disjoint train/val/test splits (rows sharing a geometry_id always land in the same split, regardless of process or seed).

## What StringForge does not own

- It is not the flux-vacuum solver. Vacuum search, period calculations, ISD sampling, flux bounding, and stability analysis live in JAXVacua.
- It is not a public release of KahlerJAX or JAXiverse. Those packages remain planned ecosystem consumers until their own releases are ready.
- It is not the owner of every derived dataset used by collaborators. Public pages distinguish hosted StringForge datasets from collaborator-generated or paper-specific data.
- It is not a monolithic umbrella package that imports every physics engine on startup. Imports stay lightweight and optional physics packages are loaded only when a workflow needs them.

## Architecture

```text
CYDatabase      -> pure I/O, HuggingFace downloads, cache, catalog queries
    |
LCSDatabase     -> mirror-convention model loading for JAXVacua workflows
    |
VacuaWriter     -> designated vacua, vault catalogues, push/fetch workflows

Vulcan          -> cluster-side production: stage locally, batch-commit
                   via HfApi.create_commit, query/stream as ML dataset
VulcanReader   -> read-side catalogue scan, run / shard fetch
VulcanMLView   -> geometry-disjoint train/val/test splits for ML
```

`KKLTDatabase` is an advanced `LCSDatabase`-style interface for a curated TDF subset. It does not duplicate the TDF geometry data; it stores logical links, conifold-class provenance, and curation tags. Actual KKLT vacuum records belong in the shared `vacua_vault` infrastructure.

`Vulcan` deliberately complements `VacuaWriter` rather than replacing it: `VacuaWriter` *designates* curated low-volume vacua into the paper-aligned `vacua_vault`, while `Vulcan` *forges* high-volume cluster output into a separate production repo with a uniform parquet schema with fixed-shape columns plus pad-to-tensor list columns (`flux`, `moduli_re`, `moduli_im`, `F_terms_*`) and deterministic geometry-disjoint train/val/test splits via `VulcanMLView`. The two share the parquet floor (`flux`, `moduli_re`, `moduli_im`, `tau_re`, `tau_im`), so a future promotion step can lift production runs into the curated vault.

## Ecosystem packages

| Package | Role | Release status |
| --- | --- | --- |
| **[JAXVacua](https://github.com/AndreasSchachner/jaxvacua)** &mdash; [docs](https://jaxvacua.readthedocs.io) | Type IIB flux vacua, complex-structure/axio-dilaton EFTs, vacuum finding, stability analysis | Public |
| **[JAXPolyLog](https://github.com/AndreasSchachner/jaxpolylog)** &mdash; [docs](https://jaxpolylog.readthedocs.io) | JAX-compatible polylogarithms with autodiff support | Public |
| **KahlerJAX** | Kähler-moduli stabilisation for 4D N=1 EFTs | Planned; not a StringForge dependency |
| **JAXiverse** | Multi-axion EFT spectra, decay constants, and couplings | Planned; not a StringForge dependency |
| **[CYTools](https://cy.tools)** | External toric Calabi-Yau geometry package | Public external dependency for selected workflows |

## Quick start

```python
from stringforge import LCSDatabase

# Query the hosted TDF catalogue. The constructor itself performs no network I/O;
# the catalogue is fetched lazily on first query.
db = LCSDatabase(dataset="tdf", cache_dir=".stringforge_cache")
models = db.query(h12=2, has_conifolds=True).head(5)
print(models[["h11", "h12", "ks_id", "triang_id", "n_conifolds"]])

# Load one catalogue row as JAXVacua-compatible data.
row = models.iloc[0]
tree = db.load(
    h11=int(row["h11"]),
    h12=int(row["h12"]),
    ks_id=int(row["ks_id"]),
    triang_id=int(row["triang_id"]),
    include_gv=False,
)

# Or construct the corresponding JAXVacua FluxVacuaFinder directly.
finder = db.load_model(
    h11=int(row["h11"]),
    h12=int(row["h12"]),
    ks_id=int(row["ks_id"]),
    triang_id=int(row["triang_id"]),
    include_gv=False,
)
```

The returned `finder` is a JAXVacua `FluxVacuaFinder`. Use the [JAXVacua documentation](https://jaxvacua.readthedocs.io) for vacuum-search, flux-sampling, period-calculation, and stability-analysis workflows.

## Vulcan: production vacuum forging

For high-volume cluster workloads, use `Vulcan` instead of `VacuaWriter` to publish runs to a separate production repo without tripping over HuggingFace's commit-rate cap.

```python
from stringforge.vulcan import Vulcan

# On a cluster worker: stage a batch locally; no HuggingFace I/O.
forge = Vulcan.from_env()                      # reads STRINGFORGE_VULCAN_*
forge.write(
    vacua_df,
    geometry={"h11": 3, "h12": 2, "ks_id": 384564, "triang_id": 0},
    tadpole_charge=12,
    solver={"name": "newton", "config_hash": "abc123"},
    provenance={"git_sha": "deadbeef", "seed": 42, "wall_clock_s": 3.4},
)

# On the head node (cron, daemon, or manual): drain pending shards
# into batched commits respecting the 90/hour budget.
report = forge.sync(max_batch=500)             # one create_commit, many files

# After sync: query, fetch a specific run, or build an ML view.
susy = forge.query(h12=2, solver_name="newton", is_susy=True)
train = forge.ml_view().as_dataframe("train")  # deterministic, geometry-disjoint split
```

CLI: `python -m stringforge.vulcan {status,sync}`. See [`Vulcan cluster runs`](documentation/source/tutorials/database_and_infrastructure/vulcan_cluster_runs.ipynb) for the full cluster best-practice walkthrough.

## Vacua vault workflow

```python
import pandas as pd

vacua = pd.DataFrame({
    "flux": [[1, 0, -2, 3, 0, 1]],
    "moduli_re": [[0.0, 0.0]],
    "moduli_im": [[2.5, 3.0]],
    "tau_re": [0.0],
    "tau_im": [4.0],
    "is_susy": [True],
})

db.designate_vacua(
    vacua,
    label="example_run",
    committed_by="A. Schachner",
    h11=int(row["h11"]),
    h12=int(row["h12"]),
    ks_id=int(row["ks_id"]),
    triang_id=int(row["triang_id"]),
)

designated = db.query_vacua(label="example_run")
print(designated[["label", "n_vacua", "created"]])
```

## Installation

**Prerequisites:** Python >= 3.12. If GPU acceleration is needed, install [JAX with CUDA support](https://github.com/jax-ml/jax#installation) first.

```bash
# Recommended once the package is public on PyPI
pip install stringforge

# Development install from a local clone
git clone https://github.com/AndreasSchachner/stringforge.git
cd stringforge
pip install -e .
```

> [!CAUTION]
> StringForge workflows that construct JAX models require `float64` precision. JAX Metal on macOS does not support the required complex `float64` operations; use the CPU backend on Mac.

## Documentation

Build the documentation locally with:

```bash
cd documentation
pip install -r requirements.txt
make html
```

The full JAXVacua API reference is available at [jaxvacua.readthedocs.io](https://jaxvacua.readthedocs.io).

## Requirements

Core dependencies installed by `pip`:

- NumPy
- Pandas and PyArrow
- HuggingFace Hub
- JAX and jaxlib
- JAXPolyLog
- JAXVacua

Optional workflow dependencies:

- [CYTools](https://cy.tools) for constructing models from Kreuzer-Skarke polytopes.
- [python-flint](https://github.com/flintlib/python-flint) for exact arithmetic in selected downstream routines.

## Citation

If you find this work useful, please cite the companion paper as the **primary** reference and the software release as the secondary reference.  When StringForge is used to drive a JAXVacua flux-vacuum search, please additionally cite the JAXVacua framework paper.

**Companion paper (preferred, in preparation).**  A single-author paper describing the conventions, dataset structure and capabilities of StringForge is in preparation.  The arXiv identifier and journal/DOI will be added here at submission; until then, cite the temporary manuscript entry below together with the software release.

```bibtex
@article{Schachner:2026stringforge,
    author = "Schachner, Andreas",
    title = "{StringForge: shared database and vacuum-storage infrastructure for differentiable type IIB flux-compactification workflows}",
    note = "In preparation; arXiv ID and journal/DOI to be added at submission.",
    year = "2026"
}
```

**Software release.**

```bibtex
@software{schachner_2026_stringforge,
  author = {Schachner, Andreas},
  title = {StringForge: shared infrastructure for string-compactification workflows},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/AndreasSchachner/stringforge}
}
```

**JAXVacua framework (cite when relevant).**  The upstream physics engine that StringForge provides infrastructure for was introduced in:

```bibtex
@article{Dubey:2023dvu,
    author = "Dubey, Abhishek and Krippendorf, Sven and Schachner, Andreas",
    title = "{JAXVacua --- a framework for sampling string vacua}",
    eprint = "2306.06160",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1007/JHEP12(2023)146",
    journal = "JHEP",
    volume = "12",
    pages = "146",
    year = "2023"
}
```

## License

StringForge is released under the [GNU General Public License v3.0](LICENSE).

## Contact

Andreas Schachner
- Email: [as3475@cornell.edu](mailto:as3475@cornell.edu)
- GitHub: [github.com/AndreasSchachner](https://github.com/AndreasSchachner)
- Website: [andreasschachner.github.io](https://andreasschachner.github.io)
