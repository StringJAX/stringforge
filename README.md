# StringForge

**Differentiable tools for string compactifications with JAX.**

StringForge is the shared data and infrastructure layer for the StringForge ecosystem of JAX-based string-compactification packages. It provides reproducible access to Calabi-Yau geometry databases, model-loading bridges into physics packages such as [JAXVacua](https://github.com/AndreasSchachner/jaxvacua), and a permanent vacua vault for storing, validating, curating, and publishing vacuum solutions.

The goal is to make large-scale string-landscape computations less fragile: geometry data should be queried through a stable interface, expensive downloads should be cached lazily, generated vacua should carry enough provenance to be reused, and downstream packages should share the same database conventions.

## Key features

- **Geometry databases.** Unified access to the hosted TDF/Kreuzer-Skarke and CICY datasets through `CYDatabase`, `TDFDatabase`, `CICYDatabase`, and `LCSDatabase`.
- **Lazy local caching.** Catalogues and parquet shards are downloaded on demand and cached under a configurable data directory, with an explicit offline mode for cluster jobs.
- **JAXVacua bridge.** `LCSDatabase` loads database rows as `jaxvacua.lcs.lcs_tree` objects or fully initialised `FluxVacuaFinder` models when JAXVacua is installed.
- **KKLT-vacua subset.** `KKLTDatabase` exposes the curated conifold-class indexed subset used for KKLT-style searches, including run-log provenance for cluster campaigns.
- **Vacua vault.** `VacuaWriter` designates, validates, queries, uploads, fetches, retracts, and purges vacuum-solution parquet files in a shared vault layout.
- **Vault validation tools.** `stringforge.vacuavault` validates parquet submissions, rebuilds catalogues, and supports curation workflows without importing physics solvers.
- **Ecosystem documentation.** The documentation explains how StringForge relates to JAXVacua, JAXPolyLog, KahlerJAX, JAXiverse, CYTools, and the shared data conventions.

## Architecture

The package architecture mirrors the boundary between shared infrastructure and physics engines:

```
CYDatabase      ← pure I/O, HuggingFace downloads, cache, catalog queries
    ↓
LCSDatabase     ← mirror-convention model loading for JAXVacua workflows
    ↓
KKLTDatabase    ← curated KKLT-vacua subset and cluster run tracking
    ↓
VacuaWriter     ← designated vacua, vault catalogues, push/fetch workflows
```

The low-level database layer is intentionally solver-free. Physics construction is deferred to sibling packages at the point where a user asks for a model object, so the same catalogues can support JAXVacua, KahlerJAX, JAXiverse, and pure data-analysis workflows.

## Packages

StringForge is an umbrella framework comprising the following packages:

| Package | Description | Status |
|---------|-------------|--------|
| **[JAXVacua](https://github.com/AndreasSchachner/jaxvacua)** | Type IIB flux vacua: complex-structure and axio-dilaton sector, vacuum finding, stability analysis | Public (this release) |
| **[JAXPolyLog](https://github.com/AndreasSchachner/jaxpolylog)** | JAX-compatible polylogarithm functions with autodiff support | Public (this release) |
| **KahlerJAX** | Numerical Kähler moduli stabilisation for 4d N=1 EFTs | Planned |
| **JAXiverse** | Multi-axion EFT: spectra, decay constants, and couplings from Calabi–Yau compactifications | Planned |

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)

from stringforge import LCSDatabase

# Query the hosted TDF catalogue. Catalogues are downloaded lazily and cached.
db = LCSDatabase(dataset="tdf", cache_dir=".stringforge_cache")
models = db.query(h11=272, h12=2, has_conifolds=True)
print(models[["h11", "h12", "ks_id", "triang_id", "n_conifolds"]].head())

# Load one catalogue row as a JAXVacua FluxVacuaFinder model.
# Hodge numbers are in the mirror convention used by JAXVacua.
row = models.iloc[0]
finder = db.load_model(
    h11=int(row["h11"]),
    h12=int(row["h12"]),
    ks_id=int(row["ks_id"]),
    triang_id=int(row["triang_id"]),
    include_gv=False,
)
```

The returned `finder` is a JAXVacua `FluxVacuaFinder`. Vacuum search, flux sampling, period calculations, and stability analysis are documented in the [JAXVacua documentation](https://jaxvacua.readthedocs.io).

### Vacua vault workflow

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

**Prerequisites:** Python ≥ 3.12. If GPU acceleration is needed, install [JAX with CUDA support](https://github.com/jax-ml/jax#installation) first.

```bash
# Create and activate a virtual environment (recommended)
python -m venv stringforge-env && source stringforge-env/bin/activate

# Install from GitHub
pip install -e "git+https://github.com/AndreasSchachner/stringforge.git#egg=stringforge"

# Or clone and install locally (recommended for development)
git clone --recurse-submodules https://github.com/AndreasSchachner/stringforge.git
cd stringforge
pip install -e .
```

> [!CAUTION]
> StringForge requires `float64` precision. [JAX Metal](https://developer.apple.com/metal/jax/) on macOS does not support `float64` and is therefore incompatible. Use the CPU backend on Mac.

> [!NOTE]
> When using the [CYTools](https://cy.tools) Docker image, check compatibility with the required NumPy and JAX versions.

## Documentation

The StringForge ecosystem documentation — including package overviews, tutorials, database guides, and API reference pages — can be built from the `documentation/` folder in this repository. The full JAXVacua API reference is available at [jaxvacua.readthedocs.io](https://jaxvacua.readthedocs.io).

To build the documentation locally:

```bash
cd documentation
pip install -r requirements.txt
make html
# Open build/html/index.html
```

## Requirements

Core dependencies (installed automatically via `pip`):

- NumPy
- Pandas and PyArrow
- [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub)
- [JAX](https://github.com/google/jax) and jaxlib for model-construction workflows

Optional:

- [CYTools](https://cy.tools) — for constructing models from Kreuzer–Skarke polytopes
- [JAXVacua](https://github.com/AndreasSchachner/jaxvacua) — for period calculations, flux EFTs, vacuum finding, and stability analysis
- [python-flint](https://github.com/flintlib/python-flint) — for exact arithmetic in selected downstream routines

## Citation

If you find this work useful, please cite:

```bibtex
@article{Schachner:2026stringforge,
    author = "Schachner, Andreas",
    title = "{StringForge --- Differentiable Tools for String Compactifications}",
    eprint = "",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "",
    journal = "",
    year = "2026"
}
```

The initial JAXVacua framework was introduced in:

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

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Contact

Andreas Schachner
- Email: [as3475@cornell.edu](mailto:as3475@cornell.edu)
- GitHub: [github.com/AndreasSchachner](https://github.com/AndreasSchachner)
- Website: [andreasschachner.github.io](https://andreasschachner.github.io)
