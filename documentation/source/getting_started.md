# Getting started

## Installation

**Prerequisites:** Python ≥ 3.12. If GPU acceleration is needed,
install [JAX with CUDA support](https://github.com/jax-ml/jax#installation)
first.

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

```{caution}
StringForge requires `float64` precision. [JAX Metal](https://developer.apple.com/metal/jax/)
on macOS does not support `float64` and is therefore incompatible. Use the CPU backend
on Mac.
```

```{note}
When using the [CYTools](https://cy.tools) Docker image, check compatibility with the
required NumPy and JAX versions.
```

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

The returned `finder` is a JAXVacua `FluxVacuaFinder`. Use the
[JAXVacua documentation](https://jaxvacua.readthedocs.io) for the vacuum-search,
flux-sampling, period-calculation, and stability-analysis workflows built on
top of this model object.

## Vacua vault

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

## Architecture

The package architecture mirrors the boundary between shared infrastructure and
physics engines:

```
CYDatabase      ← pure I/O, HuggingFace downloads, cache, catalog queries
    ↓
LCSDatabase     ← mirror-convention model loading for JAXVacua workflows
    ↓
KKLTDatabase    ← curated KKLT-vacua subset and cluster run tracking
    ↓
VacuaWriter     ← designated vacua, vault catalogues, push/fetch workflows
```

The low-level database layer is solver-free. Physics construction is deferred
to sibling packages at the point where a user requests a model object, so the
same catalogues can support JAXVacua, KahlerJAX, JAXiverse, and pure
data-analysis workflows.

See the [package pages](packages/jaxvacua) and the [tutorial index](tutorials)
for the available workflows.

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
