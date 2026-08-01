# Getting started

StringForge is the infrastructure layer for database access, model loading,
cache/offline workflows, and vacuum-solution storage.  It is not the flux-vacuum
solver itself: physics calculations are delegated to sibling packages, most
importantly [JAXVacua](https://jaxvacua.readthedocs.io).

## Installation

**Prerequisites:** Python >= 3.12. If GPU acceleration is needed, install
[JAX with CUDA support](https://github.com/jax-ml/jax#installation) first.

```bash
# Recommended once public on PyPI
pip install stringforge

# Development install from a local clone
git clone https://github.com/StringJAX/stringforge.git
cd stringforge
pip install -e .
```

```{caution}
Workflows that construct JAXVacua models require `float64` precision. JAX Metal
on macOS does not support the required complex `float64` operations; use the CPU
backend on Mac.
```

## What belongs where

| Layer | Owned by | Typical entry points |
| --- | --- | --- |
| Pure catalogue I/O | StringForge | `CYDatabase`, `TDFDatabase`, `CICYDatabase` |
| Toric CY phases (FRST / VEX) at scale | StringForge | `ToricCYDatabase`, `CYPhase`, `ToricCYPhase` |
| JAXVacua model loading | StringForge bridge | `LCSDatabase.load`, `LCSDatabase.load_model` |
| Vacuum-solution persistence | StringForge | `VacuaWriter`, `designate_vacua`, `vacuavault` |
| Cluster production runs and ML-ready datasets | StringForge | `Vulcan`, `VulcanReader`, `VulcanMLView`, `python -m stringforge.vulcan` |
| Flux-vacuum physics | JAXVacua | `FluxVacuaFinder`, ISD sampling, flux bounding |
| Kähler/axion extensions | Planned sibling packages | KahlerJAX and JAXiverse; not public release dependencies |

## Quick start: database row to JAXVacua model

```python
from stringforge import LCSDatabase

# Constructing the database object performs no network I/O.
db = LCSDatabase(dataset="tdf", cache_dir=".stringforge_cache")

# First query downloads the lightweight catalogue and filters it locally.
models = db.query(h12=2, has_conifolds=True).head(5)
print(models[["h11", "h12", "ks_id", "triang_id", "n_conifolds"]])

row = models.iloc[0]

# Load the model data as a JAXVacua lcs_tree.
tree = db.load(
    h11=int(row["h11"]),
    h12=int(row["h12"]),
    ks_id=int(row["ks_id"]),
    triang_id=int(row["triang_id"]),
    include_gv=False,
)

# Or construct a ready-to-use JAXVacua FluxVacuaFinder.
finder = db.load_model(
    h11=int(row["h11"]),
    h12=int(row["h12"]),
    ks_id=int(row["ks_id"]),
    triang_id=int(row["triang_id"]),
    include_gv=False,
)

print(type(tree).__name__)
print(type(finder).__name__)
```

The returned `finder` is a JAXVacua `FluxVacuaFinder`.  Use the JAXVacua
documentation for vacuum search, flux sampling, period calculations, and
stability analysis built on top of this object.

## Vacua vault in one minute

The vault stores accepted vacuum solutions with enough model identity and run
metadata to make them reusable.

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
```

For details, use the vault workflow tutorials and the `stringforge.vacuavault`
API reference.

## Cache and offline mode

StringForge downloads data lazily.  The constructor does not fetch anything;
queries fetch only the catalogue; model loading fetches only the shard files
required for the requested rows.

Set `offline=True` on a warmed cache for cluster jobs without network access:

```python
db = LCSDatabase(dataset="tdf", cache_dir="/path/to/cache", offline=True)
```

## Advanced curated subsets

`KKLTDatabase` is public but advanced.  It exposes the curated `kklt` TDF
subset indexed by conifold class and carries curation tags for specialised
KKLT-style searches.  Actual KKLT vacuum records are stored through the shared
vault workflow.  Most users should first learn the TDF/CICY database and vault
workflows before using it.
