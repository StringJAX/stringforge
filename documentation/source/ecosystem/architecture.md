# Architecture

This page records the package boundaries.  The guiding rule is simple:
StringForge owns shared infrastructure; sibling packages own physics engines.

## Owner / consumer matrix

### `stringforge` — shared infrastructure

| Module | Public symbols | Purpose |
| --- | --- | --- |
| [`stringforge.cy_io`](../api/stringforge.cy_io) | `CYDatabase`, `TDFDatabase`, `CICYDatabase` | Pure catalogue I/O, HuggingFace downloads, cache management, schema checks. |
| [`stringforge.lcs_database`](../api/stringforge.lcs_database) | `LCSDatabase`, `load_tdf_model`, `load_cicy_model` | Mirror-convention model loading; builds `lcs_tree` or `FluxVacuaFinder` objects when JAXVacua is available. |
| [`stringforge.kklt_database`](../api/stringforge.kklt_database) | `KKLTDatabase` | Advanced curated `kklt` TDF subset indexed by conifold class, with logical TDF links and curation tags. |
| [`stringforge.vacua_writer`](../api/stringforge.vacua_writer) | `VacuaWriter` and database delegations | Vacuum-solution persistence, designation, retraction, purge, and HuggingFace vault push/fetch. |
| [`stringforge.vacuavault`](../api/stringforge.vacuavault) | validation and curation helpers | Schema validation and catalogue rebuild tooling for the public vacua vault. |

### `jaxvacua` — flux-vacua engine

JAXVacua owns the layered physics pipeline:

```text
lcs_tree -> periods -> css -> FluxEFT -> FluxVacuaFinder
```

StringForge can construct `lcs_tree` and `FluxVacuaFinder` objects, but it does
not own period calculations, flux sampling, scalar potentials, vacuum solvers,
or mass-spectrum computations.

### Planned packages

KahlerJAX and JAXiverse are planned ecosystem consumers.  Their planning pages
describe intent only.  They are not imported by StringForge, not installed as
default dependencies, and not treated as stable public APIs in this release.

### External packages

CYTools remains the external toric geometry package used by selected workflows.
JAXPolyLog is the public differentiable polylogarithm package consumed by
JAXVacua and planned downstream physics packages.

## What StringForge is not

- Not a replacement for JAXVacua.
- Not a complete string-phenomenology framework by itself.
- Not a public release vehicle for KahlerJAX/JAXiverse.
- Not the owner of every collaborator-generated dataset referenced in examples.

## Convention boundaries

Two conventions cross package boundaries and are easy to get wrong:

- **Catalog versus mirror Hodge numbers.** `CYDatabase` exposes parquet catalogue
  convention. `LCSDatabase` exposes mirror convention to match `lcs_tree` and
  JAXVacua. Translation happens at the public API boundary.
- **Float64 precision.** Model-construction workflows assume JAX `x64` support.
  Set precision before importing JAX-heavy physics packages.

## Duck-typed model identity

`VacuaWriter` accepts either a JAXVacua model with `.periods.lcs_tree` or an
`lcs_tree`-like object directly.  This keeps StringForge free of eager physics
imports while still recording model identity and hashes for vault entries.

## Background belongs in docs

Source docstrings should document signatures, side effects, convention
boundaries, and failure modes.  Conceptual explanations belong in these Sphinx
pages, where they can be expanded with notes, warnings, and figures.
