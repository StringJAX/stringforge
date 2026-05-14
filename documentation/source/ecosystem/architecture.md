# Architecture

This page documents the internal boundaries of the ecosystem: who owns which
classes / modules, who consumes them, and how the packages fit together at the
code level. The high-level diagram is on the [overview page](overview); this
page is the technical companion.

## Owner / consumer matrix

### `stringforge` — shared infrastructure

This package owns the cross-cutting infrastructure consumed by sibling packages
and end-user pipelines.

| Module | Public symbols | Purpose |
| --- | --- | --- |
| [`stringforge.cy_io`](../api/stringforge.cy_io) | `CYDatabase`, `TDFDatabase`, `CICYDatabase`, `query_models`, `load_catalog` | Catalog I/O, query layer over the `aschachner/cy-database` HuggingFace dataset. |
| [`stringforge.lcs_database`](../api/stringforge.lcs_database) | `LCSDatabase`, `load_tdf_model`, `load_cicy_model` | Mirror-convention model construction. Extends `CYDatabase` with `lcs_tree` / `FluxVacuaFinder` builders, batch / iter / sample APIs. |
| [`stringforge.vacua_writer`](../api/stringforge.vacua_writer) | `VacuaWriter` and database delegations | Vacuum-solution persistence, designation / retraction / purge, HuggingFace `vacua_vault` push and fetch. |
| [`stringforge.vacuavault`](../api/stringforge.vacuavault) | `SCHEMA_VERSION`, `validate_parquet_file`, `validate_pr_diff`, `rebuild_catalog`, `curate_submission` | Schema validation and catalog rebuild for the public vacua vault. Pure dependency-injection — does not import jaxvacua. |

### `jaxvacua` — the flux-vacua engine

The JAXVacua package owns the layered pipeline from topological data to vacuum
solutions, as described in its README:

```
lcs_tree → periods → css → FluxEFT → FluxVacuaFinder
```

| Module | Public symbols | Purpose |
| --- | --- | --- |
| `jaxvacua.lcs` | `lcs_tree` | JAX-registered pytree carrying topological metadata + numerical leaves. |
| `jaxvacua.periods` | `periods` | Period vector, prepotential, Kähler potential. |
| `jaxvacua.css` | `css` | Complex-structure sector: gauge-kinetic matrix, special-Kähler geometry. |
| `jaxvacua.flux_eft` | `FluxEFT` | GVW superpotential, F-terms, scalar potential. |
| `jaxvacua.flux_vacua_finder` | `FluxVacuaFinder` | Newton solver, Hessian, mass spectrum. |
| `jaxvacua.flux_bounding` | `bounded_fluxes` | Flux enumeration and stochastic search. |
| `jaxvacua.sampling` | `data_sampler` | ISD-biased and uniform initial-guess sampling. |
| `jaxvacua.freezer` | `Freezer`, `ConifoldFreezer` | Light-field EFT after integrating out heavy moduli. |
| `jaxvacua.cytools_interface` | `cytools_model_data_init` | Translation layer from a `cytools.CalabiYau` to `lcs_tree` inputs. |

### `kahlerjax` — Kähler moduli stabilisation

| Module | Public symbols | Purpose |
| --- | --- | --- |
| `kahlerjax.kahler_sector_N2` | `kahler_sector` | Kähler-potential corrections (BBHL-type), N=2 alpha' corrections. |
| `kahlerjax.cytree` | `cytree` | KahlerJAX's own pytree carrying the Kähler-side data. **Distinct from** `jaxvacua.lcs.lcs_tree`. |

### `jaxiverse` — axion EFT

| Module | Public symbols | Purpose |
| --- | --- | --- |
| `jaxiverse.axions` | `jaxion` (extends `kahler_sector`) | Multi-axion EFT, decay-constant matrix, mass spectrum. |

`jaxiverse.__init__` adds a convenience method to `cytools.calabiyau.CalabiYau`
(monkey-patch); end users can write `cy.jaxion(...)` rather than constructing
the EFT manually.

### `jaxpolylog` — pure leaf

| Symbol | Used by |
| --- | --- |
| `jax_polylog`, `jax_polylog_vmap` | `jaxvacua.css`, `jaxvacua.periods`, `jaxvacua.conifold.*`, `kahlerjax.kahler_sector_N2`, and indirectly any caller of those modules. |

### `cytools` — pure leaf (external)

| Symbol | Used by |
| --- | --- |
| `cytools.Polytope`, `cytools.Cone`, `cytools.triangulation.Triangulation` | `jaxvacua.conifold`, `jaxvacua.cytools_interface`, `kahlerjax` (toric_curves, divisors, orientifolds). |
| `cytools.calabiyau.CalabiYau` | `kahlerjax.cytree`, `jaxiverse.axions`, and end-user pipelines that build models from polytopes. |

## Cross-package dependency injection: the `lcs_tree` bus

A subtle but important design point: `lcs_tree` is *defined* in `jaxvacua`, but
`stringforge.vacua_writer` consumes it via duck typing — it accepts either a
`flux_sector`-like object (with `.periods.lcs_tree`) or an `lcs_tree` directly,
and never imports the concrete class. The relevant excerpt from
`stringforge/vacua_writer.py`:

```python
tree = getattr(getattr(model, "periods", None), "lcs_tree", model)
```

This pattern lets stringforge stay free of any jaxvacua import (`stringforge`'s
`__init__.py` only lazy-tries the siblings) while still operating on jaxvacua
data. Sibling packages can plug into the vacua-vault layer the same way: pass
any object that exposes the expected attributes.

The trade-off is that the runtime contract is implicit. The
[vacua-vault schema page](../api/stringforge.vacuavault) documents what fields
are read.

## Convention boundaries

Two conventions cross package boundaries and are easy to get wrong:

- **Mirror vs. catalog convention for Hodge numbers.** `stringforge.cy_io.CYDatabase`
  speaks *catalog* convention (the parquet column names match the original
  geometry's $h^{1,1}$ and $h^{1,2}$). `stringforge.lcs_database.LCSDatabase`
  speaks *mirror* convention end-to-end on its public surface — this matches
  `lcs_tree.h11` / `lcs_tree.h12` (mirror) and the jaxvacua physics. Translation
  happens at every public method's entry / exit. See
  [the migration page](migration_from_jaxvacua) for the rename and convention
  notes.
- **`x64` precision.** All sibling packages assume `JAX_ENABLE_X64=1`. Set this
  before importing JAX in your entry point.

## Ecosystem citation

When citing usage of the broader framework, cite the JAXVacua paper
([arXiv:2306.06160](https://arxiv.org/abs/2306.06160)) plus the
package-specific papers as listed on the
[overview page](overview).
