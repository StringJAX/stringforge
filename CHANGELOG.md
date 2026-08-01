# Changelog

## Unreleased

### Added

- `ToricCYDatabase` (`stringforge.toric_db`) -- reader for the `toric` sub-dataset of `cy-database`: Calabi-Yau phases from triangulations of the 4d reflexive polytopes, in two modes. `frst` holds FRST classes (`cy()`-equivalence classes of fine, regular, star triangulations; arXiv:2310.06820) for `h11 = 1...12`; `vex` holds VEX classes (Wall classes of not-necessarily-fine star triangulations; arXiv:2512.14817) for `h11 = 2...7`. The catalogue is **sharded per `h11`** rather than a single `catalog.parquet`, so the FRST layer's >10^8 phases stay queryable: point lookups are O(1) via a per-`h11` `ks_id` index, and attribute queries stream one shard with `pyarrow` filter pushdown. Access is **local only** (`from_local`); lazy download of the sharded layout is not yet implemented.
- `CYPhase` and `ToricCYPhase` (`stringforge.cy_phase`) -- per-phase geometry objects. `CYPhase` is the construction-independent base, carrying exactly the Wall data `(h11, h21, kappa, c2)` that fixes the diffeomorphism type (Wall, *Invent. Math.* **1** (1966) 355); `ToricCYPhase` adds the polytope layer, the out-of-basis/in-basis machinery, and lazy CYTools bridges. Stored geometry is served without importing CYTools; `CYPhase.from_database` dispatches on the database's sub-dataset.
- `CICYPhase` (`stringforge.cy_phase`) -- wraps a row of the `cicy` sub-dataset as a `CYPhase`. It undoes that sub-dataset's mirror convention (its `h11` column is h^{2,1}(X) and its `chi` column is -chi(X)) so `h11`/`h12`/`euler_characteristic` agree with `ToricCYPhase`, and sets `basis_is_complete` from the Kähler-favourable flag -- verified over all 7,406 rows to be exactly the `len(c2) == h11(X)` condition and exactly the catalogue's `has_gv`. The 22 degenerate *product* entries, which record `h11 = h12 = 0` as a placeholder, are rejected with an explanatory error. `wall_hash` is deliberately **not** exposed: `basis_change` is `NULL` in every row and the configuration matrix was never ingested, so no basis identification exists and a CICY `wall_hash` would be uncomparable.
- `stringforge.toric_normalize` -- the geometry-normalisation and identifier/hash definitions for the toric sub-dataset (`TORIC_SCHEMA_VERSION`, in-basis conversion, `wall_hash`), shared by the builder and the reader so there is one source of truth.

### Fixed

- `CYDatabase.from_local` mis-resolved a path pointing *at* a sub-dataset directory whose layout has no monolithic `catalog.parquet`: the step-up guard keyed on that file alone, so `from_local(".../toric")` silently returned `.../toric/toric` and created that directory. The guard now also accepts `schema.json`.
- `LCSDatabase.load(cicy_id=...)` dimensioned the intersection-number tensor by the post-mirror-swap `h12`, which is correct for `tdf` but not for `cicy` -- whose catalogue is already in mirror convention. The quintic came back with a `(101, 101, 101)` tensor alongside a length-1 `c2`. The swap is now applied per sub-dataset, and `chi` follows the Hodge numbers.
- `LCSDatabase` hard-coded the `gv/h11_{N}/` GV layout, so `load(cicy_id=..., include_gv=True)` raised `FileNotFoundError` for `cicy`, whose `gv/` split is flat by design. Two further layers of the same failure are fixed: an absent GV family is now reported as `None` instead of a dict of `None`s (which defeated the downstream presence check), and GV truncation tolerates `None`.
- `LCSDatabase.load` passed a zero Kähler-moduli count straight through to `jaxvacua`, which built an empty (0, 0, 0) intersection tensor and then indexed into it, surfacing a bare `IndexError`. The `cicy` list's 22 degenerate *product* entries hit exactly this. The condition is now caught with a message naming the cause; the pre-existing zero-Hodge-number warning is kept for the cases that remain loadable.
- The toric `_check_schema` was a no-op that accepted **any** `schema.json` version, including a future incompatible one; it now validates against `TORIC_SCHEMA_VERSION` and raises `SchemaVersionError` on a mismatch, warning only when the file is absent or unversioned.

## 0.1.0 - 2026-05-25

Initial public-release preparation for StringForge.

### Added

- Shared Calabi-Yau database interfaces for TDF/Kreuzer-Skarke and CICY data.
- `LCSDatabase` bridge from catalogue rows to JAXVacua `lcs_tree` and `FluxVacuaFinder` objects.
- Vacua-vault persistence, validation, designation, retraction, fetch, and curation helpers.
- Advanced `KKLTDatabase` interface for a curated conifold-class indexed TDF subset.
- `stringforge.vulcan` -- cluster-side production vacuum-forging subpackage: worker-side parquet staging, head-node batched HuggingFace commits with an advisory 90-commit/hour rolling-window budget (10-commit margin below HF's 100/hour cap), `VulcanReader` query API, deterministic geometry-disjoint `VulcanMLView` train/val/test splits, and a `python -m stringforge.vulcan {status,sync}` CLI.
- Sphinx documentation with grouped tutorials, API reference pages, and package-boundary explanations.

### Release notes

- KahlerJAX and JAXiverse are described only as planned ecosystem packages; they are not installed or imported by StringForge.
- KKLT documentation is public but advanced and should not be treated as the default first-user workflow.
- Physics calculations remain delegated to JAXVacua and future sibling packages.
