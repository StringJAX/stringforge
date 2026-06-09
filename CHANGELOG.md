# Changelog

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
