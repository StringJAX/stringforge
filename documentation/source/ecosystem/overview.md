# Ecosystem overview

**StringForge** is the umbrella framework for a family of JAX-based packages targeting
string compactifications, Calabi–Yau geometry, and 4-D effective field theories.
Each sibling package owns one layer of the compactification problem; this site
documents the shared infrastructure that ties them together (catalog I/O, vacua
storage, mirror-convention API) and provides cross-references into the per-package
documentation.

## Sibling packages

| Package | Role | Status | Reference |
| --- | --- | --- | --- |
| [`jaxvacua`](../packages/jaxvacua) | Type-IIB flux vacua: complex-structure sector, vacuum finding, stability analysis. | Public | [arXiv:2306.06160](https://arxiv.org/abs/2306.06160) |
| [`jaxpolylog`](../packages/jaxpolylog) | JAX-compatible polylogarithm functions with autodiff support. | Public | — |
| [`kahlerjax`](../packages/kahlerjax) | Kähler moduli stabilisation for 4-D N=1 EFTs. | Planned (public release pending) | [arXiv:2507.00615](https://arxiv.org/abs/2507.00615) |
| [`jaxiverse`](../packages/jaxiverse) | Multi-axion EFT: spectra, decay constants, couplings from CY compactifications. | Planned (public release pending) | — |
| [`cytools`](../packages/cytools) | Toric Calabi–Yau geometry library (external dependency). | Public | [arXiv:2211.03823](https://arxiv.org/abs/2211.03823) |

## Dependency diagram

The graph below shows code-level dependencies (solid edges) and the umbrella
aggregation (the `stringforge` cluster). `cytools` and `jaxpolylog` are leaves;
`kahlerjax` is consumed by `jaxiverse`; `jaxvacua` stands as its own pillar and
is consumed only via `stringforge`'s shared infrastructure.

```{mermaid}
graph TD
  cytools --> jaxvacua
  cytools --> kahlerjax
  cytools --> jaxiverse
  jaxpolylog --> jaxvacua
  jaxpolylog --> kahlerjax
  kahlerjax --> jaxiverse
  subgraph stringforge["stringforge (umbrella)"]
    jaxvacua
    kahlerjax
    jaxiverse
    jaxpolylog
  end
```

For a finer-grained breakdown of who owns what and who consumes what,
see [the architecture page](architecture).

## Where to read what

| If you want… | Look here |
| --- | --- |
| Physics intros (CY geometries, periods, flux compactifications, moduli stabilisation, perturbatively flat vacua, supergravity). | The `Introduction` chapters in the [JAXVacua docs](https://jaxvacua.readthedocs.io/en/latest/) (linked via intersphinx — physics background is not duplicated here). |
| Database & vacua-vault tooling (catalog queries, model loading, designation, HuggingFace push). | [Database interface](../tutorials/database_and_infrastructure/database_interface), [Vacua storage](../tutorials/database_and_infrastructure/vacua_storage), [Vault workflow](../tutorials/vault_workflow) on this site. |
| Flux vacuum search, ISD sampling, flux bounding. | The `Vacuum Finding` and `Flux Bounding` chapters in the JAXVacua docs. |
| Kähler moduli stabilisation. | The KahlerJAX docs (link will appear here once public). |
| Axion physics. | The JAXiverse docs (link will appear here once public). |
| End-to-end multi-package pipeline (CY → flux vacua → Kähler stabilisation → axion spectrum). | [Ecosystem pipeline](../tutorials/ecosystem_pipeline) on this site. |
| The recent `LCSDatabase` rename and mirror-convention swap. | [Migration from jaxvacua](migration_from_jaxvacua). |

## Citing the framework

If you use any part of the StringForge ecosystem in your research, cite the
framework paper:

```bibtex
@article{Dubey:2023dvu,
    author        = "Dubey, Abhishek and Krippendorf, Sven and Schachner, Andreas",
    title         = "{JAXVacua --- a framework for sampling string vacua}",
    eprint        = "2306.06160",
    archivePrefix = "arXiv",
    primaryClass  = "hep-th",
    doi           = "10.1007/JHEP12(2023)146",
    journal       = "JHEP",
    volume        = "12",
    pages         = "146",
    year          = "2023"
}
```

Per-package citations (KahlerJAX, JAXiverse, …) are listed on each package's
overview page once the package is public.
