# Ecosystem overview

**StringForge** is the shared infrastructure layer for a family of JAX-based
packages targeting string compactifications, Calabi-Yau geometry, and effective
field theories.  It standardises the data and provenance boundary: catalogues,
caches, model-loading bridges, vault layouts, and validation.

StringForge intentionally carries minimal physics.  Physics engines consume its
data interfaces, but their computations live in their own packages.

## Package status

| Package | Role | Status | Reference |
| --- | --- | --- | --- |
| [`jaxvacua`](../packages/jaxvacua) | Type IIB flux vacua: complex-structure sector, vacuum finding, stability analysis. | Public | [arXiv:2306.06160](https://arxiv.org/abs/2306.06160) |
| [`jaxpolylog`](../packages/jaxpolylog) &mdash; [GitHub](https://github.com/AndreasSchachner/jaxpolylog) &nbsp;[docs](https://jaxpolylog.readthedocs.io) | JAX-compatible polylogarithm functions with autodiff support. | Public | -- |
| [`kahlerjax`](../packages/kahlerjax) | Planned Kähler-moduli stabilisation package. | Planned; API not stable | [arXiv:2507.00615](https://arxiv.org/abs/2507.00615) |
| [`jaxiverse`](../packages/jaxiverse) | Planned multi-axion EFT package. | Planned; API not stable | -- |
| [`cytools`](../packages/cytools) | External toric Calabi-Yau geometry library. | Public external dependency | [arXiv:2211.03823](https://arxiv.org/abs/2211.03823) |

```{important}
KahlerJAX and JAXiverse are shown to explain the intended ecosystem boundary.
They are not installed by StringForge, are not imported by StringForge, and their
public APIs should be treated as provisional until their own releases.
```

## Ecosystem flow

The figure below shows the practical boundary between StringForge and sibling
packages.  StringForge owns shared data movement and validation; the physics
packages own their package-specific calculations.

```{eval-rst}
.. raw:: html
   :file: ../_static/figures/f1_ecosystem_architecture.html
```

## Where to read what

| If you want... | Look here |
| --- | --- |
| Database queries, cache/offline behaviour, and HuggingFace layout. | [Calabi-Yau Geometry Database](../intro/database). |
| Loading JAXVacua models from database rows. | [Getting started](../getting_started) and [Database interface](../tutorials/database_and_infrastructure/database_interface). |
| Vacuum-solution storage, designation, validation, and vault publication. | [Vault workflow](../tutorials/vault_workflow), [Vacua storage](../tutorials/database_and_infrastructure/vacua_storage), and [`stringforge.vacuavault`](../api/stringforge.vacuavault). |
| Flux vacuum search, ISD sampling, flux bounding, periods, and mass spectra. | The [JAXVacua documentation](https://jaxvacua.readthedocs.io/en/latest/). |
| Advanced KKLT-style curated index and tagging. | [KKLT Database](../intro/kklt_database). |
| Planned Kähler or axion packages. | The planning pages, with the caveat that these are not first-release APIs. |

## Future optimisation tooling

PFV/fan-roots style optimisation packages are natural future consumers of the
StringForge data and vault conventions.  They are not public release components
and no install links are provided here yet.

## Citing the framework

Cite the package release together with the package-specific physics/software
papers relevant to your workflow.  For flux-vacuum calculations, cite the
JAXVacua paper in addition to StringForge.
