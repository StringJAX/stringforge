# JAXVacua

**Type IIB flux vacua: complex-structure and axio-dilaton sector, vacuum finding, stability analysis.**

JAXVacua is the core flux-vacuum engine of the StringForge ecosystem. It implements
the layered pipeline from topological data to flux-vacuum solutions, built
natively on JAX with automatic differentiation, just-in-time compilation, and
hardware acceleration throughout.

## Status

Public.

## What it owns

- **`jaxvacua.lcs.lcs_tree`** — JAX-registered pytree carrying topological data
  (Hodge numbers, intersection numbers, GV invariants, conifold info).
- **`jaxvacua.periods.periods`** — period vector, prepotential, Kähler
  potential at large complex structure.
- **`jaxvacua.css.css`** — Kähler geometry of the complex-structure sector;
  gauge-kinetic matrix.
- **`jaxvacua.flux_eft.FluxEFT`** — GVW superpotential, F-term scalar potential,
  D3-tadpole, SL(2,ℤ) duality.
- **`jaxvacua.flux_vacua_finder.FluxVacuaFinder`** — Newton solver, ISD-biased
  initial-guess sampling, Hessian / mass spectrum via `jax.hessian`.
- **`jaxvacua.flux_bounding.bounded_fluxes`** — flux enumeration and stochastic
  search with cluster parallelisation hooks.
- **`jaxvacua.sampling.data_sampler`** — flux / moduli sampling utilities.
- **`jaxvacua.freezer`** — light-field EFTs after integrating out heavy moduli
  (e.g. conifold freezing).

The full module-by-module breakdown lives in [the architecture page](../ecosystem/architecture).

## What it consumes

- [`cytools`](cytools) — `Polytope`, `Triangulation`, `Cone` for KS-polytope input.
- [`jaxpolylog`](jaxpolylog) — `jax_polylog_vmap` for the instanton sum.
- [`stringforge`](../api/index) — for catalog / vacua-vault tooling, end users typically
  go through `stringforge.lcs_database.LCSDatabase` rather than constructing
  models by hand.

## Links

- **Full documentation:** <https://jaxvacua.readthedocs.io/en/latest/>
- **Source:** <https://github.com/StringJAX/jaxvacua>
- **Reference paper:** [arXiv:2306.06160](https://arxiv.org/abs/2306.06160)
  (Dubey, Krippendorf, Schachner — JHEP 12 (2023) 146)
