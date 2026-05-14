# KahlerJAX

```{admonition} Planned
:class: note
KahlerJAX is under development and will be released publicly in a future
version of StringJAX. This page is a placeholder for the package's public
overview. The architecture page describes the planned module surface; the
links section will be populated when the docs go live.
```

**Numerical Kähler moduli stabilisation for 4D N=1 effective field theories.**

KahlerJAX extends the ecosystem to the Kähler-moduli sector of string
compactifications. It provides tools for constructing and stabilising the
Kähler potential of arbitrary four-dimensional N=1 EFTs obtained from Calabi–Yau
compactifications, including non-perturbative effects and α′-corrections.

## Status

Planned (public release pending).

## What it owns (planned surface)

- **`kahlerjax.kahler_sector_N2.kahler_sector`** — Kähler-potential corrections
  (BBHL-type), N=2 α′-corrections, instanton-induced superpotential.
- **`kahlerjax.cytree.cytree`** — KahlerJAX's own pytree carrying the Kähler-side
  data. **Distinct from** `jaxvacua.lcs.lcs_tree`.
- **`kahlerjax.toric_curves`, `kahlerjax.divisors_*`, `kahlerjax.orientifolds_utils`** —
  toric-geometry helpers feeding the Kähler-sector calculation.

## What it consumes

- [`cytools`](cytools) — `Polytope`, `Triangulation`, `CalabiYau` for the toric
  data layer (heavily used).
- [`jaxpolylog`](jaxpolylog) — `jax_polylog_vmap` for higher-order polylog
  expansions in the curve-instanton sum.

## Used by

- [`jaxiverse`](jaxiverse) — extends `kahler_sector` for axion EFTs.

## Links

- **Source / docs:** to be populated when public.
- **Reference paper:** [arXiv:2507.00615](https://arxiv.org/abs/2507.00615).
