# JAXiverse

```{admonition} Planned
:class: note
JAXiverse is under development and will be released publicly in a future
version of StringForge. This page is a placeholder for the package's public
overview.
```

**Multi-axion effective field theory: spectra, decay constants, and couplings
from Calabi–Yau compactifications.**

JAXiverse provides a framework-agnostic solver for multi-axion EFTs arising from
string compactifications. Given the inverse kinetic matrix, the charge matrix,
and the instanton scales of an N-axion system, it computes masses, decay
constants, mixing matrices, and quartic self-couplings — using a choice of
numerical strategies that range from fast hierarchical approximations to
arbitrary-precision backends.

## Status

Planned (public release pending).

## What it owns (planned surface)

- **`jaxiverse.axions.jaxion`** — multi-axion EFT, extends
  `kahlerjax.kahler_sector_N2.kahler_sector`. Computes the spectrum, decay
  constants and couplings.
- **`cytools.calabiyau.CalabiYau.jaxion`** — convenience method monkey-patched
  at import time so end users can write `cy.jaxion(...)` rather than
  constructing the EFT by hand.

## What it consumes

- [`cytools`](cytools) — `CalabiYau` for the geometric input.
- [`kahlerjax`](kahlerjax) — `kahler_sector` (subclass relationship).
- [`jaxpolylog`](jaxpolylog) — transitively, via `kahlerjax`.

## Links

- **Source / docs:** to be populated when public.
