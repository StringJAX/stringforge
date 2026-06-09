# KahlerJAX

```{admonition} Planned package; API not stable
:class: warning
KahlerJAX is not part of the first public StringForge release.  It is shown here
to explain the intended ecosystem boundary.  Do not treat the module names,
examples, or planned interfaces below as stable public API, and do not expect
StringForge to install or import KahlerJAX.
```

**Planned role:** Kähler-moduli stabilisation for four-dimensional N=1 effective
field theories.

KahlerJAX is intended to consume Calabi-Yau data and shared conventions from the
StringForge ecosystem while owning the Kähler-sector physics itself.  StringForge
will remain the infrastructure layer: it will not absorb the Kähler solver.

## Planned ownership

- Kähler-sector data containers and model construction.
- Kähler-potential corrections and non-perturbative superpotential ingredients.
- Stabilisation routines and diagnostics for Kähler moduli.

## Current release status

- Not a StringForge dependency.
- Not imported by `stringforge`.
- No install command or source link is provided here until public release.
- Public documentation will be linked once the package is released.

## Related packages

- [`cytools`](cytools) for geometric input.
- [`jaxpolylog`](jaxpolylog) for differentiable polylogarithms where needed.
- [`jaxvacua`](jaxvacua) for the complex-structure/axio-dilaton flux sector.
