# CYTools

**Toric Calabi–Yau geometry library — external dependency, not part of the
StringForge umbrella.**

[CYTools](https://cy.tools) is the upstream geometry library on which the
StringForge ecosystem builds. It provides numerical tools for working with toric
varieties, lattice polytopes, triangulations, and Calabi–Yau hypersurfaces /
complete intersections derived from them. Within the ecosystem, CYTools is the
canonical input layer: most pipelines start from a `cytools.Polytope` or
`cytools.calabiyau.CalabiYau`.

## Status

External public dependency.

## How the ecosystem uses it

- **`jaxvacua`** — `cytools.Polytope`, `cytools.triangulation.Triangulation`,
  `cytools.Cone` are imported by `jaxvacua.conifold`, `jaxvacua.cytools_interface`.
  The `cytools_interface` module translates a `cytools.CalabiYau` object into
  the inputs needed for `lcs_tree.from_cytools`.
- **`kahlerjax`** — heavy user of toric helpers (`Polytope`, `Triangulation`,
  `CalabiYau`) for divisor / orientifold computations.
- **`jaxiverse`** — uses `cytools.calabiyau.CalabiYau` and monkey-patches a
  `.jaxion(...)` convenience method onto it.
- **`stringforge`** — does not directly depend on cytools at runtime; the
  `LCSDatabase` and vacua-vault layers operate on already-loaded data.

## Compatibility notes

- When using the [CYTools Docker image](https://cy.tools), check the bundled
  NumPy version against the JAX-side requirements; mismatches can cause subtle
  array-protocol issues.
- All ecosystem packages assume `JAX_ENABLE_X64=1` is set before the first JAX
  import. CYTools itself does not require this.

## Links

- **Source / docs:** <https://cy.tools/>
- **Reference paper:** [arXiv:2211.03823](https://arxiv.org/abs/2211.03823).
