# JAXVacua

**Type IIB flux vacua: complex-structure and axio-dilaton sector.**

JAXVacua is the core package of the StringJAX ecosystem. It provides tools for
constructing and analysing flux vacua in Type IIB string compactifications,
built natively on JAX.

## Scope

- **Calabi–Yau geometry:** period vectors, prepotentials, Kähler potentials,
  gauge kinetic matrices, and instanton corrections — from Kreuzer–Skarke
  polytopes (via [CYTools](https://cy.tools)) or CICY data.
- **Moduli-space geometry:** Kähler metrics, Christoffel symbols, and curvature
  tensors via automatic differentiation.
- **Flux effective field theory:** GVW superpotential, covariant derivatives,
  F-term scalar potential, D3-tadpole, SL(2,ℤ) duality.
- **Vacuum finding:** gradient-based minimisation and Newton-type solvers with
  exact Jacobians. ISD-biased flux sampling.
- **Stability analysis:** exact Hessians and physical mass spectra via
  `jax.hessian`.
- **Ensemble generation:** Monte Carlo sampling, flux enumeration, and one-line
  wrappers for generating large vacuum datasets.

## Links

- **Full documentation:** [jaxvacua.readthedocs.io](https://jaxvacua.readthedocs.io)
- **Source code:** [github.com/AndreasSchachner/jaxvacua](https://github.com/AndreasSchachner/jaxvacua)
- **Reference paper:** [arXiv:2306.06160](https://arxiv.org/abs/2306.06160)
