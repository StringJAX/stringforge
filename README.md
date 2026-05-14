# StringForge

**Differentiable tools for string compactifications with JAX.**

StringForge is a Python framework for the systematic construction and analysis of string vacua, built natively on [JAX](https://github.com/google/jax). It provides a unified computational pipeline from Calabi–Yau compactification data to four-dimensional effective field theories, vacuum solutions, and physical observables — with automatic differentiation, just-in-time compilation, and hardware acceleration throughout.

The framework is designed to turn the study of string compactifications from a largely case-by-case enterprise into a scalable, reproducible computational discipline. It combines curated databases of geometric and topological input data with efficient numerical tools for repeated sampling, optimisation, and stability analysis, enabling ensemble-level investigations of the string landscape.

## Key features

- **Calabi–Yau geometry.** Period vectors, prepotentials, Kähler potentials, gauge kinetic matrices, and instanton corrections — evaluated from Kreuzer–Skarke polytopes (via [CYTools](https://cy.tools)) or CICY data.
- **Moduli-space geometry.** Kähler metrics, Christoffel symbols, and curvature tensors computed via automatic differentiation of the Kähler potential — no analytic formulae required beyond the prepotential.
- **Flux effective field theory.** GVW superpotential, covariant derivatives, F-term scalar potential, D3-tadpole, and SL(2,ℤ) duality — with clean separation of geometric and discrete flux data.
- **Vacuum finding.** Gradient-based minimisation and Newton-type solvers for the F-term equations, with exact Jacobians from autodiff. ISD-biased flux sampling for efficient targeting of supersymmetric vacua.
- **Stability analysis.** Exact Hessians and physical mass spectra via `jax.hessian`, without finite-difference noise.
- **Ensemble generation.** Monte Carlo sampling of flux space and the Kähler cone interior, systematic flux enumeration subject to physical bounds, and one-line wrappers for generating large vacuum datasets.
- **Reduced EFTs.** Abstract framework for integrating out heavy moduli (e.g. conifold freezing), with automatic reconstruction of the reduced superpotential and mass matrices.
- **Database interfaces.** Kreuzer–Skarke (via CYTools), CICY, and one-modulus hypergeometric families. Custom geometries supported via user-supplied prepotentials or period functions.
- **JAX-native throughout.** All core objects are JAX-registered pytrees: `jit`-compilable, `vmap`-batchable, and differentiable at arbitrary order.

## Architecture

The software architecture mirrors the layered structure of the physics:

```
periods          ← topological data, prepotential, period vector, Kähler potential
    ↓
css              ← Kähler geometry of the complex structure sector (autodiff)
    ↓
FluxEFT          ← flux background, superpotential, scalar potential
    ↓
FluxVacuaFinder  ← vacuum search, F-term solver, Hessian, mass spectrum
```

Each layer inherits from the one above and adds a new category of physics. Orthogonal standalone modules provide composable functionality: flux-vector algebra (`flux_utils`), heavy-field decoupling (`freezer`), Monte Carlo sampling (`sampling`), flux enumeration (`flux_bounding`), and database interfaces (`cytools_interface`, `cicy_prepot`).

At the base sits the `lcs_tree` — a JAX-registered pytree that separates static model metadata (Hodge numbers, model identifiers) from array-valued numerical leaves (intersection numbers, instanton invariants, cone generators). This is the point at which topological input data become portable differentiable objects.

## Packages

StringForge is an umbrella framework comprising the following packages:

| Package | Description | Status |
|---------|-------------|--------|
| **[JAXVacua](https://github.com/AndreasSchachner/jaxvacua)** | Type IIB flux vacua: complex-structure and axio-dilaton sector, vacuum finding, stability analysis | Public (this release) |
| **[JAXPolyLog](https://github.com/AndreasSchachner/jaxpolylog)** | JAX-compatible polylogarithm functions with autodiff support | Public (this release) |
| **KahlerJAX** | Numerical Kähler moduli stabilisation for 4d N=1 EFTs | Planned |
| **JAXiverse** | Multi-axion EFT: spectra, decay constants, and couplings from Calabi–Yau compactifications | Planned |

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxvacua import flux_sector

# Load a model: CP^{11169}[18] at LCS with instanton corrections
model = flux_sector(h12=2, model_ID=1, model_type="KS", maximum_degree=2)

# Choose a flux vector and a point in moduli space
fluxes = jnp.array([7, 3, -24, 0, -16, 50, 0, 3, -4, 0, 0, 0])
z = jnp.array([2.742j, 2.057j])        # complex structure moduli
tau = 6.855j                             # axio-dilaton

# Evaluate covariant derivatives D_i W (should vanish at a vacuum)
DW = model.DW(z, jnp.conj(z), tau, jnp.conj(tau), fluxes)
print("|DW| =", jnp.abs(DW))

# Evaluate the scalar potential
V = model.scalar_potential(z, jnp.conj(z), tau, jnp.conj(tau), fluxes)
print("V =", V)

# Generate an ensemble of 10,000 flux vacua
model.generate_sample(N=10000)
```

### Using Kreuzer–Skarke data via CYTools

```python
from cytools import fetch_polytopes

p = fetch_polytopes(h11=2, h12=272, limit=5, lattice="N", as_list=True)[0]
cy = p.triangulate().get_cy()

model = flux_sector(
    h12=cy.h11(), Q=cy.h11() + cy.h12() + 2,
    model_type="KS", maximum_degree=10,
    use_cytools=True, mirror_cy=cy
)
```

### Batched evaluation with `vmap`

```python
import numpy as np

N = 10_000
# Sample moduli inside the Kähler cone
generators = model.periods.generators_kahler_cone
coefficients = np.random.uniform(1, 5, (N, generators.shape[0]))
z0 = np.random.uniform(-0.5, 0.5, (N, generators.shape[1])) \
     + 1j * (coefficients @ generators)
tau0 = np.random.uniform(-0.5, 0.5, (N,)) \
       + 1j * np.random.uniform(2, 10, (N,))

# Vectorised ISD sampling
ISD_sampling = jax.vmap(
    lambda z, tau, flux: model.ISD_sampling(
        z, jnp.conj(z), tau, jnp.conj(tau),
        flux, mode="ISD+"
    )
)
fluxes_isd = ISD_sampling(z0, tau0, np.random.randint(-3, 4, (N, model.n_fluxes)))
```

## Installation

**Prerequisites:** Python ≥ 3.12. If GPU acceleration is needed, install [JAX with CUDA support](https://github.com/jax-ml/jax#installation) first.

```bash
# Create and activate a virtual environment (recommended)
python -m venv stringforge-env && source stringforge-env/bin/activate

# Install from GitHub
pip install -e "git+https://github.com/AndreasSchachner/stringforge.git#egg=stringforge"

# Or clone and install locally (recommended for development)
git clone --recurse-submodules https://github.com/AndreasSchachner/stringforge.git
cd stringforge
pip install -e .
```

> [!CAUTION]
> StringForge requires `float64` precision. [JAX Metal](https://developer.apple.com/metal/jax/) on macOS does not support `float64` and is therefore incompatible. Use the CPU backend on Mac.

> [!NOTE]
> When using the [CYTools](https://cy.tools) Docker image, check compatibility with the required NumPy and JAX versions.

## Documentation

The StringForge ecosystem documentation — including package overviews, tutorials, and an ecosystem-pipeline walkthrough — can be built from the `documentation/` folder in this repository. The full JAXVacua API reference is available at [jaxvacua.readthedocs.io](https://jaxvacua.readthedocs.io).

To build the documentation locally:

```bash
cd documentation
pip install -r requirements.txt
make html
# Open build/html/index.html
```

## Requirements

Core dependencies (installed automatically via `pip`):

- [JAX](https://github.com/google/jax) and jaxlib
- NumPy, SciPy, SymPy
- [Optax](https://github.com/deepmind/optax)
- Matplotlib, Seaborn
- h5py, Pandas, tqdm

Optional:

- [CYTools](https://cy.tools) — for constructing models from Kreuzer–Skarke polytopes
- [python-flint](https://github.com/flintlib/python-flint) — for exact arithmetic in selected routines

## Citation

If you find this work useful, please cite:

```bibtex
@article{Schachner:2026stringforge,
    author = "Schachner, Andreas",
    title = "{StringForge --- Differentiable Tools for String Compactifications}",
    eprint = "",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "",
    journal = "",
    year = "2026"
}
```

The initial JAXVacua framework was introduced in:

```bibtex
@article{Dubey:2023dvu,
    author = "Dubey, Abhishek and Krippendorf, Sven and Schachner, Andreas",
    title = "{JAXVacua --- a framework for sampling string vacua}",
    eprint = "2306.06160",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1007/JHEP12(2023)146",
    journal = "JHEP",
    volume = "12",
    pages = "146",
    year = "2023"
}
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Contact

Andreas Schachner
- Email: [as3475@cornell.edu](mailto:as3475@cornell.edu)
- GitHub: [github.com/AndreasSchachner](https://github.com/AndreasSchachner)
- Website: [andreasschachner.github.io](https://andreasschachner.github.io)