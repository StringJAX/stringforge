# Getting started

## Installation

**Prerequisites:** Python ≥ 3.12. If GPU acceleration is needed,
install [JAX with CUDA support](https://github.com/jax-ml/jax#installation)
first.

```bash
# Create and activate a virtual environment (recommended)
python -m venv stringjax-env && source stringjax-env/bin/activate

# Install from GitHub
pip install -e "git+https://github.com/AndreasSchachner/stringjax.git#egg=stringjax"

# Or clone and install locally (recommended for development)
git clone --recurse-submodules https://github.com/AndreasSchachner/stringjax.git
cd stringjax
pip install -e .
```

```{caution}
StringJAX requires `float64` precision. [JAX Metal](https://developer.apple.com/metal/jax/)
on macOS does not support `float64` and is therefore incompatible. Use the CPU backend
on Mac.
```

```{note}
When using the [CYTools](https://cy.tools) Docker image, check compatibility with the
required NumPy and JAX versions.
```

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
```

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

Each layer inherits from the one above and adds a new category of physics.
Orthogonal standalone modules provide composable functionality: flux-vector
algebra, heavy-field decoupling, Monte Carlo sampling, flux enumeration, and
database interfaces.

See the individual [package pages](packages/jaxvacua) for details on each
component.

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