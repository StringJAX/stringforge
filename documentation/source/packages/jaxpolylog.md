# JAXPolyLog

**JAX-compatible polylogarithm functions with automatic differentiation.**

JAXPolyLog provides $\mathrm{Li}_s(z)$ implementations that are compatible with
JAX's tracing model — `jit`, `vmap`, `grad`, double `jacfwd`, etc. all work.
The library is the polylog dependency for any package in the ecosystem that
evaluates instanton corrections to a prepotential or scalar potential.

## Status

Public.

## What it owns

- **`jaxpolylog.jax_polylog`** — single-input polylog evaluator with
  configurable expansion (`approx`) and convergence range (`p_range`).
- **`jaxpolylog.jax_polylog_vmap`** — `vmap`-friendly variant for evaluating
  many points / curves in a single jitted call. The form used inside the
  ecosystem.

## What it consumes

- Pure leaf — depends only on `jax`, `jaxlib`, `numpy`.

## Used by

- [`jaxvacua`](jaxvacua) — `css`, `periods`, and the conifold submodules call
  `jax_polylog_vmap` for the instanton sum.
- [`kahlerjax`](kahlerjax) — `kahler_sector_N2` calls `jax_polylog_vmap` for
  the curve-instanton sum at higher polylog order.

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxpolylog import jax_polylog

# Evaluate Li_3(z)
z = 0.5 + 0.3j
result = jax_polylog(jnp.array(z, dtype=jnp.complex128), s=3, p_range=20, approx="patch")

# Differentiate (holomorphic gradient)
dLi3_dz = jax.grad(lambda x: jax_polylog(x, s=3, p_range=20, approx="patch").real, holomorphic=False)(z)
```

## Links

- **Source:** <https://github.com/AndreasSchachner/jaxpolylog>
