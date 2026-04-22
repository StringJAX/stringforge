# JAXPolyLog

**JAX-compatible polylogarithm functions with automatic differentiation.**

JAXPolyLog provides implementations of polylogarithm functions
(Li₂, Li₃, …) that are compatible with JAX's tracing model, supporting
`jit`, `vmap`, and `grad`. These functions are used internally by JAXVacua
for evaluating instanton corrections to the prepotential.

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxpolylog import Li

# Evaluate Li_3(z)
z = 0.5 + 0.3j
result = Li(3, z)

# Differentiate with respect to z
dLi3_dz = jax.grad(lambda x: Li(3, x).real)(z)
```

## Links

- **Source code:** [github.com/AndreasSchachner/jaxpolylog](https://github.com/AndreasSchachner/jaxpolylog)
