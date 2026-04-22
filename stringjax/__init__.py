# Copyright 2022-2026 Andreas Schachner
#
# This file is part of StringJAX.
#
# StringJAX is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# StringJAX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StringJAX. If not, see <https://www.gnu.org/licenses/>.

"""
StringJAX: Differentiable tools for string compactifications with JAX.

An umbrella framework providing a unified computational pipeline from
Calabi-Yau compactification data to four-dimensional effective field
theories, vacuum solutions, and physical observables.

Public submodules:
    jaxvacua   - Type IIB flux vacua (complex-structure + axio-dilaton sector)
    jaxpolylog - JAX-compatible polylogarithm functions

Optional (private, install separately):
    kahlerjax  - Kahler moduli stabilisation
    jaxiverse  - Multi-axion EFT from string compactifications
"""

__version__ = '0.0.1'

# Re-export public submodules.
try:
    import jaxpolylog
except ImportError:
    pass

try:
    import jaxvacua
except ImportError:
    pass

# Optional private submodules — available only with `pip install .[full]`.
try:
    import kahlerjax
except ImportError:
    pass

try:
    import jaxiverse
except ImportError:
    pass