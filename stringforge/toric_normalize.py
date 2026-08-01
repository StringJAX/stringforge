r"""
Normalization and deterministic identifiers for the FRST/VEX cy-database build.

This module converts the two source conventions (FRST vs VEX; see the build plan)
into **one** stored convention — the prime toric divisors, **0-indexed** — and
provides the reproducible identifiers used throughout the build.

**Stored (normalized) convention (D2), identical for FRST and VEX.** Positions
`0 … Ntor-1` correspond to prime toric divisor labels `1 … Ntor`
(`Ntor = basis_dim + 4`; favorable ⇒ `basis_dim = h11`). Intersection *numbers*
are unchanged; only the index layout is unified:

- **FRST source → normalized** (source is indexed `0 … Ntor` with index 0 = the
  interior/origin point): drop the origin. ``c2[i] = c2_src[i+1]`` and keep the
  dropped value as scalar ``c2_origin = c2_src[0]``; from the COO drop every
  triple containing label 0 and remap the survivors ``L → L-1``;
  ``basis = basis_src - 1``.
- **VEX source → normalized** (source COO uses labels `1 … Ntor`, ``c2`` is the
  0-indexed array over `1 … Ntor`): ``c2 = c2_src`` (``c2_origin = None``); remap
  the COO ``L → L-1``; ``basis = basis_src - 1``.

After normalization the in-basis slice is a single clean convention for both
datasets (:func:`in_basis_from_stored`), validated bit-for-bit against CYTools
``in_basis=True``.

**Identifiers.** ``polytope_hash`` = ``sha256(repr(normal_form))`` (identical to
``collect_ks_id_map.hash_poly`` / cornell-dev ``hash_database.db``);
``wall_hash`` = ``sha256`` of the Wall data ``(h11, h12, canonical in-basis κ,
canonical in-basis c₂)``; ``phase_id`` = ``"{dataset}:{h11}:{ks_id}:{triang_id}"``.
``triang_id`` is assigned as the deterministic rank under
:func:`triang_sort_key`.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

CONVENTION = "prime-toric-0indexed-v1"

#: Schema version of the ``toric`` sub-dataset, written into its ``schema.json`` and checked
#: on read.  Deliberately **not** named ``SCHEMA_VERSION``: the shipping package has its own
#: :data:`stringforge.cy_io.SCHEMA_VERSION` (currently 1) governing the *repository* layout,
#: and two same-named constants with different values invite silent confusion.
#:
#: It lives here rather than in the builder because this module is the only one both the
#: writer (``build_toric_database``) and the reader (``toric_db``) can import — the builder
#: pulls in ``cytools`` at module scope and so must never be imported by the consumer.
TORIC_SCHEMA_VERSION = 2

#: What changed in each ``toric`` schema version, surfaced in the reader's error message so a
#: user with a stale cache is told what to do.
TORIC_SCHEMA_CHANGELOG = {
    1: "initial toric layout (monolithic catalogs)",
    2: "sharded per-h11 catalogs + _ksid_index; thin phase catalog (no polytope_hash/phase_id; "
       "binary wall_hash); map-verified ks_id; real provenance; streaming/resumable build.",
}

COO = List[Tuple[int, int, int, int]]  # list of (i, j, k, value), 0-indexed positions


# --------------------------------------------------------------------------- #
# polytope hash
# --------------------------------------------------------------------------- #
def polytope_hash(p: "cytools.Polytope") -> str:  # noqa: F821 - runtime type
    r"""
    Canonical content hash of a polytope: ``sha256(repr(normal_form()))``.

    Byte-for-byte the ``collect_ks_id_map.hash_poly`` / cornell-dev
    ``hash_database.db`` convention, so the `ks_id_map` join and every
    cross-dataset link (frst↔vex↔tdf) use one identical key.
    """
    return normal_form_hash(p.normal_form())


def normal_form_hash(nf: np.ndarray) -> str:
    """Hash a CYTools ``normal_form`` matrix (or array-like) with the shared convention."""
    key = repr(tuple(tuple(int(x) for x in row) for row in np.asarray(nf)))
    return hashlib.sha256(key.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# geometry normalization (D2)
# --------------------------------------------------------------------------- #
def _canonicalize_coo(triples: Sequence[Sequence[int]]) -> COO:
    """
    Canonicalize a COO tensor: sort each triple's indices ascending, sum
    duplicate index-triples, drop zero-sum entries, and sort the result.

    Deterministic and order-independent — the basis for `wall_hash`.
    """
    agg: dict[Tuple[int, int, int], int] = {}
    for row in triples:
        i, j, k, v = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        key = (i, j, k) if i <= j <= k else tuple(sorted((i, j, k)))  # type: ignore[assignment]
        agg[key] = agg.get(key, 0) + v
    return sorted((a, b, c, val) for (a, b, c), val in agg.items() if val != 0)


def normalize_geometry(
    dataset: str,
    coo_src: Sequence[Sequence[int]],
    c2_src: Sequence[int],
    basis_src: Sequence[int],
) -> dict:
    r"""
    Map one source class-record's geometry to the stored (normalized) convention.

    **Arguments:**
    - ``dataset``: ``"frst"`` or ``"vex"``.
    - ``coo_src``: out-of-basis κ as a COO list of ``[i, j, k, value]`` in the
      source label convention (FRST: labels ``0…Ntor``; VEX: labels ``1…Ntor``).
    - ``c2_src``: out-of-basis c₂ (FRST: length ``Ntor+1``, index 0 = origin;
      VEX: length ``Ntor``, 0-indexed over labels ``1…Ntor``).
    - ``basis_src``: GLSM basis divisor labels.

    **Returns:** dict with
    ``coo`` (normalized COO, 0-indexed prime-toric positions, canonicalized),
    ``c2`` (length ``Ntor``), ``c2_origin`` (FRST origin value, else ``None``),
    ``basis`` (0-indexed positions), ``oob_dim`` (= ``Ntor``),
    ``basis_dim`` (= ``len(basis)``).
    """
    if dataset == "frst":
        c2_full = [int(x) for x in c2_src]
        c2_origin: Optional[int] = c2_full[0]
        c2 = c2_full[1:]                       # drop origin → positions 0…Ntor-1
        # drop origin (label 0) triples, remap L → L-1
        remapped = [
            [int(a) - 1, int(b) - 1, int(c) - 1, int(v)]
            for a, b, c, v in coo_src
            if int(a) != 0 and int(b) != 0 and int(c) != 0
        ]
    elif dataset == "vex":
        c2_origin = None
        c2 = [int(x) for x in c2_src]          # already 0-indexed over 1…Ntor
        remapped = [
            [int(a) - 1, int(b) - 1, int(c) - 1, int(v)] for a, b, c, v in coo_src
        ]
    else:
        raise ValueError("dataset must be 'frst' or 'vex'.")

    coo = _canonicalize_coo(remapped)
    basis = [int(b) - 1 for b in basis_src]
    return {
        "coo": coo,
        "c2": c2,
        "c2_origin": c2_origin,
        "basis": basis,
        "oob_dim": len(c2),
        "basis_dim": len(basis),
    }


# --------------------------------------------------------------------------- #
# in-basis slice (operates on normalized, 0-indexed data)
# --------------------------------------------------------------------------- #
def in_basis_from_stored(
    coo: Sequence[Sequence[int]],
    c2: Sequence[int],
    basis: Sequence[int],
) -> Tuple[COO, np.ndarray]:
    """
    Restrict normalized (0-indexed) κ / c₂ to the GLSM basis — the in-basis form.

    Returns ``(rows, c2_ib)`` where ``rows`` is the in-basis COO (indices remapped
    to ``0…basis_dim-1``) and ``c2_ib = c2[basis]``. Validated == CYTools
    ``intersection_numbers(in_basis=True)`` / ``second_chern_class(in_basis=True)``.
    """
    remap = {int(x): i for i, x in enumerate(basis)}
    c2_ib = np.asarray(c2)[np.asarray(basis, dtype=int)]
    rows = [
        (remap[int(i)], remap[int(j)], remap[int(k)], int(w))
        for i, j, k, w in coo
        if int(i) in remap and int(j) in remap and int(k) in remap
    ]
    return rows, c2_ib


def canonical_in_basis(
    coo: Sequence[Sequence[int]],
    c2: Sequence[int],
    basis: Sequence[int],
) -> Tuple[COO, List[int]]:
    """Canonical (deterministic) in-basis κ (COO) and c₂ (list) from normalized data."""
    rows, c2_ib = in_basis_from_stored(coo, c2, basis)
    return _canonicalize_coo(rows), [int(x) for x in c2_ib]


# --------------------------------------------------------------------------- #
# identifiers
# --------------------------------------------------------------------------- #
def _wall_hash_obj(
    h11: int,
    h12: int,
    coo: Sequence[Sequence[int]],
    c2: Sequence[int],
    basis: Sequence[int],
) -> "hashlib._Hash":
    """The ``sha256`` object over the Wall data (shared by hex/digest accessors)."""
    canon_coo, c2_ib = canonical_in_basis(coo, c2, basis)
    key = (int(h11), int(h12), tuple(canon_coo), tuple(c2_ib))
    return hashlib.sha256(repr(key).encode())


def wall_hash(
    h11: int,
    h12: int,
    coo: Sequence[Sequence[int]],
    c2: Sequence[int],
    basis: Sequence[int],
) -> str:
    r"""
    Hex ``sha256`` of the Wall data ``(h11, h12, canonical in-basis κ, canonical
    in-basis c₂)`` computed from normalized stored data.

    A **necessary** diffeomorphism pre-filter (Wall 1966) in the fixed GLSM basis
    — not GL(h¹¹,ℤ)-canonical and torsion-blind (see the plan's `wall_hash`
    semantics). Uses ``hashlib`` (never process-salted ``hash``). For the stored
    (thin) catalog use :func:`wall_hash_digest` (32-byte binary).
    """
    return _wall_hash_obj(h11, h12, coo, c2, basis).hexdigest()


def wall_hash_digest(
    h11: int,
    h12: int,
    coo: Sequence[Sequence[int]],
    c2: Sequence[int],
    basis: Sequence[int],
) -> bytes:
    r"""
    Raw 32-byte ``sha256`` **digest** of the Wall data — the compact form stored
    in the (thinned) per-phase catalog. ``wall_hash_digest(...).hex() ==
    wall_hash(...)``.
    """
    return _wall_hash_obj(h11, h12, coo, c2, basis).digest()


def triang_sort_key(
    coo: Sequence[Sequence[int]],
    c2: Sequence[int],
    basis: Sequence[int],
    heights: Sequence,
    c2_origin: Optional[int] = None,
) -> tuple:
    """
    Fully **content-based**, reproducible total order for assigning ``triang_id``
    within one polytope: (canonical in-basis κ, canonical in-basis c₂, heights,
    content-hash tiebreak).

    The final element is a ``sha256`` of the canonical **out-of-basis** (κ, c₂,
    c₂-origin) — a compact, deterministic discriminator. It replaces the previous
    ``src_index`` tiebreak (which depended on the non-reproducible source class
    order), so ``triang_id`` is reproducible across a fresh regeneration, not just
    a same-source rebuild. An exact tie on *all* elements means two byte-identical
    class representatives (a duplicate) — the build logs it.
    """
    canon_coo, c2_ib = canonical_in_basis(coo, c2, basis)
    heights_key = tuple(float(x) for x in np.asarray(heights).ravel().tolist())
    content = repr((tuple(_canonicalize_coo(coo)), tuple(int(x) for x in c2),
                    (int(c2_origin) if c2_origin is not None else None)))
    tiebreak = hashlib.sha256(content.encode()).hexdigest()
    return (tuple(canon_coo), tuple(c2_ib), heights_key, tiebreak)


def phase_id(dataset: str, h11: int, ks_id: int, triang_id: int) -> str:
    """``"{dataset}:{h11}:{ks_id}:{triang_id}"`` — reproducible, unique per phase."""
    return f"{dataset}:{int(h11)}:{int(ks_id)}:{int(triang_id)}"
