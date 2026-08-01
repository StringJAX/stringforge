r"""
``CYPhase`` — per-phase Calabi–Yau objects for the cy-database.

A small class family: a construction-independent base plus one subclass per construction
(currently :class:`ToricCYPhase` for the FRST / VEX toric sub-dataset).

While :class:`stringforge.toric_db.ToricCYDatabase` is the *I/O* layer (query the catalogues,
``load`` a phase's geometry into a ``dict``), ``CYPhase`` is the *object* layer:
one instance represents a single Calabi–Yau phase ``(mode, h11, ks_id, triang_id)``
and serves the precomputed geometry instantly, materialising a full CYTools object
only on demand.

It is modelled on jaxvacua's :class:`jaxvacua.lcs.lcs_tree` — eager stored
attributes + factory classmethods + ``to_X()`` bridges — but written in the
``stringforge`` docstring / typing style, and is JAX-free (a plain class, like
the :mod:`stringforge.cy_io` classes; not a pytree).

**Two-speed design.**

- **Stored fast-path (numpy only, no CYTools):** intersection numbers κ (out-of-
  basis and in-basis), the second Chern class c₂, the Euler characteristic χ, the
  Hodge numbers, the GLSM charge matrix, the basis, ``wall_hash`` and
  favourability — straight from the database row.
- **CYTools fallback (lazy):** cones, GV invariants and further toric analysis
  materialise a ``cytools`` object once, cached in a backing field (``_cy``).
  ``import cytools`` happens *inside* :meth:`ToricCYPhase.to_cytools` via
  :func:`_require_cytools` (the package forbids importing ``cytools`` at module
  top; see :mod:`stringforge.tests.test_import_hygiene`).

**Class family.** :class:`CYPhase` is the construction-independent base — the Wall data
:math:`(h^{1,1}, h^{2,1}, \kappa, c_2)` and what follows from it.  :class:`ToricCYPhase`
adds the polytope layer, the out-of-basis/in-basis machinery and the CYTools bridges.
Instantiating ``CYPhase`` with toric arguments returns a ``ToricCYPhase``, so the legacy
call form and :meth:`CYPhase.from_database` are unchanged.

**FRST vs VEX (one toric class).** ``mode`` distinguishes FRST (``cytools.CalabiYau``
classes) from VEX (Wall classes, toric fans).  :meth:`ToricCYPhase.to_cytools` returns
a ``cytools.CalabiYau`` for FRST but a CYTools ``Fan`` for VEX (there is no
``.cy()`` for a VEX phase); consequently the ``CalabiYau``-only features — GV
invariants, ``mori_cone(version="cap")``, ``kahler_cone(version="cup")`` and
:meth:`ToricCYPhase.to_lcs_tree` — are unavailable for VEX and raise
``NotImplementedError``.

Example::

    from stringforge import CYPhase, ToricCYDatabase

    db = ToricCYDatabase.from_local("/path/to/build")   # the dir containing toric/
    cp = CYPhase.from_database(db, "frst", h11=3, ks_id=0, triang_id=0)

    kappa = cp.intersection_numbers(in_basis=True)     # stored, no CYTools
    c2    = cp.second_chern_class(in_basis=True)        # stored, no CYTools
    chi   = cp.euler_characteristic                     # 2 * (h11 - h12)

    cy    = cp.to_cytools()                              # lazy CYTools CalabiYau
    kc    = cp.kahler_cone(version="cup")                # K_cup (FRST only)
    assert cp.verify()                                  # stored == CYTools recompute
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

import numpy as np

from . import toric_normalize as nz

#: The two phase layers. Defined locally rather than imported from
#: :mod:`build_toric_database`, which pulls in ``cytools`` at module scope — this module must
#: stay CYTools-free on import (``stringforge/tests/test_import_hygiene.py``).
_MODES = ("frst", "vex")


# --------------------------------------------------------------------------- #
# lazy heavy-dependency imports (house ``_require_*`` style, cy_io.py:192-253)
# --------------------------------------------------------------------------- #
def _require_cytools() -> Any:
    r"""
    **Description:**
    Import and return the ``cytools`` module with experimental features enabled
    (required for the VEX ``vc()`` / ``Fan`` path), raising a clear error if
    ``cytools`` is absent.
    """
    try:
        import cytools
    except ImportError:
        raise ImportError(
            "The 'cytools' package is required for CYPhase.to_cytools() and the "
            "cone / GV methods.  Install it with:  pip install cytools  "
            "(see https://cy.tools)."
        )
    cytools.config.enable_experimental_features()
    return cytools


def _require_lcs_tree() -> Any:
    r"""
    **Description:**
    Import and return :class:`jaxvacua.lcs.lcs_tree`, raising a clear error if
    ``jaxvacua`` is absent.  Used by :meth:`CYPhase.to_lcs_tree`.
    """
    try:
        from jaxvacua.lcs import lcs_tree
        return lcs_tree
    except ImportError:
        raise ImportError(
            "The 'jaxvacua' package is required for CYPhase.to_lcs_tree().  "
            "Install it with:  pip install jaxvacua"
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _dense(coo: Any, n: int) -> np.ndarray:
    r"""
    **Description:**
    Expand a COO intersection tensor ``[[i, j, k, value], ...]`` into a dense,
    fully symmetric ``(n, n, n)`` integer array.

    Args:
        coo (ArrayLike): COO triples with values, shape ``(nnz, 4)``.
        n (int): Side length of the dense tensor.

    Returns:
        np.ndarray: The symmetric dense tensor.
    """
    tensor = np.zeros((n, n, n), dtype=int)
    for i, j, k, v in np.asarray(coo, dtype=int).reshape(-1, 4):
        for a, b, c in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            tensor[a, b, c] = v
    return tensor


# --------------------------------------------------------------------------- #
# CYPhase
# --------------------------------------------------------------------------- #
class CYPhase:
    r"""
    **Description:**
    One Calabi–Yau threefold geometry, independent of how it was constructed.

    This base class carries exactly the data that is meaningful for *any* CY threefold —
    the **Wall data** :math:`(h^{1,1},\, h^{2,1},\, \kappa_{ijk},\, c_2)` together with what
    follows from it (:math:`\chi`, the Hodge pair, dense/COO views).  That is not an arbitrary
    boundary: by Wall's theorem (Wall 1966) those data, plus torsion, fix the diffeomorphism
    type of a smooth simply-connected CY threefold, and they are precisely what
    ``wall_hash`` fingerprints.

    Construction-specific material lives in the subclasses:

    - :class:`ToricCYPhase` — polytope, GLSM basis, triangulation ``heights``, favourability,
      the out-of-basis/in-basis machinery, and every CYTools bridge.
    - ``CICYPhase`` — the CICY-list identifier and its ancillary data.

    **Instantiating :class:`CYPhase` with toric arguments returns a**
    :class:`ToricCYPhase` (see :meth:`__new__`), so existing call sites and
    :meth:`from_database` keep working unchanged.

    Args:
        construction (str): Which construction this geometry came from — ``"toric"`` or
            ``"cicy"``.
        h11 (int): Hodge number :math:`h^{1,1}`.
        h12 (int): Hodge number :math:`h^{2,1}` (:math:`= h^{1,2}` by Hodge symmetry).
        intnums_coo (ArrayLike): Triple intersection numbers as COO rows
            ``(i, j, k, value)``, in whatever index convention the construction supplies.
        c2 (Sequence[int]): Second Chern class, same index convention as ``intnums_coo``.
        basis_rank (int | None): Number of divisor classes the stored geometry spans.
            Defaults to ``len(c2)``.
        basis_is_complete (bool): Whether those classes span all of :math:`H^{1,1}(X)`.
            ``False`` means the stored geometry is a *proper subspace* and in-basis
            quantities describe only part of the topology — the toric non-favorable case
            (``fav_N=False``) and the CICY non-Kähler-favourable case.
        chi (int | None): Euler characteristic.  Defaults to :math:`2(h^{1,1} - h^{2,1})`.

    Raises:
        ValueError: If ``construction`` is unknown, or a supplied ``chi`` contradicts the
            Hodge numbers.
    """

    #: Keyword arguments that only a toric phase can supply.  Their presence is what makes
    #: ``CYPhase(...)`` dispatch to :class:`ToricCYPhase` (back-compatibility with the
    #: pre-split 17-argument constructor).
    _TORIC_ONLY_KWARGS = frozenset({
        "dataset", "mode", "ks_id", "triang_id", "heights", "vertices", "glsm_basis",
        "glsm_charge_matrix", "fav_N", "fav_M", "trilayer", "polytope_hash", "c2_origin",
        "oob_dim", "basis_dim", "phase_id", "wall_hash",
    })

    _CONSTRUCTIONS = ("toric", "cicy")

    def __new__(cls, **kwargs: Any) -> "CYPhase":
        r"""Dispatch a bare ``CYPhase(...)`` call to the right subclass.

        Only fires when :class:`CYPhase` itself is instantiated *and* toric-only keywords are
        present; subclasses construct normally.  Python then calls the chosen subclass's
        ``__init__`` with the original keywords, so the legacy call form is untouched.
        """
        if cls is CYPhase and (cls._TORIC_ONLY_KWARGS & set(kwargs)):
            return object.__new__(ToricCYPhase)
        return object.__new__(cls)

    def __init__(
        self,
        *,
        construction: str,
        h11: int,
        h12: int,
        intnums_coo: Any,
        c2: Any,
        basis_rank: Optional[int] = None,
        basis_is_complete: bool = True,
        chi: Optional[int] = None,
    ) -> None:
        if construction not in self._CONSTRUCTIONS:
            raise ValueError(
                f"construction must be one of {self._CONSTRUCTIONS}; got {construction!r}."
            )
        self.construction = construction
        self.h11 = int(h11)
        self.h12 = int(h12)
        self._coo = np.asarray(intnums_coo, dtype=int).reshape(-1, 4)
        self._c2 = np.asarray(c2, dtype=int)
        self.basis_rank = int(basis_rank) if basis_rank is not None else int(self._c2.shape[0])
        self.basis_is_complete = bool(basis_is_complete)
        expected_chi = 2 * (self.h11 - self.h12)
        if chi is not None and int(chi) != expected_chi:
            raise ValueError(
                f"chi={chi} contradicts the Hodge numbers: for a CY threefold "
                f"chi = 2*(h11 - h12) = {expected_chi}."
            )
        self._chi = expected_chi

    # -- universal, construction-independent geometry ---------------------- #
    @property
    def euler_characteristic(self) -> int:
        r"""**Description:** Euler characteristic :math:`\chi = 2\,(h^{1,1} - h^{2,1})`."""
        return self._chi

    @property
    def hodge_numbers(self) -> Tuple[int, int]:
        r"""**Description:** The pair :math:`(h^{1,1},\, h^{2,1})`."""
        return (self.h11, self.h12)

    def to_dense(self, coo: Optional[Any] = None, n: Optional[int] = None) -> np.ndarray:
        r"""
        **Description:**
        Expand COO intersection numbers into a dense, fully symmetric tensor.

        Args:
            coo (ArrayLike | None): COO rows; defaults to this geometry's stored κ.
            n (int | None): Side length; defaults to :attr:`basis_rank`.

        Returns:
            np.ndarray: Symmetric ``(n, n, n)`` integer tensor.
        """
        return _dense(self._coo if coo is None else coo,
                      self.basis_rank if n is None else int(n))

    def intersection_numbers(self, in_basis: bool = False, format: str = "coo") -> np.ndarray:
        r"""
        **Description:**
        The stored triple intersection numbers.

        ``in_basis`` is a **toric** notion (the stored toric form is indexed by *prime toric
        divisors*, a superset of a basis) and is only available on :class:`ToricCYPhase`;
        other constructions store κ already in their divisor basis.

        Args:
            in_basis (bool): Must be ``False`` on the base class.
            format (str): ``"coo"`` or ``"dense"``.

        Returns:
            np.ndarray: ``(nnz, 4)`` COO rows, or a symmetric dense tensor.

        Raises:
            NotImplementedError: If ``in_basis=True`` on a construction without an
                out-of-basis form.
            ValueError: On an unknown ``format``.
        """
        if in_basis:
            raise NotImplementedError(
                f"in_basis=True is a toric notion (prime-toric positions sliced to the GLSM "
                f"basis) and is not defined for construction={self.construction!r}, whose "
                f"kappa is already stored in its divisor basis."
            )
        if format == "coo":
            return self._coo
        if format == "dense":
            return self.to_dense()
        raise ValueError(f"format must be 'coo' or 'dense'; got {format!r}.")

    def second_chern_class(self, in_basis: bool = False) -> np.ndarray:
        r"""
        **Description:**
        The stored second Chern class.  See :meth:`intersection_numbers` on ``in_basis``.
        """
        if in_basis:
            raise NotImplementedError(
                f"in_basis=True is not defined for construction={self.construction!r}."
            )
        return self._c2

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(construction={self.construction!r}, h11={self.h11}, "
            f"h12={self.h12}, basis_rank={self.basis_rank}, "
            f"basis_is_complete={self.basis_is_complete})"
        )

    # -- dispatching factories --------------------------------------------- #
    #: Maps a consumer database's ``dataset`` name to the subclass that reads it.  Adding a
    #: construction means adding one entry here, not editing the dispatch logic.  Values are
    #: resolved lazily by name because the subclasses are defined further down this module.
    _DB_DISPATCH = {"toric": "ToricCYPhase"}

    @classmethod
    def _subclass_for_db(cls, db: Any) -> type:
        r"""Resolve the subclass that reads ``db``, by its ``dataset`` name."""
        name = getattr(db, "dataset", None)
        try:
            return globals()[cls._DB_DISPATCH[name]]
        except KeyError:
            known = ", ".join(sorted(cls._DB_DISPATCH))
            raise TypeError(
                f"CYPhase cannot be built from a {type(db).__name__} "
                f"(dataset={name!r}); no CYPhase subclass reads that sub-dataset. "
                f"Supported: {known}."
            ) from None

    @classmethod
    def from_database(cls, db: Any, *args: Any, **kwargs: Any) -> "CYPhase":
        r"""
        **Description:**
        Build the right :class:`CYPhase` subclass from a consumer database.

        Called on :class:`CYPhase` it dispatches on ``db.dataset`` via :data:`_DB_DISPATCH`;
        subclasses override with their own signature (see
        :meth:`ToricCYPhase.from_database`).
        """
        if cls is CYPhase:
            return cls._subclass_for_db(db).from_database(db, *args, **kwargs)
        raise NotImplementedError(f"{cls.__name__} does not implement from_database().")

    @classmethod
    def from_row(cls, *args: Any, **kwargs: Any) -> "CYPhase":
        r"""**Description:** As :meth:`from_database`, from already-fetched rows.

        There is no database to dispatch on here, so the base implementation targets the
        toric layout (the only row schema that existed when this was written).  Call the
        subclass directly when the construction is known.
        """
        if cls is CYPhase:
            return ToricCYPhase.from_row(*args, **kwargs)
        raise NotImplementedError(f"{cls.__name__} does not implement from_row().")


class ToricCYPhase(CYPhase):
    r"""
    **Description:**
    One Calabi–Yau phase from the FRST / VEX cy-database.

    A ``CYPhase`` carries the stored (normalized, prime-toric 0-indexed) geometry
    of a single phase and exposes it through a numpy fast-path, while lazily
    materialising a CYTools object for cone / GV computations.  See the module
    docstring for the two-speed design and the FRST/VEX differences.

    Args:
        dataset (str): ``"frst"`` or ``"vex"``.
        h11 (int): Hodge number :math:`h^{1,1}`.
        h12 (int): Hodge number :math:`h^{2,1}` (:math:`= h^{1,2}` by Hodge symmetry;
            spelled :math:`h^{2,1}` throughout this module and in the stored columns).
        ks_id (int): Canonical Kreuzer–Skarke polytope index.
        triang_id (int): Phase (class) index within the polytope.
        heights (Sequence[float]): Triangulation heights (stored verbatim; VEX
            heights may be non-integer).
        intnums_coo (ArrayLike): Out-of-basis κ as COO triples ``(nnz, 4)`` in the
            normalized (prime-toric 0-indexed) convention.
        c2 (Sequence[int]): Out-of-basis c₂ (length ``oob_dim``), normalized.
        c2_origin (int | None): FRST origin-divisor c₂ value (``None`` for VEX).
        vertices (ArrayLike): Polytope vertices.
        glsm_basis (Sequence[int]): GLSM basis positions (0-indexed).
        glsm_charge_matrix (ArrayLike): GLSM charge matrix of the polytope.
        fav_N (bool): Whether the polytope is favorable.
        fav_M (bool): Whether the mirror polytope is favorable.
        trilayer (bool): Trilayer flag (as stored).
        wall_hash (bytes): Wall-data diffeomorphism pre-filter fingerprint — the raw
            32-byte sha256 digest as stored in the catalog (use ``.hex()`` for display).
        polytope_hash (str): ``sha256(repr(normal_form))`` content hash.
        oob_dim (int | None): Out-of-basis dimension ``= Ntor``.  Defaults to
            ``len(c2)``.
        basis_dim (int | None): ``len(glsm_basis)`` (``= h11`` iff favorable).
            Defaults to ``len(glsm_basis)``.
        phase_id (str | None): ``"{dataset}:{h11}:{ks_id}:{triang_id}"``.
            Computed if omitted.

    Example::

        cp = CYPhase.from_database(db, mode="frst", h11=3, ks_id=0, triang_id=0)
        cp.intersection_numbers(in_basis=True, format="dense")
    """

    def __init__(
        self,
        *,
        h11: int,
        h12: int,
        ks_id: int,
        triang_id: int,
        heights: Any,
        intnums_coo: Any,
        c2: Any,
        vertices: Any,
        glsm_basis: Any,
        glsm_charge_matrix: Any,
        fav_N: bool,
        fav_M: bool,
        trilayer: bool,
        wall_hash: Any,
        polytope_hash: str,
        mode: Optional[str] = None,
        dataset: Optional[str] = None,
        c2_origin: Optional[int] = None,
        oob_dim: Optional[int] = None,
        basis_dim: Optional[int] = None,
        phase_id: Optional[str] = None,
        chi: Optional[int] = None,
    ) -> None:
        # `mode` is canonical; `dataset` is the pre-2026-07-30 spelling, kept working because
        # the inherited `CYDatabase.dataset` means the *sub-dataset* ("toric") — one attribute
        # name for two disjoint vocabularies across one API.  See :attr:`dataset`.
        if mode is None:
            if dataset is None:
                raise ValueError("ToricCYPhase requires mode='frst' or mode='vex'.")
            mode = dataset
        elif dataset is not None and dataset != mode:
            raise ValueError(
                f"mode={mode!r} and the deprecated dataset={dataset!r} disagree; pass only mode."
            )
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}; got {mode!r}.")
        self.mode = mode

        # polytope-level data (needed before super().__init__ so basis_dim can default)
        self.vertices = np.asarray(vertices)
        self.glsm_basis = [int(x) for x in glsm_basis]
        self.glsm_charge_matrix = np.asarray(glsm_charge_matrix)
        self.fav_N = bool(fav_N)
        self.fav_M = bool(fav_M)
        self.trilayer = bool(trilayer)
        self.basis_dim = int(basis_dim) if basis_dim is not None else len(self.glsm_basis)

        # universal Wall data.  `basis_is_complete` is exactly toric favourability: for
        # fav_N=False the stored geometry spans only the basis_dim toric classes of h11.
        super().__init__(
            construction="toric", h11=h11, h12=h12, intnums_coo=intnums_coo, c2=c2,
            basis_rank=self.basis_dim, basis_is_complete=bool(fav_N), chi=chi,
        )

        self.ks_id = int(ks_id)
        self.triang_id = int(triang_id)

        # geometry (stored, normalized prime-toric 0-indexed)
        self.heights = [float(x) for x in np.asarray(heights).ravel().tolist()]
        self.c2_origin = None if c2_origin is None else int(c2_origin)

        # identifiers
        self.wall_hash = wall_hash
        self.polytope_hash = polytope_hash
        self.phase_id = phase_id or nz.phase_id(mode, h11, ks_id, triang_id)

        # dimensions
        self.oob_dim = int(oob_dim) if oob_dim is not None else int(self._c2.shape[0])

        # lazy CYTools backing field (materialised by :meth:`to_cytools`)
        self._cy: Any = None
        # emit the non-favorable "toric part only" warning at most once per instance
        self._warned_nonfav: bool = False

    # -- factories ------------------------------------------------------------ #
    @classmethod
    def from_database(cls, db: Any, mode: Optional[str] = None, h11: Optional[int] = None,
                      ks_id: Optional[int] = None, triang_id: Optional[int] = None,
                      h12: Optional[int] = None) -> "CYPhase":
        r"""
        **Description:**
        Build a :class:`CYPhase` from a :class:`toric_db.ToricCYDatabase` by combining
        ``db.load(mode, ...)`` (per-phase geometry from the ``mode`` layer) with the
        **shared** ``db.get_polytope(...)`` (per-polytope data).

        Accepts positional or keyword form::

            CYPhase.from_database(db, "frst", 10, 42, 0)
            CYPhase.from_database(db, mode="frst", h11=10, ks_id=42, triang_id=0)

        **The unique key is** ``(mode, h11, ks_id, triang_id)`` — all four are required and a
        :class:`ValueError` names any that are missing.  Two points worth being explicit about:

        - ``mode`` is **not** optional.  ``triang_id`` is enumerated independently per mode, so
          ``(h11, ks_id, triang_id)`` alone is ambiguous wherever a polytope has both layers:
          at ``h11=4, ks_id=1`` for instance, ``triang_id=0`` names a different geometry in
          ``frst`` than in ``vex`` (verified — the two have different ``wall_hash``).
        - ``h12`` is **not** needed for identification.  ``ks_id`` is the *global per-h11* KS
          emission index, so it is already unique within an ``h11`` (audited: ``#unique ks_id ==
          #rows`` in every built bucket, e.g. 568 078/568 078 at ``h11=10``), and ``h12`` is a
          *function* of ``(h11, ks_id)``.  Do not confuse it with the *within-Hodge-pair* index
          used by some other tooling (``hash_database``'s ``KS_index``), which does need both:
          ``global ks_id = offset(h12) + within-pair index``.  If you pass ``h12`` it is used as
          a **cross-check** and a mismatch raises, which is the useful thing it can do here.

        Args:
            db (ToricCYDatabase): An open consumer database.
            mode (str): ``"frst"`` or ``"vex"`` — which phase layer to load from. Required.
            h11 (int): Hodge number :math:`h^{1,1}`. Required.
            ks_id (int): Canonical Kreuzer–Skarke polytope index (unique within ``h11``). Required.
            triang_id (int): Phase index within the polytope, per mode. Required.
            h12 (int): Optional cross-check on :math:`h^{2,1}`; redundant for identification.

        Returns:
            CYPhase: The phase object.

        Raises:
            ValueError: If any part of the key is missing, if ``mode`` is not ``"frst"``/``"vex"``,
                or if a supplied ``h12`` disagrees with the stored value.
        """
        missing = [n for n, v in (("mode", mode), ("h11", h11), ("ks_id", ks_id),
                                  ("triang_id", triang_id)) if v is None]
        if missing:
            raise ValueError(
                f"CYPhase.from_database: incomplete key — missing {missing}. A phase is identified "
                f"by (mode, h11, ks_id, triang_id); e.g. "
                f"CYPhase.from_database(db, mode='frst', h11=10, ks_id=42, triang_id=0). "
                f"(h12 is optional and only cross-checked: it is determined by (h11, ks_id).)")
        if mode not in _MODES:
            raise ValueError(f"CYPhase.from_database: mode must be one of {_MODES}, got {mode!r}")
        geom = db.load(mode, h11, ks_id, triang_id, in_basis=False)
        poly = db.get_polytope(h11=h11, ks_id=ks_id)
        if h12 is not None and int(h12) != int(geom["h12"]):
            raise ValueError(
                f"CYPhase.from_database: h12={h12} does not match the stored h12="
                f"{int(geom['h12'])} for (mode={mode!r}, h11={h11}, ks_id={ks_id}). "
                f"ks_id is the global per-h11 KS emission index, so (h11, ks_id) already fixes "
                f"h12 — check whether your ks_id came from a within-Hodge-pair numbering.")
        return cls(
            dataset=mode,
            h11=h11,
            h12=geom["h12"],
            ks_id=ks_id,
            triang_id=triang_id,
            heights=geom["heights"],
            intnums_coo=geom["intnums_coo"],
            c2=geom["c2"],
            c2_origin=geom["c2_origin"],
            vertices=poly["vertices"],
            glsm_basis=poly["glsm_basis"],
            glsm_charge_matrix=poly["glsm_charge_matrix"],
            fav_N=poly["fav_N"],
            fav_M=poly["fav_M"],
            trilayer=poly["trilayer"],
            wall_hash=geom["wall_hash"],
            polytope_hash=geom["polytope_hash"],
            oob_dim=poly["oob_dim"],
            basis_dim=poly["basis_dim"],
            phase_id=geom["phase_id"],
        )

    @classmethod
    def from_row(cls, geom_row: Any, polytope_row: Any, dataset: str) -> "CYPhase":
        r"""
        **Description:**
        Build a :class:`CYPhase` directly from a ``geom/`` row and a
        ``polytope/`` row (e.g. two aligned Parquet / ``dict`` records), without
        an open database.

        The ``geom_row`` may carry κ either as a single ``"intnums_coo"`` array
        or as the four split columns ``intnums_coo_{i,j,k,v}``.

        Args:
            geom_row (Mapping): Per-phase geometry row.
            polytope_row (Mapping): Per-polytope row.
            dataset (str): ``"frst"`` or ``"vex"``.

        Returns:
            CYPhase: The phase object.
        """
        coo = cls._coo_from_row(geom_row)
        h11 = int(polytope_row.get("h11", geom_row["h11"]))
        h12 = int(geom_row["h12"])
        c2 = geom_row["c2"]
        glsm_basis = polytope_row["glsm_basis"]
        wall_hash = geom_row.get("wall_hash", polytope_row.get("wall_hash"))
        if wall_hash is None:                       # E2: compute it (we have the normalized geometry)
            wall_hash = nz.wall_hash_digest(h11, h12, coo, c2, glsm_basis)
        return cls(
            dataset=dataset,
            h11=h11,
            h12=h12,
            ks_id=int(geom_row["ks_id"]),
            triang_id=int(geom_row["triang_id"]),
            heights=geom_row["heights"],
            intnums_coo=coo,
            c2=c2,
            c2_origin=geom_row.get("c2_origin"),
            vertices=polytope_row["vertices"],
            glsm_basis=glsm_basis,
            glsm_charge_matrix=polytope_row["glsm_charge_matrix"],
            fav_N=polytope_row["fav_N"],
            fav_M=polytope_row["fav_M"],
            trilayer=polytope_row["trilayer"],
            wall_hash=wall_hash,
            polytope_hash=polytope_row["polytope_hash"],
            oob_dim=polytope_row.get("oob_dim"),
            basis_dim=polytope_row.get("basis_dim"),
            phase_id=geom_row.get("phase_id"),
        )

    @staticmethod
    def _coo_from_row(row: Any) -> np.ndarray:
        """Extract a ``(nnz, 4)`` COO array from a row (single array or split columns)."""
        if row.get("intnums_coo") is not None:
            return np.asarray(row["intnums_coo"], dtype=int).reshape(-1, 4)
        cols = (row["intnums_coo_i"], row["intnums_coo_j"],
                row["intnums_coo_k"], row["intnums_coo_v"])
        return np.asarray(list(zip(*[[int(x) for x in c] for c in cols])),
                          dtype=int).reshape(-1, 4)

    # -- stored fast-path (no CYTools) --------------------------------------- #
    def intersection_numbers(self, in_basis: bool = False, format: str = "coo") -> np.ndarray:
        r"""
        **Description:**
        Return the stored triple intersection numbers κ.

        Args:
            in_basis (bool): If ``True``, restrict to the GLSM basis via
                :func:`normalize.in_basis_from_stored` (== CYTools
                ``intersection_numbers(in_basis=True)``); otherwise return the
                out-of-basis prime-toric tensor.
            format (str): ``"coo"`` for COO triples ``(nnz, 4)`` or ``"dense"``
                for a symmetric ``(n, n, n)`` array.

        Returns:
            np.ndarray: κ in the requested layout.  For a **non-favorable** phase
            the in-basis form spans only the ``basis_dim`` toric classes (< ``h11``);
            a :class:`UserWarning` is emitted (see :meth:`covers_full_h11`).

        Raises:
            ValueError: If ``format`` is not ``"coo"`` or ``"dense"``.
        """
        if in_basis:
            self._warn_if_incomplete_basis("intersection numbers")
            rows, _ = nz.in_basis_from_stored(self._coo, self._c2, self.glsm_basis)
            coo = np.asarray(rows, dtype=int).reshape(-1, 4)
            n = self.basis_dim
        else:
            coo = self._coo
            n = self.oob_dim
        if format == "coo":
            return coo
        if format == "dense":
            return _dense(coo, n)
        raise ValueError(f"format must be 'coo' or 'dense'; got {format!r}.")

    def to_dense(self, coo: Optional[Any] = None,  # type: ignore[override]
                 n: Optional[int] = None) -> np.ndarray:
        r"""
        **Description:**
        Expand COO intersection numbers into a dense symmetric tensor.

        The side length must be overridden relative to the base class.  A toric phase stores κ
        in the **out-of-basis** prime-toric indexing, whose positions run up to ``oob_dim - 1``,
        while :attr:`~CYPhase.basis_rank` is the smaller ``basis_dim``.  Defaulting to
        ``basis_rank`` — as the base class does, correctly, for constructions that store κ
        already in a divisor basis — would raise ``IndexError`` here.

        Args:
            coo (ArrayLike | None): COO rows; defaults to the stored out-of-basis κ.
            n (int | None): Side length; defaults to :attr:`oob_dim`.  **Pass ``n`` explicitly
                when passing an in-basis ``coo``**, otherwise the result is padded out to
                ``oob_dim``.

        Returns:
            np.ndarray: Symmetric ``(n, n, n)`` integer tensor.
        """
        return super().to_dense(coo, self.oob_dim if n is None else int(n))

    def second_chern_class(self, in_basis: bool = False) -> np.ndarray:
        r"""
        **Description:**
        Return the stored second Chern class c₂.

        Args:
            in_basis (bool): If ``True``, restrict to the GLSM basis
                (== CYTools ``second_chern_class(in_basis=True)``); otherwise
                return the out-of-basis prime-toric vector (length ``oob_dim``).

        Returns:
            np.ndarray: c₂ in the requested basis.  For a **non-favorable** phase
            the in-basis form spans only the ``basis_dim`` toric classes (< ``h11``);
            a :class:`UserWarning` is emitted (see :meth:`covers_full_h11`).
        """
        if in_basis:
            self._warn_if_incomplete_basis("second Chern class")
            _, c2_ib = nz.in_basis_from_stored(self._coo, self._c2, self.glsm_basis)
            return np.asarray(c2_ib, dtype=int)
        return np.asarray(self._c2, dtype=int)

    @property
    def euler_characteristic(self) -> int:
        r"""**Description:** Euler characteristic :math:`\chi = 2\,(h^{1,1} - h^{2,1})`."""
        return 2 * (self.h11 - self.h12)

    @property
    def hodge_numbers(self) -> Tuple[int, int]:
        r"""**Description:** The Hodge numbers ``(h11, h12)``."""
        return (self.h11, self.h12)

    @property
    def is_favorable(self) -> bool:
        r"""**Description:** Whether the polytope is favorable (stored ``fav_N``)."""
        return self.fav_N

    @property
    def covers_full_h11(self) -> bool:
        r"""
        **Description:**
        Whether the stored geometry spans **all** of :math:`H^{1,1}(X)`.

        ``True`` iff the polytope is favorable (``fav_N``). For a **non-favorable**
        polytope ``h11(X) > h11(V)``: some prime toric divisors are reducible (a
        2-face divisor splits into ``g+1`` irreducible components on :math:`X`), so
        the stored κ/c₂ cover only the ``basis_dim = h11(V)`` toric classes and the
        remaining ``h11 - basis_dim`` non-toric classes are **not** represented. Full
        :math:`h^{1,1}(X)` support (unfolding the reducible divisors) is deferred —
        see :meth:`full_intersection_numbers`.
        """
        return bool(self.fav_N)

    @property
    def dataset(self) -> str:
        r"""
        **Description:**
        Deprecated alias for :attr:`mode`.

        Renamed because the inherited :attr:`stringforge.cy_io.CYDatabase.dataset` means the
        *sub-dataset* (``"toric"``), so one attribute name carried two disjoint vocabularies
        across a single API.  Use :attr:`mode` (``"frst"`` / ``"vex"``).
        """
        warnings.warn(
            "CYPhase.dataset is deprecated; use CYPhase.mode ('frst'/'vex'). "
            "The name collided with CYDatabase.dataset, which is the sub-dataset ('toric').",
            DeprecationWarning, stacklevel=2,
        )
        return self.mode

    @property
    def class_kind(self) -> str:
        r"""
        **Description:**
        The equivalence used to define this phase's class: ``"cy-class"`` (FRST,
        CYTools ``cy()``-equivalence) or ``"wall-class"`` (VEX, in-basis (κ, c₂)
        Wall-data dedup).
        """
        return "cy-class" if self.mode == "frst" else "wall-class"

    def _warn_if_incomplete_basis(self, what: str) -> None:
        """Warn once per instance if this non-favorable phase's in-basis geometry
        covers only the toric subspace (``basis_dim < h11``)."""
        if not self.fav_N and not self._warned_nonfav:
            self._warned_nonfav = True
            warnings.warn(
                f"CYPhase {self.phase_id}: non-favorable polytope (fav_N=False) — the "
                f"in-basis {what} spans only the {self.basis_dim} toric H^(1,1) classes "
                f"of h11={self.h11}; the {self.h11 - self.basis_dim} non-toric classes "
                f"(reducible 2-face divisors) are not stored. Full h11(X) support is "
                f"deferred — see CYPhase.covers_full_h11 / full_intersection_numbers().",
                UserWarning,
                stacklevel=3,
            )

    def full_intersection_numbers(self, format: str = "coo") -> np.ndarray:
        r"""
        **Description:**
        Triple intersection numbers over the **full** :math:`H^{1,1}(X)` basis
        (dimension ``h11``).

        For **favorable** phases this equals :meth:`intersection_numbers` with
        ``in_basis=True``. For **non-favorable** phases it requires unfolding the
        reducible 2-face divisors into their ``h11(X)+4`` irreducible components,
        which is **not yet implemented** (deferred; the stored data covers only the
        ``basis_dim`` toric classes).

        Args:
            format (str): ``"coo"`` or ``"dense"`` (favorable case only).

        Returns:
            np.ndarray: κ over the full ``h11``-dimensional basis (favorable only).

        Raises:
            NotImplementedError: For a non-favorable phase (:attr:`covers_full_h11`
                is ``False``).
        """
        if not self.covers_full_h11:
            raise NotImplementedError(
                f"CYPhase {self.phase_id}: full h11(X)={self.h11} intersection numbers "
                f"for a non-favorable polytope require unfolding reducible 2-face "
                f"divisors (h11(X) > h11(V)={self.basis_dim}); this is deferred. Use "
                f"intersection_numbers(in_basis=True) for the {self.basis_dim}-dim toric part."
            )
        return self.intersection_numbers(in_basis=True, format=format)

    def full_second_chern_class(self) -> np.ndarray:
        r"""
        **Description:**
        Second Chern class over the **full** :math:`H^{1,1}(X)` basis. Favorable:
        equals :meth:`second_chern_class` with ``in_basis=True``. Non-favorable:
        **deferred** (needs 2-face divisor unfolding).

        Returns:
            np.ndarray: c₂ over the full ``h11``-dimensional basis (favorable only).

        Raises:
            NotImplementedError: For a non-favorable phase.
        """
        if not self.covers_full_h11:
            raise NotImplementedError(
                f"CYPhase {self.phase_id}: full h11(X) second Chern class for a "
                f"non-favorable polytope is deferred (needs 2-face divisor unfolding). "
                f"Use second_chern_class(in_basis=True) for the toric part."
            )
        return self.second_chern_class(in_basis=True)

    # -- CYTools fallback (lazy import INSIDE) -------------------------------- #
    def to_cytools(self) -> Any:
        r"""
        **Description:**
        Materialise and cache the CYTools object for this phase.

        **FRST** returns a ``cytools.CalabiYau``
        (``Polytope(vertices, deterministic_glsm_basis=True).triangulate(heights).cy()``).
        **VEX** returns a CYTools ``Fan``
        (``Polytope(...).vc().triangulate(heights)`` — a not-necessarily-fine star
        triangulation, with *no* ``.cy()``) and emits a ``UserWarning``: the
        ``CalabiYau``-only features (GV invariants, ``mori_cone(version="cap")``,
        ``kahler_cone(version="cup")``, :meth:`to_lcs_tree`) are unavailable for
        VEX and raise ``NotImplementedError``.  The stored fast-path attributes
        remain valid regardless.

        Returns:
            cytools.CalabiYau | cytools.vector_config.fan.Fan: The materialised
            object (cached in ``self._cy``).
        """
        if self._cy is None:
            cytools = _require_cytools()
            p = cytools.Polytope(self.vertices, deterministic_glsm_basis=True)
            if self.mode == "frst":
                self._cy = p.triangulate(heights=self.heights).cy()
            else:  # vex
                warnings.warn(
                    "CYPhase.to_cytools(): for a VEX phase this returns a CYTools "
                    "`Fan` (from `Polytope.vc().triangulate(heights)`), not a "
                    "`CalabiYau`.  CalabiYau-only features — GV invariants, "
                    "mori_cone(version='cap'), kahler_cone(version='cup') and "
                    "to_lcs_tree() — are unavailable for VEX and raise "
                    "NotImplementedError.  Stored fast-path attributes remain valid.",
                    UserWarning,
                    stacklevel=2,
                )
                self._cy = p.vc().triangulate(heights=self.heights)
        return self._cy

    def _require_calabiyau(self, feature: str) -> Any:
        """Return the CYTools ``CalabiYau`` (FRST); raise for VEX (a toric ``Fan``)."""
        if self.mode != "frst":
            raise NotImplementedError(
                f"{feature} requires a CYTools CalabiYau, available only for FRST "
                f"phases; this is a '{self.mode}' phase (a toric Fan).  Use the "
                f"stored fast-path or CYPhase.to_cytools() for Fan-level operations."
            )
        return self.to_cytools()

    def mori_cone(self, version: str = "toric", in_basis: bool = True) -> Any:
        r"""
        **Description:**
        The Mori cone of this phase, materialising CYTools on demand.

        Args:
            version (str): ``"toric"`` → the toric Mori cone
                (FRST: ``CalabiYau.toric_mori_cone``; VEX: ``Fan.mori_cone`` with
                ``pushed_down=True``).  ``"cap"`` → ``CalabiYau.mori_cone_cap``
                (the single-geometry combinatorial Mori-cone cap; **FRST only**).
            in_basis (bool): Whether to express the cone in the GLSM basis.

        Returns:
            cytools.cone.Cone: The requested Mori cone.

        Raises:
            NotImplementedError: For ``version="cap"`` on a VEX phase.
            ValueError: If ``version`` is not ``"toric"`` or ``"cap"``.
        """
        if version == "toric":
            cy = self.to_cytools()
            if self.mode == "frst":
                return cy.toric_mori_cone(in_basis=in_basis)
            return cy.mori_cone(pushed_down=True, in_basis=in_basis)  # VEX Fan
        if version == "cap":
            cy = self._require_calabiyau("mori_cone(version='cap')")
            return cy.mori_cone_cap(in_basis=in_basis)
        raise ValueError(f"version must be 'toric' or 'cap'; got {version!r}.")

    def kahler_cone(self, version: str = "toric", in_basis: bool = True) -> Any:
        r"""
        **Description:**
        The Kähler cone of this phase, materialising CYTools on demand.

        Args:
            version (str): ``"toric"`` → the toric Kähler cone
                (FRST: ``CalabiYau.toric_kahler_cone``, always in the GLSM basis;
                VEX: ``Fan.kahler_cone`` with ``pushed_down=True``).  ``"cup"`` →
                :math:`K_{\cup}`, the dual of the Mori-cone cap,
                ``Cone(cy.mori_cone_cap(in_basis=True).extremal_rays()).dual()``
                (definition from ``jaxvacua.cytools_interface.compute_K_cup``;
                a jaxvacua-internal derivation, **FRST only**).
            in_basis (bool): Whether to express the cone in the GLSM basis
                (ignored for FRST ``version="toric"``, which is inherently
                in-basis).

        Returns:
            cytools.cone.Cone: The requested Kähler cone.

        Raises:
            NotImplementedError: For ``version="cup"`` on a VEX phase.
            ValueError: If ``version`` is not ``"toric"`` or ``"cup"``.
        """
        if version == "toric":
            cy = self.to_cytools()
            if self.mode == "frst":
                return cy.toric_kahler_cone()
            return cy.kahler_cone(pushed_down=True, in_basis=in_basis)  # VEX Fan
        if version == "cup":
            cy = self._require_calabiyau("kahler_cone(version='cup')")
            cytools = _require_cytools()
            m_cap = cy.mori_cone_cap(in_basis=in_basis)
            return cytools.Cone(m_cap.extremal_rays()).dual()
        raise ValueError(f"version must be 'toric' or 'cup'; got {version!r}.")

    def gv_invariants(self, **kwargs: Any) -> Any:
        r"""
        **Description:**
        Gopakumar–Vafa invariants via ``cytools.CalabiYau.compute_gvs`` (**FRST
        only**; requires the external ``cygv`` backend).

        Args:
            **kwargs: Forwarded to ``CalabiYau.compute_gvs`` (e.g. ``max_deg``).

        Returns:
            The GV invariants as returned by CYTools.

        Raises:
            NotImplementedError: For a VEX phase (no ``CalabiYau``).
        """
        cy = self._require_calabiyau("gv_invariants()")
        return cy.compute_gvs(**kwargs)

    # -- bridges / checks ---------------------------------------------------- #
    def to_lcs_tree(self, **kwargs: Any) -> Any:
        r"""
        **Description:**
        Build a jaxvacua :class:`jaxvacua.lcs.lcs_tree` from this phase — the
        bridge to the flux-vacua / mirror (complex-structure) side (**FRST
        only**).

        The ``lcs_tree`` stores the **mirror** CY's Hodge numbers, so ``h11`` and
        ``h12`` are **exchanged**: ``lcs_tree.h11 = self.h12`` and
        ``lcs_tree.h12 = self.h11``.  This swap is performed canonically inside
        ``jaxvacua.cytools_interface.cytools_model_data_init`` (lines 168-169),
        which ``lcs_tree.from_cytools`` calls — matching the convention used by
        ``stringforge.lcs_database.LCSDatabase.load`` — so this method delegates
        to it rather than re-implementing the swap.  κ, c₂ and the cone data are
        the CY's own (Kähler-side) geometry, passed through unswapped.

        Args:
            **kwargs: Forwarded to ``lcs_tree.from_cytools`` (e.g.
                ``maximum_degree``, ``limit``, ``time_out``, ``compute_gws``).

        Returns:
            jaxvacua.lcs.lcs_tree: The LCS-tree for the mirror CY.

        Raises:
            NotImplementedError: For a VEX phase (no ``CalabiYau``).
        """
        cy = self._require_calabiyau("to_lcs_tree()")
        lcs_tree = _require_lcs_tree()
        return lcs_tree.from_cytools(cy, **kwargs)

    def verify(self) -> bool:
        r"""
        **Description:**
        Self-verify the stored geometry: rebuild the triangulation from the stored
        ``heights`` via CYTools and compare κ (and, for FRST, c₂) to the stored values.

        The comparison is made **out-of-basis** (basis-independent), then re-sliced with
        *this phase's stored* :attr:`glsm_basis`.  Because the build stores the
        **deterministic** GLSM basis, that agrees with CYTools' own ``in_basis=True``;
        slicing with the stored basis is used regardless so the check also holds for
        buckets built before 2026-07-29, whose basis came from the FRST generators'
        non-deterministic ``Polytope(vertices)`` call and differed from the deterministic
        one for ~31% of h11=10 polytopes.

        For VEX only κ is checked (c₂ is stored verbatim from the fan); calling this
        materialises the ``Fan`` and therefore emits the VEX ``UserWarning``.

        Returns:
            bool: ``True`` iff the recompute matches the stored geometry.
        """
        cy = self.to_cytools()
        basis_src = [int(b) + 1 for b in self.glsm_basis]        # stored is 0-indexed
        if self.mode == "frst":
            coo_src = cy.intersection_numbers(in_basis=False, format="coo").tolist()
            c2_src = [int(x) for x in cy.second_chern_class(in_basis=False).tolist()]
        else:
            # VEX: a Fan, pushed down to the prime toric divisors. Fan.intersection_numbers has
            # no ``format`` argument -- it returns a {index-tuple: value} dict -- so rebuild the
            # COO exactly as the generator did (run_frst_class.py:266-268).
            d_int = cy.intersection_numbers(pushed_down=True, in_basis=False, as_np_array=False)
            keys = np.array(list(d_int.keys()))
            vals = np.array(list(d_int.values()))
            coo_src = np.rint(np.append(keys.T, [vals], axis=0).T).astype(int).tolist()
            c2_src = [int(x) for x in np.rint(np.array(cy.c2())).astype(int).tolist()]
        norm = nz.normalize_geometry(self.mode, coo_src, c2_src, basis_src)

        # (a) basis-independent: the normalized out-of-basis tensor itself
        kappa_ok = (nz._canonicalize_coo(norm["coo"])
                    == nz._canonicalize_coo(np.asarray(self._coo).tolist()))
        c2_ok = [int(x) for x in norm["c2"]] == [int(x) for x in self._c2]
        # (b) the in-basis slice, taken with the SAME (stored) basis on both sides
        r_ib, r_c2 = nz.in_basis_from_stored(norm["coo"], norm["c2"], self.glsm_basis)
        mine_ib = self.intersection_numbers(in_basis=True, format="coo").tolist()
        ib_ok = (sorted(tuple(int(x) for x in row) for row in r_ib)
                 == sorted(tuple(int(x) for x in row) for row in mine_ib)
                 and np.array_equal(np.asarray(r_c2, dtype=int),
                                    self.second_chern_class(in_basis=True)))
        if self.mode != "frst":
            c2_ok = True                       # VEX c₂ stored verbatim from the fan
        return bool(kappa_ok and c2_ok and ib_ok)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(mode={self.mode!r}, h11={self.h11}, h12={self.h12}, "
            f"ks_id={self.ks_id}, triang_id={self.triang_id}, fav_N={self.fav_N}, "
            f"materialised={self._cy is not None})"
        )


# --------------------------------------------------------------------------- #
# CICYPhase
# --------------------------------------------------------------------------- #
class CICYPhase(CYPhase):
    r"""
    **Description:**
    One Calabi–Yau threefold from the complete-intersection (CICY) list.

    A CICY threefold is a complete intersection of hypersurfaces in a product of projective
    spaces, specified by a configuration matrix; the list of 7,890 such threefolds is from
    Candelas–Dale–Lütken–Schimmrigk, *Nucl. Phys. B* **298** (1988) 493.  This class wraps the
    ``cicy`` sub-dataset's stored geometry as a :class:`CYPhase`.

    **Conventions.**  The ``cicy`` catalogue and geometry are stored in *mirror* convention —
    its ``h11`` column is :math:`h^{2,1}(X)` and its ``chi`` column is :math:`-\chi(X)`.  This
    class undoes both, so :attr:`h11`, :attr:`h12` and
    :attr:`~CYPhase.euler_characteristic` follow the same convention as
    :class:`ToricCYPhase`: ``h11`` is :math:`h^{1,1}(X)`, ``h12`` is :math:`h^{2,1}(X)`, and
    :math:`\chi = 2(h^{1,1} - h^{2,1})`.

    .. warning::

       **No basis identification is available for this sub-dataset.**  The configuration matrix
       was never ingested, ``basis_change`` is ``NULL`` in every row, and there is no GLSM or
       weight matrix, so nothing records *which* divisor class each :math:`\kappa` index refers
       to.  Consequently:

       - :math:`\kappa` and :math:`c_2` are usable **internally** (contractions, :math:`\chi`,
         Kähler-cone data) but are **not comparable** to another geometry's, not even another
         CICY's.
       - ``wall_hash`` is deliberately **not** exposed.  Its purpose is cross-geometry
         comparison in a *fixed, known* basis; a CICY ``wall_hash`` would be uncomparable and
         actively misleading.

    **Completeness of the stored basis.**  Only for the *Kähler-favourable* rows does the
    stored index set have :math:`\mathrm{len}(c_2) = h^{1,1}(X)`, i.e. actually span
    :math:`H^{1,1}(X)`.  Verified over the whole local build (7,406 rows), the partition is
    exact:

    .. list-table::
       :header-rows: 1
       :widths: 12 30 58

       * - Rows
         - Condition
         - Handling
       * - 4,511
         - ``len(c2) == h11(X)`` :math:`\Leftrightarrow` Kähler favourable
           :math:`\Leftrightarrow` catalogue ``has_gv``
         - ``basis_is_complete = True``
       * - 2,873
         - ``len(c2) < h11(X)``
         - ``basis_is_complete = False``; the geometry is a proper subspace
       * - 22
         - ``h11 == 0`` recorded — the degenerate *product* CICYs
         - **Rejected**: a CY threefold cannot have :math:`h^{1,1} = 0`, so these are
           placeholders rather than Hodge numbers

    Args:
        cicy_id (int): Row identifier in the CICY list.
        h11 (int): :math:`h^{1,1}(X)` — already un-swapped.
        h12 (int): :math:`h^{2,1}(X)` — already un-swapped.
        intnums_coo (ArrayLike): COO rows ``(i, j, k, value)`` in the stored index set.
        c2 (Sequence[int]): Second Chern class, same index set.
        favorable (bool | None): The list's ``favorable`` flag.
        kahler_favorable (bool | None): The list's ``Kahler favorable`` flag.  When given it
            sets ``basis_is_complete``; the two are the same property.
        a_matrix (ArrayLike | None): Stored ``a_matrix``.
        kahler_generators (ArrayLike | None): Stored Kähler-cone generators.
        mori_rays (ArrayLike | None): Stored Mori-cone rays.
        gv (dict | None): GV/GW invariants, when loaded.

    Raises:
        ValueError: If ``h11 <= 0`` (a degenerate product row), or if ``kahler_favorable`` is
            ``True`` but ``len(c2) != h11``, which would mean the stored data contradicts the
            flag.
    """

    def __init__(
        self,
        *,
        cicy_id: int,
        h11: int,
        h12: int,
        intnums_coo: Any,
        c2: Any,
        favorable: Optional[bool] = None,
        kahler_favorable: Optional[bool] = None,
        a_matrix: Optional[Any] = None,
        kahler_generators: Optional[Any] = None,
        mori_rays: Optional[Any] = None,
        gv: Optional[Any] = None,
        **base: Any,
    ) -> None:
        if int(h11) <= 0:
            raise ValueError(
                f"cicy_id={cicy_id} has h11={h11} recorded, which is not a valid Calabi-Yau "
                f"threefold Hodge number. The CICY list's 22 degenerate 'product' entries "
                f"store h11 = h12 = 0 as a placeholder; they are not supported."
            )
        base.setdefault("basis_is_complete",
                        True if kahler_favorable is None else bool(kahler_favorable))
        super().__init__(construction="cicy", h11=h11, h12=h12,
                         intnums_coo=intnums_coo, c2=c2, **base)
        if kahler_favorable and self.basis_rank != self.h11:
            raise ValueError(
                f"cicy_id={cicy_id} is flagged Kahler favorable, which requires "
                f"len(c2) == h11(X), but len(c2)={self.basis_rank} and h11={self.h11}. "
                f"The stored geometry contradicts the flag."
            )
        self.cicy_id = int(cicy_id)
        self.favorable = None if favorable is None else bool(favorable)
        self.kahler_favorable = (None if kahler_favorable is None
                                 else bool(kahler_favorable))
        self.a_matrix = a_matrix
        self.kahler_generators = kahler_generators
        self.mori_rays = mori_rays
        self.gv = gv

    @classmethod
    def from_database(cls, db: Any, cicy_id: Optional[int] = None,  # type: ignore[override]
                      **kwargs: Any) -> "CICYPhase":
        r"""
        **Description:**
        Build a :class:`CICYPhase` from an ``LCSDatabase`` opened on the ``cicy`` sub-dataset.

        The database returns the row in *mirror* convention; this method un-swaps the Hodge
        labels so the resulting object follows the :class:`ToricCYPhase` convention.

        Args:
            db: An :class:`stringforge.lcs_database.LCSDatabase` with ``dataset="cicy"``.
                ``CICYDatabase`` is catalogue-only and cannot be used here.
            cicy_id (int): The CICY-list identifier.
            **kwargs: Forwarded to ``db.load`` (e.g. ``include_gv=True``).

        Returns:
            CICYPhase: The geometry, with ``h11`` :math:`= h^{1,1}(X)`.

        Raises:
            TypeError: If ``db`` cannot load cicy geometry.
            ValueError: If ``cicy_id`` is missing, or names a degenerate product row.
        """
        if cicy_id is None:
            cicy_id = kwargs.pop("cicy_id", None)
        if cicy_id is None:
            raise ValueError("CICYPhase.from_database requires cicy_id.")
        if getattr(db, "dataset", None) != "cicy" or not hasattr(db, "load"):
            raise TypeError(
                f"CICYPhase.from_database needs an LCSDatabase opened on the 'cicy' "
                f"sub-dataset; got {type(db).__name__} with "
                f"dataset={getattr(db, 'dataset', None)!r}. Note that CICYDatabase is "
                f"catalogue-only (no load method) -- use "
                f"LCSDatabase.from_local(..., dataset='cicy')."
            )
        tree = db.load(cicy_id=int(cicy_id), **kwargs)
        return cls.from_row(tree, cicy_id=int(cicy_id))

    @classmethod
    def from_row(cls, tree: Any, cicy_id: Optional[int] = None,  # type: ignore[override]
                 **_: Any) -> "CICYPhase":
        r"""
        **Description:**
        Build a :class:`CICYPhase` from an already-loaded ``lcs_tree``.

        ``lcs_tree`` is in mirror convention: its ``h11`` is :math:`h^{2,1}(X)` and its
        ``h12`` — the number of Kähler moduli, which dimensions :math:`\kappa` — is
        :math:`h^{1,1}(X)`.  Both are swapped back here.  When the row's ``extra_data``
        carries the un-swapped Hodge numbers they are cross-checked against the swap.
        """
        extra = dict(getattr(tree, "extra_data", None) or {})
        h11 = int(tree.h12)                       # mirror h12 == number of Kahler moduli
        h12 = int(tree.h11)
        # extra_data stores the un-swapped truth; verified equal to the swap on all 7,406
        # rows of the local build, so a mismatch means the row is malformed.
        for key, want in (("h11", h11), ("h12", h12)):
            if extra.get(key) is not None and int(extra[key]) != want:
                raise ValueError(
                    f"cicy row {cicy_id}: extra_data[{key!r}]={extra[key]} contradicts the "
                    f"mirror-swapped catalogue value {want}. Refusing to guess the "
                    f"convention."
                )
        return cls(
            cicy_id=int(cicy_id if cicy_id is not None
                        else extra.get("cicy_id", extra.get("CICY ID", -1))),
            h11=h11, h12=h12,
            intnums_coo=tree.intnums_coo, c2=tree.c2,
            favorable=extra.get("favorable"),
            kahler_favorable=extra.get("Kahler favorable"),
            a_matrix=getattr(tree, "a_matrix", None),
            kahler_generators=getattr(tree, "kahler_generators", None),
            mori_rays=getattr(tree, "mori_rays", None),
            gv=getattr(tree, "gv", None),
        )

    def __repr__(self) -> str:
        return (
            f"CICYPhase(cicy_id={self.cicy_id}, h11={self.h11}, h12={self.h12}, "
            f"basis_rank={self.basis_rank}, "
            f"kahler_favorable={self.kahler_favorable})"
        )


# ``cicy`` geometry is loaded through LCSDatabase, whose ``dataset`` is likewise "cicy".
CYPhase._DB_DISPATCH["cicy"] = "CICYPhase"

__all__: List[str] = ["CYPhase", "ToricCYPhase", "CICYPhase"]
