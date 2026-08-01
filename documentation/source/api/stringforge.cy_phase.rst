stringforge.cy_phase
====================

.. currentmodule:: stringforge.cy_phase

.. automodule:: stringforge.cy_phase
   :no-members:

Overview
--------

Where :class:`stringforge.toric_db.ToricCYDatabase` is the *I/O* layer — query a catalogue,
``load`` a phase's geometry into a ``dict`` — ``CYPhase`` is the *object* layer: one instance is
one Calabi–Yau phase, serving the stored geometry immediately and materialising a full CYTools
object only when asked.

The class family is split along a principled boundary.  What every construction shares is
exactly the **Wall data** :math:`(h^{1,1}, h^{2,1}, \kappa, c_2)`, which by Wall's theorem
(*Invent. Math.* **1** (1966) 355, together with the torsion data) fixes the diffeomorphism type
of a simply connected six-manifold.  That, and what follows from it, is the base class.
Everything requiring a polytope lives on the toric subclass.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Class
     - Carries
   * - :class:`CYPhase`
     - ``construction``, ``h11``, ``h12``, :math:`\kappa` (COO), :math:`c_2`,
       ``euler_characteristic``, ``hodge_numbers``, ``to_dense()``, ``basis_rank``,
       ``basis_is_complete``.  No polytope, no CYTools, no ``in_basis`` notion.
   * - :class:`ToricCYPhase`
     - Adds ``mode``, ``ks_id``, ``triang_id``, ``heights``, ``vertices``, ``glsm_basis``,
       ``glsm_charge_matrix``, the favourability flags, ``wall_hash``, ``polytope_hash``, the
       out-of-basis/in-basis machinery, and the CYTools bridges (``to_cytools``, ``mori_cone``,
       ``kahler_cone``, ``gv_invariants``, ``to_lcs_tree``, ``verify``).
   * - :class:`CICYPhase`
     - Adds ``cicy_id``, the ``favorable`` / ``kahler_favorable`` flags, and the stored
       ``a_matrix``, ``kahler_generators`` and ``mori_rays``.  Deliberately **no**
       ``wall_hash`` and no ``in_basis`` notion -- see the warning below.

.. autosummary::
    :toctree: ../_autosummary
    :template: custom-class-template.rst

    CYPhase
    ToricCYPhase
    CICYPhase

Two-speed access
----------------

Stored fast path — numpy only, no CYTools import:

* :meth:`CYPhase.intersection_numbers`
* :meth:`CYPhase.second_chern_class`
* :meth:`CYPhase.to_dense`
* :attr:`CYPhase.euler_characteristic`, :attr:`CYPhase.hodge_numbers`

Lazy CYTools fallback — materialises one object, then caches it:

* :meth:`ToricCYPhase.to_cytools`
* :meth:`ToricCYPhase.mori_cone`, :meth:`ToricCYPhase.kahler_cone`
* :meth:`ToricCYPhase.gv_invariants`
* :meth:`ToricCYPhase.to_lcs_tree`
* :meth:`ToricCYPhase.verify` — recomputes from CYTools and compares against the stored values

Construction:

* :meth:`CYPhase.from_database` — dispatches to the subclass matching ``db.dataset``
* :meth:`CYPhase.from_row`

.. warning::

   **Non-favorable phases.**  When ``fav_N`` is ``False`` the stored basis spans a proper
   subspace of :math:`H^{1,1}(X)`, so ``basis_is_complete`` is ``False`` and
   :attr:`ToricCYPhase.covers_full_h11` is ``False``.  In-basis quantities then describe only
   the toric part and warn when accessed; :meth:`ToricCYPhase.full_intersection_numbers` and
   :meth:`ToricCYPhase.full_second_chern_class` raise
   :class:`NotImplementedError` rather than return a partial answer silently.

.. note::

   **VEX phases are fans, not hypersurfaces.**  For ``mode="vex"`` the triangulation need not be
   fine, so there is no ``cytools`` ``CalabiYau``: :meth:`ToricCYPhase.to_cytools` returns a
   CYTools ``Fan``.  The ``CalabiYau``-only features — GV invariants,
   ``mori_cone(version="cap")``, ``kahler_cone(version="cup")`` and ``to_lcs_tree`` — raise
   :class:`NotImplementedError` for VEX.


Complete-intersection CYs
-------------------------

:class:`CICYPhase` wraps a row of the ``cicy`` sub-dataset -- a complete intersection in a
product of projective spaces, from the list of Candelas, Dale, Lütken and Schimmrigk
(*Nucl. Phys. B* **298** (1988) 493).

Two conventions differ from ``toric`` and are corrected on load, so that a ``CICYPhase`` and a
``ToricCYPhase`` mean the same thing by ``h11``, ``h12`` and ``euler_characteristic``:

* the ``cicy`` catalogue's ``h11`` column is :math:`h^{2,1}(X)` (mirror convention), and
* its ``chi`` column is :math:`-\chi(X)`.

.. warning::

   **There is no basis identification for this sub-dataset.**  The configuration matrix was
   never ingested, ``basis_change`` is ``NULL`` in all 7,406 rows, and no GLSM or weight matrix
   is stored, so nothing records which divisor class each :math:`\kappa` index denotes.
   :math:`\kappa` and :math:`c_2` are therefore usable *internally* but are **not comparable**
   across geometries, and ``wall_hash`` is deliberately not exposed -- a CICY ``wall_hash``
   would be uncomparable and actively misleading.

The stored index set spans all of :math:`H^{1,1}(X)` only for the Kähler-favourable rows.
Verified across the whole local build, the partition is exact:

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Rows
     - Condition
     - Handling
   * - 4,511
     - ``len(c2) == h11(X)``, equivalently Kähler favourable, equivalently the catalogue's
       ``has_gv``
     - ``basis_is_complete = True``
   * - 2,873
     - ``len(c2) < h11(X)``
     - ``basis_is_complete = False``; the stored geometry is a proper subspace
   * - 22
     - ``h11 == 0`` recorded -- the degenerate *product* entries
     - **Rejected.**  A CY threefold cannot have :math:`h^{1,1} = 0`, so these are placeholders

Because the ``cicy`` geometry lives in the ``lcs_data`` split rather than a catalogue,
:meth:`CICYPhase.from_database` takes an :class:`stringforge.lcs_database.LCSDatabase` opened on
``cicy``; :class:`stringforge.cy_io.CICYDatabase` is catalogue-only and is rejected with a
message naming the replacement.
