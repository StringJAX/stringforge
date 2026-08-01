stringforge.toric_db
====================

.. currentmodule:: stringforge.toric_db

.. automodule:: stringforge.toric_db
   :no-members:

.. note::

   Access is currently **local only** — build or download the ``toric/`` tree and open it with
   :meth:`ToricCYDatabase.from_local`.  Lazy per-shard download from the Hub is not yet
   implemented.

Overview
--------

``ToricCYDatabase`` exposes the ``toric`` sub-dataset of ``cy-database``: Calabi–Yau
hypersurface phases obtained from the Kreuzer–Skarke list (arXiv:hep-th/0002240), together with
the polytope data they share.  Two kinds of phase live side by side, distinguished by ``mode``:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - ``mode``
     - What one row is
   * - ``frst``
     - A distinct CYTools ``CalabiYau`` class of a **fine**, regular, star triangulation — the
       standard notion of a toric CY hypersurface phase.  Classes are deduplicated by CYTools
       ``cy()``-equivalence, following arXiv:2310.06820.  Available for
       :math:`h^{1,1} = 1 \dots 12`.
   * - ``vex``
     - A distinct **Wall class** — equal in-basis :math:`(\kappa, c_2)` — of a *not necessarily
       fine* star triangulation of the polytope's vector configuration, following
       arXiv:2512.14817.  Available for :math:`h^{1,1} = 2 \dots 7`.

.. important::

   The two modes are built over the **same polytopes**, which is why the polytope layer is
   shared and stored once.  Every VEX polytope is also an FRST polytope, so
   ``vex`` :math:`\subseteq` ``frst`` **as sets of polytopes**.

   This is *not* a containment on phases, and the class counts are not comparable as one.  The
   two modes apply **different equivalences** to **different triangulation families**:
   ``cy()``-equivalence on fine star triangulations versus :math:`(\kappa, c_2)`-equivalence on
   not-necessarily-fine ones.  Because the non-fine family is much larger, a polytope routinely
   has more VEX classes than FRST classes — at :math:`h^{1,1} = 4` the totals are 1,774 FRST
   against 2,536 VEX.

.. note::

   Neither notion is yet a *diffeomorphism* class: both are equivalences in a fixed GLSM basis.
   ``wall_hash`` is a **necessary** diffeomorphism pre-filter (Wall's theorem), not a sufficient
   one; the full :math:`GL(h^{1,1}, \mathbb{Z})`-canonical identification is not implemented.

Unlike the other sub-datasets, ``toric`` has **no monolithic** ``catalog.parquet``: both the
phase catalogue and the geometry are sharded per :math:`h^{1,1}`, which is what makes billions
of rows tractable.  Point lookups stay :math:`O(1)` through a per-:math:`h^{1,1}`
``_ksid_index`` (``ks_id`` :math:`\to` ``(part, row0, n)``); attribute queries stream the
relevant shard with ``pyarrow`` filter pushdown.

.. warning::

   ``ks_id`` is the Kreuzer–Skarke emission order and is unique only **within** a given
   :math:`h^{1,1}`.  A phase key therefore needs all four of
   ``(mode, h11, ks_id, triang_id)``; :meth:`~stringforge.cy_phase.CYPhase.from_database`
   raises if the key it is given is not unique.

.. autosummary::
    :toctree: ../_autosummary
    :template: custom-class-template.rst

    ToricCYDatabase

Curated method index
--------------------

Discovery:

* :meth:`ToricCYDatabase.info`
* :meth:`ToricCYDatabase.query`
* :meth:`ToricCYDatabase.query_polytopes`

Point lookups:

* :meth:`ToricCYDatabase.load`
* :meth:`ToricCYDatabase.get_polytope`
* :meth:`ToricCYDatabase.from_local`

Deliberately unsupported
------------------------

The base class's monolithic-catalogue members cannot work on a sharded layout and are closed
with an explanatory :class:`NotImplementedError` rather than left to fail obscurely — chiefly
:meth:`ToricCYDatabase.query_conifolds` (the ``toric`` sub-dataset carries no conifold index;
see :class:`stringforge.kklt_database.KKLTDatabase` for that).

**Vacua-vault namespace.** There is no ``toric/`` namespace in the shared vacua vault, so
vacua found on a toric phase currently have no canonical remote location: a phase routed
through :meth:`~stringforge.cy_phase.ToricCYPhase.to_lcs_tree` does not carry its
``(h11, ks_id, triang_id, mode)`` identity into the resulting ``lcs_tree``, and
``VacuaWriter``'s remote-path resolution returns ``None`` rather than misfiling it under
``tdf/``.  Defining such a namespace is a vault data-format decision: unlike ``tdf/``, a toric
key needs :math:`h^{1,1}` (``ks_id`` is unique only within one :math:`h^{1,1}`) and ``mode``.
Until it is defined, treat vault persistence of toric vacua as unsupported.
