---
license: cc-by-4.0
pretty_name: CY Vacua Vault
language:
  - en
tags:
  - string-theory
  - flux-compactification
  - calabi-yau
  - moduli-stabilisation
  - physics
  - mathematical-physics
  - numerical-data
task_categories:
  - tabular-classification
  - tabular-regression
size_categories:
  - 10K<n<100M
configs:
  - config_name: default
    data_files:
      - split: curated
        path: "*/*/*/*.parquet"
      - split: community
        path: "*/*/*/community/*.parquet"
---

# CY Vacua Vault

A community-curated dataset of **flux vacuum solutions** for Type IIB string theory
compactifications on Calabi–Yau threefolds. Solutions are stored per-model and
cross-referenced against the geometry catalogue
[`aschachner/cy-database`](https://huggingface.co/datasets/aschachner/cy-database).

**Scope.** This repository stores **flux-sector vacua** — solutions of the
complex-structure + axio-dilaton flux equations (flux vectors, stabilised
moduli, tadpole values, classification flags, metadata). It does **not**
store the underlying geometries; those live in the geometry repo above.

**KKLT / de Sitter constructions are out of scope here.** Vacua that
additionally stabilise the *Kähler* moduli — KKLT-type non-perturbative
superpotentials, de Sitter uplifts — are maintained in a **separate
dedicated repository**. This vault is limited to the flux
(complex-structure + dilaton) sector; the overall Calabi–Yau volume is left
unfixed (see the mass-units note in §3.2).

Maintained for use with the [`stringforge`](https://github.com/aschachner/stringforge)
package (database/vault I/O) together with
[`jaxvacua`](https://github.com/aschachner/jaxvacua) (flux-vacuum physics), but
the parquet schema is library-agnostic — any tool that can read Apache Arrow can
consume the data.

---

## 1. Dataset summary

Each row in the vault is a single flux vacuum, i.e. a choice of integer
fluxes $f, h \in H^3(Y, \mathbb{Z})$ together with the stabilised complex
structure moduli $z \in \mathbb{C}^{h^{1,2}}$ and axio-dilaton $\tau \in
\mathbb{C}$ satisfying

$$
D_A W(z, \tau; f, h) \;=\; 0 \quad \text{(SUSY)} \qquad
\text{or} \qquad \partial_A V(z, \tau; f, h) \;=\; 0 \quad \text{(non-SUSY)},
$$

with the tadpole constraint $|f^T \Sigma h| \le N_{\max}$. Solutions are
grouped by the Calabi–Yau model they belong to. Curated (reviewed)
datasets live directly in the model directory; pending contributions are
staged in a `community/` sub-folder of the same model directory. Once a
maintainer curates a community submission it is promoted out of
`community/` into the model directory (see §6 for the lifecycle).

---

## 2. Repository layout

Curated datasets sit **directly** in each model directory. `community/`
is a submission queue for pending (unreviewed) contributions; once
reviewed, files move out of `community/` up to the model level.

**Homogeneity invariant.** Each parquet file in the vault contains
rows describing **exactly one model** — every row of a given file
shares the same `(ks_id, triang_id)` (TDF), `cicy_id` (CICY), or
`model_ID` (local jaxvacua models).
The directory layout (`<namespace>/h12_N/<identifier>/<label>.parquet`)
makes one-file-per-model the natural unit, and the catalogue,
server-side validation, and the `validate_parquet_file` auto-load
path all rely on this assumption. Mixed-model files are rejected
by `validate_parquet_file` with a clear top-level error.

```
vacua_vault/
├── catalog.parquet                    # Auto-generated index of all datasets
├── schema.json                        # Schema version + column spec
├── retractions.parquet                # Audit trail of retracted entries
├── README.md                          # This file
├── tdf/                               # Trilayer, double-favourable Kreuzer–Skarke models
│   ├── h12_2/
│   │   ├── ks_00029_tri_0000/
│   │   │   ├── SUSY_Nmax34.parquet                # curated, lives at model level
│   │   │   ├── nonSUSY_ISD-_Nmax200.parquet       # curated
│   │   │   ├── community/                          # pending submissions queue
│   │   │   │   ├── alice-hf_racetrack_W0_small.parquet
│   │   │   │   └── bob-hf_nonSUSY_critpts.parquet
│   │   │   └── _rejected/                          # (optional, split-mode only)
│   │   │       └── alice-hf_racetrack_W0_small.rejected.parquet
│   │   └── ...
│   └── h12_3/
│       └── ...
├── cicy/                              # Complete Intersection CY models
│   └── cicy_7807/
│       ├── SUSY_Nmax20.parquet                    # curated
│       └── community/
│           └── ...
└── local/                            # Local jaxvacua models (addressed by h12 + model_ID,
    └── h12_2/                         #   not a cy-database id — see §2.1)
        └── model_1/
            ├── ISDplus_demo.parquet               # curated
            └── community/
                └── alice-hf_extra_ISDplus.parquet
```

### 2.1 Path semantics

| Segment | Meaning |
|---------|---------|
| `tdf` / `cicy` | Model rooted in the geometry repo (`aschachner/cy-database`). |
| `local` | Local **jaxvacua** model, addressed by its bundled-model index `(h12, model_ID)` rather than a `cy-database` id. |
| `h12_{h12}` | Hodge number $h^{1,2}$ — fixes the dimension of the complex structure sector. |
| `ks_{ks_id}_tri_{triang_id}` | KS polytope + triangulation index (TDF models). |
| `cicy_{cicy_id}` | CICY configuration matrix index. |
| `model_{model_ID}` | jaxvacua-internal model index (the `local/` namespace). |
| **Curated files at model level** | Reviewed, canonical datasets. Downloaded by default. |
| `community/` | Pending, unreviewed contributions. Filename prefixed with the contributor's HF username. |
| `_rejected/` | (Opt-in, split-mode only.) Split-off rows that failed row-level validation. Excluded from default queries. |

### 2.2 Filename conventions

Filenames are **free-form human-friendly labels**. Classification (SUSY
status, sampling method, $N_{\max}$, post-selection qualifiers, …) lives
in the catalogue and in parquet-level key-value metadata, not in the
filename.

- **Curated files**: `{label}[_v{n}].parquet` where `{label}` matches
  the slug regex `^[A-Za-z0-9][A-Za-z0-9_-]*$`.
  e.g. `SUSY_Nmax34.parquet`, `racetrack_small_W0.parquet`,
  `nonSUSY_ISD-_Nmax200.parquet`.
- **Community files** (inside `community/`): `{hf_username}_{label}[_v{n}].parquet`
  e.g. `community/alice-hf_racetrack_candidates.parquet`.
- **Rejected files** (inside `_rejected/`, split-mode):
  `{original_filename}.rejected.parquet`.
- **Versioned files**: `_v{n}` is **auto-appended** by
  `push_vacua_to_hub(if_exists="new_version")` on collisions — contributors
  don't write it manually.

**Reserved names** (CI rejects them): `catalog.parquet`,
`retractions.parquet`, `schema.json`, `README.md`, and any filename
starting with `_`.

The `{hf_username}` prefix on community files avoids collisions between
independent contributions. When a community submission is curated, the
file is moved up to the model directory and the `{hf_username}_` prefix
is stripped; the original contributor is recorded in the catalogue as
`original_contributor` so attribution survives the rename.

**Collision policy** (`push_vacua_to_hub(if_exists=...)`):

| `if_exists=` | Behaviour |
|---|---|
| `"error"` (default) | Refuse upload; pick a different label or use `"new_version"`. |
| `"append"` | Load existing + concatenate + dedupe + rewrite (single-writer only). |
| `"new_version"` | Auto-suffix with `_v2`, `_v3`, ... — always safe. |

---

## 3. Schema

Every parquet file has the same column layout (see `schema.json` for the
machine-readable spec).

### 3.1 Required columns

| Column | Type | Description |
|---|---|---|
| `flux` | `list<int>` (length `2·(h12+1)`) | Integer flux vector $(f, h)$ stacked. |
| `moduli_re` | `list<float>` (length `h12`) | Real part of stabilised $z$. |
| `moduli_im` | `list<float>` (length `h12`) | Imaginary part of stabilised $z$. |
| `tau_re` | `float` | Real part of $\tau$ (axion). |
| `tau_im` | `float` | Imaginary part of $\tau$ (dilaton, $>0$). |
| `label` | `string` | Human-readable dataset label. |
| `committed_by` | `string` | Contributor identifier (ORCID preferred, HF username accepted). |
| `commit_date` | `date` | Date of commit (ISO 8601). |

### 3.2 Optional derived columns (recommended)

| Column | Type | Description |
|---|---|---|
| `W_re`, `W_im` | `float` | Superpotential $W(z, \tau)$. |
| `F_terms_re`, `F_terms_im` | `list<float>` | $D_A W$ components. |
| `N_flux` | `float` | Tadpole $|f^T \Sigma h|$. |
| `residual` | `float` | Newton residual of the solution. |
| `is_susy` | `bool` | Whether $|DW| < 10^{-6}$. |
| `is_minimum` | `bool` | Whether the Hessian is positive-definite. |
| `V` | `float` | Scalar potential at the solution. |
| `g_s` | `float` | String coupling $g_s = 1/\mathrm{Im}\,\tau$. |
| `mass2` | `list<float>` | Sorted real scalar mass-squared spectrum — eigenvalues of the canonically-normalised no-scale mass matrix (sign-carrying; negative entries are tachyonic). **FluxEFT normalisation — see note.** |
| `m_gravitino` | `float` | Gravitino mass $m_{3/2} = e^{K/2}\lvert W\rvert$. Same normalisation. |

**Mass units.** `mass2` and `m_gravitino` are in the **FluxEFT no-scale
normalisation, not** $M_{\mathrm{Pl}}$. At the flux-EFT level the overall
Calabi–Yau (Einstein-frame) volume $\mathcal{V}_E$ — a function of the
*Kähler* moduli — is not stabilised, so the FluxEFT Kähler potential omits
the $-2\log\mathcal{V}_E$ term and both quantities carry an unfixed overall
volume factor:
$m^2_{\mathrm{phys}} = \mathcal{V}_E^{-2}\,$`mass2` and
$m_{3/2,\mathrm{phys}} = \mathcal{V}_E^{-1}\,$`m_gravitino`. Only the
dimensionless ratio $m/m_{3/2}$ (in which $\mathcal{V}_E$ cancels) is
volume-independent and physical. These columns can be (re)computed from
`flux` + stabilised moduli with `db.complete_missing(df, model=...)`.

### 3.3 Metadata / provenance columns

| Column | Type | Description |
|---|---|---|
| `tags` | `string` (JSON array) | Free-form tags for filtering (e.g. `["SUSY", "ISD-", "Nmax200"]`). |
| `extra_data` | `string` (JSON object) | Per-solution metadata (e.g. `{"moduli_max": 100, "solver": "hybrid"}`). |
| `designated_id` | `int64` | Monotonic ID assigned on upload. |
| `model_ID` | `int64` (nullable) | jaxvacua-internal model index for `local/` models (catalogue-derived from the path); `NULL` for `tdf` / `cicy`. |
| `status` | `string` | Lifecycle state: `"pending"` (in `community/`), `"curated"` (at model level), `"rejected"` (split-mode failure), `"retracted"` (post-merge retraction). |
| `original_contributor` | `string` | HF username of the original contributor, preserved when a community submission is curated and the filename prefix is stripped. |
| `retracted` | `bool` | `True` if the entry has been retracted. |
| `retracted_reason` | `string` | Reason (required if `retracted=True`). |
| `retracted_at` | `timestamp` | When the retraction occurred. |

**Classification columns** (auto-populated from parquet-level
key-value metadata by `rebuild_catalog`; see §2.2):

| Column | Type | Description |
|---|---|---|
| `susy` | `string` | `"SUSY"`, `"nonSUSY"`, `"mixed"`, or `"unknown"`. |
| `method` | `string` | `"enumerate"`, `"sample"`, `"ISD+"`, `"ISD-"`, `"F"`, `"H"`, `"newton"`, `"hybrid"`, or a custom method string. |
| `nmax` | `int` (nullable) | Tadpole bound. `NULL` means unbounded or unknown. |
| `qualifiers` | `string` (JSON object) | Free-form dict of extra classifiers (e.g. `{"moduli_max": 100, "post_select": "racetrack"}`). |
| `classification_source` | `string` | `"parquet_metadata"` (contributor set it), `"filename_guess"` (best-effort fallback), or `"manual"` (maintainer override). |
| `version` | `int` | Version number from `_v{n}` suffix; `1` if absent. |
| `superseded_by` | `string` | Relative path of a compacted file that replaced this one; empty string if current. |

Schema evolution is tracked in `schema.json`. Breaking changes bump a major
version; additive changes bump a minor version.

---

## 4. Relationship to `aschachner/cy-database`

`cy-database` (the geometry repo) and `vacua_vault` (this repo) are
**complementary**:

| | `aschachner/cy-database` | `aschachner/vacua_vault` (this repo) |
|---|---|---|
| **Contents** | CY geometries: intersection numbers, $c_2$, Kähler cone data, GV/GW invariants, conifold data | Flux vacuum solutions: fluxes, stabilised moduli, tadpole, classification |
| **Update cadence** | Static / slow (append-only as new triangulations are added) | Fast (new solutions continuously produced) |
| **Size** | Could reach millions of models | Grows with sampling effort per model |
| **Key identifiers** | `(ks_id, triang_id)` for TDF, `cicy_id` for CICY | Same keys (plus `(h12, model_ID)` for local jaxvacua models) + `label` + `committed_by` |

A row in `vacua_vault` is only meaningful relative to a row in `cy-database`:
the flux vector lives in the integer lattice $H^3(Y, \mathbb{Z})$ defined by
the model's period data. The recommended workflow is:

1. Load the model via `stringforge.LCSDatabase(dataset="tdf").load_model(ks_id=29, triang_id=0)`.
2. Load solutions via `db.fetch_vacua_from_hub(ks_id=29, triang_id=0, ...)`.
3. Reconstruct the moduli physics using the loaded model + stored fluxes.

---

## 5. Using the dataset

### 5.1 With `stringforge` (recommended)

```python
from stringforge import LCSDatabase

db = LCSDatabase(dataset="tdf")

# Browse what's available
catalog = db.list_hub_vacua(h12=2)
print(catalog)

# Fetch curated solutions for a specific model (default)
df = db.fetch_vacua_from_hub(
    ks_id=29, triang_id=0,
    label="SUSY_Nmax34",       # optional filter
    include_community=False,    # default: only curated data
    cache=True,                 # cache locally
)

# Include pending community submissions (use with care)
df_all = db.fetch_vacua_from_hub(
    ks_id=29, triang_id=0,
    include_community=True,
)
print(df[["flux", "moduli_im", "tau_im", "N_flux", "is_susy"]].head())
```

### 5.2 With `datasets` or raw parquet

```python
from datasets import load_dataset

# Curated split: reviewed datasets only
ds_curated = load_dataset("aschachner/vacua_vault", split="curated")

# Community split: pending submissions
ds_pending = load_dataset("aschachner/vacua_vault", split="community")
```

```python
import pandas as pd
# Curated file — sits at model level
df = pd.read_parquet(
    "hf://datasets/aschachner/vacua_vault/tdf/h12_2/"
    "ks_00029_tri_0000/SUSY_Nmax34.parquet"
)

# Community file — inside community/ with HF-username prefix
df_pending = pd.read_parquet(
    "hf://datasets/aschachner/vacua_vault/tdf/h12_2/"
    "ks_00029_tri_0000/community/alice-hf_racetrack_W0_small.parquet"
)
```

### 5.3 Validating downloaded data

```python
from stringforge import LCSDatabase
db = LCSDatabase(dataset="tdf")
model = db.load_model(ks_id=29, triang_id=0)
report = db.validate_vacua(df, model=model, F_term_tol=1e-6)
```

---

## 6. How to contribute

We welcome contributions from the community! Contributions go through a
**pull-request workflow** on HuggingFace with automated validation on the PR
branch.

### 6.1 Prerequisites

- A HuggingFace account (free).
- A contributor identifier — ORCID preferred; HF username accepted.
- Your solutions must reference a geometry that is already in
  `aschachner/cy-database`. If your model is not yet catalogued, contribute it
  there first (see the geometry repo's contributor guide).

### 6.2 Workflow

1. **Prepare your dataset locally.** Use `jaxvacua` to generate and designate
   solutions into your local `vacua_vault/`:

   ```python
   # After running flux searches...
   db.designate_vacua(
       results_df,
       label="my_new_dataset",
       committed_by="Your Name (ORCID:0000-...)",
       model=model,
       tags=["SUSY", "Nmax200"],
   )
   ```

2. **Push as a pull request:**

   ```python
   db.push_vacua_to_hub(
       results_df,
       label="my_new_dataset",
       committed_by="Your Name (ORCID:0000-...)",
       model=model,
       tags=["SUSY", "Nmax200"],
       create_pr=True,                   # opens a PR instead of pushing to main
       partial_failure="strict",          # or "split" (see §7)
   )
   # Prints a PR URL — share with reviewers.
   ```

   This creates a branch on your HF fork, uploads
   `tdf/h12_{N}/ks_{X}_tri_{Y}/community/{your_hf_username}_my_new_dataset.parquet`,
   and opens a PR against `main`.

3. **Wait for CI validation.** The PR-branch CI (see §7 for full details)
   validates your submission and posts two comments:
   - A **validation report** (pass/fail per file, error counts, row indices).
   - A **catalogue diff** (*"this PR adds 1,247 rows across 1 dataset"*).

4. **Respond to feedback.** If CI fails, fix the issues locally and push
   again — CI re-runs automatically. If a reviewer requests changes,
   update your dataset and repush.

5. **Merge.** After CI is green and a maintainer approves, the PR is merged.
   Your dataset goes live under `community/`. The repository-wide catalogue
   is regenerated automatically on `main` (see §8).

6. **(Optional) Promotion to curated.** After review, a maintainer may
   promote your contribution out of `community/` up to the model directory.
   The `{hf_username}_` prefix is stripped but your attribution is preserved
   in the catalogue's `original_contributor` column.

### 6.3 What a good contribution looks like

- **Reproducible.** Include sufficient metadata in `extra_data` to reproduce
  the solution (solver, tolerance, sampling seed, pipeline parameters).
- **Deduplicated.** Run `db.load_local_vacua(model=model)` locally and
  deduplicate against your existing solutions before pushing.
- **Focused label.** One label per physical question
  (e.g. `racetrack_small_W0`, `nonSUSY_dS_candidates`), not a generic dump.
- **Tagged.** Tags should describe filters a user might query on (mode, Nmax,
  solver type, post-selection criteria).
- **Within tadpole bound.** All solutions must satisfy the D3-brane tadpole.
  Solutions that saturate it should be flagged in `extra_data`.

### 6.4 What is *not* a valid contribution

- Solutions for a geometry that is not in `aschachner/cy-database`
  (contribute the geometry first).
- Solutions that fail client-side validation (fix them first).
- Duplicate solutions already in the catalogue (check with
  `db.list_hub_vacua(...)` before pushing).
- Solutions whose provenance cannot be verified (no `committed_by`).

### 6.5 Retraction policy

If a solution in the vault is later found to be incorrect, the contributor or
maintainer can retract it:

```python
db.retract_designated(
    designated_id=12345,
    reason="F-term residual exceeds 1e-6 with corrected period data",
)
```

Retraction is **reversible**: the file stays on disk, the catalogue entry is
flagged `retracted=True`, and a row is appended to `retractions.parquet` at
the repo root. Default queries skip retracted entries.

Irreversible purges (file deletion) are reserved for maintainer action under
explicit policy (see `CONTRIBUTING.md` on the repo). They require both
`dry_run=False` and an explicit `confirm=True` safeguard.

---

## 7. Validation and failure handling

Every contribution goes through **two layers** of validation:

**Layer 1 — Client-side, before push.**
`push_vacua_to_hub()` runs locally:
- Schema version compatibility (matches `schema.json`).
- Flux vectors are integers of the expected length.
- Tadpole constraint $|f^T \Sigma h| \le N_{\max}$ (when $N_{\max}$ is known).
- F-terms for SUSY-flagged rows: $|DW| < $ tol.
- No duplicates against the local vault (and, if `check_remote=True`, the
  remote catalogue).
- `committed_by` is non-empty.

Failures raise `ValidationError` locally — nothing is uploaded.

**Layer 2 — Server-side, on the PR branch.**
A CI workflow runs `python -m stringforge.vacuavault validate` on every PR. It
repeats all Layer 1 checks (no trust in client-side validation),
reconstructs the model from identity fields, verifies F-terms to a tight
tolerance, and cross-checks against the existing catalogue. Output:
- A per-file summary comment on the PR.
- A machine-readable `validation_report.json` attached as a PR artifact.
- A blocking status check — merges are blocked until CI is green.

### 7.1 Failure modes and remediation

| Case | What happens | How to fix |
|---|---|---|
| **Whole-dataset failure** (schema error, malformed parquet, identity mismatch) | CI comments the root-cause error. PR merge blocked. | Fix locally, re-push. CI auto-reruns. |
| **Partial failure, `partial_failure="strict"` (default)** | CI lists the failing row indices with errors. PR merge blocked. | Fix or drop the bad rows, re-push. |
| **Partial failure, `partial_failure="split"`** | Valid rows go to the main file; failing rows move to `_rejected/{label}.rejected.parquet` with a per-row `error` column. Main file passes CI; rejected file is catalogued with `status="rejected"`. | No fix required — but review the rejected file and consider retracting/removing it in a follow-up PR. |
| **Duplicate against existing catalogue** | CI reports which existing entries your rows match. | Remove the duplicates locally or include only novel rows. |
| **Post-merge discovery of bad data** | Use `retract_designated()` (see §6.5). | File stays on disk (audit); catalogue entry flagged; `retractions.parquet` updated. |

### 7.2 Communication channels

| Channel | Purpose | Who reads |
|---|---|---|
| PR comment (CI bot) | Primary feedback, immediate, inline | Contributor + reviewers |
| `validation_report.json` (PR artifact) | Machine-readable diagnostics | Tooling, scripts |
| Catalogue `status` column | Per-dataset state: `"pending"`, `"curated"`, `"rejected"`, `"retracted"` | Downloaders (via query filters) |
| `retractions.parquet` (repo root) | Permanent audit trail of retractions | Maintainers, auditors |
| HF Community tab | Open-ended questions (not blocking) | Anyone |

---

## 8. Catalogue automation

The repository-wide catalogue (`catalog.parquet` at the repo root) is
**never hand-edited**. It is a derived index — regenerated automatically from
the parquet files in the repo.

### 8.1 PR preview (non-destructive)

During PR CI (§7), after validation passes, a *preview* of the post-merge
catalogue is computed and posted as a PR comment. Example:

*This PR adds 2 datasets (1,247 rows total) to
`tdf/h12_2/ks_00029_tri_0000/community/` and proposes 0 retractions.*

Reviewers see the impact at a glance before merging.

### 8.2 Post-merge rebuild (automatic)

On every push to `main` (i.e. after a PR merge), a separate CI workflow
triggers and runs:

```
python -m stringforge.vacuavault rebuild_catalog
```

This scans all `**/*.parquet` files, extracts per-file metadata (path,
label, contributor, row count, schema version, checksum, status), and
commits a refreshed `catalog.parquet` under the `[catalog-bot]` author.
No human intervention is required.

### 8.3 Manual fallback

If the automated workflow fails, the maintainer can rebuild the catalogue
locally:

```bash
git clone https://huggingface.co/datasets/aschachner/vacua_vault
cd vacua_vault
python -m stringforge.vacuavault rebuild_catalog --repo-path .
git commit -am "rebuild catalog" && git push
```

Users browsing the catalogue use `db.list_hub_vacua(...)` which reads
`catalog.parquet` directly — they never need to know how it was produced.

---

## 9. Related datasets & code

- **Geometry catalogue:** [`aschachner/cy-database`](https://huggingface.co/datasets/aschachner/cy-database)
  — intersection numbers, $c_2$, Kähler cone, GV/GW invariants, conifold data.
- **Code (database/vault I/O):** [`aschachner/stringforge`](https://github.com/aschachner/stringforge)
  — `LCSDatabase`, the vault writer, and the server-side `vacuavault` validation /
  catalogue tooling.
- **Code (physics):** [`aschachner/jaxvacua`](https://github.com/aschachner/jaxvacua)
  — flux vacuum generation, period integration, and post-selection.

---

## 10. Citation

If you use this dataset in a publication, please cite **both** the dataset
and the paper(s) that produced the contribution you used. Dataset citation:

```bibtex
@misc{vacua_vault_2026,
  author       = {Schachner, A. and community contributors},
  title        = {{CY Vacua Vault: a community-curated database of flux
                  vacuum solutions for Calabi--Yau compactifications}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/aschachner/vacua_vault}},
}
```

For individual contributions, the `committed_by`, `label`, and optional
`extra_data.paper` fields identify the source work.

---

## 11. Licence

Content of this dataset is released under **CC-BY-4.0**. You are free to
use, share, and adapt the data with attribution to the contributor(s) (as
recorded in `committed_by`) and to this dataset.

Individual contributions may additionally reference a paper under a separate
licence; cite the paper where applicable.

---

## 12. Changelog

See `CHANGELOG.md` for a per-release summary of schema changes, new
curated datasets, and notable community contributions.

---

*Dataset maintainer: [@aschachner](https://huggingface.co/aschachner).
Issues and questions: open a discussion on this repository's
[Community](https://huggingface.co/datasets/aschachner/vacua_vault/discussions) tab.*
