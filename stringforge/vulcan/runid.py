# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Run-ID template engine for Vulcan.

A ``run_id`` is the per-write identifier carried on every row of a
Vulcan shard.  Templates are ordinary Python format strings with
``{name}`` placeholders; an additional ``{name?}`` form marks an
optional placeholder that falls back to ``"na"`` when the substitution
is not provided.

The rendered string must satisfy
:data:`stringforge.vacuavault.schema.LABEL_SLUG_RE`, so a Vulcan
run_id is admissible as a vault label after future promotion.  The
constraints are: the first character must be in ``[A-Za-z0-9]``;
subsequent characters must be in ``[A-Za-z0-9_-]``; and an optional
trailing ``_v\d+`` version suffix is permitted.  Substitution values
are coerced via :func:`_slugify` before insertion.  Note that
:func:`_slugify` strips leading and trailing ``-`` but does *not*
strip a leading ``_``, so a substitution rendering with a leading
``_`` will fail validation.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from string import Formatter
from typing import Any, Callable, Mapping, Optional, Union

from .schema import is_safe_run_id

# Per the plan: default template carries date, project, Hodge numbers,
# KS+triang IDs, and seed.  Optional placeholders are marked with `?`
# so they fall back to "na" when an upstream caller has nothing to put
# in them (e.g. a CICY model has no ks_id).
DEFAULT_TEMPLATE: str = (
    "{date}_{project}_h11-{h11?}_h12-{h12?}_"
    "ks-{ks_id?}_triang-{triang_id?}_seed-{seed?}"
)

#: Sentinel used to render an optional placeholder when no value is
#: available.  Chosen to be slug-safe and visually distinct.
OPTIONAL_PLACEHOLDER_FALLBACK: str = "na"

_OPTIONAL_SUFFIX_RE = re.compile(r"^(.*)\?$")

# Vulcan templates accept plain placeholders only -- attribute access
# (``{x.y}``) and item access (``{x[0]}``) lead to format errors that
# do not name the offending placeholder.  We reject them upfront so the
# error message points to the actual problem.
_FORBIDDEN_FIELD_CHARS_RE = re.compile(r"[.\[\]]")

# Slug coercion: collapse anything outside [A-Za-z0-9_-] into "-".
# Leading/trailing "-" are stripped so we never produce "-foo" or
# "foo--bar".
_SLUGIFY_RE = re.compile(r"[^A-Za-z0-9_\-]+")


def _slugify(value: Any) -> str:
    r"""
    Coerce ``value`` into a slug-safe string.  Returns
    :data:`OPTIONAL_PLACEHOLDER_FALLBACK` for ``None``.
    """
    if value is None:
        return OPTIONAL_PLACEHOLDER_FALLBACK
    text = str(value).strip()
    if not text:
        return OPTIONAL_PLACEHOLDER_FALLBACK
    text = _SLUGIFY_RE.sub("-", text)
    text = text.strip("-")
    return text or OPTIONAL_PLACEHOLDER_FALLBACK


def _utc_date_slug(now: Optional[datetime] = None) -> str:
    r"""
    Return ``YYYY-MM-DD-HHMMSS`` for the current UTC moment.
    The format is intentionally slug-safe -- no dots, colons, or
    timezone suffixes.
    """
    dt = now if now is not None else datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d-%H%M%S")


def _split_optional_keys(template: str) -> tuple[str, set[str]]:
    r"""
    **Description:**
    Convert ``{name?}`` markers into ``{name}`` and return the cleaned
    template plus the set of names that were marked optional.

    Reject placeholders that use attribute access (``{x.y}``) or item
    access (``{x[0]}``) -- those are technically valid Python format
    syntax but produce confusing errors when paired with the slugify
    pipeline.  Vulcan templates accept plain identifiers only.

    Args:
        template: A run-id template string.

    Returns:
        tuple[str, set[str]]: ``(cleaned_template, optional_names)``.

    Raises:
        ValueError: If the template contains an attribute- or
            item-access placeholder.
    """
    cleaned_parts: list[str] = []
    optional: set[str] = set()
    for literal, field_name, format_spec, conversion in Formatter().parse(template):
        cleaned_parts.append(literal)
        if field_name is None:
            continue
        match = _OPTIONAL_SUFFIX_RE.match(field_name)
        name = match.group(1) if match is not None else field_name
        if _FORBIDDEN_FIELD_CHARS_RE.search(name):
            raise ValueError(
                f"Vulcan run_id template does not support attribute or "
                f"item access in placeholders: {{{field_name}}}.  "
                f"Use a plain identifier and pass nested data via "
                f"extra_kwargs."
            )
        if match is not None:
            optional.add(name)
        # Rebuild the field spec.
        spec = ""
        if conversion:
            spec += f"!{conversion}"
        if format_spec:
            spec += f":{format_spec}"
        cleaned_parts.append("{" + name + spec + "}")
    return "".join(cleaned_parts), optional


def render(
    template: Optional[Union[str, Callable[..., str]]] = None,
    *,
    project: str = "",
    now: Optional[datetime] = None,
    **substitutions: Any,
) -> str:
    r"""
    **Description:**
    Render a Vulcan ``run_id`` from a template and a set of
    substitutions.

    The rendered identifier is guaranteed to satisfy
    :data:`stringforge.vacuavault.schema.LABEL_SLUG_RE` so it is
    admissible as a vault label after a future promotion step.  Non
    slug-safe substitution values are coerced via :func:`_slugify`
    before insertion; trailing whitespace and disallowed punctuation
    collapse into single dashes.

    Args:
        template:
            * ``str`` -- format string with ``{name}`` and/or
              ``{name?}`` placeholders.  Plain identifiers only;
              attribute access (``{x.y}``) and item access
              (``{x[0]}``) are rejected.
            * Callable -- invoked with ``**substitutions``; the
              returned string is used verbatim and must satisfy
              :data:`~stringforge.vacuavault.schema.LABEL_SLUG_RE`.
            * ``None`` -- fall back to :func:`uuid.uuid4`'s ``.hex``.
        project: Default value for ``{project}`` when not present in
            ``substitutions``.
        now: Override the wall-clock for tests.  Defaults to
            :func:`datetime.datetime.now` in UTC.
        **substitutions: Values to interpolate.  Standard keys include
            ``h11``, ``h12``, ``ks_id``, ``triang_id``, ``conifold_id``,
            ``cicy_id``, ``seed``, ``solver``.  Unknown keys are
            forwarded to the template engine.

    Returns:
        str: A slug-safe run identifier.

    Raises:
        KeyError: A required (non-``?``) placeholder is unsupplied.
        ValueError: The rendered string is not a valid run_id, or
            the template uses attribute / item access.
    """
    if template is None:
        return uuid.uuid4().hex

    if callable(template):
        # Callable templates are expected to produce the final form
        # themselves -- we validate but do not transform.  This is the
        # contract that lets a callable enforce a stricter naming
        # scheme than slugify would; the template-string path applies
        # slugify because positional substitutions are the only way
        # for plain users to guarantee a valid output.
        rendered = template(**substitutions)
        if not isinstance(rendered, str) or not is_safe_run_id(rendered):
            raise ValueError(
                f"callable run_id_template returned an invalid slug: "
                f"{rendered!r}.  Returned value must satisfy LABEL_SLUG_RE; "
                f"slugify is not applied to callable results."
            )
        return rendered

    cleaned_template, optional_names = _split_optional_keys(template)

    # Build the substitution map: always inject {date} and {project};
    # callers may override either.
    env: dict[str, Any] = {
        "date": _utc_date_slug(now=now),
        "project": project,
    }
    env.update(substitutions)

    # Slugify every substitution.  This is also what fills optional
    # placeholders with the fallback when their value is missing.
    slug_env = {k: _slugify(v) for k, v in env.items()}
    for opt in optional_names:
        slug_env.setdefault(opt, OPTIONAL_PLACEHOLDER_FALLBACK)

    try:
        rendered = cleaned_template.format(**slug_env)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(
            f"Vulcan run_id template requires placeholder {missing!r}; "
            f"pass it via extra_kwargs= or mark it optional as "
            f"{{{missing}?}}"
        ) from None

    if not is_safe_run_id(rendered):
        raise ValueError(
            f"rendered run_id is not a valid slug: {rendered!r}.  "
            f"LABEL_SLUG_RE requires the first character to be in "
            f"[A-Za-z0-9], subsequent characters in [A-Za-z0-9_-], "
            f"with an optional trailing _v\\d+ version suffix.  Note "
            f"that _slugify strips leading/trailing '-' but does NOT "
            f"strip a leading '_', so a substitution rendering with a "
            f"leading '_' will fail this check.  Check that no "
            f"template literals or substitutions introduce disallowed "
            f"characters."
        )

    return rendered
