# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for the Vulcan run-id template engine
(:mod:`stringforge.vulcan.runid`).

Coverage:

* default template renders with all keys supplied;
* ``{name?}`` optional placeholder falls back to ``"na"`` when missing;
* missing non-optional placeholder raises ``KeyError`` naming the field;
* callable template branch;
* ``None`` template falls back to ``uuid.uuid4().hex``;
* every rendered slug satisfies the vault LABEL_SLUG_RE.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vacuavault.schema import LABEL_SLUG_RE
from stringforge.vulcan.runid import (
    DEFAULT_TEMPLATE,
    OPTIONAL_PLACEHOLDER_FALLBACK,
    render,
)
from stringforge.vulcan.schema import is_safe_run_id


FIXED_DATE = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_default_template_renders_with_all_keys():
    rid = render(
        DEFAULT_TEMPLATE,
        project="fluxscan",
        now=FIXED_DATE,
        h11=2, h12=2, ks_id=1234, triang_id=0, seed=42,
    )
    assert rid == "2026-06-01-120000_fluxscan_h11-2_h12-2_ks-1234_triang-0_seed-42"
    assert is_safe_run_id(rid)


def test_default_template_with_missing_optionals_falls_back():
    rid = render(
        DEFAULT_TEMPLATE,
        project="fluxscan",
        now=FIXED_DATE,
        h12=2, ks_id=1234,  # h11, triang_id, seed missing
    )
    fb = OPTIONAL_PLACEHOLDER_FALLBACK
    assert rid == f"2026-06-01-120000_fluxscan_h11-{fb}_h12-2_ks-1234_triang-{fb}_seed-{fb}"
    assert is_safe_run_id(rid)


def test_missing_required_placeholder_raises():
    template = "{date}_{required_field}"
    with pytest.raises(KeyError, match="required_field"):
        render(template, now=FIXED_DATE)


def test_optional_marker_strips_question_mark_in_output():
    template = "{date}_{maybe?}"
    rid = render(template, now=FIXED_DATE)
    assert "?" not in rid
    assert rid.endswith(f"_{OPTIONAL_PLACEHOLDER_FALLBACK}")


def test_callable_template():
    template = lambda **kw: f"custom-{kw.get('ks_id', 0)}"
    rid = render(template, ks_id=99)
    assert rid == "custom-99"


def test_callable_template_with_invalid_slug_raises():
    # Callable templates are NOT slugified -- the callable is expected
    # to produce a final slug-safe form.  An invalid return therefore
    # raises rather than being silently coerced.
    with pytest.raises(ValueError, match="LABEL_SLUG_RE"):
        render(lambda **kw: "has spaces!")
    with pytest.raises(ValueError):
        render(lambda **kw: "")


def test_none_template_yields_uuid_hex():
    rid = render(None)
    # uuid4().hex is a 32-character lowercase hex string -- always slug-safe.
    assert len(rid) == 32
    assert all(c in "0123456789abcdef" for c in rid)


def test_slug_coerces_non_alphanumeric():
    template = "{date}_{label}"
    rid = render(template, now=FIXED_DATE, label="paper:2306.06160")
    # Both ':' and '.' collapse to '-', so the rendered slug must
    # still satisfy LABEL_SLUG_RE.
    assert is_safe_run_id(rid)
    assert ":" not in rid and "." not in rid


def test_extra_kwargs_supplied_by_caller():
    template = "{date}_{variant}_h12-{h12?}"
    rid = render(template, now=FIXED_DATE, variant="isd-only", h12=2)
    assert rid == "2026-06-01-120000_isd-only_h12-2"


def test_template_rejects_attribute_access():
    with pytest.raises(ValueError, match="attribute or item access"):
        render("{date}_{geometry.h11}", now=FIXED_DATE)


def test_template_rejects_item_access():
    with pytest.raises(ValueError, match="attribute or item access"):
        render("{date}_{geom[0]}", now=FIXED_DATE)


def test_render_produces_label_slug_re_match():
    # Sweep a few representative input shapes.  All must satisfy
    # the vault label regex.
    samples = [
        dict(template=DEFAULT_TEMPLATE, project="fluxscan",
             h11=2, h12=2, ks_id=1, triang_id=0, seed=7),
        dict(template="{date}_{project}_solver-{solver}",
             project="vulcan", solver="agm"),
        dict(template="{date}_{label}", project="vulcan",
             label="paper:2407.99999"),
    ]
    for kw in samples:
        rid = render(now=FIXED_DATE, **kw)
        assert LABEL_SLUG_RE.match(rid), f"failed for {kw}: {rid!r}"
