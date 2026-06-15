# Configuration file for the Sphinx documentation builder.
# StringForge — unified ecosystem documentation.

import os
import re
import sys
import logging
import warnings

from sphinx.ext import autosummary as _sphinx_autosummary

sys.path.insert(0, os.path.abspath("../.."))

warnings.filterwarnings("ignore", message=r".*Two neighbors is broken!")
logging.getLogger("cvxpy").setLevel(logging.CRITICAL)

_logging_log = logging.Logger._log


def _filtered_log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
    """Drop noisy optional-solver import messages during autodoc imports."""
    if "Encountered unexpected exception importing solver" in str(msg):
        return
    return _logging_log(self, level, msg, args, exc_info, extra, stack_info, stacklevel)


logging.Logger._log = _filtered_log

# -- Project information -----------------------------------------------------

project = "StringForge"
copyright = "2026, Andreas Schachner"
author = "Andreas Schachner"
version = "0.1.1"
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
    "myst_nb",
]

# -- Autosummary -------------------------------------------------------------

autosummary_generate = True
autodoc_default_options = {}
napoleon_use_rtype = False
napoleon_custom_sections = [("Returns", "params_style")]

# -- Intersphinx -------------------------------------------------------------

# Cross-package links go here.  Sibling-package mappings are added when the
# corresponding public documentation exposes a stable objects.inv inventory.
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3",                    None),
    "numpy":      ("https://numpy.org/doc/stable/",                None),
    "jax":        ("https://docs.jax.dev/en/latest/",              None),
    "jaxpolylog": ("https://jaxpolylog.readthedocs.io/en/latest/", None),
    # "jaxvacua":  ("https://jaxvacua.readthedocs.io/en/latest/",    None),
    # "cytools":   ("https://cy.tools/",                             None),  # add when stable RTD URL exists
    # "kahlerjax": ("https://kahlerjax.readthedocs.io/en/latest/",   None),  # add when public
    # "jaxiverse": ("https://jaxiverse.readthedocs.io/en/latest/",   None),  # add when public
}

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]
suppress_warnings = [
    # Some package docstrings are Markdown-flavoured and contain Google-style
    # argument sections that docutils parses too aggressively.  Keep the
    # rendered API pages, but do not fail strict documentation builds on those
    # parser diagnostics.
    "docutils",
]

# -- MyST / notebook settings ------------------------------------------------

nb_execution_mode = "off"
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/stringforge-figures.css"]
html_title = "StringForge"

html_theme_options = {
    "repository_url": "https://github.com/StringJAX/stringforge",
    "repository_branch": "main",
    "path_to_docs": "documentation/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "show_toc_level": 2,
}

toc_object_entries_show_parents = "hide"


def _strip_leading_description_marker(lines):
    """Remove the project's leading Markdown-style description marker."""
    lines = list(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        marker = "**Description:**"
        if stripped == marker:
            del lines[idx]
        elif stripped.startswith(marker):
            indent = line[: len(line) - len(line.lstrip())]
            lines[idx] = indent + stripped[len(marker):].lstrip()
        break
    return lines


def _extract_project_summary(doc, settings):
    """Extract autosummary text without parsing project docstring markers."""
    first_stanza = []
    content_started = False
    for line in _strip_leading_description_marker(doc):
        stripped = line.strip()
        if not content_started:
            if not stripped:
                continue
            content_started = True

        if (
            not stripped
            or stripped.startswith("```")
            or stripped in {"Args:", "Arguments:", "Returns:", "Raises:", "Example:", "Examples:"}
            or stripped.startswith(".. ")
        ):
            break
        first_stanza.append(stripped)

    if not first_stanza:
        return ""

    summary = " ".join(first_stanza).strip()
    return re.sub(r"::$", ".", summary)


_sphinx_autosummary.extract_summary = _extract_project_summary


def _normalise_project_docstring(app, what, name, obj, options, lines):
    """Adapt the project's Markdown-flavoured docstrings for Sphinx/ReST."""
    lines[:] = _strip_leading_description_marker(lines)

    normalised = []
    in_fence = False
    fence_indent = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = False
                normalised.append("")
            else:
                language = stripped[3:].strip() or "text"
                fence_indent = line[: len(line) - len(line.lstrip())]
                normalised.append(f"{fence_indent}.. code-block:: {language}")
                normalised.append("")
                in_fence = True
            continue

        if in_fence:
            normalised.append(f"{fence_indent}    {line}")
        else:
            normalised.append(line)

    lines[:] = normalised


def setup(app):
    app.connect("autodoc-process-docstring", _normalise_project_docstring)
