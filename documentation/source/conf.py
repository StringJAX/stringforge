# Configuration file for the Sphinx documentation builder.
# StringForge — unified ecosystem documentation.

import os
import sys

# -- Project information -----------------------------------------------------

project = "StringForge"
copyright = "2026, Andreas Schachner"
author = "Andreas Schachner"
version = "0.1.0"
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

# -- Intersphinx -------------------------------------------------------------

# Cross-package links go here.  Sibling-package mappings are added as each
# package's docs go public; until then those references will report missing
# at build time but will not block the build.
intersphinx_mapping = {
    "python":   ("https://docs.python.org/3", None),
    "numpy":    ("https://numpy.org/doc/stable/", None),
    "jax":      ("https://docs.jax.dev/en/latest/", None),
    "jaxvacua": ("https://jaxvacua.readthedocs.io/en/latest/", None),
    # "cytools":   ("https://cy.tools/",                             None),  # add when stable RTD URL exists
    # "kahlerjax": ("https://kahlerjax.readthedocs.io/en/latest/",    None),  # add when public
    # "jaxiverse": ("https://jaxiverse.readthedocs.io/en/latest/",    None),  # add when public
}

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]

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
html_title = "StringForge"

html_theme_options = {
    "repository_url": "https://github.com/AndreasSchachner/stringforge",
    "use_repository_button": True,
    "use_issues_button": True,
    "show_toc_level": 2,
}
