# Configuration file for the Sphinx documentation builder.
# StringJAX — unified ecosystem documentation.

import os
import sys

# -- Project information -----------------------------------------------------

project = "StringJAX"
copyright = "2026, Andreas Schachner"
author = "Andreas Schachner"
version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

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
html_title = "StringJAX"

html_theme_options = {
    "repository_url": "https://github.com/AndreasSchachner/stringjax",
    "use_repository_button": True,
    "use_issues_button": True,
    "show_toc_level": 2,
}
