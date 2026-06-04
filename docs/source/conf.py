import os
import sys
import shutil

__location__ = os.path.dirname(__file__)

sys.path.insert(0, os.path.abspath("../../src"))

# ---- auto-generate API ----
try:
    from sphinx.ext import apidoc

    output_dir = os.path.join(__location__, "api")
    module_dir = os.path.abspath("../../src/qewton")

    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        pass

    apidoc.main([
        "-f",
        "-o", output_dir,
        module_dir
    ])

except Exception as e:
    print("apidoc failed:", e)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Qewton"
copyright = "2026, Nick Heilenkötter, Tom Freudenberg"
author = "Nick Heilenkötter, Tom Freudenberg"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon"
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Qewton"