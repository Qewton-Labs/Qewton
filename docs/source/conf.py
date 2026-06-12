import os
import sys
import shutil
from pathlib import Path
from m2r2 import convert

__location__ = os.path.dirname(__file__)

sys.path.insert(0, os.path.abspath("../../src"))


def process_signature(app, what, name, obj, options, signature, return_annotation):
    if signature:
        signature = signature.replace("typing.Annotated", "")
    if return_annotation:
        return_annotation = return_annotation.replace("typing.Annotated", "")

    return signature, return_annotation


def insert_readme_as_module_doc(app, what, name, obj, options, lines):
    """
    Replace module docstring with README.md if it exists.
    """
    if what != "module":
        return

    try:
        module_path = Path(obj.__file__).parent
        readme = module_path / "README.md"
        if Path(obj.__file__).stem != "__init__":
            return

        if readme.exists():
            lines.clear()  # remove __init__.py docstring
            readme_text = readme.read_text(encoding="utf-8")
            rst_text = convert(readme_text)
            lines.extend(rst_text.splitlines())
            # lines += readme.read_text(encoding="utf-8").splitlines()

    except Exception:
        pass


def setup(app):
    app.connect("autodoc-process-docstring", insert_readme_as_module_doc)
    app.connect("autodoc-process-signature", process_signature)


# ---- auto-generate API ----
try:
    from sphinx.ext import apidoc

    output_dir = os.path.join(__location__, "api")
    module_dir = os.path.abspath("../../src/qewton")

    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError:
        pass

    apidoc.main(["-f", "-o", output_dir, module_dir])

    # Put read me in the front:
    for rst_file in Path(output_dir).glob("*.rst"):
        text = rst_file.read_text(encoding="utf-8")

        marker = "Module contents\n---------------"
        if marker not in text:
            continue

        # Remove the "Module contents" title
        # text = text.replace("Module contents\n---------------\n\n", "")

        # Extract module contents section
        before, module_section = text.split(marker, 1)

        # Put it right after the title
        lines = before.splitlines()

        title_end = 2  # title + underline
        new_text = (
            "\n".join(lines[: title_end + 1])
            + "\n\n"
            + marker
            + module_section
            + "\n"
            + "\n".join(lines[title_end + 1 :])
        )

        rst_file.write_text(new_text, encoding="utf-8")

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
    "sphinx.ext.napoleon",
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
