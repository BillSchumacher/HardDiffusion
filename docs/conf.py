# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import django

sys.path.insert(0, os.path.abspath(".."))
os.environ["DJANGO_SETTINGS_MODULE"] = "HardDiffusion.settings"
django.setup()

# -- Project information -----------------------------------------------------

project = "Hard Diffusion"
copyright = "2023, Bill Schumacher"
author = "Bill Schumacher"

# The full version, including alpha/beta/rc tags
release = "0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
    "myst_parser",
]

source_suffix = [".rst", ".md"]
autodoc_mock_imports = ["django"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to docs directory, that match files and
# directories to ignore when looking for docs files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "../**/migrations/*",
    "../manage.py",
    "../run_pylint.py",
    "../hard_diffusion/*",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "django": (
        "https://docs.djangoproject.com/en/4.1/",
        "http://docs.djangoproject.com/en/4.1/_objects/",
    ),
}
# -- Options for HTML output -------------------------------------------------
html_baseurl = "https://billschumacher.github.io/HardDiffusion/"
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"
html_theme_options = {
    "show_powered_by": True,
    "github_user": "BillSchumacher",
    "github_repo": "HardDiffusion",
    "github_banner": True,
    "show_related": False,
    "note_bg": "#FFF59C",
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autosummary_generate = True
myst_enable_extensions = ["colon_fence"]
