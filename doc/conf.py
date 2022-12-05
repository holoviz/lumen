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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import json
import pathlib

from datetime import date

import param

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

import panel

from nbsite.shared_conf import setup
from nbsite.util import base_version  # noqa
from panel.io.convert import BOKEH_VERSION, PY_VERSION
from panel.io.resources import CDN_DIST

import lumen

project = 'Lumen'
copyright = f'2020-{date.today().year}, HoloViz Developers'
author = 'HoloViz Developers'
description = 'Declarative data intake, processing, visualization and dashboarding.'

# The full version, including alpha/beta/rc tags
version = release = base_version(lumen.__version__)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_nb',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'nbsite.pyodide'
]

PANEL_ROOT = pathlib.Path(panel.__file__).parent

pn_version = release = base_version(panel.__version__)
js_version = json.loads((PANEL_ROOT / 'package.json').read_text())['version']

if panel.__version__ != pn_version and (PANEL_ROOT / 'dist' / 'wheels').is_dir():
    py_version = panel.__version__.replace("-dirty", "")
    panel_req = f'./wheels/panel-{py_version}-py3-none-any.whl'
    bokeh_req = f'./wheels/bokeh-{BOKEH_VERSION}-py3-none-any.whl'
else:
    panel_req = f'{CDN_DIST}wheels/panel-{PY_VERSION}-py3-none-any.whl'
    bokeh_req = f'{CDN_DIST}wheels/bokeh-{BOKEH_VERSION}-py3-none-any.whl'

nbsite_pyodide_conf = {
    'requirements': [bokeh_req, panel_req, 'pandas', 'pyodide-http', 'holoviews>=1.15.1']
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_enable_extensions = [
    "colon_fence",
    "substitution",
]

html_css_files = [
    'custom.css',
    'dataframe.css'
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo_horizontal.png"
html_favicon = "_static/favicon.ico"

html_theme_options = {
    "github_url": "https://github.com/holoviz/lumen",
    "icon_links": [
        {
            "name": "Discourse",
            "url": "https://discourse.holoviz.org/c/lumen/14",
            "icon": "fab fa-discourse",
        },
    ],
    "navbar_end": ["navbar-icon-links"],
    "pygment_light_style": "material",
    "pygment_dark_style": "material"
}

html_context = {
    "default_mode": "light",
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Adding substitutions to the documentation
myst_substitutions = {
  "version": version,
}
