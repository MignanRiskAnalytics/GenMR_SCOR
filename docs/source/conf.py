# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'GenMR'
copyright = '2025, Arnaud Mignan'
author = 'Arnaud Mignan'
release = '1.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

nb_execution_mode = 'off'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

source_suffix = {
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

bibtex_bibfiles = ['refs.bib']
bibtex_reference_style = 'author_year'


import os
import sys
sys.path.insert(0, os.path.abspath('../..'))