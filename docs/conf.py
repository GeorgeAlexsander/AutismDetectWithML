# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project Imports ---------------------------------------------------------
import os
import sys

# -- Path Setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'AutismDetectWithML'
copyright = '2024, George Flores'
author = 'George Flores'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx_design',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',  # Adiciona links para o código-fonte
    'ansys_sphinx_theme.extension.autoapi',
    'ansys_sphinx_theme.extension.linkcode',
]

autoapi_dirs = ['../src']  # Specify the directories for autoapi to scan

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = "ansys_sphinx_theme"
html_theme_options = {
        "logo": {
        "image_light": "images/logo_light.png",
        "image_dark": "images/logo_dark_alt.png",
        "show_breadcrumbs": True,
    },
}


html_static_path = ['_static']

# GitHub context for documentation
html_context = {
    "github_user": "GeorgeAlexsander",
    "github_repo": "AutismDetectWithML",
    "github_version": "main",
}
