"""Sphinx configuration for the torch-dae documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "torch-dae: an AI skill-based framework for Audio Embedding Models"
author = "Stefano Giacomelli"
copyright = "2026, Stefano Giacomelli"
version = "0.1.0"
release = version

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = False
autodoc_typehints = "none"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
intersphinx_mapping: dict[str, tuple[str, str | None]] = {}
myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "furo"
html_title = project
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_static_path = ["_static"]
html_css_files = ["custom.css"]
