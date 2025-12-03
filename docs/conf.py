"""Sphinx configuration for the Repuragent documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project package is importable (adds repo root to sys.path)
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# -- Project information -----------------------------------------------------
project = "Repuragent"
author = "Repuragent Team"
copyright = f"{datetime.now():%Y}, {author}"
version = ""
release = ""

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# Treat Markdown files as sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Ensure static/template directories exist to avoid warnings
for folder_name in ("_static", "_templates"):
    path = DOCS_DIR / folder_name
    path.mkdir(exist_ok=True)
