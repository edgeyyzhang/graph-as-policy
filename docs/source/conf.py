# Configuration file for the Sphinx documentation builder.
#
# Layout and conventions follow docs.skypilot.co (pydata-sphinx-theme,
# MyST markdown sources, sphinx-design components).
#
# Build:  ./docs/build.sh          (one-shot, warnings are errors)
#         ./docs/build.sh --watch  (live-reload via sphinx-autobuild)

import re
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "graph-as-policy"
copyright = "2026, the graph-as-policy authors"
author = "the graph-as-policy authors"

# Single-source the version from pyproject.toml (regex keeps this working on
# Python 3.10, which has no tomllib).
_pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
_version_match = re.search(
    r'^version\s*=\s*"([^"]+)"', _pyproject.read_text(), re.MULTILINE
)
version = _version_match.group(1) if _version_match else "0.0.0"
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "notfound.extension",
    "sphinx.ext.intersphinx",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST -----------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "linkify",
]
# Generate #anchors for h1-h3 so pages can deep-link into each other.
myst_heading_anchors = 3
# Shorthand external link schemes (SkyPilot convention, adapted):
#   <gh-engine:path> -> engine repo file, <gh-skills:path> -> registry repo file
myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "gh-engine": {
        "url": "https://github.com/graph-robots/graph-as-policy/tree/main/{{path}}",
        "title": "{{path}}",
    },
    "gh-skills": {
        "url": "https://github.com/graph-robots/open-robot-skills/tree/main/{{path}}",
        "title": "open-robot-skills/{{path}}",
    },
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "graph-as-policy documentation"
html_short_title = "GaP"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"

html_theme_options = {
    # Site-wide banner: GaP is pre-1.0 and in public beta. Shown on every page.
    "announcement": (
        "🧪 <strong>GaP Beta Code release (1 July 2026)</strong> — under active "
        "development and now in beta testing. Send comments and suggestions to "
        '<a href="mailto:kych@berkeley.edu">kych@berkeley.edu</a>; an updated '
        "version is planned by 1 Aug 2026."
    ),
    "logo": {
        "text": "GaP",
        "alt_text": "GaP — graph-as-policy",
    },
    "show_toc_level": 2,
    "navigation_depth": 4,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button-field"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "use_edit_page_button": True,
    "header_links_before_dropdown": 6,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/graph-robots/graph-as-policy",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_context = {
    "github_user": "graph-robots",
    "github_repo": "graph-as-policy",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Extension options -------------------------------------------------------

# Strip leading "$ " prompts when copying shell blocks.
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
