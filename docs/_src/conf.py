# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Haystack'
copyright = '2020, deepset'
author = 'deepset'

# The full version, including alpha/beta/rc tags



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 
		'IPython.sphinxext.ipython_console_highlighting', 
		'sphinx_rtd_theme', 
		'sphinx_tabs.tabs', 
		'sphinx_copybutton', 
		'nbsphinx', 
		'sphinx.ext.autosectionlabel',
                'sphinx_markdown_builder',
                'recommonmark']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build/*']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# This fixes weird spacing between bullet points in lists
html4_writer = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../static']

# -- Added configuration -----------------------------------------------------

# Define master file which is by default contents.rst
master_doc = "index"

# Logo for the title
html_logo="img/logo.png"
 
# Custom css
#html_context = {"css_files":["_static/custom.css"]}

# Additional layouts
html_additional_pages = {"index": "pages/index.html"}

#The file extensions of source files.
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

# -- Add autodocs for __init__() methods -------------------------------------

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
	# Custom css
	app.add_stylesheet("rtd_theme.css")
	app.connect("autodoc-skip-member", skip)

