*******************************************************
# Haystack — Docstrings Generation
*******************************************************

Setup Pydoc-Markdown
============

Pydoc-Markdown is a tool and library to create Python API documentation in Markdown format based on lib2to3, allowing it to parse your Python code without executing it ([link](https://pydoc-markdown.readthedocs.io/en/latest/)).

Pydoc-Markdown can be installed from PyPI ([Get Started](https://pydoc-markdown.readthedocs.io/en/latest/docs/getting-started/))

``
$ pipx install 'pydoc-markdown>=3.0.0,<4.0.0'
$ pydoc-markdown --version
``

Configuration
============

Pydoc will read the configuration from a `.yml` file which is located in the current working directory. Our files contains three main sections:

- **loader**: A list of plugins that load API objects from python source files.
    - **type**: Loader for python source files
    - **search_path**: Location of source files 
    - **ignore_when_discovered**: Define which files should be ignored
- **processor**: A list of plugins that process API objects to modify their docstrings (e.g. to adapt them from a documentation format to Markdown or to remove items that should not be rendered into the documentation).
    - **ignore_when_discovered**: Define which API objects should be ignored
    - **documented_only**: Only documented API objects
    - **do_not_filter_modules**: Do not filter module objects
    - **skip_empty_modules**: Skip modules without content
- **renderer**: A plugin that produces the output files.
    - **type**: Define the renderer which you want to use. We are using the Markdown renderer as it can be configured in very detail.
    - **descriptive_class_title**: Remove the word "Object" from class titles. 
    - **filename**: file name of the generated file

Geneate Docstrings
============

Every .yml file will generate a new markdown file. Run one of the following commands to generate the needed output:

- **Document store**: `pydoc-markdown pydoc-markdown-document-store.yml`
- **File converters**: `pydoc-markdown pydoc-markdown-file-converters.yml`
- **Preprocessor**: `pydoc-markdown pydoc-markdown-preprocessor.yml`
- **Reder**: `pydoc-markdown pydoc-markdown-reader.yml`
- **Retriever**: `pydoc-markdown pydoc-markdown-retriever.yml`
