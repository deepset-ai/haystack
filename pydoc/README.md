# :ledger: Looking for the docs?

You can find Haystack's documentation at https://docs.haystack.deepset.ai/.

# API Reference

We use Pydoc-Markdown to create Markdown files from the docstrings in our code. There is a Github Action that regenerates the API pages when tags are pushed or when manually triggered.

If you want to generate a new Markdown file for a new Haystack module, create a `.yml` file in `pydoc` which configures how Pydoc-Markdown will generate the page and commit it to main.

All the updates to docstrings get pushed to documentation when a new version is released.

### Configuration

Pydoc will read the configuration from a `.yml` file which is located under `pydoc`. Our files contain three main sections:

- **loaders**: A list of plugins that load API objects from python source files.
    - **type**: Loader for python source files (we use `haystack_pydoc_tools.loaders.CustomPythonLoader`)
    - **search_path**: Location of source files (relative to the config file)
    - **modules**: Modules which are used for generating the markdown file
    - **ignore_when_discovered**: Define which files should be ignored
- **processors**: A list of plugins that process API objects to modify their docstrings (e.g. to adapt them from a documentation format to Markdown or to remove items that should not be rendered into the documentation).
    - **type: filter**: Filter for specific modules
    - **documented_only**: Only documented API objects
    - **do_not_filter_modules**: Do not filter module objects
    - **skip_empty_modules**: Skip modules without content
    - **type: smart**: Smart processor for docstring processing
    - **type: crossref**: Cross-reference processor
- **renderer**: A plugin that produces the output files. We use a custom DocusaurusRenderer based on the Markdown renderer. It makes sure the Markdown files comply with Docusaurus requirements.
    - **type**: Define the renderer which you want to use. We are using the DocusaurusRenderer to make sure the files display properly in Docusaurus.
    - **description**: Add a short description of the page. It shows up right below the page title.
    - **id**: This is the Docusaurus page ID to make sure the doc lands in the right section of Haystack docs.
    - **title**: The title of the doc as it will appear on the website. Make sure you always add "API" at the end.
    - **markdown**:
        - **descriptive_class_title**: Remove the word "Object" from class titles.
        - **descriptive_module_title**: Adding the word "Module" before the module name.
        - **add_method_class_prefix**: Add the class name as a prefix to method names.
        - **add_member_class_prefix**: Add the class name as a prefix to member names.
        - **filename**: File name of the generated file, use underscores to separate each word.
