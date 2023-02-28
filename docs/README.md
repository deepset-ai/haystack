# :ledger: Looking for the docs?

You can find Haystack's documentation at https://docs.haystack.deepset.ai/.

# :computer: How to update docs?

## Overview, Components, Pipeline Nodes, and Guides

You can find these docs on the Haystack Docs page: https://docs.haystack.deepset.ai/docs/get_started. If you want to contribute, and we welcome every contribution, do the following:
1. Make sure you're on the right version (check the version expanding list in the top left corner).
2. Use the "Suggest Edits" link you can find in the top right corner of every page.
3. Suggest a change right in the docs and click **Submit Suggested Edits**.
4. Optionally, leave us a comment and submit your change.

Once we take care of it, you'll get an email telling you the change's been merged, or not. If not, we'll give you the reason why.

Make sure to check our [Contribution Guidelines](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md).

## Tutorials

The Tutorials live in a separate repo: https://github.com/deepset-ai/haystack-tutorials. For instructions on how to contribute to tutorials, see [Contributing to Tutorials](https://github.com/deepset-ai/haystack-tutorials/blob/main/Contributing.md#contributing-to-haystack-tutorials).

## API Reference

We use Pydoc-Markdown to create Markdown files from the docstrings in our code. There is a Github Action that regenerates the API pages with each commit.

If you want to generate a new Markdown file for a new Haystack module, create a `.yml` file in `docs/src/api/api` which configures how Pydoc-Markdown will generate the page and commit it to main.

All the updates to doctrings get pushed to documentation when you commit to the main branch.

### Configuration

Pydoc will read the configuration from a `.yml` file which is located under `/haystack/docs/_src/api/pydoc`. Our files contain three main sections:

- **loader**: A list of plugins that load API objects from python source files.
    - **type**: Loader for python source files
    - **search_path**: Location of source files
    - **modules**: Module which are used for generating the markdown file
    - **ignore_when_discovered**: Define which files should be ignored
- **processor**: A list of plugins that process API objects to modify their docstrings (e.g. to adapt them from a documentation format to Markdown or to remove items that should not be rendered into the documentation).
    - **type: filter**: Filter for specific modules
    - **documented_only**: Only documented API objects
    - **do_not_filter_modules**: Do not filter module objects
    - **skip_empty_modules**: Skip modules without content
- **renderer**: A plugin that produces the output files. We use a custom ReadmeRenderer based on the Markdown renderer. It makes sure the Markdown files comply with ReadMe requirements.
    - **type**: Define the renderer which you want to use. We are using the ReadmeRenderer to make sure the files display properly in ReadMe.
    - **excerpt**: Add a short description of the page. It shows up right below the page title.
    - **category**: This is the ReadMe category ID to make sure the doc lands in the right section of Haystack docs.
    - **title**: The title of the doc as it will appear on the website. Make sure you always add "API" at the end.
    - **slug**: The page slug, each word should be separated with a dash.
    - **order**: Pages are ordered alphabetically. This defines where in the TOC the page lands.
    - markdown:
        - **descriptive_class_title**: Remove the word "Object" from class titles.
        - **descriptive_module_title**: Adding the word “Module” before the module name.
        - **add_method_class_prefix**: Add the class name as a prefix to method names.
        - **add_member_class_prefix**: Add the class name as a prefix to member names.
        - **filename**: File name of the generated file, use underscores to separate each word.
