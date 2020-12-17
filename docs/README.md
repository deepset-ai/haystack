# :ledger: Looking for the docs?
You find them here here: 
#### https://haystack.deepset.ai/docs/intromd


# :computer: How to update docs?

## Usage / Guides etc.

Will be automatically deployed with every commit to the master branch

## API Reference 

We use Pydoc-Markdown to create markdown files from the docstrings in our code.

### Update docstrings
Execute this in `/haystack/docs/_src/api/api`:
```
pip install 'pydoc-markdown>=3.0.0,<4.0.0'
pydoc-markdown pydoc-markdown-document-store.yml
pydoc-markdown pydoc-markdown-file-converters.yml
pydoc-markdown pydoc-markdown-preprocessor.yml
pydoc-markdown pydoc-markdown-reader.yml
pydoc-markdown pydoc-markdown-generator.yml
pydoc-markdown pydoc-markdown-retriever.yml
pydoc-markdown pydoc-markdown-pipelines.yml
```

(Or run one of the commands above to update the docstrings only for a single module)

### Configuration

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
