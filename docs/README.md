# :ledger: Looking for the docs?
You find them here here: 
#### https://haystack.deepset.ai/overview/intro


# :computer: How to update docs?

## Overview and Usage

We move the Overview and Usage docs to the [haystack-website](https://github.com/deepset-ai/haystack-website) repository. You will find the docs in the folder `docs`. Please make sure to only edit the newest version of the docs. We will release the docs together with the Haystack version. 
We are open for contibutions to our documentation. Please make sure to check our [Contribution Guidelines](https://github.com/deepset-ai/haystack/blob/master/CONTRIBUTING.md). You will find a step by step introduction to our docs [here](https://github.com/deepset-ai/haystack-website/tree/source).

## Tutorials

The Tutorials live in the folder `tutorials`. They are created as colab notebooks which can be used by users to explore new haystack features. To include tutorials into the docs website, markdowns files need to be generated from the notebook. This can be done by running the script `/docs/_src/tutorials/tutorials/convert_ipynb.py`. Just run `python convert_ipynb.py` and the script will update all existing notebooks. Furthermore, plaese make sure to update the `headers.py` file with headers for the new tutorials. These headers are important for the docs website workflow. After the markdown files are generated successfully, you can raise a PR. We will review it and as soons as the markdown file is merged to master, it can be added to our website. Please follow the steps described [here](https://github.com/deepset-ai/haystack-website/tree/source) under `Tutorial & Reference Docs`. 

## API Reference 

We use Pydoc-Markdown to create markdown files from the docstrings in our code.

### Update docstrings

Execute the following commands in `/haystack/docs/_src/api/api`:
```
pip install 'pydoc-markdown==3.11.0'
./generate_docstrings.sh
```

If you want to generate a new markdown file for a new haystack module, please create a `.yml` which is inline with the following configuration and a a new line to `generate_docstrings.sh` for the module. After you ran the generate_docstrings.sh again, there should be a new markdown file for the module. To include it into the docs website, push it to master and follow the steps described [here](https://github.com/deepset-ai/haystack-website/tree/source) under `Tutorial & Reference Docs`. 

### Configuration

Pydoc will read the configuration from a `.yml` file which is located in the current working directory. Our files contains three main sections:

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
- **renderer**: A plugin that produces the output files.
    - **type**: Define the renderer which you want to use. We are using the Markdown renderer as it can be configured in very detail.
    - **descriptive_class_title**: Remove the word "Object" from class titles. 
    - **descriptive_module_title**:  Adding the word “Module” before the module name
    - **add_method_class_prefix**: Add the class name as a prefix to method names
    - **add_member_class_prefix**: Add the class name as a prefix to member names
    - **filename**: file name of the generated file
