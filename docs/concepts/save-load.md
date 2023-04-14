# Save and Load your Pipelines

Pipelines can be serialized to Python dictionaries, that can be then dumped to JSON or to any other suitable format, like YAML, TOML, HCL, etc. These pipelines can then be loaded back.

Here is an example of Pipeline saving and loading:

```python
from haystack.pipelines import Pipeline, save_pipelines, load_pipelines

pipe1 = Pipeline()
pipe2 = Pipeline()

# .. assemble the pipelines ...

# Save the pipelines
save_pipelines(
    pipelines={
        "pipe1": pipe1,
        "pipe2": pipe2,
    },
    path="my_pipelines.json",
    _writer=json.dumps
)

# Load the pipelines
new_pipelines = load_pipelines(
    path="my_pipelines.json",
    _reader=json.loads
)

assert new_pipelines["pipe1"] == pipe1
assert new_pipelines["pipe2"] == pipe2
```

Note how the save/load functions accept a `_writer`/`_reader` function: this choice frees us from committing strongly to a specific template language, and although a default will be set (be it YAML, TOML, HCL or anything else) the decision can be overridden by passing another explicit reader/writer function to the `save_pipelines`/`load_pipelines` functions.

This is how the resulting file will look like, assuming a JSON writer was chosen.

`my_pipeline.json`

```python
{
    "pipelines": {
        "pipe1": {
            # All the components that would be added with a
            # Pipeline.add_component() call
            "components": {
                "first_addition": {
                    "type": "AddValue",
                    "init_parameters": {
                        "add": 1,
                        "input": "value",
                        "output": "value"
                    },
                    "run_parameters": {
                        "add": 6
                    },
                },
                "double": {
                    "type": "Double",
                    "init_parameters": {
                        "input": "value",
                        "output": "value"
                    }
                },
                "second_addition": {
                    "type": "AddValue",
                    "init_parameters": {
                        "add": 1,
                        "input": "value",
                        "output": "value"
                    },
                },
                # This is how instances of the same component are reused
                "third_addition": {
                    "refer_to": "pipe1.first_addition"
                },
            },
            # All the components that would be made with a
            # Pipeline.connect() call
            "connections": [
                ("first_addition", "double"),
                ("double", "second_addition"),
                ("second_addition", "third_addition"),
            ],
            # All other Pipeline.__init__() parameters go here.
            "metadata": {"type": "test pipeline", "author": "me"},
            "max_loops_allowed": 100,
        },
        "pipe2": {
            "components": {
                "first_addition": {
                    # We can reference components from other pipelines too!
                    "refer_to": "pipe1.first_addition",
                    # Additional parameters for this specific
                    # component location only
                    "run_parameters": {
                        "add": 4
                    }
                },
                "double": {
                    "type": "Double",
                    "init_parameters": {
                        "input": "value",
                        "output": "value"
                    }
                },
                "second_addition": {
                    "refer_to": "pipe1.second_addition"
                },
            },
            "connections": [
                ("first_addition", "double"),
                ("double", "second_addition"),
            ],
            "metadata": {"type": "another test pipeline", "author": "you"},
            "max_loops_allowed": 100,
        },
    },
    # A list of "dependencies" for the application.
    # Used to ensure all external components are present when loading.
    "dependencies": ["my_custom_components_module"],
}
```
