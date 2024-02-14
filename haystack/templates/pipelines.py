import re
from pathlib import Path
from typing import Dict, Any, Set, Optional

import yaml
from jinja2 import meta, TemplateSyntaxError
from jinja2.nativetypes import NativeEnvironment

from haystack import default_to_dict, Pipeline
from haystack.core.component import Component
from haystack.core.errors import PipelineValidationError


class PipelineTemplate:
    """
    The PipelineTemplate class enables the straightforward creation of flexible and configurable pipelines using
    Jinja2 templated YAML files. Specifically designed to simplify the setup of complex data processing pipelines for
    a range of NLP tasks—including question answering, retriever augmented generation (RAG), document indexing, among
    others - PipelineTemplate empowers users to dynamically generate pipeline configurations from templates and
    customize components as necessary. Its design philosophy centers on providing an accessible, yet powerful, tool
    for constructing pipelines that accommodate both common use cases and specialized requirements with ease.


    The class enables two primary use cases:

    1. Building a pipeline directly using all default components specified in a predefined or custom template.
    2. Customizing pipelines by overriding default components with custom component settings, integrating user-provided
    component instances, and adjusting component parameters conditionally.

    Examples of usage:

    - **Default Build**: Instantiating a pipeline with default settings for a "question answering" (qa) task.
      ```python
      from haystack.templates import PipelineTemplate

      # Create a pipeline with default components for a QA task
      pipe = PipelineTemplate("qa").build()
      print(pipe.run(data={"question": "What's the capital of Bosnia and Herzegovina? Be brief"}))
      ```

    - **Custom Component Settings**: Customizing a pipeline by overriding a component, such as integrating a
    streaming-capable generator for real-time feedback.
      ```python
      from haystack.components.generators import OpenAIGenerator
      from haystack.components.generators.utils import print_streaming_chunk
      from haystack.templates import PipelineTemplate

      # Customize the pipeline with a streaming-capable generator
      streaming_pipe = PipelineTemplate("qa").override("generator",
                                                               OpenAIGenerator(
                                                                   streaming_callback=print_streaming_chunk)).build()
      streaming_pipe.run(data={"question": "What's the capital of Germany? Tell me about it"})
      ```

    - **Customizing for Specific Tasks**: Building a pipeline for document indexing with specific components tailored
    to the task.
      ```python
      from haystack.components.embedders import SentenceTransformersDocumentEmbedder
      from haystack.templates import PipelineTemplate

      # Customize the pipeline for document indexing with specific components, include PDF file converter
      ptb = PipelineTemplate("indexing", template_params={"use_pdf_file_converter": True})
      ptb.override("embedder", SentenceTransformersDocumentEmbedder(progress_bar=True))
      pipe = ptb.build()

      result = pipe.run(data={
          "sources": ["some_text_file.txt", "another_pdf_file.pdf"]})
      print(result)
      ```

    The `PipelineTemplate` is designed to offer both ease of use for common pipeline configurations and the
    flexibility to customize and extend pipelines as required by advanced users and specific use cases.

    :param pipeline_template: The name of a predefined pipeline template or a custom Jinja2 template string. This
    parameter is crucial for determining the structure and components of the pipeline to be built and connected.
    :param template_params: An optional dictionary for passing additional parameters to the Jinja2 template, allowing
    for further customization of the pipeline configuration.

    :raises ValueError: If the specified `pipeline_template` is invalid or if Jinja2 template syntax errors are detected
    :raises PipelineValidationError: When attempting to override a non-existent component or if the component instance
    provided does not meet the expected specifications.
    """

    template_file_extension = ".yaml.jinja2"

    def __init__(self, pipeline_template: str, template_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a PipelineTemplate.

        :param pipeline_template: The pipeline template to use. This can be either a predefined pipeline type
        (e.g. 'rag', 'indexing', etc.) or a custom jinja2 template string defining the pipeline itself.
        :param template_params: An optional dictionary of parameters to use when rendering the pipeline template.
        """
        if not pipeline_template:
            raise ValueError(
                "pipeline_name is required and has to be either:\n"
                "a) predefined pipeline type (e.g. 'rag', 'indexing', etc.)\n"
                "b) custom jinja2 template defining the pipeline itself.\n"
                f"Available pipeline types are: {self._list_templates()}"
            )
        available_templates = self._list_templates()
        if pipeline_template not in available_templates and not self._contains_jinja2_syntax(pipeline_template):
            raise ValueError(
                f"Pipeline template '{pipeline_template}' not found. Available pipeline types are: {available_templates}.\n"
                "Users can also provide a custom jinja2 template defining the pipeline itself."
            )
        if pipeline_template in available_templates:
            template_path = f"{Path(__file__).resolve().parent}/{pipeline_template}{self.template_file_extension}"
            if not Path(template_path).exists():
                raise ValueError(f"Pipeline template '{template_path}' not found.")
            self.template_path = template_path

            with open(template_path, "r") as file:
                self.template_text = file.read()
        else:
            self.template_path = None
            self.template_text = pipeline_template

        env = NativeEnvironment()
        try:
            self.template = env.from_string(self.template_text)
        except TemplateSyntaxError as e:
            raise ValueError(
                f"Invalid pipeline template '{(self.template_path if self.template_path else self.template_text)}': {e}"
            ) from e
        self.templated_variables = self._extract_variables(env)
        self.components = {}
        self.template_params = template_params or {}

    def override(self, component_name: str, component_instance):
        """
        Overrides a component specified in the pipeline template with a custom component instance.

        :param component_name: The name of the component within the template to override.
        :param component_instance: The instance of the component to use as an override. Must be an instance
        of a class annotated with `@component`.

        :return: The instance of `PipelineTemplate` to allow for method chaining.

        :raises PipelineValidationError: If the `component_name` does not exist in the template or if
        `component_instance` is not a valid component.
        """
        # check if the component_name is allowed in the template
        if component_name not in self.templated_variables:
            raise PipelineValidationError(
                f"Component '{component_name}' is not defined in the pipeline template '{self.template_path}'."
            )
        if not isinstance(component_instance, Component):
            raise PipelineValidationError(
                f"'{type(component_instance)}' doesn't seem to be a component. Is this class decorated with @component?"
            )
        component_dict: Dict[str, Any]
        if hasattr(component_instance, "to_dict"):
            component_dict = component_instance.to_dict()
        else:
            component_dict = default_to_dict(component_instance)
        self.components[component_name] = component_dict
        return self

    def build(self):
        """
        Constructs a `Pipeline` instance based on the template and any overridden components.

        :return: An instance of `Pipeline` constructed from the rendered template and custom component configurations.
        """
        rendered_yaml = self.template.render(**self.components, **self.template_params)
        pipeline_yaml = yaml.safe_load(rendered_yaml)
        return Pipeline.from_dict(pipeline_yaml)

    def _extract_variables(self, env: NativeEnvironment) -> Set[str]:
        """
        Extracts all variables from a list of Jinja template strings.

        :param env: A Jinja native environment.
        :return: A set of variable names extracted from the template strings.
        """
        variables = set()
        ast = env.parse(self.template_text)
        variables.update(meta.find_undeclared_variables(ast))
        return variables

    def _list_templates(self):
        """
        Lists all available pipeline templates by scanning the directory for files with the template file extension.

        :return: A list of strings representing the names of available pipeline templates, excluding the file extension.
        """
        directory = Path(__file__).resolve().parent
        jinja_files = [f for f in directory.iterdir() if f.is_file() and f.name.endswith(self.template_file_extension)]
        correct_template_names = [f.name.rsplit(self.template_file_extension, 1)[0] for f in jinja_files]
        return correct_template_names

    @staticmethod
    def _contains_jinja2_syntax(potential_jinja_template: str):
        """
        Determines if a given string contains Jinja2 templating syntax.

        :param potential_jinja_template: The string to check for Jinja2 syntax.

        :return: `True` if Jinja2 syntax is found, otherwise `False`.
        """
        # Patterns to look for: {{ var }}, {% block %}, {# comment #}
        patterns = [r"\{\{.*?\}\}", r"\{%.*?%\}", r"\{#.*?#\}"]
        combined_pattern = re.compile("|".join(patterns))
        return bool(combined_pattern.search(potential_jinja_template))
