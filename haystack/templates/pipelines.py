from typing import Dict, Any, Set, Optional

import yaml
from jinja2 import meta, TemplateSyntaxError
from jinja2.nativetypes import NativeEnvironment

from haystack import Pipeline
from haystack.core.component import Component
from haystack.core.errors import PipelineValidationError
from haystack.core.serialization import component_to_dict
from haystack.templates.source import TemplateSource


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
      from haystack.templates import PipelineTemplate, TemplateSource, PredefinedTemplate

      # Create a pipeline with default components for a QA task
      ts = TemplateSource.from_predefined(PredefinedTemplate.QA)
      pipe = PipelineTemplate(ts).build()
      print(pipe.run(data={"question": "What's the capital of Bosnia and Herzegovina? Be brief"}))
      ```

    - **Custom Component Settings**: Customizing a pipeline by overriding a component, such as integrating a
    streaming-capable generator for real-time feedback.
      ```python
      from haystack.components.generators import OpenAIGenerator
      from haystack.components.generators.utils import print_streaming_chunk
      from haystack.templates import PipelineTemplate, TemplateSource, PredefinedTemplate

      # Customize the pipeline with a streaming-capable generator
      ts = TemplateSource.from_predefined(PredefinedTemplate.QA)
      streaming_pipe = PipelineTemplate(ts).override("generator",
                                                               OpenAIGenerator(
                                                                   streaming_callback=print_streaming_chunk)).build()
      streaming_pipe.run(data={"question": "What's the capital of Germany? Tell me about it"})
      ```

    - **Customizing for Specific Tasks**: Building a pipeline for document indexing with specific components tailored
    to the task.
      ```python
      from haystack.components.embedders import SentenceTransformersDocumentEmbedder
      from haystack.templates import PipelineTemplate, TemplateSource, PredefinedTemplate

      # Customize the pipeline for document indexing with specific components, include PDF file converter
      ts = TemplateSource.from_predefined(PredefinedTemplate.INDEXING)
      ptb = PipelineTemplate(ts, template_params={"use_pdf_file_converter": True})
      ptb.override("embedder", SentenceTransformersDocumentEmbedder(progress_bar=True))
      pipe = ptb.build()

      result = pipe.run(data={
          "sources": ["some_text_file.txt", "another_pdf_file.pdf"]})
      print(result)
      ```

    The `PipelineTemplate` is designed to offer both ease of use for common pipeline configurations and the
    flexibility to customize and extend pipelines as required by advanced users and specific use cases.
    """

    template_file_extension = ".yaml.jinja2"

    def __init__(self, pipeline_template: TemplateSource, template_params: Optional[Dict[str, Any]] = None):
        """
        Initialize a PipelineTemplate.

        :param pipeline_template: The template source to use. See `TemplateSource` for available methods to load
        templates.
        :param template_params: An optional dictionary of parameters to use when rendering the pipeline template.
        """
        self.template_text = pipeline_template.template
        env = NativeEnvironment()
        try:
            self.template = env.from_string(self.template_text)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid pipeline template, template syntax error: {e.message}") from e
        self.templated_variables = self._extract_variables(env)
        self.components: Dict[str, Any] = {}
        self.template_params = template_params or {}

    def override(self, component_name: str, component_instance: Component) -> "PipelineTemplate":
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
            raise PipelineValidationError(f"Component '{component_name}' is not defined in the pipeline template")
        if not isinstance(component_instance, Component):
            raise PipelineValidationError(
                f"'{type(component_instance)}' doesn't seem to be a component. Is this class decorated with @component?"
            )
        self.components[component_name] = component_to_dict(component_instance)
        return self

    def list_variables(self) -> Set[str]:
        """
        Lists all templated variables in the pipeline template.

        :return: a list of strings representing the names of templated variables in the pipeline template.
        """
        return self.templated_variables

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
