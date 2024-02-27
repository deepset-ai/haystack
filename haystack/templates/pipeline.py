from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union

import requests
import yaml
from jinja2 import meta, TemplateSyntaxError, Environment, PackageLoader

from haystack import Pipeline
from haystack.core.component import Component
from haystack.core.errors import PipelineValidationError
from haystack.core.serialization import component_to_dict


TEMPLATE_FILE_EXTENSION = ".yaml.jinja2"
TEMPLATE_HOME_DIR = Path(__file__).resolve().parent / "predefined"


class PredefinedPipeline(Enum):
    """
    Enumeration of predefined pipeline templates that can be used to create a `PipelineTemplate`.
    """

    # When type is empty, the template source must be provided to the PipelineTemplate before calling build()
    EMPTY = "empty"

    # Maintain 1-to-1 mapping between the enum name and the template file name in templates directory
    GENERATIVE_QA = "generative_qa"
    RAG = "rag"
    INDEXING = "indexing"


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
      from haystack.templates import PipelineTemplate, PredefinedPipeline

      # Create a pipeline with default components for an extractive QA task
      pipe = PipelineTemplate.from_predefined(PredefinedPipeline.GENERATIVE_QA).build()
      print(pipe.run(data={"question": "What's the capital of Bosnia and Herzegovina? Be brief"}))
      ```

    - **Custom Component Settings**: Customizing a pipeline by overriding a component, such as integrating a
    streaming-capable generator for real-time feedback.
      ```python
      from haystack.components.generators import OpenAIGenerator
      from haystack.components.generators.utils import print_streaming_chunk
      from haystack.templates import PipelineTemplate, PredefinedPipeline

      # Customize the pipeline with a streaming-capable question-answering generator
      streaming_pipe = (
        PipelineTemplate.from_predefined(PredefinedTemplate.GENERATIVE_QA)
        .override("generator", OpenAIGenerator(streaming_callback=print_streaming_chunk))
        .build()
      )
      streaming_pipe.run(data={"question": "What's the capital of Germany? Tell me about it"})
      ```

    - **Customizing for Specific Tasks**: Building a pipeline for document indexing with specific components tailored
    to the task.
      ```python
      from haystack.components.embedders import SentenceTransformersDocumentEmbedder
      from haystack.templates import PipelineTemplate, PredefinedPipeline

      # Customize the pipeline for document indexing with specific components, include PDF file converter
      pt = PipelineTemplate.from_predefined(PredefinedTemplate.INDEXING)
      pt.override("embedder", SentenceTransformersDocumentEmbedder(progress_bar=True))
      pipe = pt.build(template_params={"use_pdf_file_converter": True})

      result = pipe.run(data={"sources": ["some_text_file.txt", "another_pdf_file.pdf"]})
      print(result)
      ```

    The `PipelineTemplate` is designed to offer both ease of use for common pipeline configurations and the
    flexibility to customize and extend pipelines as required by advanced users and specific use cases.
    """

    def __init__(self, template_content: str):
        """
        Initialize a PipelineTemplate. Besides calling the constructor directly, a set of utility methods is provided
        for conveniently create an instance of `PipelineTemplate` from different sources. See `from_string`,
        `from_file`, `from_predefined` and `from_url`.

        :param template_content: The raw template source to use in the template.
        """
        env = Environment(
            loader=PackageLoader("haystack.templates", "predefined"), trim_blocks=True, lstrip_blocks=True
        )
        try:
            self._template = env.from_string(template_content)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid pipeline template: {e.message}") from e

        # Store the list of undefined variables in the template. Components' names will be part of this list
        self.template_variables = meta.find_undeclared_variables(env.parse(template_content))
        self.component_overrides: Dict[str, Any] = {}
        self._template_content = template_content

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
        if component_name not in self.template_variables:
            raise PipelineValidationError(f"Component '{component_name}' is not defined in the pipeline template")

        if not isinstance(component_instance, Component):
            raise PipelineValidationError(
                f"'{type(component_instance)}' is not a Component. Is this class decorated with @component?"
            )

        self.component_overrides[component_name] = yaml.safe_dump(component_to_dict(component_instance))
        return self

    def build(self, template_params: Optional[Dict[str, Any]] = None):
        """
        Constructs a `Pipeline` instance based on the template and any overridden components.

        :return: An instance of `Pipeline` constructed from the rendered template and custom component configurations.
        """
        template_params = template_params or {}
        rendered = self._template.render(**self.component_overrides, **template_params)
        return Pipeline.loads(rendered)

    @classmethod
    def from_string(cls, template_str: str) -> "PipelineTemplate":
        """
        Create a PipelineTemplate from a string.
        :param template_str: The template string to use. Must contain valid Jinja syntax.
        :return: An instance of `PipelineTemplate `.
        """
        return cls(template_str)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "PipelineTemplate":
        """
        Create a PipelineTemplate from a file.
        :param file_path: The path to the file containing the template. Must contain valid Jinja2 syntax.
        :return: An instance of `PipelineTemplate `.
        """
        with open(file_path, "r") as file:
            return cls(file.read())

    @classmethod
    def from_predefined(cls, predefined_pipeline: PredefinedPipeline) -> "PipelineTemplate":
        """
        Create a PipelineTemplate from a predefined template. See `PredefinedPipeline` for available options.
        :param predefined_pipeline: The predefined pipeline to use.
        :return: An instance of `PipelineTemplate `.
        """
        if predefined_pipeline == PredefinedPipeline.EMPTY:
            # This is temporary, to ease the refactoring
            raise ValueError("Please provide a PipelineType value")

        template_path = f"{TEMPLATE_HOME_DIR}/{predefined_pipeline.value}{TEMPLATE_FILE_EXTENSION}"
        return cls.from_file(template_path)

    @classmethod
    def from_url(cls, url: str) -> "PipelineTemplate":
        """
        Create a PipelineTemplate from a URL.
        :param url: The URL to fetch the template from. Must contain valid Jinja2 syntax.
        :return: An instance of `PipelineTemplate `.
        """
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return cls(response.text)

    @property
    def template_content(self) -> str:
        """
        Returns the raw template string as a read-only property.
        """
        return self._template_content
