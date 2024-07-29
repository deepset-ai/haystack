# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from jinja2 import PackageLoader, TemplateSyntaxError, meta
from jinja2.sandbox import SandboxedEnvironment

TEMPLATE_FILE_EXTENSION = ".yaml.jinja2"
TEMPLATE_HOME_DIR = Path(__file__).resolve().parent / "predefined"


class PredefinedPipeline(Enum):
    """
    Enumeration of predefined pipeline templates that can be used to create a `PipelineTemplate`.
    """

    # Maintain 1-to-1 mapping between the enum name and the template file name in templates directory
    GENERATIVE_QA = "generative_qa"
    RAG = "rag"
    INDEXING = "indexing"
    CHAT_WITH_WEBSITE = "chat_with_website"


class PipelineTemplate:
    """
    The PipelineTemplate enables the creation of flexible and configurable pipelines.

    The PipelineTemplate class enables the straightforward creation of flexible and configurable pipelines using
    Jinja2 templated YAML files. Specifically designed to simplify the setup of complex data processing pipelines for
    a range of NLP tasks—including question answering, retriever augmented generation (RAG), document indexing, among
    others - PipelineTemplate empowers users to dynamically generate pipeline configurations from templates and
    customize components as necessary. Its design philosophy centers on providing an accessible, yet powerful, tool
    for constructing pipelines that accommodate both common use cases and specialized requirements with ease.

    Examples of usage:

    - **Default Build**: Instantiating a pipeline with default settings for a "question answering" (qa) task.
      ```python
      from haystack.templates import PipelineTemplate, PredefinedPipeline

      # Create a pipeline with default components for an extractive QA task
      pipe = PipelineTemplate.from_predefined(PredefinedPipeline.GENERATIVE_QA).build()
      print(pipe.run(data={"question": "What's the capital of Bosnia and Herzegovina? Be brief"}))
      ```

    - **Customizing for Specific Tasks**: Building a pipeline for document indexing with specific components tailored
    to the task.
      ```python
      from haystack.components.embedders import SentenceTransformersDocumentEmbedder
      from haystack.templates import PipelineTemplate, PredefinedPipeline

      # Customize the pipeline for document indexing with specific components, include PDF file converter
      pt = PipelineTemplate.from_predefined(PredefinedTemplate.INDEXING)
      pipe = pt.build(template_params={"use_pdf_file_converter": True})

      result = pipe.run(data={"sources": ["some_text_file.txt", "another_pdf_file.pdf"]})
      print(result)
      ```

    The `PipelineTemplate` is designed to offer both ease of use for common pipeline configurations and the
    flexibility to customize and extend pipelines as required by advanced users and specific use cases.
    """

    def __init__(self, template_content: str):
        """
        Initialize a PipelineTemplate.

        Besides calling the constructor directly, a set of utility methods is provided to conveniently create an
        instance of `PipelineTemplate` from different sources. See `from_string`, `from_file`, `from_predefined`
        and `from_url`.

        :param template_content: The raw template source to use in the template.
        """
        env = SandboxedEnvironment(
            loader=PackageLoader("haystack.core.pipeline", "predefined"), trim_blocks=True, lstrip_blocks=True
        )
        try:
            self._template = env.from_string(template_content)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid pipeline template: {e.message}") from e

        # Store the list of undefined variables in the template. Components' names will be part of this list
        self.template_variables = meta.find_undeclared_variables(env.parse(template_content))
        self._template_content = template_content

    def render(self, template_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Constructs a `Pipeline` instance based on the template.

        :param template_params: An optional dictionary of parameters to use when rendering the pipeline template.

        :returns: An instance of `Pipeline` constructed from the rendered template and custom component configurations.
        """
        template_params = template_params or {}
        return self._template.render(**template_params)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "PipelineTemplate":
        """
        Create a PipelineTemplate from a file.

        :param file_path: The path to the file containing the template. Must contain valid Jinja2 syntax.
        :returns: An instance of `PipelineTemplate`.
        """
        with open(file_path, "r") as file:
            return cls(file.read())

    @classmethod
    def from_predefined(cls, predefined_pipeline: PredefinedPipeline) -> "PipelineTemplate":
        """
        Create a PipelineTemplate from a predefined template.

        See `PredefinedPipeline` for available options.

        :param predefined_pipeline: The predefined pipeline to use.
        :returns: An instance of `PipelineTemplate `.
        """
        template_path = f"{TEMPLATE_HOME_DIR}/{predefined_pipeline.value}{TEMPLATE_FILE_EXTENSION}"
        return cls.from_file(template_path)

    @property
    def template_content(self) -> str:
        """
        Returns the raw template string as a read-only property.
        """
        return self._template_content
