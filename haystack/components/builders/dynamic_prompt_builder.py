# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Dict, List, Optional, Set

from jinja2 import Template, meta

from haystack import component, logging

logger = logging.getLogger(__name__)


@component
class DynamicPromptBuilder:
    """
    DynamicPromptBuilder is designed to construct dynamic prompts for the pipeline.

    Users can change the prompt template at runtime by providing a new template for each pipeline run invocation
    if needed.

    Usage example:
    ```python
    from typing import List
    from haystack.components.builders import DynamicPromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack import Pipeline, component, Document
    from haystack.utils import Secret

    prompt_builder = DynamicPromptBuilder(runtime_variables=["documents"])
    llm = OpenAIGenerator(api_key=Secret.from_token("<your-api-key>"), model="gpt-3.5-turbo")


    @component
    class DocumentProducer:

        @component.output_types(documents=List[Document])
        def run(self, doc_input: str):
            return {"documents": [Document(content=doc_input)]}


    pipe = Pipeline()
    pipe.add_component("doc_producer", DocumentProducer())
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("doc_producer.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    template = "Here is the document: {{documents[0].content}} \\n Answer: {{query}}"
    result = pipe.run(
        data={
            "doc_producer": {"doc_input": "Hello world, I live in Berlin"},
            "prompt_builder": {
                "prompt_source": template,
                "template_variables": {"query": "Where does the speaker live?"},
            },
        }
    )
    print(result)

    >> {'llm': {'replies': ['The speaker lives in Berlin.'],
    >> 'meta': [{'model': 'gpt-3.5-turbo-0613',
    >> 'index': 0,
    >> 'finish_reason': 'stop',
    >> 'usage': {'prompt_tokens': 28,
    >> 'completion_tokens': 6,
    >> 'total_tokens': 34}}]}}

    Note how in the example above, we can dynamically change the prompt template by providing a new template to the
    run method of the pipeline. This dynamic prompt generation is in contrast to the static prompt generation
    using `PromptBuilder`, where the prompt template is fixed for the pipeline's lifetime and cannot be changed
    for each pipeline run invocation.

    """

    def __init__(self, runtime_variables: Optional[List[str]] = None):
        """
        Constructs a DynamicPromptBuilder component.

        :param runtime_variables:
            A list of template variable names you can use in prompt construction. For example,
            if `runtime_variables` contains the string `documents`, the component will create an input called
            `documents` of type `Any`. These variable names are used to resolve variables and their values during
            pipeline execution. The values associated with variables from the pipeline runtime are then injected into
            template placeholders of a prompt text template that is provided to the `run` method.
        """
        warnings.warn(
            "`DynamicPromptBuilder` is deprecated and will be removed in Haystack 2.4.0."
            "Use `PromptBuilder` instead.",
            DeprecationWarning,
        )

        runtime_variables = runtime_variables or []

        # setup inputs
        run_input_slots = {"prompt_source": str, "template_variables": Optional[Dict[str, Any]]}
        kwargs_input_slots = {var: Optional[Any] for var in runtime_variables}
        component.set_input_types(self, **run_input_slots, **kwargs_input_slots)

        # setup outputs
        component.set_output_types(self, prompt=str)

        self.runtime_variables = runtime_variables

    def run(self, prompt_source: str, template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Executes the dynamic prompt building process.

        Depending on the provided type of `prompt_source`, this method either processes a list of `ChatMessage`
        instances or a string template. In the case of `ChatMessage` instances, the last user message is treated as a
        template and rendered with the resolved pipeline variables and any additional template variables provided.

        For a string template, it directly applies the template variables to render the final prompt. You can provide
        additional template variables directly to this method, that are then merged with the variables resolved from
        the pipeline runtime.

        :param prompt_source:
            A string template.
        :param template_variables:
            An optional dictionary of template variables. Template variables provided at initialization are required
            to resolve pipeline variables, and these are additional variables users can provide directly to this method.
        :param kwargs:
            Additional keyword arguments, typically resolved from a pipeline, which are merged with the provided
            template variables.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated prompt text after rendering the string template.
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}
        if not template_variables_combined:
            raise ValueError(
                "The DynamicPromptBuilder run method requires template variables, but none were provided. "
                "Please provide an appropriate template variable to enable prompt generation."
            )

        template = self._validate_template(prompt_source, set(template_variables_combined.keys()))
        result = template.render(template_variables_combined)
        return {"prompt": result}

    def _validate_template(self, template_text: str, provided_variables: Set[str]):
        """
        Checks if all the required template variables are provided to the pipeline `run` method.

        If all the required template variables are provided, returns a Jinja2 template object.
        Otherwise, raises a ValueError.

        :param template_text:
            A Jinja2 template as a string.
        :param provided_variables:
            A set of provided template variables.
        :returns:
            A Jinja2 template object if all the required template variables are provided.
        :raises ValueError:
            If all the required template variables are not provided.
        """
        template = Template(template_text)
        ast = template.environment.parse(template_text)
        required_template_variables = meta.find_undeclared_variables(ast)
        filled_template_vars = required_template_variables.intersection(provided_variables)
        if len(filled_template_vars) != len(required_template_variables):
            raise ValueError(
                f"The {self.__class__.__name__} requires specific template variables that are missing. "
                f"Required variables: {required_template_variables}. Only the following variables were "
                f"provided: {provided_variables}. Please provide all the required template variables."
            )
        return template
