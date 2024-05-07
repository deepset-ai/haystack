from typing import Any, Dict, List, Optional, Set

from jinja2 import Template, meta

from haystack import component, default_to_dict, logging

logger = logging.getLogger(__name__)


@component
class PromptBuilder:
    """
    PromptBuilder is a component that renders a prompt from a template string using Jinja2 templates.

    It is designed to construct prompts for the pipeline using static or dynamic templates: Users can change
    the prompt template at runtime by providing a new template for each pipeline run invocation if needed.

    Usage example with static prompt template:
    ```python
    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```

    Usage example of overriding the static template at runtime:
    ```python
    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")

    summary_template = "Translate to {{ target_language }} and summarize the following context. Context: {{ snippet }}; Summary:"
    builder.run(target_language="spanish", snippet="I can't speak spanish.", template=summary_template)
    ```

    Usage example with dynamic prompt template:
    ```python
    from typing import List
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack import Pipeline, component, Document
    from haystack.utils import Secret

    prompt_builder = PromptBuilder(variables=["documents"])
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
                "template": template,
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
    run method of the pipeline.

    """

    def __init__(
        self,
        template: Optional[str] = None,
        variables: Optional[List[str]] = None,
        required_variables: Optional[List[str]] = None,
    ):
        """
        Constructs a PromptBuilder component.

        :param template:
            A Jinja2 template string that will be used to render the prompt text. If not provided, the template
            must be provided at runtime using the `template` parameter of the `run` method.
        :param variables:
            A list of template variable names you can use in prompt construction. For example,
            if `variables` contains the string `documents`, the component will create an input called
            `documents` of type `Any`. These variable names are used to resolve variables and their values during
            pipeline execution. The values associated with variables from the pipeline runtime are then injected into
            template placeholders of a prompt text template that is provided to the `run` method.
            If not provided, variables are inferred from `template`.
        :param required_variables:
            A list of required template variable names you can use in prompt. These variables are required to be
            provided at runtime. If not provided, an exception will be raised.
        """
        self._default_template_string = template
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = set(required_variables or [])
        self.default_template: Optional[Template] = None
        if template:
            self.default_template = Template(template)
            if not variables:
                # infere variables from template
                ast = self.default_template.environment.parse(template)
                default_template_variables = meta.find_undeclared_variables(ast)
                variables = list(default_template_variables)

        variables = variables or []

        # setup inputs
        run_input_slots = {"template": Optional[str], "template_variables": Optional[Dict[str, Any]]}
        kwargs_input_slots = {var: Optional[Any] for var in variables}
        component.set_input_types(self, **run_input_slots, **kwargs_input_slots)

        # setup outputs
        component.set_output_types(self, prompt=str)

    def run(self, template: Optional[str] = None, template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Executes the dynamic prompt building process.

        Depending on the provided type of `template`, this method either processes a list of `ChatMessage`
        instances or a string template. In the case of `ChatMessage` instances, the last user message is treated as a
        template and rendered with the resolved pipeline variables and any additional template variables provided.

        For a string template, it directly applies the template variables to render the final prompt. You can provide
        additional template variables directly to this method, that are then merged with the variables resolved from
        the pipeline runtime.

        :param template:
            A string template.
        :param template_variables:
            An optional dictionary of template variables. Template variables provided at initialization are required
            to resolve pipeline variables, and these are additional variables users can provide directly to this method.
        :param prompt_source:
            A string template (deprecated). Will be removed in future releases. Use `template` instead.
        :param kwargs:
            Additional keyword arguments, typically resolved from a pipeline, which are merged with the provided
            template variables.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated prompt text after rendering the string template.
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}
        compiled_template = self._validate_template(template, set(template_variables_combined.keys()))
        result = compiled_template.render(template_variables_combined)
        return {"prompt": result}

    def _validate_template(self, template_text: Optional[str], provided_variables: Set[str]):
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
        if isinstance(template_text, str):
            template = Template(template_text)
        elif self.default_template is not None:
            template = self.default_template
        else:
            raise ValueError(
                "The PromptBuilder run method requires a template, but none was provided. "
                "Please provide an appropriate template to enable prompt generation."
            )

        missing_required_vars = self.required_variables.difference(provided_variables)
        if missing_required_vars:
            raise ValueError(
                f"The PromptBuilder requires specific template variables that are missing. "
                f"Required variables: {self.required_variables}. Only the following variables were "
                f"provided: {provided_variables}. Please provide all the required template variables."
            )
        return template

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the component.

        :returns:
            Serialized dictionary representation of the component.
        """
        return default_to_dict(
            self,
            template=self._default_template_string,
            variables=self._variables,
            required_variables=self._required_variables,
        )
