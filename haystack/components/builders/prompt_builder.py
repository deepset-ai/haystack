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
        :param required_variables: An optional list of input variables that must be provided at all times.
            If not provided, an exception will be raised.
        """
        self._template_string = template
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = required_variables or []
        self.template: Optional[Template] = None
        if template:
            self.template = Template(template)
            if not variables:
                # infere variables from template
                ast = self.template.environment.parse(template)
                template_variables = meta.find_undeclared_variables(ast)
                variables = list(template_variables)

        variables = variables or []

        # setup inputs
        static_input_slots = {"template": Optional[str], "template_variables": Optional[Dict[str, Any]]}
        variable_input_slots = {var: Optional[Any] for var in variables}
        component.set_input_types(self, **static_input_slots, **variable_input_slots)

    @component.output_types(prompt=str)
    def run(self, template: Optional[str] = None, template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Executes the prompt building process.

        It applies the template variables to render the final prompt. You can provide variables either via pipeline
        (set through `variables` or inferred from `template` at initialization) or via additional template variables
        set directly to this method. On collision, the variables provided directly to this method take precedence.

        :param template:
            An optional string template to overwrite PromptBuilder's default template. If None, the default template
            provided at initialization is used.
        :param template_variables:
            An optional dictionary of template variables. These are additional variables users can provide directly
            to this method in contrast to pipeline variables.
        :param kwargs:
            Pipeline variables (typically resolved from a pipeline) which are merged with the provided template variables.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated prompt text after rendering the prompt template.
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}
        compiled_template = self._validate_template(template, set(template_variables_combined.keys()))
        result = compiled_template.render(template_variables_combined)
        return {"prompt": result}

    def _validate_template(self, template_text: Optional[str], provided_variables: Set[str]):
        """
        Checks if the template is valid and all the required template variables are provided.

        If all the required template variables are provided, returns a Jinja2 template object.
        Otherwise, raises a ValueError.

        :param template_text:
            A Jinja2 template as a string.
        :param provided_variables:
            A set of provided template variables.
        :returns:
            A Jinja2 template object if all the required template variables are provided.
        :raises ValueError:
            If no template is provided or if all the required template variables are not provided.
        """
        if isinstance(template_text, str):
            template = Template(template_text)
        elif self.template is not None:
            template = self.template
        else:
            raise ValueError(
                "The PromptBuilder run method requires a template, but none was provided. "
                "Please provide an appropriate template to enable prompt generation."
            )

        missing_variables = [var for var in self.required_variables if var not in provided_variables]
        if missing_variables:
            missing_vars_str = ", ".join(missing_variables)
            raise ValueError(
                f"Missing required input variables in PromptBuilder: {missing_vars_str}. "
                f"Required variables: {self.required_variables}. Provided variables: {provided_variables}."
            )

        return template

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the component.

        :returns:
            Serialized dictionary representation of the component.
        """
        return default_to_dict(
            self, template=self._template_string, variables=self._variables, required_variables=self._required_variables
        )
