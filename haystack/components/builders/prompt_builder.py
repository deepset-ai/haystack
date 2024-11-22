# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional, Set, Union

from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment

from haystack import component, default_to_dict
from haystack.utils import Jinja2TimeExtension


@component
class PromptBuilder:
    """

    Renders a prompt filling in any variables so that it can send it to a Generator.

    The prompt uses Jinja2 template syntax.
    The variables in the default template are used as PromptBuilder's input and are all optional.
    If they're not provided, they're replaced with an empty string in the rendered prompt.
    To try out different prompts, you can replace the prompt template at runtime by
    providing a template for each pipeline run invocation.

    ### Usage examples

    #### On its own

    This example uses PromptBuilder to render a prompt template and fill it with `target_language`
    and `snippet`. PromptBuilder returns a prompt with the string "Translate the following context to Spanish.
    Context: I can't speak Spanish.; Translation:".
    ```python
    from haystack.components.builders import PromptBuilder

    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```

    #### In a Pipeline

    This is an example of a RAG pipeline where PromptBuilder renders a custom prompt template and fills it
    with the contents of the retrieved documents and a query. The rendered prompt is then sent to a Generator.
    ```python
    from haystack import Pipeline, Document
    from haystack.utils import Secret
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.builders.prompt_builder import PromptBuilder

    # in a real world use case documents could come from a retriever, web, or any other source
    documents = [Document(content="Joe lives in Berlin"), Document(content="Joe is a software engineer")]
    prompt_template = \"\"\"
        Given these documents, answer the question.
        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        Question: {{query}}
        Answer:
        \"\"\"
    p = Pipeline()
    p.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    p.add_component(instance=OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY")), name="llm")
    p.connect("prompt_builder", "llm")

    question = "Where does Joe live?"
    result = p.run({"prompt_builder": {"documents": documents, "query": question}})
    print(result)
    ```

    #### Changing the template at runtime (prompt engineering)

    You can change the prompt template of an existing pipeline, like in this example:
    ```python
    documents = [
        Document(content="Joe lives in Berlin", meta={"name": "doc1"}),
        Document(content="Joe is a software engineer", meta={"name": "doc1"}),
    ]
    new_template = \"\"\"
        You are a helpful assistant.
        Given these documents, answer the question.
        Documents:
        {% for doc in documents %}
            Document {{ loop.index }}:
            Document name: {{ doc.meta['name'] }}
            {{ doc.content }}
        {% endfor %}

        Question: {{ query }}
        Answer:
        \"\"\"
    p.run({
        "prompt_builder": {
            "documents": documents,
            "query": question,
            "template": new_template,
        },
    })
    ```
    To replace the variables in the default template when testing your prompt,
    pass the new variables in the `variables` parameter.

    #### Overwriting variables at runtime

    To overwrite the values of variables, use `template_variables` during runtime:
    ```python
    language_template = \"\"\"
    You are a helpful assistant.
    Given these documents, answer the question.
    Documents:
    {% for doc in documents %}
        Document {{ loop.index }}:
        Document name: {{ doc.meta['name'] }}
        {{ doc.content }}
    {% endfor %}

    Question: {{ query }}
    Please provide your answer in {{ answer_language | default('English') }}
    Answer:
    \"\"\"
    p.run({
        "prompt_builder": {
            "documents": documents,
            "query": question,
            "template": language_template,
            "template_variables": {"answer_language": "German"},
        },
    })
    ```
    Note that `language_template` introduces variable `answer_language` which is not bound to any pipeline variable.
    If not set otherwise, it will use its default value 'English'.
    This example overwrites its value to 'German'.
    Use `template_variables` to overwrite pipeline variables (such as documents) as well.

    """

    def __init__(
        self,
        template: str,
        required_variables: Optional[Union[List[str], Literal["*"]]] = None,
        variables: Optional[List[str]] = None,
    ):
        """
        Constructs a PromptBuilder component.

        :param template:
            A prompt template that uses Jinja2 syntax to add variables. For example:
            `"Summarize this document: {{ documents[0].content }}\\nSummary:"`
            It's used to render the prompt.
            The variables in the default template are input for PromptBuilder and are all optional,
            unless explicitly specified.
            If an optional variable is not provided, it's replaced with an empty string in the rendered prompt.
        :param required_variables: List variables that must be provided as input to PromptBuilder.
            If a variable listed as required is not provided, an exception is raised.
            If set to "*", all variables found in the prompt are required. Optional.
        :param variables:
            List input variables to use in prompt templates instead of the ones inferred from the
            `template` parameter. For example, to use more variables during prompt engineering than the ones present
            in the default template, you can provide them here.
        """
        self._template_string = template
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = required_variables or []
        try:
            # The Jinja2TimeExtension needs an optional dependency to be installed.
            # If it's not available we can do without it and use the PromptBuilder as is.
            self._env = SandboxedEnvironment(extensions=[Jinja2TimeExtension])
        except ImportError:
            self._env = SandboxedEnvironment()

        self.template = self._env.from_string(template)
        if not variables:
            # infer variables from template
            ast = self._env.parse(template)
            template_variables = meta.find_undeclared_variables(ast)
            variables = list(template_variables)
        variables = variables or []
        self.variables = variables

        # setup inputs
        for var in self.variables:
            if self.required_variables == "*" or var in self.required_variables:
                component.set_input_type(self, var, Any)
            else:
                component.set_input_type(self, var, Any, "")

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the component.

        :returns:
            Serialized dictionary representation of the component.
        """
        return default_to_dict(
            self, template=self._template_string, variables=self._variables, required_variables=self._required_variables
        )

    @component.output_types(prompt=str)
    def run(self, template: Optional[str] = None, template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Renders the prompt template with the provided variables.

        It applies the template variables to render the final prompt. You can provide variables via pipeline kwargs.
        In order to overwrite the default template, you can set the `template` parameter.
        In order to overwrite pipeline kwargs, you can set the `template_variables` parameter.

        :param template:
            An optional string template to overwrite PromptBuilder's default template. If None, the default template
            provided at initialization is used.
        :param template_variables:
            An optional dictionary of template variables to overwrite the pipeline variables.
        :param kwargs:
            Pipeline variables used for rendering the prompt.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated prompt text after rendering the prompt template.

        :raises ValueError:
            If any of the required template variables is not provided.
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}
        self._validate_variables(set(template_variables_combined.keys()))

        compiled_template = self.template
        if template is not None:
            compiled_template = self._env.from_string(template)

        result = compiled_template.render(template_variables_combined)
        return {"prompt": result}

    def _validate_variables(self, provided_variables: Set[str]):
        """
        Checks if all the required template variables are provided.

        :param provided_variables:
            A set of provided template variables.
        :raises ValueError:
            If any of the required template variables is not provided.
        """
        if self.required_variables == "*":
            required_variables = sorted(self.variables)
        else:
            required_variables = self.required_variables
        missing_variables = [var for var in required_variables if var not in provided_variables]
        if missing_variables:
            missing_vars_str = ", ".join(missing_variables)
            raise ValueError(
                f"Missing required input variables in PromptBuilder: {missing_vars_str}. "
                f"Required variables: {required_variables}. Provided variables: {provided_variables}."
            )
