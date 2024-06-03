# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Set

from jinja2 import Template, meta

from haystack import component, default_to_dict


@component
class PromptBuilder:
    """
    PromptBuilder is a component that renders a prompt from a template string using Jinja2 templates.

    For prompt engineering, users can switch the template at runtime by providing a template for each pipeline run invocation.

    The template variables found in the default template string are used as input types for the component and are all optional,
    unless explicitly specified. If an optional template variable is not provided as an input, it will be replaced with
    an empty string in the rendered prompt. Use `variables` and `required_variables` to change the default variable behavior.

    ### Usage examples

    #### On its own

    Below is an example of using the `PromptBuilder` to render a prompt template and fill it with `target_language` and `snippet`.
    The PromptBuilder returns a prompt with the string "Translate the following context to spanish. Context: I can't speak spanish.; Translation:".
    ```python
    from haystack.components.builders import PromptBuilder

    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```

    #### In a Pipeline

    Below is an example of a RAG pipeline where we use a `PromptBuilder` to render a custom prompt template and fill it with the
    contents of retrieved Documents and a query. The rendered prompt is then sent to a Generator.
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

    #### Changing the template at runtime (Prompt Engineering)

    `PromptBuilder` allows you to switch the prompt template of an existing pipeline.
    Below's example builds on top of the existing pipeline of the previous section.
    The existing pipeline is invoked with a new prompt template:
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
    If you want to use different variables during prompt engineering than in the default template,
    you can do so by setting `PromptBuilder`'s `variables` init parameter accordingly.

    #### Overwriting variables at runtime

    In case you want to overwrite the values of variables, you can use `template_variables` during runtime as illustrated below:
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
    If not set otherwise, it would evaluate to its default value 'English'.
    In this example we are overwriting its value to 'German'.
    `template_variables` allows you to overwrite pipeline variables (such as documents) as well.

    """

    def __init__(
        self, template: str, required_variables: Optional[List[str]] = None, variables: Optional[List[str]] = None
    ):
        """
        Constructs a PromptBuilder component.

        :param template:
            A Jinja2 template string that is used to render the prompt, e.g.:
            `"Summarize this document: {{ documents[0].content }}\\nSummary:"`
        :param required_variables: An optional list of input variables that must be provided at runtime.
            If a required variable is not provided at runtime, an exception will be raised.
        :param variables:
            An optional list of input variables to be used in prompt templates instead of the ones inferred from `template`.
            For example, if you want to use more variables during prompt engineering than the ones present in the default
            template, you can provide them here.
        """
        self._template_string = template
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = required_variables or []
        self.template = Template(template)
        if not variables:
            # infere variables from template
            ast = self.template.environment.parse(template)
            template_variables = meta.find_undeclared_variables(ast)
            variables = list(template_variables)

        variables = variables or []

        # setup inputs
        static_input_slots = {"template": Optional[str], "template_variables": Optional[Dict[str, Any]]}
        component.set_input_types(self, **static_input_slots)
        for var in variables:
            if var in self.required_variables:
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
        if isinstance(template, str):
            compiled_template = Template(template)

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
        missing_variables = [var for var in self.required_variables if var not in provided_variables]
        if missing_variables:
            missing_vars_str = ", ".join(missing_variables)
            raise ValueError(
                f"Missing required input variables in PromptBuilder: {missing_vars_str}. "
                f"Required variables: {self.required_variables}. Provided variables: {provided_variables}."
            )
