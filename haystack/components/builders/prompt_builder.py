# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Jinja2TimeExtension
from haystack.utils.jinja2_extensions import _extract_template_variables_and_assignments
from haystack.utils.jinja2_sandbox import HaystackSandboxedEnvironment

logger = logging.getLogger(__name__)


@component
class PromptBuilder:
    """

    Renders a prompt filling in any variables so that it can send it to a Generator.

    The prompt uses Jinja2 template syntax.
    The variables in the default template are used as PromptBuilder's input and are all required by default.
    To make any subset of variables optional, set `required_variables` to an explicit list of the variables that
    should remain required. Optional variables are replaced with an empty string in the rendered prompt.
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
    with the contents of the retrieved documents and a query. The rendered prompt is then sent to a ChatGenerator.
    ```python
    from haystack import Pipeline, Document
    from haystack.utils import Secret
    from haystack.components.generators.chat import OpenAIChatGenerator
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
    p.add_component(instance=OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY")), name="llm")
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
        required_variables: list[str] | Literal["*"] | None = "*",
        variables: list[str] | None = None,
        max_length: int | None = None,
    ) -> None:
        """
        Constructs a PromptBuilder component.

        :param template:
            A prompt template that uses Jinja2 syntax to add variables. For example:
            `"Summarize this document: {{ documents[0].content }}\\nSummary:"`
            It's used to render the prompt.
            The variables in the default template are input for PromptBuilder and are all required by default.
        :param required_variables: List variables that must be provided as input to PromptBuilder.
            Defaults to `"*"`, which marks every variable found in the prompt as required.
            Pass an explicit list to only require a subset of the variables; any variable not listed becomes
            optional and is replaced with an empty string in the rendered prompt when missing.
            Set to `None` to mark every variable as optional.
        :param variables:
            List input variables to use in prompt templates instead of the ones inferred from the
            `template` parameter. For example, to use more variables during prompt engineering than the ones present
            in the default template, you can provide them here.
        :param max_length:
            Maximum length of the rendered prompt in characters. When set, prompts longer than this
            are truncated intelligently: content from the `documents` variable is shortened first
            (dropping least-relevant documents and truncating remaining ones) while preserving
            other variables such as `question` or `history` at the end of the prompt.
            This prevents `ValidationError` from models with limited context windows
            (e.g. 4095 tokens ≈ 16380 characters) when many documents are retrieved.
            If no `documents` variable is present, the prompt is truncated from the
            middle while preserving its suffix. Leave as `None` (default) to disable truncation.
        """
        self._template_string = template
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = required_variables or []
        self.max_length = max_length
        try:
            # The Jinja2TimeExtension needs an optional dependency to be installed.
            # If it's not available we can do without it and use the PromptBuilder as is.
            self._env = HaystackSandboxedEnvironment(extensions=[Jinja2TimeExtension])
        except ImportError:
            self._env = HaystackSandboxedEnvironment()

        self.template = self._env.from_string(template)

        if not variables:
            assigned_variables, template_variables = _extract_template_variables_and_assignments(
                env=self._env, template=template
            )
            variables = list(template_variables - assigned_variables)

        variables = variables or []
        self.variables = variables

        if len(self.variables) > 0 and required_variables is None:
            logger.warning(
                "PromptBuilder has {length} prompt variables and `required_variables` is explicitly set to `None`. "
                "This treats all prompt variables as optional, which may lead to unintended behavior in "
                "multi-branch pipelines. Only set `required_variables` to `None` if you intentionally want all "
                "variables to be optional.",
                length=len(self.variables),
            )

        # setup inputs
        for var in self.variables:
            if self.required_variables == "*" or var in self.required_variables:
                component.set_input_type(self, var, Any)
            else:
                component.set_input_type(self, var, Any, "")

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary representation of the component.

        :returns:
            Serialized dictionary representation of the component.
        """
        return default_to_dict(
            self,
            template=self._template_string,
            variables=self._variables,
            required_variables=self._required_variables,
            max_length=self.max_length,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptBuilder":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary to deserialize and create the component.

        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(prompt=str)
    def run(
        self, template: str | None = None, template_variables: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
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

        # Handle intelligent truncation when max_length is set
        if self.max_length is not None:
            # First try document-aware truncation
            if "documents" in template_variables_combined and isinstance(
                template_variables_combined["documents"], list
            ):
                initial_prompt = compiled_template.render(template_variables_combined)
                if len(initial_prompt) > self.max_length:
                    truncated_docs = self._truncate_documents(
                        template_variables_combined["documents"],
                        template_variables_combined,
                        compiled_template,
                        self.max_length,
                    )
                    template_variables_combined = {**template_variables_combined, "documents": truncated_docs}

            # Render and check if still over limit (or no documents to truncate)
            result_str = compiled_template.render(template_variables_combined)
            if len(result_str) > self.max_length:
                # Generic fallback: keep suffix (question/history) intact, truncate from middle
                # This preserves the end of the prompt where the question typically lives
                orig_len = len(result_str)
                tail_len = min(1000, self.max_length // 3)
                head_len = self.max_length - tail_len - 3
                if head_len > 0:
                    result_str = result_str[:head_len] + "..." + result_str[-tail_len:]
                else:
                    result_str = result_str[: self.max_length]
                logger.warning(
                    "Prompt truncated from {orig_len} to {new_len} characters (max_length={max_len}); "
                    "applied middle-truncation preserving suffix",
                    orig_len=orig_len,
                    new_len=len(result_str),
                    max_len=self.max_length,
                )
                return {"prompt": result_str}

        result = compiled_template.render(template_variables_combined)
        return {"prompt": result}

    def _truncate_documents(
        self, documents: list[Any], variables: dict[str, Any], compiled_template: Any, max_length: int
    ) -> list[Any]:
        """
        Truncate documents to fit prompt within max_length.

        Strategy: keep most-relevant (first) documents, drop least-relevant last ones,
        and if a single document still overflows, truncate its content.
        """
        if not documents:
            return documents

        # Try dropping least relevant docs first
        truncated = list(documents)
        # First, try to find minimal number of docs that fits by dropping from the end
        while truncated:
            candidate_vars = {**variables, "documents": truncated}
            prompt = compiled_template.render(candidate_vars)
            if len(prompt) <= max_length:
                if len(truncated) != len(documents):
                    logger.info(
                        "Prompt truncated from {orig_docs} to {new_docs} documents to fit max_length={max_len}",
                        orig_docs=len(documents),
                        new_docs=len(truncated),
                        max_len=max_length,
                    )
                return truncated

            # If even with current count it doesn't fit, try truncating the last doc's content
            last = truncated[-1]
            content = getattr(last, "content", None)
            if content is None:
                # For plain string docs or other types, fallback to string representation
                try:
                    content = str(last)
                    is_string_doc = True
                except Exception:
                    # Drop it if we can't handle
                    truncated = truncated[:-1]
                    continue
            else:
                is_string_doc = False

            # If last doc content is substantial, truncate it instead of dropping entirely
            # Only do this when we are down to trying to fit with this doc
            prompt_without_last = compiled_template.render({**variables, "documents": truncated[:-1]})
            remaining_budget = max_length - len(prompt_without_last)
            # Account for template overhead around each doc (approx 10 chars) and ensure non-negative
            if remaining_budget > 50 and len(content) > remaining_budget:
                # Truncate content to fit remaining budget
                # Keep beginning of document (most relevant part)
                allowed_content_len = max(0, remaining_budget - 20)
                truncated_content = content[:allowed_content_len]
                if is_string_doc:
                    new_last: Any = truncated_content
                else:
                    try:
                        new_last = replace(last, content=truncated_content)
                    except Exception:
                        # Fallback: try to create new doc with same type
                        try:
                            new_last = last.__class__(content=truncated_content, meta=getattr(last, "meta", {}))
                        except Exception:
                            new_last = truncated_content
                truncated = truncated[:-1] + [new_last]
                # Check if this truncated version fits
                candidate_vars = {**variables, "documents": truncated}
                prompt = compiled_template.render(candidate_vars)
                if len(prompt) <= max_length:
                    logger.info(
                        "Prompt truncated by shortening last document content from {orig_len} to {new_len} chars",
                        orig_len=len(content),
                        new_len=len(truncated_content),
                    )
                    return truncated
                # If still doesn't fit even after truncating, drop the doc entirely
                truncated = truncated[:-1]
            else:
                # Drop last doc and continue
                truncated = truncated[:-1]

        # All documents dropped but still over? Return empty list;
        # generic fallback in run() will handle remaining overflow
        return []

    def _validate_variables(self, provided_variables: set[str]) -> None:
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
