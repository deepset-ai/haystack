# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Set, Union

from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses.chat_message import ChatMessage, ChatRole, TextContent
from haystack.lazy_imports import LazyImport
from haystack.utils.jinja2_extensions import ChatMessageExtension, Jinja2TimeExtension, for_template

logger = logging.getLogger(__name__)

with LazyImport() as arrow_import:
    import arrow


@component
class ChatPromptBuilder:
    """
    Renders a chat prompt from a template string using Jinja2 syntax.

    It constructs prompts using static or dynamic templates, which you can update for each pipeline run.

    Template variables in the template are optional unless specified otherwise.
    If an optional variable isn't provided, it defaults to an empty string. Use `variable` and `required_variables`
    to define input types and required variables.

    ### Usage examples

    #### With static prompt template

    ```python
    template = [ChatMessage.from_user("Translate to {{ target_language }}. Context: {{ snippet }}; Translation:")]
    builder = ChatPromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```

    #### Overriding static template at runtime

    ```python
    template = [ChatMessage.from_user("Translate to {{ target_language }}. Context: {{ snippet }}; Translation:")]
    builder = ChatPromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")

    msg = "Translate to {{ target_language }} and summarize. Context: {{ snippet }}; Summary:"
    summary_template = [ChatMessage.from_user(msg)]
    builder.run(target_language="spanish", snippet="I can't speak spanish.", template=summary_template)
    ```

    #### With dynamic prompt template

    ```python
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack import Pipeline
    from haystack.utils import Secret

    # no parameter init, we don't use any runtime template variables
    prompt_builder = ChatPromptBuilder()
    llm = OpenAIChatGenerator(api_key=Secret.from_token("<your-api-key>"), model="gpt-4o-mini")

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    location = "Berlin"
    language = "English"
    system_message = ChatMessage.from_system("You are an assistant giving information to tourists in {{language}}")
    messages = [system_message, ChatMessage.from_user("Tell me about {{location}}")]

    res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location, "language": language},
                                        "template": messages}})
    print(res)

    >> {'llm': {'replies': [ChatMessage(content="Berlin is the capital city of Germany and one of the most vibrant
    and diverse cities in Europe. Here are some key things to know...Enjoy your time exploring the vibrant and dynamic
    capital of Germany!", role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-4o-mini',
    'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 27, 'completion_tokens': 681, 'total_tokens':
    708}})]}}


    messages = [system_message, ChatMessage.from_user("What's the weather forecast for {{location}} in the next
    {{day_count}} days?")]

    res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location, "day_count": "5"},
                                        "template": messages}})

    print(res)
    >> {'llm': {'replies': [ChatMessage(content="Here is the weather forecast for Berlin in the next 5
    days:\\n\\nDay 1: Mostly cloudy with a high of 22°C (72°F) and...so it's always a good idea to check for updates
    closer to your visit.", role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-4o-mini',
    'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 37, 'completion_tokens': 201,
    'total_tokens': 238}})]}}
    ```

    """

    def __init__(
        self,
        template: Optional[Union[List[ChatMessage], str]] = None,
        required_variables: Optional[Union[List[str], Literal["*"]]] = None,
        variables: Optional[List[str]] = None,
    ):
        """
        Constructs a ChatPromptBuilder component.

        :param template:
            A list of `ChatMessage` objects or a string template. The component looks for Jinja2 template syntax and
            renders the prompt with the provided variables. Provide the template in either
            the `init` method` or the `run` method.
        :param required_variables:
            List variables that must be provided as input to ChatPromptBuilder.
            If a variable listed as required is not provided, an exception is raised.
            If set to "*", all variables found in the prompt are required. Optional.
        :param variables:
            List input variables to use in prompt templates instead of the ones inferred from the
            `template` parameter. For example, to use more variables during prompt engineering than the ones present
            in the default template, you can provide them here.
        """
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = required_variables or []
        self.template = template
        variables = variables or []

        self._env = SandboxedEnvironment(extensions=[ChatMessageExtension])
        if arrow_import.is_successful():
            self._env.add_extension(Jinja2TimeExtension)
        self._env.filters["for_template"] = for_template

        if template and not variables:
            if isinstance(template, list):
                for message in template:
                    if message.is_from(ChatRole.USER) or message.is_from(ChatRole.SYSTEM):
                        # infer variables from template
                        if message.text is None:
                            raise ValueError(f"The provided ChatMessage has no text. ChatMessage: {message}")
                        ast = self._env.parse(message.text)
                        template_variables = meta.find_undeclared_variables(ast)
                        variables += list(template_variables)
            elif isinstance(template, str):
                ast = self._env.parse(template)
                variables = list(meta.find_undeclared_variables(ast))
        self.variables = variables

        if len(self.variables) > 0 and required_variables is None:
            logger.warning(
                "ChatPromptBuilder has {length} prompt variables, but `required_variables` is not set. "
                "By default, all prompt variables are treated as optional, which may lead to unintended behavior in "
                "multi-branch pipelines. To avoid unexpected execution, ensure that variables intended to be required "
                "are explicitly set in `required_variables`.",
                length=len(self.variables),
            )

        # setup inputs
        for var in self.variables:
            if self.required_variables == "*" or var in self.required_variables:
                component.set_input_type(self, var, Any)
            else:
                component.set_input_type(self, var, Any, "")

    @component.output_types(prompt=List[ChatMessage])
    def run(
        self,
        template: Optional[Union[List[ChatMessage], str]] = None,
        template_variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Renders the prompt template with the provided variables.

        It applies the template variables to render the final prompt. You can provide variables with pipeline kwargs.
        To overwrite the default template, you can set the `template` parameter.
        To overwrite pipeline kwargs, you can set the `template_variables` parameter.

        :param template:
            An optional list of `ChatMessage` objects or a string template to overwrite ChatPromptBuilder's default
            template.
            If `None`, the default template provided at initialization is used.
        :param template_variables:
            An optional dictionary of template variables to overwrite the pipeline variables.
        :param kwargs:
            Pipeline variables used for rendering the prompt.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated list of `ChatMessage` objects after rendering the templates.
        :raises ValueError:
            If `chat_messages` is empty or contains elements that are not instances of `ChatMessage`.
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}

        if template is None:
            template = self.template

        if not template:
            raise ValueError(
                f"The {self.__class__.__name__} requires a non-empty list of ChatMessage instances or a string"
                f"template. Please provide a valid list of ChatMessage instances or a string template to render "
                f"the prompt."
            )

        if isinstance(template, list) and not all(isinstance(message, ChatMessage) for message in template):
            raise ValueError(
                f"The {self.__class__.__name__} expects a list containing only ChatMessage instances. "
                f"The provided list contains other types. Please ensure that all elements in the list "
                f"are ChatMessage instances."
            )

        processed_messages = []
        if isinstance(template, list):
            for message in template:
                if message.is_from(ChatRole.USER) or message.is_from(ChatRole.SYSTEM):
                    self._validate_variables(set(template_variables_combined.keys()))
                    if message.text is None:
                        raise ValueError(f"The provided ChatMessage has no text. ChatMessage: {message}")
                    compiled_template = self._env.from_string(message.text)
                    rendered_text = compiled_template.render(template_variables_combined)
                    # deep copy the message to avoid modifying the original message
                    rendered_message: ChatMessage = deepcopy(message)
                    rendered_message._content = [TextContent(text=rendered_text)]
                    processed_messages.append(rendered_message)
                else:
                    processed_messages.append(message)
        elif isinstance(template, str):
            processed_messages = self._render_chat_messages_from_str_template(template, template_variables_combined)

        return {"prompt": processed_messages}

    def _render_chat_messages_from_str_template(
        self, template: str, template_variables: Dict[str, Any]
    ) -> List[ChatMessage]:
        """
        Renders a chat message from a string template.

        This must be used in conjunction with the `ChatMessageExtension` Jinja2 extension
        and the `for_template` filter.
        """
        compiled_template = self._env.from_string(template)
        rendered = compiled_template.render(template_variables)
        messages = []
        for line in rendered.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                messages.append(ChatMessage.from_dict(json.loads(line)))
            except Exception:
                contextlib.suppress(Exception)

        # Fallback for templates without message tags
        if not messages:
            content = rendered.strip()
            if content:
                messages.append(ChatMessage(_role=ChatRole("user"), _content=[TextContent(text=content)]))

        return messages

    def _validate_variables(self, provided_variables: Set[str]):
        """
        Checks if all the required template variables are provided.

        :param provided_variables:
            A set of provided template variables.
        :raises ValueError:
            If no template is provided or if all the required template variables are not provided.
        """
        if self.required_variables == "*":
            required_variables = sorted(self.variables)
        else:
            required_variables = self.required_variables
        missing_variables = [var for var in required_variables if var not in provided_variables]
        if missing_variables:
            missing_vars_str = ", ".join(missing_variables)
            raise ValueError(
                f"Missing required input variables in ChatPromptBuilder: {missing_vars_str}. "
                f"Required variables: {required_variables}. Provided variables: {provided_variables}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the component.

        :returns:
            Serialized dictionary representation of the component.
        """

        template: Optional[Union[List[Dict[str, Any]], str]] = None
        if isinstance(self.template, list):
            template = [m.to_dict() for m in self.template]
        elif isinstance(self.template, str):
            template = self.template

        return default_to_dict(
            self, template=template, variables=self._variables, required_variables=self._required_variables
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatPromptBuilder":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary to deserialize and create the component.

        :returns:
            The deserialized component.
        """
        init_parameters = data["init_parameters"]
        template = init_parameters.get("template")
        if template:
            if isinstance(template, list):
                init_parameters["template"] = [ChatMessage.from_dict(d) for d in template]
            elif isinstance(template, str):
                init_parameters["template"] = template

        return default_from_dict(cls, data)
