# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Dict, List, Optional, Set

from jinja2 import Template, meta

from haystack import component, logging
from haystack.dataclasses.chat_message import ChatMessage, ChatRole

logger = logging.getLogger(__name__)


@component
class DynamicChatPromptBuilder:
    """
    DynamicChatPromptBuilder is designed to construct dynamic prompts from a list of `ChatMessage` instances.

    It integrates with Jinja2 templating for dynamic prompt generation. It considers any user or system message in the
    list potentially containing a template and renders it with variables provided to the constructor. Additional
    template variables can be feed into the component/pipeline `run` method and will be merged before rendering the
    template.

    Usage example:
    ```python
    from haystack.components.builders import DynamicChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack import Pipeline
    from haystack.utils import Secret

    # no parameter init, we don't use any runtime template variables
    prompt_builder = DynamicChatPromptBuilder()
    llm = OpenAIChatGenerator(api_key=Secret.from_token("<your-api-key>"), model="gpt-3.5-turbo")

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    location = "Berlin"
    language = "English"
    system_message = ChatMessage.from_system("You are an assistant giving information to tourists in {{language}}")
    messages = [system_message, ChatMessage.from_user("Tell me about {{location}}")]

    res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location, "language": language},
                                        "prompt_source": messages}})
    print(res)

    >> {'llm': {'replies': [ChatMessage(content="Berlin is the capital city of Germany and one of the most vibrant
    and diverse cities in Europe. Here are some key things to know...Enjoy your time exploring the vibrant and dynamic
    capital of Germany!", role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-3.5-turbo-0613',
    'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 27, 'completion_tokens': 681, 'total_tokens':
    708}})]}}


    messages = [system_message, ChatMessage.from_user("What's the weather forecast for {{location}} in the next
    {{day_count}} days?")]

    res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location, "day_count": "5"},
                                        "prompt_source": messages}})

    print(res)
    >> {'llm': {'replies': [ChatMessage(content="Here is the weather forecast for Berlin in the next 5
    days:\n\nDay 1: Mostly cloudy with a high of 22°C (72°F) and...so it's always a good idea to check for updates
    closer to your visit.", role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-3.5-turbo-0613',
    'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 37, 'completion_tokens': 201,
    'total_tokens': 238}})]}}
    ```

    Note that the weather forecast in the example above is fictional, but it can be easily connected to a weather
    API to provide real weather forecasts.
    """

    def __init__(self, runtime_variables: Optional[List[str]] = None):
        """
        Constructs a DynamicChatPromptBuilder component.

        :param runtime_variables:
            A list of template variable names you can use in chat prompt construction. For example,
            if `runtime_variables` contains the string `documents`, the component will create an input called
            `documents` of type `Any`. These variable names are used to resolve variables and their values during
            pipeline execution. The values associated with variables from the pipeline runtime are then injected into
            template placeholders of a ChatMessage that is provided to the `run` method.
        """
        warnings.warn(
            "`DynamicChatPromptBuilder` is deprecated and will be removed in Haystack 2.4.0."
            "Use `ChatPromptBuilder` instead.",
            DeprecationWarning,
        )
        runtime_variables = runtime_variables or []

        # setup inputs
        default_inputs = {"prompt_source": List[ChatMessage], "template_variables": Optional[Dict[str, Any]]}
        additional_input_slots = {var: Optional[Any] for var in runtime_variables}
        component.set_input_types(self, **default_inputs, **additional_input_slots)

        # setup outputs
        component.set_output_types(self, prompt=List[ChatMessage])
        self.runtime_variables = runtime_variables

    def run(self, prompt_source: List[ChatMessage], template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Executes the dynamic prompt building process by processing a list of `ChatMessage` instances.

        Any user message or system message is inspected for templates and rendered with the variables provided to the
        constructor. You can provide additional template variables directly to this method, which are then merged with
        the variables provided to the constructor.

        :param prompt_source:
            A list of `ChatMessage` instances. All user and system messages are treated as potentially having templates
            and are rendered with the provided template variables - if templates are found.
        :param template_variables:
            A dictionary of template variables. Template variables provided at initialization are required
            to resolve pipeline variables, and these are additional variables users can provide directly to this method.
        :param kwargs:
            Additional keyword arguments, typically resolved from a pipeline, which are merged with the provided
            template variables.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated list of `ChatMessage` instances after rendering the found templates.
        :raises ValueError:
            If `chat_messages` is empty or contains elements that are not instances of `ChatMessage`.
        :raises ValueError:
            If the last message in `chat_messages` is not from a user.
        """
        if not prompt_source:
            raise ValueError(
                f"The {self.__class__.__name__} requires a non-empty list of ChatMessage instances. "
                f"Please provide a valid list of ChatMessage instances to render the prompt."
            )
        if not all(isinstance(message, ChatMessage) for message in prompt_source):
            raise ValueError(
                f"The {self.__class__.__name__} expects a list containing only ChatMessage instances. "
                f"The provided list contains other types. Please ensure that all elements in the list "
                f"are ChatMessage instances."
            )

        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables = {**kwargs, **template_variables}
        if not template_variables:
            logger.warning(
                "The DynamicChatPromptBuilder run method requires template variables, but none were provided. "
                "Please provide an appropriate template variable to enable correct prompt generation."
            )
        processed_messages = []
        for message in prompt_source:
            if message.is_from(ChatRole.USER) or message.is_from(ChatRole.SYSTEM):
                template = self._validate_template(message.content, set(template_variables.keys()))
                rendered_content = template.render(template_variables)
                rendered_message = (
                    ChatMessage.from_user(rendered_content)
                    if message.is_from(ChatRole.USER)
                    else ChatMessage.from_system(rendered_content)
                )
                processed_messages.append(rendered_message)
            else:
                processed_messages.append(message)
        return {"prompt": processed_messages}

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
