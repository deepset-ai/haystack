import logging
from typing import Dict, Any, Optional, List, Set

from jinja2 import Template, meta

from haystack import component
from haystack.dataclasses.chat_message import ChatMessage, ChatRole

logger = logging.getLogger(__name__)


@component
class DynamicChatPromptBuilder:
    """
    DynamicChatPromptBuilder is designed to construct dynamic prompts from a list of `ChatMessage` instances. It
    integrates with Jinja2 templating for dynamic prompt generation.

    DynamicChatPromptBuilder assumes that the last user message in the list contains a template and renders it with resolved
    pipeline variables and any additional template variables provided.

    You can provide additional template variables directly to the pipeline `run` method. They are then merged with the
    variables resolved from the pipeline runtime. This allows for greater flexibility and customization of the
    generated prompts based on runtime conditions and user inputs.

    The following example demonstrates how to use DynamicChatPromptBuilder to generate a chat prompt:

    ```python
    from haystack.components.builders import DynamicChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack import Pipeline

    # no parameter init, we don't use any runtime template variables
    prompt_builder = DynamicChatPromptBuilder()
    llm = OpenAIChatGenerator(api_key="<your-api-key>", model_name="gpt-3.5-turbo")

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")

    location = "Berlin"
    system_message = ChatMessage.from_system("You are a helpful assistant giving out valuable information to tourists.")
    messages = [system_message, ChatMessage.from_user("Tell me about {{location}}")]


    res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location}, "prompt_source": messages}})
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

    The primary advantage of using DynamicChatPromptBuilder is showcased in the examples provided above.
    DynamicChatPromptBuilder allows dynamic customization of prompt messages without the need to reload or recreate the
    pipeline for each invocation.

    In the example above, the first query asks for general information about Berlin, and the second query requests
    the weather forecast for Berlin in the next few days. DynamicChatPromptBuilder efficiently handles these distinct
    prompt structures by adjusting pipeline run parameters invocations, as opposed to a regular PromptBuilder, which
    would require recreating or reloading the pipeline for each distinct type of query, leading to inefficiency and
    potential service disruptions, especially in server environments where continuous service is vital.

    Note that the weather forecast in the example above is fictional, but it can be easily connected to a weather
    API to provide real weather forecasts.
    """

    def __init__(self, runtime_variables: Optional[List[str]] = None):
        """
        Initializes DynamicChatPromptBuilder with the provided variable names. These variable names are used to resolve
        variables and their values during pipeline runtime execution. For example, if `runtime_variables` contains
        `documents`, your instance of DynamicChatPromptBuilder will expect an input called `documents`.
        The values associated with variables from the pipeline runtime are then injected into template placeholders
        of a ChatMessage that is provided to the `run` method.

        :param runtime_variables: A list of template variable names you can use in chat prompt construction.
        :type runtime_variables: Optional[List[str]]
        """
        runtime_variables = runtime_variables or []

        # setup inputs
        run_input_slots = {"prompt_source": List[ChatMessage], "template_variables": Optional[Dict[str, Any]]}
        kwargs_input_slots = {var: Optional[Any] for var in runtime_variables}
        component.set_input_types(self, **run_input_slots, **kwargs_input_slots)

        # setup outputs
        component.set_output_types(self, prompt=List[ChatMessage])
        self.runtime_variables = runtime_variables

    def run(self, prompt_source: List[ChatMessage], template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Executes the dynamic prompt building process by processing a list of `ChatMessage` instances.
        The last user message is treated as a template and rendered with the resolved pipeline variables and any
        additional template variables provided.

        You can provide additional template variables directly to this method, which are then merged with the variables
        resolved from the pipeline runtime.

        :param prompt_source: A list of `ChatMessage` instances. We make an assumption that the last user message has
        the template for the chat prompt
        :type prompt_source: List[ChatMessage]

        :param template_variables: An optional dictionary of template variables. Template variables provided at
        initialization are required to resolve pipeline variables, and these are additional variables users can
        provide directly to this method.
        :type template_variables: Optional[Dict[str, Any]]

        :param kwargs: Additional keyword arguments, typically resolved from a pipeline, which are merged with the
        provided template variables.

        :return: A dictionary containing the key "prompt", which holds either the updated list of `ChatMessage`
        instances or the rendered string template, forming the complete dynamic prompt.
        :rtype: Dict[str, List[ChatMessage]
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}
        if not template_variables_combined:
            logger.warning(
                "The DynamicChatPromptBuilder run method requires template variables, but none were provided. "
                "Please provide an appropriate template variable to enable correct prompt generation."
            )
        result: List[ChatMessage] = self._process_chat_messages(prompt_source, template_variables_combined)
        return {"prompt": result}

    def _process_chat_messages(self, prompt_source: List[ChatMessage], template_variables: Dict[str, Any]):
        """
        Processes a list of :class:`ChatMessage` instances to generate a chat prompt.

        It takes the last user message in the list, treats it as a template, and renders it with the provided
        template variables. The resulting message replaces the last user message in the list, forming a complete,
        templated chat prompt.

        :param prompt_source: A list of `ChatMessage` instances to be processed. The last message is expected
        to be from a user and is treated as a template.
        :type prompt_source: List[ChatMessage]

        :param template_variables: A dictionary of template variables used for rendering the last user message.
        :type template_variables: Dict[str, Any]

        :return: A list of `ChatMessage` instances, where the last user message has been replaced with its
        templated version.
        :rtype: List[ChatMessage]

        :raises ValueError: If `chat_messages` is empty or contains elements that are not instances of
        `ChatMessage`.
        :raises ValueError: If the last message in `chat_messages` is not from a user.
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

        last_message: ChatMessage = prompt_source[-1]
        if last_message.is_from(ChatRole.USER):
            template = self._validate_template(last_message.content, set(template_variables.keys()))
            templated_user_message = ChatMessage.from_user(template.render(template_variables))
            return prompt_source[:-1] + [templated_user_message]
        else:
            logger.warning(
                "DynamicChatPromptBuilder was not provided with a user message as the last message in "
                "chat conversation, no templating will be applied."
            )
            return prompt_source

    def _validate_template(self, template_text: str, provided_variables: Set[str]):
        """
        Checks if all the required template variables are provided to the pipeline `run` method.
        If all the required template variables are provided, returns a Jinja2 template object.
        Otherwise, raises a ValueError.

        :param template_text: A Jinja2 template as a string.
        :param provided_variables: A set of provided template variables.
        :type provided_variables: Set[str]
        :return: A Jinja2 template object if all the required template variables are provided.
        :raises ValueError: If all the required template variables are not provided.
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
