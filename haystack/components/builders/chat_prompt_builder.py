# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Set

from jinja2 import Template, meta

from haystack import component, logging
from haystack.dataclasses.chat_message import ChatMessage, ChatRole

logger = logging.getLogger(__name__)


@component
class ChatPromptBuilder:
    """
    ChatPromptBuilder is a component that renders a chat prompt from a template string using Jinja2 templates.

    It is designed to construct prompts for the pipeline using static or dynamic templates: Users can change
    the prompt template at runtime by providing a new template for each pipeline run invocation if needed.

    The template variables found in the init template string are used as input types for the component and are all optional,
    unless explicitly specified. If an optional template variable is not provided as an input, it will be replaced with
    an empty string in the rendered prompt. Use `variable` and `required_variables` to specify the input types and
    required variables.

    Usage example with static prompt template:
    ```python
    template = [ChatMessage.from_user("Translate to {{ target_language }}. Context: {{ snippet }}; Translation:")]
    builder = ChatPromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```

    Usage example of overriding the static template at runtime:
    ```python
    template = [ChatMessage.from_user("Translate to {{ target_language }}. Context: {{ snippet }}; Translation:")]
    builder = ChatPromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")

    summary_template = [ChatMessage.from_user("Translate to {{ target_language }} and summarize. Context: {{ snippet }}; Summary:")]
    builder.run(target_language="spanish", snippet="I can't speak spanish.", template=summary_template)
    ```

    Usage example with dynamic prompt template:
    ```python
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack import Pipeline
    from haystack.utils import Secret

    # no parameter init, we don't use any runtime template variables
    prompt_builder = ChatPromptBuilder()
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
                                        "template": messages}})
    print(res)

    >> {'llm': {'replies': [ChatMessage(content="Berlin is the capital city of Germany and one of the most vibrant
    and diverse cities in Europe. Here are some key things to know...Enjoy your time exploring the vibrant and dynamic
    capital of Germany!", role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-3.5-turbo-0613',
    'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 27, 'completion_tokens': 681, 'total_tokens':
    708}})]}}


    messages = [system_message, ChatMessage.from_user("What's the weather forecast for {{location}} in the next
    {{day_count}} days?")]

    res = pipe.run(data={"prompt_builder": {"template_variables": {"location": location, "day_count": "5"},
                                        "template": messages}})

    print(res)
    >> {'llm': {'replies': [ChatMessage(content="Here is the weather forecast for Berlin in the next 5
    days:\n\nDay 1: Mostly cloudy with a high of 22°C (72°F) and...so it's always a good idea to check for updates
    closer to your visit.", role=<ChatRole.ASSISTANT: 'assistant'>, name=None, meta={'model': 'gpt-3.5-turbo-0613',
    'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 37, 'completion_tokens': 201,
    'total_tokens': 238}})]}}
    ```

    Note how in the example above, we can dynamically change the prompt template by providing a new template to the
    run method of the pipeline.

    """

    def __init__(
        self,
        template: Optional[List[ChatMessage]] = None,
        required_variables: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
    ):
        """
        Constructs a ChatPromptBuilder component.

        :param template:
            A list of `ChatMessage` instances. All user and system messages are treated as potentially having jinja2
            templates and are rendered with the provided template variables. If not provided, the template
            must be provided at runtime using the `template` parameter of the `run` method.
        :param required_variables: An optional list of input variables that must be provided at all times.
            If not provided, an exception will be raised.
        :param variables:
            A list of template variable names you can use in prompt construction. For example,
            if `variables` contains the string `documents`, the component will create an input called
            `documents` of type `Any`. These variable names are used to resolve variables and their values during
            pipeline execution. The values associated with variables from the pipeline runtime are then injected into
            template placeholders of a prompt text template that is provided to the `run` method.
            If not provided, variables are inferred from `template`.
        """
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = required_variables or []
        self.template = template
        variables = variables or []
        if template and not variables:
            for message in template:
                if message.is_from(ChatRole.USER) or message.is_from(ChatRole.SYSTEM):
                    # infere variables from template
                    msg_template = Template(message.content)
                    ast = msg_template.environment.parse(message.content)
                    template_variables = meta.find_undeclared_variables(ast)
                    variables += list(template_variables)

        # setup inputs
        static_input_slots = {"template": Optional[str], "template_variables": Optional[Dict[str, Any]]}
        component.set_input_types(self, **static_input_slots)
        for var in variables:
            if var in self.required_variables:
                component.set_input_type(self, var, Any)
            else:
                component.set_input_type(self, var, Any, "")

    @component.output_types(prompt=List[ChatMessage])
    def run(
        self,
        template: Optional[List[ChatMessage]] = None,
        template_variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Executes the prompt building process.

        It applies the template variables to render the final prompt. You can provide variables either via pipeline
        (set through `variables` or inferred from `template` at initialization) or via additional template variables
        set directly to this method. On collision, the variables provided directly to this method take precedence.

        :param template:
            An optional list of ChatMessages to overwrite ChatPromptBuilder's default template. If None, the default template
            provided at initialization is used.
        :param template_variables:
            An optional dictionary of template variables. These are additional variables users can provide directly
            to this method in contrast to pipeline variables.
        :param kwargs:
            Pipeline variables (typically resolved from a pipeline) which are merged with the provided template variables.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated list of `ChatMessage` instances after rendering the found templates.
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
                f"The {self.__class__.__name__} requires a non-empty list of ChatMessage instances. "
                f"Please provide a valid list of ChatMessage instances to render the prompt."
            )

        if not all(isinstance(message, ChatMessage) for message in template):
            raise ValueError(
                f"The {self.__class__.__name__} expects a list containing only ChatMessage instances. "
                f"The provided list contains other types. Please ensure that all elements in the list "
                f"are ChatMessage instances."
            )

        processed_messages = []
        for message in template:
            if message.is_from(ChatRole.USER) or message.is_from(ChatRole.SYSTEM):
                self._validate_variables(set(template_variables_combined.keys()))
                compiled_template = Template(message.content)
                rendered_content = compiled_template.render(template_variables_combined)
                rendered_message = (
                    ChatMessage.from_user(rendered_content)
                    if message.is_from(ChatRole.USER)
                    else ChatMessage.from_system(rendered_content)
                )
                processed_messages.append(rendered_message)
            else:
                processed_messages.append(message)

        return {"prompt": processed_messages}

    def _validate_variables(self, provided_variables: Set[str]):
        """
        Checks if all the required template variables are provided.

        :param provided_variables:
            A set of provided template variables.
        :raises ValueError:
            If no template is provided or if all the required template variables are not provided.
        """
        missing_variables = [var for var in self.required_variables if var not in provided_variables]
        if missing_variables:
            missing_vars_str = ", ".join(missing_variables)
            raise ValueError(
                f"Missing required input variables in ChatPromptBuilder: {missing_vars_str}. "
                f"Required variables: {self.required_variables}. Provided variables: {provided_variables}."
            )
