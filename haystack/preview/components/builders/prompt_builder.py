from typing import Dict, Any, Optional, List

from jinja2 import Template, meta

from haystack.preview import component
from haystack.preview import default_to_dict
from haystack.preview.dataclasses.chat_message import ChatMessage, ChatRole


@component
class PromptBuilder:
    """
    A component for building prompts using template strings or template variables.

    The `PromptBuilder` can be initialized with a template string or template variables to dynamically build
    a prompt. There are two distinct use cases:

    1. **Static Template**: When initialized with a `template` string, the `PromptBuilder` will always use
       this template for rendering prompts throughout its lifetime.

    2. **Dynamic Templates**: When initialized with `template_variables` and receiving messages in
       `ChatMessage` format, it allows for different templates for each message, enabling prompt templating
       on a per-user message basis.

    :param template: (Optional) A template string to be rendered using Jinja2 syntax, e.g.,
                     "What's the weather like in {{ location }}?". This template will be used for all
                     prompts. Defaults to None.
    :param template_variables: (Optional) A list of all template variables to be used as input.
                              This parameter enables dynamic templating based on user messages. Defaults to None.

    :raises ValueError: If neither `template` nor `template_variables` are provided.


    Usage (static templating):

    ```python
    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```

    The above template is used for all messages that are passed to the `PromptBuilder`.


    Usage (dynamic templating):
    ```python

    from haystack.preview.dataclasses.chat_message import ChatMessage
    template = "What's the weather like in {{ location }}?"

    prompt_builder = PromptBuilder(template_variables=["location", "time"])
    messages = [ChatMessage.from_system("Always start response to user with Herr Blagojevic.
    Respond in German even if some input data is in other languages"),
                ChatMessage.from_user(template)]

    response = pipe.run(data={"prompt_builder": {"location": location, "messages": messages}})
    ```

    In this example, only the last user message is templated. The template_variables parameter in the PromptBuilder
    initialization specifies all potential template variables, yet only the variables utilized in the template are
    required in the run method. For instance, since the time variable isn't used in the template, it's not
    necessary in the run method invocation above.


    Note:
        The behavior of `PromptBuilder` is determined by the initialization parameters. A static template
        provides a consistent prompt structure, while template variables offer dynamic templating per user
        message.

    """

    def __init__(self, template: Optional[str] = None, template_variables: Optional[List[str]] = None):
        """
        Initialize the component with either a template string or template variables.
        If template is given PromptBuilder will parse the template string and use the template variables
        as input types. Conversely, if template_variables are given, PromptBuilder will directly use
        them as input variables.

        If neither template nor template_variables are provided, an error will be raised. If both are provided,
        an error will be raised as well.

        :param template: Template string to be rendered.
        :param template_variables: List of template variables to be used as input types.
        """
        if template_variables and template:
            raise ValueError("template and template_variables cannot be provided at the same time.")

        # dynamic per-user message templating
        if template_variables:
            # treat vars as optional input slots
            dynamic_input_slots = {var: Optional[Any] for var in template_variables}
            self.template = None

        # static templating
        else:
            if not template:
                raise ValueError("Either template or template_variables must be provided.")
            self.template = Template(template)
            ast = self.template.environment.parse(template)
            static_template_variables = meta.find_undeclared_variables(ast)
            # treat vars as required input slots - as per design
            dynamic_input_slots = {var: Any for var in static_template_variables}

        # always provide all serialized vars, so we can serialize
        # the component regardless of the initialization method (static vs. dynamic)
        self.template_variables = template_variables
        self._template_string = template

        optional_input_slots = {"messages": Optional[List[ChatMessage]]}
        component.set_input_types(self, **optional_input_slots, **dynamic_input_slots)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, template=self._template_string, template_variables=self.template_variables)

    @component.output_types(prompt=str)
    def run(self, messages: Optional[List[ChatMessage]] = None, **kwargs):
        """
        Build and return the prompt based on the provided messages and template or template variables.
        If `messages` are provided, the template will be applied to the last user message.

        :param messages: (Optional) List of `ChatMessage` instances, used for dynamic templating
        when `template_variables` are provided.
        :param kwargs: Additional keyword arguments representing template variables.
        """

        if messages:
            # apply the template to the last user message only
            last_message: ChatMessage = messages[-1]
            if last_message.is_from(ChatRole.USER):
                template = Template(last_message.content)
                return {"prompt": messages[:-1] + [ChatMessage.from_user(template.render(kwargs))]}
            else:
                return {"prompt": messages}
        else:
            if self.template:
                return {"prompt": self.template.render(kwargs)}
            else:
                raise ValueError(
                    "PromptBuilder was initialized with template_variables, but no ChatMessage(s) were provided."
                )
