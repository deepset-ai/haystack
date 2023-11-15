from typing import Dict, Any, Optional, List

from jinja2 import Template, meta

from haystack.preview import component
from haystack.preview import default_to_dict
from haystack.preview.dataclasses.chat_message import ChatMessage, ChatRole


@component
class PromptBuilder:
    """
    Constructs prompts for language models using a specified template or dynamic template variables.

    This component operates in two modes:

    - **Static Template Mode**: Utilizes a single, fixed template for all prompts. This mode is ideal for
      scenarios like non-chat language model (LLM) generation, where the prompt structure remains constant.

    - **Dynamic Template Mode**: Applies a set of template variables to incoming `ChatMessage` objects.
      This mode is tailored for interactive chat environments, allowing customization of each prompt based on user input.

    :param template: A Jinja2 template string for static templating. Defaults to None.
    :param template_variables: A list of variable names for dynamic templating. Defaults to None.

    :raises ValueError: Raised if neither `template` nor `template_variables` are provided, or if both are simultaneously provided.

    **Example Usage**:

    *Static Template Mode*:
    ```python
    template = "Translate '{{ snippet }}' to {{ target_language }}."
    builder = PromptBuilder(template=template)
    prompt = builder.run(target_language="Spanish", snippet="Hello, world!")
    ```

    *Dynamic Template Mode*:
    ```python
    from haystack.preview.dataclasses.chat_message import ChatMessage

    builder = PromptBuilder(template_variables=["location"])
    messages = [ChatMessage.from_user("What's the weather like in {{ location }}?")]
    prompt = builder.run(messages=messages, location="Paris")
    ```
    """

    def __init__(self, template: Optional[str] = None, template_variables: Optional[List[str]] = None):
        """
        Initializes the `PromptBuilder`.

        :param template: Jinja2 template string for static mode.
        :param template_variables: Variable names for dynamic mode.
        """
        if template_variables and template:
            raise ValueError("template and template_variables cannot be provided at the same time.")
        if not template_variables and not template:
            raise ValueError("Either template or template_variables must be provided.")

        # dynamic per-user message templating
        if template_variables:
            # treat vars as optional input slots
            input_slots = {var: Optional[Any] for var in template_variables}
            self.template = None
            component.set_output_types(self, prompt=List[ChatMessage])

        # static templating
        if template:
            self.template = Template(template)
            ast = self.template.environment.parse(template)
            static_template_variables = meta.find_undeclared_variables(ast)
            # treat vars as required input slots - as per design
            input_slots = {var: Any for var in static_template_variables}
            component.set_output_types(self, prompt=str)

        # always provide all serialized vars, so we can serialize
        # the component regardless of the initialization method (static vs. dynamic)
        self._template_string = template
        self._template_variables = template_variables

        optional_input_slots = {"messages": Optional[List[ChatMessage]], "template_variables": Optional[List[str]]}
        component.set_input_types(self, **optional_input_slots, **input_slots)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, template=self._template_string, template_variables=self._template_variables)

    def run(
        self,
        messages: Optional[List[ChatMessage]] = None,
        template_variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Generates a prompt based on the provided messages and template variables.

        If `messages` are not provided, the method uses the static template defined during initialization,
        populating it with `kwargs` to generate a prompt string.

        If `messages` are provided (List[ChatMessage]), the last user message is templated. Besides the initial
        template variables (resolved automatically), additional variables can optionally be provided through
        `template_variables`.

        :param messages: List of `ChatMessage` instances for dynamic mode.
        :param template_variables: Dictionary of additional variables for dynamic mode.
        :return: Generated prompt (str) or list of `ChatMessage` instances.
        """
        if messages:
            # apply the template to the last user message only
            last_message: ChatMessage = messages[-1]
            if last_message.is_from(ChatRole.USER):
                template = Template(last_message.content)
                if template_variables or kwargs:
                    # merge template_variables and kwargs
                    template_variables_final = {**(kwargs or {}), **(template_variables or {})}
                    templated_user_message = ChatMessage.from_user(template.render(template_variables_final))
                    # return all previous messages + templated user message
                    new_messages = messages[:-1] + [templated_user_message]
                    return {"prompt": new_messages}
                return {"prompt": messages}
            else:
                return {"prompt": messages}
        else:
            if self.template:
                return {"prompt": self.template.render(kwargs)}
            else:
                raise ValueError(
                    "PromptBuilder was initialized with template_variables, but no ChatMessage(s) were provided."
                )
