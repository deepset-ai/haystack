from typing import Any, Dict, List, Optional

from jinja2 import Template, meta

from haystack import component, default_to_dict


@component
class PromptBuilder:
    """
    PromptBuilder is a component that renders a prompt from a template string using Jinja2 templates.

    The template variables found in the template string are used as input types for the component and are all optional,
    unless explicitly specified. If an optional template variable is not provided as an input, it will be replaced with
    an empty string in the rendered prompt.

    Usage example:
    ```python
    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```
    """

    def __init__(self, template: str, required_variables: Optional[List[str]] = None):
        """
        Constructs a PromptBuilder component.

        :param template: A Jinja2 template string, e.g. "Summarize this document: {documents}\\nSummary:"
        :param required_variables: An optional list of input variables that must be provided at all times.
        """
        self._template_string = template
        self.template = Template(template)
        self.required_variables = required_variables or []
        ast = self.template.environment.parse(template)
        template_variables = meta.find_undeclared_variables(ast)

        for var in template_variables:
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
        return default_to_dict(self, template=self._template_string)

    @component.output_types(prompt=str)
    def run(self, **kwargs):
        """
        Renders the prompt template with the provided variables.

        :param kwargs:
            The variables that will be used to render the prompt template.

        :returns: A dictionary with the following keys:
            - `prompt`: The updated prompt text after rendering the prompt template.
        """
        missing_variables = [var for var in self.required_variables if var not in kwargs]
        if missing_variables:
            missing_vars_str = ", ".join(missing_variables)
            raise ValueError(f"Missing required input variables in PromptBuilder: {missing_vars_str}")

        return {"prompt": self.template.render(kwargs)}
