from typing import Dict, Any

from jinja2 import Template, meta

from haystack.preview import component
from haystack.preview import default_to_dict


@component
class PromptBuilder:
    """
    PromptBuilder is a component that renders a prompt from a template string using Jinja2 engine.
    The template variables found in the template string are used as input types for the component and are all required.

    Usage:
    ```python
    template = "Translate the following context to {{ target_language }}. Context: {{ snippet }}; Translation:"
    builder = PromptBuilder(template=template)
    builder.run(target_language="spanish", snippet="I can't speak spanish.")
    ```
    """

    def __init__(self, template: str):
        """
        Initialize the component with a template string.

        :param template: Jinja2 template string, e.g. "Summarize this document: {documents}\nSummary:"
        :type template: str
        """
        self._template_string = template
        self.template = Template(template)
        ast = self.template.environment.parse(template)
        template_variables = meta.find_undeclared_variables(ast)
        component.set_input_types(self, **{var: Any for var in template_variables})

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, template=self._template_string)

    @component.output_types(prompt=str)
    def run(self, **kwargs):
        return {"prompt": self.template.render(kwargs)}
