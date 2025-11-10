# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from jinja2 import Environment, nodes


class Jinja2TemplateVariableExtractor:
    """
    A utility class for extracting declared variables from Jinja2 templates.
    """

    def __init__(self, env: Optional[Environment] = None):
        self.env = env or Environment()

    def _extract_from_text(self, template_str: Optional[str], role: Optional[str] = None) -> set[str]:
        """
        Extract declared variables from a Jinja2 template string.

        :param template_str: The Jinja2 template string to analyze.
        :param env: The Jinja2 Environment. Defaults to None.

        :returns:
        A set of variable names used in the template.
        """
        try:
            ast = self.env.parse(template_str)
        except Exception as e:
            raise RuntimeError(f"Failed to parse Jinja2 template: {e}")

        # Collect all variables assigned inside the template via {% set %}
        assigned_variables = set()

        for node in ast.find_all(nodes.Assign):
            if isinstance(node.target, nodes.Name):
                assigned_variables.add(node.target.name)
            elif isinstance(node.target, (nodes.List, nodes.Tuple)):
                for name_node in node.target.items:
                    if isinstance(name_node, nodes.Name):
                        assigned_variables.add(name_node.name)

        return assigned_variables
