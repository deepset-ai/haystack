# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from jinja2 import Environment, nodes


def extract_declared_variables(template_str: str, env: Optional[Environment] = None) -> set:
    """
    Extract declared variables from a Jinja2 template string.

    Args:
        template_str (str): The Jinja2 template string to analyze.
        env (Environment, optional): The Jinja2 Environment. Defaults to None.

    Returns:
        A list of variable names used in the template.
    """
    env = env or Environment()

    try:
        ast = env.parse(template_str)
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
