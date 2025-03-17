# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional

from haystack.core.component import component


@component
class FString:
    """
    Takes a template string and a list of variables in input and returns the formatted string in output.
    """

    def __init__(self, template: str, variables: Optional[List[str]] = None):
        self.template = template
        self.variables = variables or []
        if "template" in self.variables:
            raise ValueError("The variable name 'template' is reserved and cannot be used.")
        component.set_input_types(self, **dict.fromkeys(self.variables, Any))

    @component.output_types(string=str)
    def run(self, template: Optional[str] = None, **kwargs):
        """
        Takes a template string and a list of variables in input and returns the formatted string in output.

        If the template is not given, the component will use the one given at initialization.
        """
        if not template:
            template = self.template
        return {"string": template.format(**kwargs)}
