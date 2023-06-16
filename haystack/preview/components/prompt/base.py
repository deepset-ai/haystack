from typing import Any
from dataclasses import make_dataclass, asdict, is_dataclass

from banks import Prompt
from banks.env import with_env
from jinja2 import meta

from haystack.preview import ComponentInput


@with_env
class PromptInputMixin:
    @property
    def input_type(self):
        ast = PromptInputMixin.env.parse(self.prompt)
        fields = meta.find_undeclared_variables(ast)
        return make_dataclass("Input", fields=[(f, Any) for f in fields], bases=(ComponentInput,))

    def _render_prompt(self, data) -> str:
        variables = {}
        if is_dataclass(data):
            data_dict = asdict(data)
            for field, value in data_dict.items():
                if value is None:
                    raise ValueError(f"{field} is None")
            variables = data_dict
        return Prompt(self.prompt).text(variables)
