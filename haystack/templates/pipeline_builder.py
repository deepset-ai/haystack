import re
from pathlib import Path
from typing import Dict, Any, Set

import yaml
from jinja2 import meta, TemplateSyntaxError
from jinja2.nativetypes import NativeEnvironment

from haystack import default_to_dict, Pipeline
from haystack.core.component import Component
from haystack.core.errors import PipelineValidationError


def ternary_filter(condition, true_val, false_val):
    return true_val if condition else false_val


class PipelineTemplateBuilder:
    template_file_extension = ".yaml.jinja2"

    def __init__(self, pipeline_template: str):
        if not pipeline_template:
            raise ValueError(
                "pipeline_name is required and has to be either:\n"
                "a) predefined pipeline type (e.g. 'rag', 'indexing', etc.)\n"
                "b) custom jinja2 template defining the pipeline itself.\n"
                f"Available pipeline types are: {self.list_templates()}"
            )
        available_templates = self.list_templates()
        if pipeline_template not in available_templates and not self.contains_jinja2_syntax(pipeline_template):
            raise ValueError(
                f"Pipeline template '{pipeline_template}' not found. Available pipeline types are: {available_templates}.\n"
                "Users can also provide a custom jinja2 template defining the pipeline itself."
            )
        if pipeline_template in available_templates:
            template_path = f"{Path(__file__).resolve().parent}/{pipeline_template}{self.template_file_extension}"
            if not Path(template_path).exists():
                raise ValueError(f"Pipeline template '{template_path}' not found.")
            self.template_path = template_path

            with open(template_path, "r") as file:
                self.template_text = file.read()
        else:
            self.template_path = None
            self.template_text = pipeline_template

        env = NativeEnvironment()
        env.filters["ternary"] = ternary_filter
        try:
            self.template = env.from_string(self.template_text)
        except TemplateSyntaxError as e:
            raise ValueError(
                f"Invalid pipeline template '{(self.template_path if self.template_path else self.template_text)}': {e}"
            ) from e
        self.templated_components = self._extract_variables(env)
        self.components = {}
        self.user_kwargs = {}

    def with_component(self, component_name: str, component_instance):
        # check if the component_name is allowed in the template
        if component_name not in self.templated_components:
            raise PipelineValidationError(
                f"Component '{component_name}' is not defined in the pipeline template '{self.template_path}'."
            )
        if not isinstance(component_instance, Component):
            raise PipelineValidationError(
                f"'{type(component_instance)}' doesn't seem to be a component. Is this class decorated with @component?"
            )
        component_dict: Dict[str, Any]
        if hasattr(component_instance, "to_dict"):
            component_dict = component_instance.to_dict()
        else:
            component_dict = default_to_dict(component_instance)
        self.components[component_name] = component_dict
        return self

    def with_kwargs(self, **kwargs):
        self.user_kwargs = kwargs or {}
        return self

    def build(self):
        rendered_yaml = self.template.render(**self.components, **self.user_kwargs)
        pipeline_yaml = yaml.safe_load(rendered_yaml)
        return Pipeline.from_dict(pipeline_yaml)

    def _extract_variables(self, env: NativeEnvironment) -> Set[str]:
        """
        Extracts all variables from a list of Jinja template strings.
        :param env: A Jinja native environment.
        :return: A set of variable names extracted from the template strings.
        """
        variables = set()
        ast = env.parse(self.template_text)
        variables.update(meta.find_undeclared_variables(ast))
        return variables

    def list_templates(self):
        directory = Path(__file__).resolve().parent
        jinja_files = [f for f in directory.iterdir() if f.is_file() and f.name.endswith(self.template_file_extension)]
        correct_template_names = [f.name.rsplit(self.template_file_extension, 1)[0] for f in jinja_files]
        return correct_template_names

    @staticmethod
    def contains_jinja2_syntax(s):
        # Patterns to look for: {{ var }}, {% block %}, {# comment #}
        patterns = [r"\{\{.*?\}\}", r"\{%.*?%\}", r"\{#.*?#\}"]
        combined_pattern = re.compile("|".join(patterns))
        return bool(combined_pattern.search(s))
