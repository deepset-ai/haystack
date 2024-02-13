from pathlib import Path
from typing import Dict, Any, Set

import yaml
from jinja2.nativetypes import NativeEnvironment

from haystack import default_to_dict, Pipeline
from haystack.core.errors import PipelineValidationError
from jinja2 import Template, meta, TemplateSyntaxError

from haystack.core.component import Component


class PipelineTemplateBuilder:
    def __init__(self, template_path: str):
        self.components = {}
        with open(template_path, "r") as file:
            self.template_text = file.read()
        env = NativeEnvironment()
        try:
            env.parse(self.template_text)
            self.template = Template(self.template_text)
        except TemplateSyntaxError as e:
            raise ValueError(f"Invalid pipeline template '{template_path}': {e}") from e
        self.templated_components = self._extract_variables(env)

    @classmethod
    def for_type(cls, pipeline_type: str):
        # perhaps try a few loading options here like this default relative path template file resolution
        template_path = f"{Path(__file__).resolve().parent}/{pipeline_type}.yaml"
        return cls(template_path)

    def with_component(self, component_name: str, component_instance):
        # check if the component_name is allowed in the template
        if component_name not in self.templated_components:
            raise PipelineValidationError(
                f"Component '{component_name}' is not allowed in the template '{self.template}'."
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

    def build(self):
        rendered_yaml = self.template.render(**self.components)
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
