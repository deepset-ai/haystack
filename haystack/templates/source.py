import re
from enum import Enum
from pathlib import Path
from typing import Union

import requests

TEMPLATE_FILE_EXTENSION = ".yaml.jinja2"
TEMPLATE_HOME_DIR = Path(__file__).resolve().parent


class PredefinedTemplate(Enum):
    """
    Enumeration of predefined pipeline templates that can be used to create a `PipelineTemplate` using `TemplateSource`.
    See `TemplateSource.from_predefined` for usage.
    """

    # maintain 1-to-1 mapping between the enum name and the template file name in templates directory
    QA = "qa"
    RAG = "rag"
    INDEXING = "indexing"


class TemplateSource:
    """
    TemplateSource loads template content from various inputs, including strings, files, predefined templates, and URLs.
    The class provides mechanisms to load templates dynamically and ensure they contain valid Jinja2 syntax.

    TemplateSource is used by `PipelineTemplate` to load pipeline templates from various sources.
    For example:
    ```python
    # Load a predefined indexing pipeline template
    ts = TemplateSource.from_predefined(PredefinedTemplate.INDEXING)
    pipeline = PipelineTemplate(ts)

    # Load a custom pipeline template from a file
    ts = TemplateSource.from_file("path/to/custom_template.yaml.jinja2")
    pipeline = PipelineTemplate(ts)
    ```

    Similar methods are available to load templates from strings and URLs.
    """

    def __init__(self, template: str):
        """
        Initialize a TemplateSource.
        :param template: The template string to use.
        """
        self._template = template

    @classmethod
    def from_str(cls, template_str: str) -> "TemplateSource":
        """
        Create a TemplateSource from a string.
        :param template_str: The template string to use. Must contain valid Jinja2 syntax.
        :return: An instance of `TemplateSource`.
        """
        if not cls._contains_jinja2_syntax(template_str):
            raise ValueError("The provided template does not contain Jinja2 syntax.")
        return cls(template_str)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "TemplateSource":
        """
        Create a TemplateSource from a file.
        :param file_path: The path to the file containing the template. Must contain valid Jinja2 syntax.
        :return: An instance of `TemplateSource`.
        """
        with open(file_path, "r") as file:
            return cls.from_str(file.read())

    @classmethod
    def from_predefined(cls, predefined_template: PredefinedTemplate) -> "TemplateSource":
        """
        Create a TemplateSource from a predefined template. See `PredefinedTemplate` for available options.
        :param predefined_template: The name of the predefined template to use.
        :return: An instance of `TemplateSource`.
        """
        template_path = f"{TEMPLATE_HOME_DIR}/{predefined_template.value}{TEMPLATE_FILE_EXTENSION}"
        return cls.from_file(template_path)

    @classmethod
    def from_url(cls, url: str) -> "TemplateSource":
        """
        Create a TemplateSource from a URL.
        :param url: The URL to fetch the template from. Must contain valid Jinja2 syntax.
        :return: An instance of `TemplateSource`.
        """
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return cls.from_str(response.text)

    @property
    def template(self) -> str:
        """
        Returns the raw template string as a read-only property.
        """
        return self._template

    @staticmethod
    def _contains_jinja2_syntax(potential_jinja_template: str) -> bool:
        """
        Determines if a given string contains Jinja2 templating syntax.

        :param potential_jinja_template: The string to check for Jinja2 syntax.

        :return: `True` if Jinja2 syntax is found, otherwise `False`.
        """
        # Patterns to look for: {{ var }}, {% block %}, {# comment #}
        patterns = [r"\{\{.*?\}\}", r"\{%.*?%\}", r"\{#.*?#\}"]
        combined_pattern = re.compile("|".join(patterns))
        return bool(combined_pattern.search(potential_jinja_template))
