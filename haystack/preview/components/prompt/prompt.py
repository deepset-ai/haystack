import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Literal

from haystack.preview import component
from haystack.preview.components.prompt.models.base import get_model


logger = logging.getLogger(__name__)


PromptTemplates = Literal["question-answering",]


IMPLEMENTATION_MODULES = ["haystack.preview"]


PREDEFINED_TEMPLATES = {
    "question-answering": "Given the context please answer the question. Context: {{ documents }}; Question: {{ question }}; Answer:"
}


@component
class Prompt:
    """
    The Prompt class is the central abstraction in Haystack's large language model (LLM) support. Prompt
    supports multiple NLP tasks out of the box. You can use it to perform tasks, such as
    summarization, question answering, question generation, and more, using a single, unified model within the Haystack
    framework.

    Prompt also supports multiple model invocation layers:
    - Hugging Face transformers (all text2text-generation models)
    - OpenAI InstructGPT models
    - Azure OpenAI InstructGPT models

    We recommend using LLMs fine-tuned on a collection of datasets phrased as instructions, otherwise we find that the
    LLM does not "follow" prompt instructions well. This is why we recommend using T5 flan or OpenAI InstructGPT models.

    For more details, see [Prompt](https://docs.haystack.deepset.ai/docs/prompt_node).
    """

    def __init__(
        self,
        template: Optional[PromptTemplates] = None,
        custom_template: Optional[str] = None,
        output: Optional[str] = "answers",
        model_name_or_path: str = "google/flan-t5-base",
        model_implementation: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        implementation_modules: Optional[List[str]] = None,
    ):
        """
        Creates a Prompt instance.

        The inputs of this component correspond to the variables of the template you provide.
        For example, if your prompt template expects `{{ question }}` and `{{ documents }}`, the inputs this component
        expects are `['question', 'documents']`. Inputs are automatically inferred from the template text.

        :param outputs: the outputs that this component will generate.
        :param template: The template of the prompt. Must be the name of a predefined prompt template. To provide a
            custom template, use `custom_template`.
        :param custom_template: A custom template of the prompt. To use Haystack's pre-defined templates, use `template`.
        :param model_name: The name of the model to use, like a HF model identifier or an OpenAI model name.
        :param model_implementation: force a specific implementation for the model
            (see `haystack.preview.components.prompt.models`). If not given, Haystack will find an implementation for
            your model automatically.
        :param model_params: Parameters to be passed to the model implementation, like API keys, init parameters, etc.
        :param implementation_modules: if you have external model implementations, add the module where they can be
            imported from, like `haystack.preview`.
        """
        if template:
            self.template = PREDEFINED_TEMPLATES[template]
        elif custom_template:
            self.template = custom_template
        else:
            raise ValueError("Provide either a template or a custom_template for this Prompt.")

        self.inputs = self.find_inputs()
        self.outputs = [output]
        self.model = None
        self.model_name_or_path = model_name_or_path
        self.model_implementation = model_implementation
        self.model_params = model_params
        self.implementation_modules = implementation_modules or IMPLEMENTATION_MODULES

    def warm_up(self):
        if not self.model:
            self.model = get_model(
                model_name_or_path=self.model_name_or_path,
                model_implementation=self.model_implementation,
                model_params=self.model_params,
                modules_to_search=self.implementation_modules,
            )

    def run(
        self, name: str, data: List[Tuple[str, Any]], parameters: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        It takes in a prompt, and returns a list of responses using the underlying invocation layer.

        :param prompt: The prompt to use for the invocation. It can be a single prompt or a list of prompts.
        :param kwargs: Additional keyword arguments to pass to the invocation layer.
        :return: A list of model generated responses for the prompt or prompts.
        """
        data = {key: value for key, value in data}
        params = parameters.get(name, {})
        model_params = params.pop("model_params", {})
        output = self.prompt(validate_inputs=False, model_params=model_params, **data, **params)
        return ({self.outputs[0]: output}, parameters)

    def prompt(
        self, validate_inputs: bool = True, model_params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[str]:
        """
        It takes in a prompt, and returns a list of responses using the underlying model.

        :param prompt: The prompt to use for the invocation.
        :param kwargs: the variables to be rendered into the prompt.
        :return: A list of model generated responses for the prompt.
        """
        self.warm_up()
        if validate_inputs:
            self.validate_inputs(given_inputs=list(kwargs.keys()))

        prompt = self.template
        for placeholder in re.findall(r"\{\{\s*[a-zA-Z0-9_]*\s*\}\}", prompt):  # Note that () were removed
            value_name = placeholder[2:-2].strip()
            prompt.replace(placeholder, kwargs.get(value_name))
        return self.model.invoke(prompt=prompt, model_params=model_params or {})

    def find_inputs(self) -> List[str]:
        """
        Find out which inputs this template requires.

        :param template: the template this Prompt uses.
        :returns: a list of strings corresponding to the variables found in the template.
        """
        return re.findall(r"\{\{\s*([a-zA-Z0-9_]*)\s*\}\}", self.template)

    def validate_inputs(self, given_inputs: List[str]) -> None:
        """
        Make sure the template variables and the input variables match.

        :param given_inputs: the inputs to validate
        :raises ValueError if the template is using variables that are not found
            in the inputs list or vice versa.
        """
        if sorted(self.inputs) != sorted(given_inputs):
            raise ValueError(
                "The values given to this Prompt do not match the variables it expects:\n- Expected variables: "
                f"{self.inputs}\n- Given variables: {given_inputs}\nConnect this node to another node that outputs "
                "these variables, pass this value to `Prompt.prompt()`, or change template."
            )
