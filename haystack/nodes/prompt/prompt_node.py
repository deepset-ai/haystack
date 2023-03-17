import ast
from collections import defaultdict
import copy
import logging
from abc import ABC
import os
import re
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Type
from uuid import uuid4

import torch
import yaml

from haystack import MultiLabel
from haystack.environment import HAYSTACK_PROMPT_TEMPLATE_ALLOWED_FUNCTIONS
from haystack.errors import NodeError
from haystack.nodes.base import BaseComponent
from haystack.nodes.other.shaper import (  # pylint: disable=unused-import
    Shaper,
    join_documents_to_string as join,  # used as shaping function
    format_document,
    format_answer,
    format_string,
)
from haystack.nodes.prompt.providers import PromptModelInvocationLayer
from haystack.schema import Answer, Document
from haystack.telemetry_2 import send_event

logger = logging.getLogger(__name__)


class BasePromptTemplate(BaseComponent):
    outgoing_edges = 1

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        raise NotImplementedError("This method should never be implemented in the derived class")

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        raise NotImplementedError("This method should never be implemented in the derived class")


PROMPT_TEMPLATE_ALLOWED_FUNCTIONS = ast.literal_eval(
    os.environ.get(HAYSTACK_PROMPT_TEMPLATE_ALLOWED_FUNCTIONS, '["join", "to_strings", "replace", "enumerate", "str"]')
)
PROMPT_TEMPLATE_SPECIAL_CHAR_ALIAS = {"new_line": "\n", "tab": "\t", "double_quote": '"', "carriage_return": "\r"}
PROMPT_TEMPLATE_STRIPS = ["'", '"']
PROMPT_TEMPLATE_STR_REPLACE = {'"': "'"}


def to_strings(items: List[Union[str, Document, Answer]], pattern=None, str_replace=None) -> List[str]:
    results = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            results.append(format_string(item, str_replace=str_replace))
        elif isinstance(item, Document):
            results.append(format_document(document=item, pattern=pattern, str_replace=str_replace, idx=idx))
        elif isinstance(item, Answer):
            results.append(format_answer(answer=item, pattern=pattern, str_replace=str_replace, idx=idx))
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")
    return results


class PromptTemplateValidationError(NodeError):
    """
    Error raised when a prompt template is invalid.
    """

    pass


class _ValidationVisitor(ast.NodeVisitor):
    """
    This class is used to validate the prompt text for a prompt template.
    It checks that the prompt text is a valid f-string and that it only uses allowed functions.
    Useful information extracted from the AST is stored in the class attributes (e.g. `prompt_params` and `used_functions`)
    """

    def __init__(self, prompt_template_name: str):
        self.used_names: List[str] = []
        self.comprehension_targets: List[str] = []
        self.used_functions: List[str] = []
        self.prompt_template_name = prompt_template_name

    @property
    def prompt_params(self) -> List[str]:
        """
        The names of the variables used in the prompt text.
        E.g. for the prompt text `f"Hello {name}"`, the prompt_params would be `["name"]`
        """
        return list(set(self.used_names) - set(self.used_functions) - set(self.comprehension_targets))

    def visit_Name(self, node: ast.Name) -> None:
        """
        Stores the name of the variable used in the prompt text. This also includes function and method names.
        E.g. for the prompt text `f"Hello {func(name)}"`, the used_names would be `["func", "name"]`
        """
        self.used_names.append(node.id)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        """
        Stores the name of the variable used in comprehensions.
        E.g. for the prompt text `f"Hello {[name for name in names]}"`, the comprehension_targets would be `["name"]`
        """
        super().generic_visit(node)
        if isinstance(node.target, ast.Name):
            self.comprehension_targets.append(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            self.comprehension_targets.extend([elt.id for elt in node.target.elts if isinstance(elt, ast.Name)])

    def visit_Call(self, node: ast.Call) -> None:
        """
        Stores the name of functions and methods used in the prompt text and validates that only allowed functions are used.
        E.g. for the prompt text `f"Hello {func(name)}"`, the used_functions would be `["func"]`

        raises: PromptTemplateValidationError if an invalid function is used in the prompt text
        """
        super().generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in PROMPT_TEMPLATE_ALLOWED_FUNCTIONS:
            # functions: func(args, kwargs)
            self.used_functions.append(node.func.id)
        elif isinstance(node.func, ast.Attribute) and node.func.attr in PROMPT_TEMPLATE_ALLOWED_FUNCTIONS:
            # methods: instance.method(args, kwargs)
            self.used_functions.append(node.func.attr)
        else:
            raise PromptTemplateValidationError(
                f"Invalid function in prompt text for prompt template {self.prompt_template_name}. "
                f"Allowed functions are {PROMPT_TEMPLATE_ALLOWED_FUNCTIONS}."
            )


class _FstringParamsTransformer(ast.NodeTransformer):
    """
    This class is used to transform an AST for f-strings into a format that can be used by the PromptTemplate.
    It replaces all f-string expressions with a unique id and stores the corresponding expression in a dictionary.

    The stored expressions can be evaluated using the `eval` function given the `prompt_params` (see _ValidatorVisitor) .
    PromptTemplate determines the number of prompts to generate and renders them using the evaluated expressions.
    """

    def __init__(self):
        self.prompt_params_functions: Dict[str, ast.Expression] = {}

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Optional[ast.AST]:
        """
        Replaces the f-string expression with a unique id and stores the corresponding expression in a dictionary.
        If the expression is the raw `documents` variable, it is encapsulated into a call to `documents_to_strings` to ensure that the documents get rendered correctly.
        """
        super().generic_visit(node)

        # Keep special char variables as is. They are available via globals.
        if isinstance(node.value, ast.Name) and node.value.id in PROMPT_TEMPLATE_SPECIAL_CHAR_ALIAS:
            return node

        id = uuid4().hex
        if isinstance(node.value, ast.Name) and node.value.id in ["documents", "answers"]:
            call = ast.Call(func=ast.Name(id="to_strings", ctx=ast.Load()), args=[node.value], keywords=[])
            self.prompt_params_functions[id] = ast.fix_missing_locations(ast.Expression(body=call))
        else:
            self.prompt_params_functions[id] = ast.fix_missing_locations(ast.Expression(body=node.value))
        return ast.FormattedValue(
            value=ast.Name(id=id, ctx=ast.Load()), conversion=node.conversion, format_spec=node.format_spec
        )


class PromptTemplate(BasePromptTemplate, ABC):
    """
    PromptTemplate is a template for a prompt you feed to the model to instruct it what to do. For example, if you want the model to perform sentiment analysis, you simply tell it to do that in a prompt. Here's what such prompt template may look like:

    ```python
        PromptTemplate(name="sentiment-analysis",
                   prompt_text="Give a sentiment for this context. Answer with positive, negative
                   or neutral. Context: {documents}; Answer:")
    ```

    Optionally, you can declare prompt parameters using f-string syntax in the PromptTemplate. Prompt parameters are input parameters that need to be filled in
    the prompt_text for the model to perform the task. For example, in the template above, there's one prompt parameter, `documents`. You declare prompt parameters by adding variables to the prompt text. These variables should be in the format: `{variable}`. In the template above, the variable is `{documents}`.

    At runtime, these variables are filled in with arguments passed to the `fill()` method of the PromptTemplate. So in the example above, the `{documents}` variable will be filled with the Documents whose sentiment you want the model to analyze.

    Note that other than strict f-string syntax, you can safely use the following backslash characters in text parts of the prompt text: `\n`, `\t`, `\r`.
    If you want to use them in f-string expressions, use `new_line`, `tab`, `carriage_return` instead.
    Double quotes (e.g. `"`) will be automatically replaced with single quotes (e.g. `'`) in the prompt text. If you want to use double quotes in the prompt text, use `{double_quote}` instead.

    For more details on how to use PromptTemplate, see
    [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node).
    """

    def __init__(
        self, name: str, prompt_text: str, output_shapers: Optional[Union[List[Shaper], List[Dict[str, Any]]]] = None
    ):
        """
         Creates a PromptTemplate instance.

        :param name: The name of the prompt template (for example, sentiment-analysis, question-generation). You can specify your own name but it must be unique.
        :param prompt_text: The prompt text, including prompt parameters.
        :param output_shapers: A list of shapers that will be applied to the output of the model.
                For example, if you want to convert the output of the model to a Answer object, you can use a shaper using the `string_to_answer` function.
                Note, that the last shaper in the list must only have one output and PromptNode will use this value as the output_variable.
        """
        super().__init__()

        # use case when PromptTemplate is loaded from a YAML file, we need to start and end the prompt text with quotes
        for strip in PROMPT_TEMPLATE_STRIPS:
            prompt_text = prompt_text.strip(strip)
        replacements = {
            **{v: "{" + k + "}" for k, v in PROMPT_TEMPLATE_SPECIAL_CHAR_ALIAS.items()},
            **PROMPT_TEMPLATE_STR_REPLACE,
        }
        for old, new in replacements.items():
            prompt_text = prompt_text.replace(old, new)

        self._ast_expression = ast.parse(f'f"{prompt_text}"', mode="eval")

        ast_validator = _ValidationVisitor(prompt_template_name=name)
        ast_validator.visit(self._ast_expression)

        ast_transformer = _FstringParamsTransformer()
        self._ast_expression = ast.fix_missing_locations(ast_transformer.visit(self._ast_expression))
        self._prompt_params_functions = ast_transformer.prompt_params_functions
        self._used_functions = ast_validator.used_functions

        self.name = name
        self.prompt_text = prompt_text
        self.prompt_params = sorted(
            param for param in ast_validator.prompt_params if param not in PROMPT_TEMPLATE_SPECIAL_CHAR_ALIAS
        )
        self.globals = {
            **{k: v for k, v in globals().items() if k in PROMPT_TEMPLATE_ALLOWED_FUNCTIONS},
            **PROMPT_TEMPLATE_SPECIAL_CHAR_ALIAS,
        }
        self.output_shapers: List[Shaper] = []
        if output_shapers:
            for shaper in output_shapers:
                if isinstance(shaper, dict):
                    self.output_shapers.append(Shaper(**shaper))
                elif isinstance(shaper, Shaper):
                    self.output_shapers.append(shaper)
                else:
                    raise TypeError(
                        f"Expected output_shapers to be a list of dicts or Shapers, but got {type(shaper)}."
                    )
        if self.output_shapers and len(self.output_shapers[-1].outputs) != 1:
            raise PromptTemplateValidationError(
                f"The last shaper in the output shapers for prompt template {self.name} must have only one output."
            )

    @property
    def output_variable(self) -> Optional[str]:
        return self.output_shapers[-1].outputs[0] if self.output_shapers else None

    def prepare(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Prepares and verifies the prompt template with input parameters.

        :param args: Non-keyword arguments to fill the parameters in the prompt text of a PromptTemplate.
        :param kwargs: Keyword arguments to fill the parameters in the prompt text of a PromptTemplate.
        :return: A dictionary with the prompt text and the prompt parameters.
        """
        params_dict = {}
        # attempt to resolve args first
        if args:
            if len(args) != len(self.prompt_params):
                logger.warning(
                    "For %s, expected %s arguments, instead got %s arguments %s",
                    self.name,
                    self.prompt_params,
                    len(args),
                    args,
                )
            for prompt_param, arg in zip(self.prompt_params, args):
                params_dict[prompt_param] = [arg] if isinstance(arg, str) else arg
        # then attempt to resolve kwargs
        if kwargs:
            for param in self.prompt_params:
                if param in kwargs:
                    params_dict[param] = kwargs[param]

        if set(params_dict.keys()) != set(self.prompt_params):
            available_params = set(list(params_dict.keys()) + list(set(kwargs.keys())))
            raise ValueError(f"Expected prompt parameters {self.prompt_params} but got {list(available_params)}.")

        template_dict = {"_at_least_one_prompt": True}
        for id, call in self._prompt_params_functions.items():
            template_dict[id] = eval(  # pylint: disable=eval-used
                compile(call, filename="<string>", mode="eval"), self.globals, params_dict
            )

        return template_dict

    def post_process(self, prompt_output: List[str], **kwargs) -> List[Any]:
        """
        Post-processes the output of the prompt template.
        :param args: Non-keyword arguments to use for post-processing the prompt output.
        :param kwargs: Keyword arguments to use for post-processing the prompt output.
        :return: A dictionary with the post-processed output.
        """
        if self.output_shapers:
            invocation_context = kwargs
            invocation_context["results"] = prompt_output
            for shaper in self.output_shapers:
                shaper.run(invocation_context=invocation_context)
            return invocation_context[shaper.outputs[0]]
        else:
            return prompt_output

    def fill(self, *args, **kwargs) -> Iterator[str]:
        """
        Fills the parameters defined in the prompt text with the arguments passed to it and returns the iterator prompt text.

        You can pass non-keyword (args) or keyword (kwargs) arguments to this method. If you pass non-keyword arguments, their order must match the left-to-right
        order of appearance of the parameters in the prompt text. For example, if the prompt text is:
        `Come up with a question for the given context and the answer. Context: {documents};
        Answer: {answers}; Question:`, then the first non-keyword argument fills the `{documents}` variable
        and the second non-keyword argument fills the `{answers}` variable.

        If you pass keyword arguments, the order of the arguments doesn't matter. Variables in the
        prompt text are filled with the corresponding keyword argument.

        :param args: Non-keyword arguments to fill the parameters in the prompt text. Their order must match the order of appearance of the parameters in prompt text.
        :param kwargs: Keyword arguments to fill the parameters in the prompt text.
        :return: An iterator of prompt texts.
        """
        template_dict = self.prepare(*args, **kwargs)

        # the prompt context values should all be lists, as they will be split as one
        prompt_context_copy = {k: v if isinstance(v, list) else [v] for k, v in template_dict.items()}
        max_len = max(len(v) for v in prompt_context_copy.values())
        if max_len > 1:
            for key, value in prompt_context_copy.items():
                if len(value) == 1:
                    prompt_context_copy[key] = value * max_len

        for prompt_context_values in zip(*prompt_context_copy.values()):
            template_input = {key: prompt_context_values[idx] for idx, key in enumerate(prompt_context_copy.keys())}
            prompt_prepared: str = eval(  # pylint: disable=eval-used
                compile(self._ast_expression, filename="<string>", mode="eval"), self.globals, template_input
            )
            yield prompt_prepared

    def __repr__(self):
        return f"PromptTemplate(name={self.name}, prompt_text={self.prompt_text}, prompt_params={self.prompt_params})"


class PromptModel(BaseComponent):
    """
    The PromptModel class is a component that uses a pre-trained model to perform tasks based on a prompt. Out of
    the box, it supports model invocation layers for:
    - Hugging Face transformers (all text2text-generation models)
    - OpenAI InstructGPT models
    - Azure OpenAI InstructGPT models

    Although it is possible to use PromptModel to make prompt invocations on the underlying model, use
    PromptNode to interact with the model. PromptModel instances are a way for multiple
    PromptNode instances to use a single PromptNode, and thus save computational resources.

    For more details, refer to [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node).
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-base",
        max_length: Optional[int] = 100,
        api_key: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
        invocation_layer_class: Optional[Type[PromptModelInvocationLayer]] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        """
        Creates an instance of PromptModel.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum length of the output text generated by the model.
        :param api_key: The API key to use for the model.
        :param use_auth_token: The Hugging Face token to use.
        :param use_gpu: Whether to use GPU or not.
        :param devices: The devices to use where the model is loaded.
        :param invocation_layer_class: The custom invocation layer class to use. If None, known invocation layers are used.
        :param model_kwargs: Additional keyword arguments passed to the underlying model.

        Note that Azure OpenAI InstructGPT models require two additional parameters: azure_base_url (The URL for the
        Azure OpenAI API endpoint, usually in the form `https://<your-endpoint>.openai.azure.com') and
        azure_deployment_name (The name of the Azure OpenAI API deployment). These parameters should be supplied
        in the `model_kwargs` dictionary.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.api_key = api_key
        self.use_auth_token = use_auth_token
        self.use_gpu = use_gpu
        self.devices = devices

        self.model_kwargs = model_kwargs if model_kwargs else {}
        self.model_invocation_layer = self.create_invocation_layer(invocation_layer_class=invocation_layer_class)

    def create_invocation_layer(
        self, invocation_layer_class: Optional[Type[PromptModelInvocationLayer]]
    ) -> PromptModelInvocationLayer:
        kwargs = {
            "api_key": self.api_key,
            "use_auth_token": self.use_auth_token,
            "use_gpu": self.use_gpu,
            "devices": self.devices,
        }
        all_kwargs = {**self.model_kwargs, **kwargs}

        if invocation_layer_class:
            return invocation_layer_class(
                model_name_or_path=self.model_name_or_path, max_length=self.max_length, **all_kwargs
            )
        # search all invocation layer classes and find the first one that supports the model,
        # then create an instance of that invocation layer
        for invocation_layer in PromptModelInvocationLayer.invocation_layer_providers:
            if invocation_layer.supports(self.model_name_or_path, **all_kwargs):
                return invocation_layer(
                    model_name_or_path=self.model_name_or_path, max_length=self.max_length, **all_kwargs
                )
        raise ValueError(
            f"Model {self.model_name_or_path} is not supported - no matching invocation layer found."
            f" Currently supported invocation layers are: {PromptModelInvocationLayer.invocation_layer_providers}"
            f" You can implement and provide custom invocation layer for {self.model_name_or_path} by subclassing "
            "PromptModelInvocationLayer."
        )

    def invoke(self, prompt: Union[str, List[str]], **kwargs) -> List[str]:
        """
        It takes in a prompt, and returns a list of responses using the underlying invocation layer.

        :param prompt: The prompt to use for the invocation. It can be a single prompt or a list of prompts.
        :param kwargs: Additional keyword arguments to pass to the invocation layer.
        :return: A list of model generated responses for the prompt or prompts.
        """
        output = self.model_invocation_layer.invoke(prompt=prompt, **kwargs)
        return output

    def _ensure_token_limit(self, prompt: str) -> str:
        """Ensure that length of the prompt and answer is within the maximum token length of the PromptModel.

        :param prompt: Prompt text to be sent to the generative model.
        """
        return self.model_invocation_layer._ensure_token_limit(prompt=prompt)

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:
        raise NotImplementedError("This method should never be implemented in the derived class")

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        raise NotImplementedError("This method should never be implemented in the derived class")

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)


def get_predefined_prompt_templates() -> List[PromptTemplate]:
    return [
        PromptTemplate(
            name="question-answering",
            prompt_text="Given the context please answer the question. Context: {join(documents)}; Question: "
            "{query}; Answer:",
            output_shapers=[Shaper(func="strings_to_answers", outputs=["answers"], inputs={"strings": "results"})],
        ),
        PromptTemplate(
            name="question-answering-per-document",
            prompt_text="Given the context please answer the question. Context: {documents}; Question: "
            "{query}; Answer:",
            output_shapers=[Shaper(func="strings_to_answers", outputs=["answers"], inputs={"strings": "results"})],
        ),
        PromptTemplate(
            name="question-answering-with-references",
            prompt_text="Create a concise and informative answer (no more than 50 words) for a given question "
            "based solely on the given documents. You must only use information from the given documents. "
            "Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[number] notation. "
            "If multiple documents contain the answer, cite those documents like ‘as stated in Document[number], Document[number], etc.’. "
            "If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’\n"
            "{join(documents, delimiter=new_line, pattern=new_line+'Document[$idx]: $content', str_replace={new_line: ' ', '[': '(', ']': ')'})} \n Question: {query}; Answer: ",
            output_shapers=[
                Shaper(
                    func="strings_to_answers",
                    inputs={"strings": "results"},
                    outputs=["answers"],
                    params={"reference_pattern": r"Document\[(\d+)\]"},
                )
            ],
        ),
        PromptTemplate(
            name="question-generation",
            prompt_text="Given the context please generate a question. Context: {documents}; Question:",
            output_shapers=[Shaper(func="rename", outputs=["query"], inputs={"value": "results"})],
        ),
        PromptTemplate(
            name="conditioned-question-generation",
            prompt_text="Please come up with a question for the given context and the answer. "
            "Context: {documents}; Answer: {answers}; Question:",
            output_shapers=[Shaper(func="rename", outputs=["query"], inputs={"value": "results"})],
        ),
        PromptTemplate(name="summarization", prompt_text="Summarize this document: {documents} Summary:"),
        PromptTemplate(
            name="question-answering-check",
            prompt_text="Does the following context contain the answer to the question? "
            "Context: {documents}; Question: {query}; Please answer yes or no! Answer:",
            output_shapers=[Shaper(func="strings_to_answers", outputs=["answers"], inputs={"strings": "results"})],
        ),
        PromptTemplate(
            name="sentiment-analysis",
            prompt_text="Please give a sentiment for this context. Answer with positive, "
            "negative or neutral. Context: {documents}; Answer:",
        ),
        PromptTemplate(
            name="multiple-choice-question-answering",
            prompt_text="Question:{query} ; Choose the most suitable option to answer the above question. "
            "Options: {options}; Answer:",
            output_shapers=[Shaper(func="strings_to_answers", outputs=["answers"], inputs={"strings": "results"})],
        ),
        PromptTemplate(
            name="topic-classification",
            prompt_text="Categories: {options}; What category best describes: {documents}; Answer:",
        ),
        PromptTemplate(
            name="language-detection",
            prompt_text="Detect the language in the following context and answer with the "
            "name of the language. Context: {documents}; Answer:",
        ),
        PromptTemplate(
            name="translation",
            prompt_text="Translate the following context to {target_language}. Context: {documents}; Translation:",
        ),
        PromptTemplate(
            name="zero-shot-react",
            prompt_text="You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions "
            "correctly, you have access to the following tools:\n\n"
            "{tool_names_with_descriptions}\n\n"
            "To answer questions, you'll need to go through multiple steps involving step-by-step thinking and "
            "selecting appropriate tools and their inputs; tools will respond with observations. When you are ready "
            "for a final answer, respond with the `Final Answer:`\n\n"
            "Use the following format:\n\n"
            "Question: the question to be answered\n"
            "Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.\n"
            "Tool: pick one of {tool_names} \n"
            "Tool Input: the input for the tool\n"
            "Observation: the tool will respond with the result\n"
            "...\n"
            "Final Answer: the final answer to the question, make it short (1-5 words)\n\n"
            "Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass\n"
            "---\n\n"
            "Question: {query}\n"
            "Thought: Let's think step-by-step, I first need to ",
        ),
    ]


class PromptNode(BaseComponent):
    """
    The PromptNode class is the central abstraction in Haystack's large language model (LLM) support. PromptNode
    supports multiple NLP tasks out of the box. You can use it to perform tasks, such as
    summarization, question answering, question generation, and more, using a single, unified model within the Haystack
    framework.

    One of the benefits of PromptNode is that you can use it to define and add additional prompt templates
     the model supports. Defining additional prompt templates makes it possible to extend the model's capabilities
    and use it for a broader range of NLP tasks in Haystack. Prompt engineers define templates
    for each NLP task and register them with PromptNode. The burden of defining templates for each task rests on
    the prompt engineers, not the users.

    Using an instance of the PromptModel class, you can create multiple PromptNodes that share the same model, saving
    the memory and time required to load the model multiple times.

    PromptNode also supports multiple model invocation layers:
    - Hugging Face transformers (all text2text-generation models)
    - OpenAI InstructGPT models
    - Azure OpenAI InstructGPT models

    However, users are not limited to above-mentioned models only as there is a built-in ability to register
    additional custom model invocation layers.

    We recommend using LLMs fine-tuned on a collection of datasets phrased as instructions, otherwise we find that the
    LLM does not "follow" prompt instructions well. This is why we recommend using T5 flan or OpenAI InstructGPT models.

    For more details, see  [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node).
    """

    outgoing_edges: int = 1

    def __init__(
        self,
        model_name_or_path: Union[str, PromptModel] = "google/flan-t5-base",
        default_prompt_template: Optional[Union[str, PromptTemplate]] = None,
        output_variable: Optional[str] = None,
        max_length: Optional[int] = 100,
        api_key: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
        stop_words: Optional[List[str]] = None,
        top_k: int = 1,
        model_kwargs: Optional[Dict] = None,
    ):
        """
        Creates a PromptNode instance.

        :param model_name_or_path: The name of the model to use or an instance of the PromptModel.
        :param default_prompt_template: The default prompt template to use for the model.
        :param output_variable: The name of the output variable in which you want to store the inference results.
            If not set, PromptNode uses PromptTemplate's output_variable. If PromptTemplate's output_variable is not set, default name is `results`.
        :param max_length: The maximum length of the generated text output.
        :param api_key: The API key to use for the model.
        :param use_auth_token: The authentication token to use for the model.
        :param use_gpu: Whether to use GPU or not.
        :param devices: The devices to use for the model.
        :param top_k: The number of independently generated texts to return per prompt. For example, if you set top_k=3, the model's going to generate three answers to the query.
        :param stop_words: Stops text generation if any of the stop words is generated.
        :param model_kwargs: Additional keyword arguments passed when loading the model specified in `model_name_or_path`.

        Note that Azure OpenAI InstructGPT models require two additional parameters: azure_base_url (The URL for the
        Azure OpenAI API endpoint, usually in the form `https://<your-endpoint>.openai.azure.com') and
        azure_deployment_name (The name of the Azure OpenAI API deployment).
        These parameters should be supplied in the `model_kwargs` dictionary.

        """
        send_event("PromptNode initialized")
        super().__init__()
        self.prompt_templates: Dict[str, PromptTemplate] = {pt.name: pt for pt in get_predefined_prompt_templates()}  # type: ignore
        self.default_prompt_template: Union[str, PromptTemplate, None] = default_prompt_template
        self.output_variable: Optional[str] = output_variable
        self.model_name_or_path: Union[str, PromptModel] = model_name_or_path
        self.prompt_model: PromptModel
        self.stop_words: Optional[List[str]] = stop_words
        self.top_k: int = top_k
        if isinstance(self.default_prompt_template, str) and not self.is_supported_template(
            self.default_prompt_template
        ):
            raise ValueError(
                f"Prompt template {self.default_prompt_template} is not supported. "
                f"Select one of: {self.get_prompt_template_names()} "
                f"or first register a new prompt template using the add_prompt_template method."
            )

        if isinstance(model_name_or_path, str):
            self.prompt_model = PromptModel(
                model_name_or_path=model_name_or_path,
                max_length=max_length,
                api_key=api_key,
                use_auth_token=use_auth_token,
                use_gpu=use_gpu,
                devices=devices,
                model_kwargs=model_kwargs,
            )
        elif isinstance(model_name_or_path, PromptModel):
            self.prompt_model = model_name_or_path
        else:
            raise ValueError("model_name_or_path must be either a string or a PromptModel object")

    def __call__(self, *args, **kwargs) -> List[Any]:
        """
        This method is invoked when the component is called directly, for example:
        ```python
            PromptNode pn = ...
            sa = pn.set_default_prompt_template("sentiment-analysis")
            sa(documents=[Document("I am in love and I feel great!")])
        ```
        """
        if "prompt_template" in kwargs:
            prompt_template = kwargs["prompt_template"]
            kwargs.pop("prompt_template")
            return self.prompt(prompt_template, *args, **kwargs)
        else:
            return self.prompt(self.default_prompt_template, *args, **kwargs)

    def prompt(self, prompt_template: Optional[Union[str, PromptTemplate]], *args, **kwargs) -> List[Any]:
        """
        Prompts the model and represents the central API for the PromptNode. It takes a prompt template,
        a list of non-keyword and keyword arguments, and returns a list of strings - the responses from the underlying model.

        If you specify the optional prompt_template parameter, it takes precedence over the default PromptTemplate for this PromptNode.

        :param prompt_template: The name or object of the optional PromptTemplate to use.
        :return: A list of strings as model responses.
        """
        send_event("PromptNode.prompt()", event_properties={"template": str(prompt_template)})
        results = []
        # we pop the prompt_collector kwarg to avoid passing it to the model
        prompt_collector: List[str] = kwargs.pop("prompt_collector", [])

        # kwargs override model kwargs
        kwargs = {**self._prepare_model_kwargs(), **kwargs}
        template_to_fill = self.get_prompt_template(prompt_template)
        if template_to_fill:
            # prompt template used, yield prompts from inputs args
            for prompt in template_to_fill.fill(*args, **kwargs):
                kwargs_copy = copy.copy(kwargs)
                # and pass the prepared prompt and kwargs copy to the model
                prompt = self.prompt_model._ensure_token_limit(prompt)
                prompt_collector.append(prompt)
                logger.debug("Prompt being sent to LLM with prompt %s and kwargs %s", prompt, kwargs_copy)
                output = self.prompt_model.invoke(prompt, **kwargs_copy)
                results.extend(output)

            kwargs["prompts"] = prompt_collector
            results = template_to_fill.post_process(results, **kwargs)
        else:
            # straightforward prompt, no templates used
            for prompt in list(args):
                kwargs_copy = copy.copy(kwargs)
                prompt = self.prompt_model._ensure_token_limit(prompt)
                prompt_collector.append(prompt)
                logger.debug("Prompt being sent to LLM with prompt %s and kwargs %s ", prompt, kwargs_copy)
                output = self.prompt_model.invoke(prompt, **kwargs_copy)
                results.extend(output)
        return results

    def add_prompt_template(self, prompt_template: PromptTemplate) -> None:
        """
        Adds a prompt template to the list of supported prompt templates.
        :param prompt_template: PromptTemplate object to be added.
        :return: None
        """
        if prompt_template.name in self.prompt_templates:
            raise ValueError(
                f"Prompt template {prompt_template.name} already exists. "
                f"Select a different name for this prompt template."
            )

        self.prompt_templates[prompt_template.name] = prompt_template  # type: ignore

    def remove_prompt_template(self, prompt_template: str) -> PromptTemplate:
        """
        Removes a prompt template from the list of supported prompt templates.
        :param prompt_template: Name of the prompt template to be removed.
        :return: PromptTemplate object that was removed.
        """
        if prompt_template not in self.prompt_templates:
            raise ValueError(f"Prompt template {prompt_template} does not exist")

        return self.prompt_templates.pop(prompt_template)

    def set_default_prompt_template(self, prompt_template: Union[str, PromptTemplate]) -> "PromptNode":
        """
        Sets the default prompt template for the node.
        :param prompt_template: The prompt template to be set as default.
        :return: The current PromptNode object.
        """
        if not self.is_supported_template(prompt_template):
            raise ValueError(f"{prompt_template} not supported, select one of: {self.get_prompt_template_names()}")

        self.default_prompt_template = prompt_template
        return self

    def get_prompt_templates(self) -> List[PromptTemplate]:
        """
        Returns the list of supported prompt templates.
        :return: List of supported prompt templates.
        """
        return list(self.prompt_templates.values())

    def get_prompt_template_names(self) -> List[str]:
        """
        Returns the list of supported prompt template names.
        :return: List of supported prompt template names.
        """
        return list(self.prompt_templates.keys())

    def is_supported_template(self, prompt_template: Union[str, PromptTemplate]) -> bool:
        """
        Checks if a prompt template is supported.
        :param prompt_template: The prompt template to be checked.
        :return: True if the prompt template is supported, False otherwise.
        """
        template_name = prompt_template if isinstance(prompt_template, str) else prompt_template.name
        return template_name in self.prompt_templates

    def get_prompt_template(self, prompt_template: Union[str, PromptTemplate, None] = None) -> Optional[PromptTemplate]:
        """
        Returns a prompt template by name.
        :param prompt_template_name: The name of the prompt template to be returned.
        :return: The prompt template object.
        """
        prompt_template = prompt_template or self.default_prompt_template
        if prompt_template is None:
            return None

        if isinstance(prompt_template, PromptTemplate):
            return prompt_template

        if prompt_template in self.prompt_templates:
            return self.prompt_templates[prompt_template]

        if re.fullmatch(r"[-a-zA-Z0-9_]+", prompt_template):
            raise ValueError(
                f"{prompt_template} not supported, please select one of: {self.get_prompt_template_names()} or pass a PromptTemplate instance for prompting."
            )

        prompt_template_parsed = yaml.safe_load(prompt_template)
        if isinstance(prompt_template_parsed, dict):
            return PromptTemplate(**prompt_template_parsed)

        # it's a prompt_text
        prompt_text = prompt_template_parsed
        output_shapers = []
        default_prompt_template = self.get_prompt_template()
        if default_prompt_template:
            output_shapers = default_prompt_template.output_shapers
        return PromptTemplate(name="custom-at-query-time", prompt_text=prompt_text, output_shapers=output_shapers)

    def prompt_template_params(self, prompt_template: str) -> List[str]:
        """
        Returns the list of parameters for a prompt template.
        :param prompt_template: The name of the prompt template.
        :return: The list of parameters for the prompt template.
        """
        if not self.is_supported_template(prompt_template):
            raise ValueError(f"{prompt_template} not supported, select one of: {self.get_prompt_template_names()}")

        return list(self.prompt_templates[prompt_template].prompt_params)

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
    ) -> Tuple[Dict, str]:
        """
        Runs the PromptNode on these inputs parameters. Returns the output of the prompt model.
        The parameters `query`, `file_paths`, `labels`, `documents` and `meta` are added to the invocation context
        before invoking the prompt model. PromptNode uses these variables only if they are present as
        parameters in the PromptTemplate.

        :param query: The PromptNode usually ignores the query, unless it's used as a parameter in the
        prompt template.
        :param file_paths: The PromptNode usually ignores the file paths, unless they're used as a parameter
        in the prompt template.
        :param labels: The PromptNode usually ignores the labels, unless they're used as a parameter in the
        prompt template.
        :param documents: The documents to be used for the prompt.
        :param meta: PromptNode usually ignores meta information, unless it's used as a parameter in the
        PromptTemplate.
        :param invocation_context: The invocation context to be used for the prompt.
        """
        # prompt_collector is an empty list, it's passed to the PromptNode that will fill it with the rendered prompts,
        # so that they can be returned by `run()` as part of the pipeline's debug output.
        prompt_collector: List[str] = []

        invocation_context = invocation_context or {}
        if query and "query" not in invocation_context.keys():
            invocation_context["query"] = query

        if file_paths and "file_paths" not in invocation_context.keys():
            invocation_context["file_paths"] = file_paths

        if labels and "labels" not in invocation_context.keys():
            invocation_context["labels"] = labels

        if documents and "documents" not in invocation_context.keys():
            invocation_context["documents"] = documents

        if meta and "meta" not in invocation_context.keys():
            invocation_context["meta"] = meta

        if "prompt_template" not in invocation_context.keys():
            invocation_context["prompt_template"] = self.get_prompt_template(prompt_template)

        prompt_template_resolved: PromptTemplate = invocation_context["prompt_template"]
        results = self(prompt_collector=prompt_collector, **invocation_context)

        output_variable = self.output_variable or prompt_template_resolved.output_variable or "results"
        invocation_context[output_variable] = results
        invocation_context["prompts"] = prompt_collector
        final_result: Dict[str, Any] = {
            output_variable: results,
            "invocation_context": invocation_context,
            "_debug": {"prompts_used": prompt_collector},
        }
        return final_result, "output_1"

    def run_batch(  # type: ignore
        self,
        queries: Optional[List[str]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        invocation_contexts: Optional[List[Dict[str, Any]]] = None,
        prompt_templates: Optional[List[Union[str, PromptTemplate]]] = None,
    ):
        """
        Runs PromptNode in batch mode.

        - If you provide a list containing a single query (and/or invocation context)...
            - ... and a single list of Documents, the query is applied to each Document individually.
            - ... and a list of lists of Documents, the query is applied to each list of Documents and the results
              are aggregated per Document list.

        - If you provide a list of multiple queries (and/or multiple invocation contexts)...
            - ... and a single list of Documents, each query (and/or invocation context) is applied to each Document individually.
            - ... and a list of lists of Documents, each query (and/or invocation context) is applied to its corresponding list of Documents
              and the results are aggregated per query-Document pair.

        - If you provide no Documents, then each query (and/or invocation context) is applied directly to the PromptTemplate.

        :param queries: List of queries.
        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.
        :param invocation_contexts: List of invocation contexts.
        """
        inputs = PromptNode._flatten_inputs(queries, documents, invocation_contexts, prompt_templates)
        all_results: Dict[str, List] = defaultdict(list)
        for query, docs, invocation_context, prompt_template in zip(
            inputs["queries"], inputs["documents"], inputs["invocation_contexts"], inputs["prompt_templates"]
        ):
            prompt_template = self.get_prompt_template(self.default_prompt_template)
            output_variable = self.output_variable or prompt_template.output_variable or "results"
            results = self.run(
                query=query, documents=docs, invocation_context=invocation_context, prompt_template=prompt_template
            )[0]
            all_results[output_variable].append(results[output_variable])
            all_results["invocation_contexts"].append(results["invocation_context"])
            all_results["_debug"].append(results["_debug"])
        return all_results, "output_1"

    def _prepare_model_kwargs(self):
        # these are the parameters from PromptNode level
        # that are passed to the prompt model invocation layer
        return {"stop_words": self.stop_words, "top_k": self.top_k}

    @staticmethod
    def _flatten_inputs(
        queries: Optional[List[str]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        invocation_contexts: Optional[List[Dict[str, Any]]] = None,
        prompt_templates: Optional[List[Union[str, PromptTemplate]]] = None,
    ) -> Dict[str, List]:
        """Flatten and copy the queries, documents, and invocation contexts into lists of equal length.

        - If you provide a list containing a single query (and/or invocation context)...
            - ... and a single list of Documents, the query is applied to each Document individually.
            - ... and a list of lists of Documents, the query is applied to each list of Documents and the results
              are aggregated per Document list.

        - If you provide a list of multiple queries (and/or multiple invocation contexts)...
            - ... and a single list of Documents, each query (and/or invocation context) is applied to each Document individually.
            - ... and a list of lists of Documents, each query (and/or invocation context) is applied to its corresponding list of Documents
              and the results are aggregated per query-Document pair.

        - If you provide no Documents, then each query (and/or invocation context) is applied to the PromptTemplate.

        :param queries: List of queries.
        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.
        :param invocation_contexts: List of invocation contexts.
        """
        # Check that queries, and invocation_contexts are of the same length if provided
        input_queries: List[Any]
        input_invocation_contexts: List[Any]
        input_prompt_templates: List[Any]
        if queries is not None and invocation_contexts is not None:
            if len(queries) != len(invocation_contexts):
                raise ValueError("The input variables queries and invocation_contexts should have the same length.")
            input_queries = queries
            input_invocation_contexts = invocation_contexts
        elif queries is not None and invocation_contexts is None:
            input_queries = queries
            input_invocation_contexts = [None] * len(queries)
        elif queries is None and invocation_contexts is not None:
            input_queries = [None] * len(invocation_contexts)
            input_invocation_contexts = invocation_contexts
        else:
            input_queries = [None]
            input_invocation_contexts = [None]

        if prompt_templates is not None:
            if len(prompt_templates) != len(input_queries):
                raise ValueError("The input variables prompt_templates and queries should have the same length.")
            input_prompt_templates = prompt_templates
        else:
            input_prompt_templates = [None] * len(input_queries)

        multi_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list)
        single_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], Document)

        # Docs case 1: single list of Documents
        # -> apply each query (and invocation_contexts) to all Documents
        inputs: Dict[str, List] = defaultdict(list)
        if documents is not None:
            if single_docs_list:
                for query, invocation_context, prompt_template in zip(
                    input_queries, input_invocation_contexts, input_prompt_templates
                ):
                    for doc in documents:
                        inputs["queries"].append(query)
                        inputs["invocation_contexts"].append(invocation_context)
                        inputs["documents"].append([doc])
                        inputs["prompt_templates"].append(prompt_template)
            # Docs case 2: list of lists of Documents
            # -> apply each query (and invocation_context) to corresponding list of Documents,
            # if queries contains only one query, apply it to each list of Documents
            elif multi_docs_list:
                total_queries = input_queries.copy()
                total_invocation_contexts = input_invocation_contexts.copy()
                total_prompt_templates = input_prompt_templates.copy()
                if len(total_queries) == 1 and len(total_invocation_contexts) == 1 and len(total_prompt_templates) == 1:
                    total_queries = input_queries * len(documents)
                    total_invocation_contexts = input_invocation_contexts * len(documents)
                    total_prompt_templates = input_prompt_templates * len(documents)
                if (
                    len(total_queries) != len(documents)
                    or len(total_invocation_contexts) != len(documents)
                    or len(total_prompt_templates) != len(documents)
                ):
                    raise ValueError("Number of queries must be equal to number of provided Document lists.")
                for query, invocation_context, prompt_template, cur_docs in zip(
                    total_queries, total_invocation_contexts, total_prompt_templates, documents
                ):
                    inputs["queries"].append(query)
                    inputs["invocation_contexts"].append(invocation_context)
                    inputs["documents"].append(cur_docs)
                    inputs["prompt_templates"].append(prompt_template)
        elif queries is not None or invocation_contexts is not None or prompt_templates is not None:
            for query, invocation_context, prompt_template in zip(
                input_queries, input_invocation_contexts, input_prompt_templates
            ):
                inputs["queries"].append(query)
                inputs["invocation_contexts"].append(invocation_context)
                inputs["documents"].append([None])
                inputs["prompt_templates"].append(prompt_template)
        return inputs
