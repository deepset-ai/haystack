from typing import Optional, List, Union, Tuple, Dict, Iterator, Any
import logging
import os
import ast
import json
from abc import ABC
from uuid import uuid4

from haystack.errors import NodeError
from haystack.environment import HAYSTACK_PROMPT_TEMPLATE_ALLOWED_FUNCTIONS
from haystack.nodes.base import BaseComponent
from haystack.nodes.prompt.shapers import (  # pylint: disable=unused-import
    BaseOutputParser,
    AnswerParser,
    to_strings,
    join,  # used as shaping function
    format_document,
    format_answer,
    format_string,
)
from haystack.schema import Document, MultiLabel

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ALLOWED_FUNCTIONS = json.loads(
    os.environ.get(HAYSTACK_PROMPT_TEMPLATE_ALLOWED_FUNCTIONS, '["join", "to_strings", "replace", "enumerate", "str"]')
)
PROMPT_TEMPLATE_SPECIAL_CHAR_ALIAS = {"new_line": "\n", "tab": "\t", "double_quote": '"', "carriage_return": "\r"}
PROMPT_TEMPLATE_STRIPS = ["'", '"']
PROMPT_TEMPLATE_STR_REPLACE = {'"': "'"}


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


class PromptTemplateValidationError(NodeError):
    """
    The error raised when a PromptTemplate is invalid.
    """

    pass


class _ValidationVisitor(ast.NodeVisitor):
    """
    This class is used to validate the prompt text for a PromptTemplate.
    It checks that the prompt text is a valid f-string and that it only uses allowed functions.
    Useful information extracted from the AST is stored in the class attributes (for example, `prompt_params` and `used_functions`).
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
        For example, for the prompt text `f"Hello {name}"`, the prompt_params is `["name"]`.
        """
        return list(set(self.used_names) - set(self.used_functions) - set(self.comprehension_targets))

    def visit_Name(self, node: ast.Name) -> None:
        """
        Stores the name of the variable used in the prompt text. This also includes function and method names.
        For example, for the prompt text `f"Hello {func(name)}"`, the used_names are `["func", "name"]`.
        """
        self.used_names.append(node.id)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        """
        Stores the name of the variable used in comprehensions.
        For example, for the prompt text `f"Hello {[name for name in names]}"`, the comprehension_targets is `["name"]`.
        """
        super().generic_visit(node)
        if isinstance(node.target, ast.Name):
            self.comprehension_targets.append(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            self.comprehension_targets.extend([elt.id for elt in node.target.elts if isinstance(elt, ast.Name)])

    def visit_Call(self, node: ast.Call) -> None:
        """
        Stores the name of functions and methods used in the prompt text and validates that only allowed functions are used.
        For example, for the prompt text `f"Hello {func(name)}"`, the used_functions is `["func"]`.

        raises: PromptTemplateValidationError if the prompt text contains an invalid function.
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
    Transforms an AST for f-strings into a format the PromptTemplate can use.
    It replaces all f-string expressions with a unique ID and stores the corresponding expressions in a dictionary.

    You can evaluate the stored expressions using the `eval` function given the `prompt_params` (see _ValidatorVisitor).
    PromptTemplate determines the number of prompts to generate and renders them using the evaluated expressions.
    """

    def __init__(self):
        self.prompt_params_functions: Dict[str, ast.Expression] = {}

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Optional[ast.AST]:
        """
        Replaces the f-string expression with a unique ID and stores the corresponding expression in a dictionary.
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
    PromptTemplate is a template for the prompt you feed to the model to instruct it what to do. For example, if you want the model to perform sentiment analysis, you simply tell it to do that in a prompt. Here's what a prompt template may look like:

    ```python
        PromptTemplate(name="sentiment-analysis",
                   prompt_text="Give a sentiment for this context. Answer with positive, negative"
                   "or neutral. Context: {documents}; Answer:")
    ```

    Optionally, you can declare prompt parameters using f-string syntax in the PromptTemplate. Prompt parameters are input parameters that need to be filled in
    the prompt_text for the model to perform the task. For example, in the template above, there's one prompt parameter, `documents`.

    You declare prompt parameters by adding variables to the prompt text. These variables should be in the format: `{variable}`. In the template above, the variable is `{documents}`.

    At runtime, the variables you declared in prompt text are filled in with the arguments passed to the `fill()` method of the PromptTemplate. So in the example above, the `{documents}` variable will be filled with the Documents whose sentiment you want the model to analyze.

    Note that other than strict f-string syntax, you can safely use the following backslash characters in the text parts of the prompt text: `\n`, `\t`, `\r`.
    In f-string expressions, use `new_line`, `tab`, `carriage_return` instead.
    Double quotes (`"`) are automatically replaced with single quotes (`'`) in the prompt text. If you want to use double quotes in the prompt text, use `{double_quote}` instead.

    For more details on how to use PromptTemplate, see
    [PromptTemplates](https://docs.haystack.deepset.ai/docs/prompt_node#prompttemplates).
    """

    def __init__(
        self, name: str, prompt_text: str, output_parser: Optional[Union[BaseOutputParser, Dict[str, Any]]] = None
    ):
        """
         Creates a PromptTemplate instance.

        :param name: The name of the prompt template (for example, "sentiment-analysis", "question-generation"). You can specify your own name but it must be unique.
        :param prompt_text: The prompt text, including prompt parameters.
        :param output_parser: A parser that applied to the model output.
                For example, to convert the model output to an Answer object, you can use `AnswerParser`.
                Instead of BaseOutputParser instances, you can also pass dictionaries defining the output parsers. For example:
                ```
                output_parser={"type": "AnswerParser", "params": {"pattern": "Answer: (.*)"}},
                ```
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
        self.prompt_params: List[str] = sorted(
            param for param in ast_validator.prompt_params if param not in PROMPT_TEMPLATE_SPECIAL_CHAR_ALIAS
        )
        self.globals = {
            **{k: v for k, v in globals().items() if k in PROMPT_TEMPLATE_ALLOWED_FUNCTIONS},
            **PROMPT_TEMPLATE_SPECIAL_CHAR_ALIAS,
        }
        self.output_parser: Optional[BaseOutputParser] = None
        if isinstance(output_parser, BaseOutputParser):
            self.output_parser = output_parser
        elif isinstance(output_parser, dict):
            output_parser_type = output_parser["type"]
            output_parser_params = output_parser.get("params", {})
            self.output_parser = BaseComponent._create_instance(output_parser_type, output_parser_params)

    @property
    def output_variable(self) -> Optional[str]:
        return self.output_parser.output_variable if self.output_parser else None

    def prepare(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Prepares and verifies the PromtpTemplate with input parameters.

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

        if not set(self.prompt_params).issubset(params_dict.keys()):
            available_params = {*params_dict.keys(), *kwargs.keys()}
            provided = set(self.prompt_params).intersection(available_params)
            message = f"only {list(provided)}" if provided else "none of these parameters"
            raise ValueError(
                f"Expected prompt parameters {self.prompt_params} to be provided but got "
                f"{message}. Make sure to provide all template parameters."
            )

        template_dict = {"_at_least_one_prompt": True}
        for id, call in self._prompt_params_functions.items():
            template_dict[id] = eval(  # pylint: disable=eval-used
                compile(call, filename="<string>", mode="eval"), self.globals, params_dict
            )

        return template_dict

    def post_process(self, prompt_output: List[str], **kwargs) -> List[Any]:
        """
        Post-processes the output of the PromptTemplate.
        :param args: Non-keyword arguments to use for post-processing the prompt output.
        :param kwargs: Keyword arguments to use for post-processing the prompt output.
        :return: A dictionary with the post-processed output.
        """
        if self.output_parser:
            invocation_context = kwargs
            invocation_context["results"] = prompt_output
            self.output_parser.run(invocation_context=invocation_context)
            return invocation_context[self.output_parser.outputs[0]]
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

        :param args: Non-keyword arguments to fill the parameters in the prompt text. Their order must match the order of appearance of the parameters in the prompt text.
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


def get_predefined_prompt_templates() -> List[PromptTemplate]:
    return [
        PromptTemplate(
            name="question-answering",
            prompt_text="Given the context please answer the question. Context: {join(documents)}; Question: "
            "{query}; Answer:",
            output_parser=AnswerParser(),
        ),
        PromptTemplate(
            name="question-answering-per-document",
            prompt_text="Given the context please answer the question. Context: {documents}; Question: "
            "{query}; Answer:",
            output_parser=AnswerParser(),
        ),
        PromptTemplate(
            name="question-answering-with-references",
            prompt_text="Create a concise and informative answer (no more than 50 words) for a given question "
            "based solely on the given documents. You must only use information from the given documents. "
            "Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[number] notation. "
            "If multiple documents contain the answer, cite those documents like ‘as stated in Document[number], Document[number], etc.’. "
            "If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’\n"
            "{join(documents, delimiter=new_line, pattern=new_line+'Document[$idx]: $content', str_replace={new_line: ' ', '[': '(', ']': ')'})} \n Question: {query}; Answer: ",
            output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]"),
        ),
        PromptTemplate(
            name="question-answering-with-document-scores",
            prompt_text="Answer the following question using the paragraphs below as sources. "
            "An answer should be short, a few words at most.\n"
            "Paragraphs:\n{documents}\n"
            "Question: {query}\n\n"
            "Instructions: Consider all the paragraphs above and their corresponding scores to generate "
            "the answer. While a single paragraph may have a high score, it's important to consider all "
            "paragraphs for the same answer candidate to answer accurately.\n\n"
            "After having considered all possibilities, the final answer is:\n",
        ),
        PromptTemplate(
            name="question-generation",
            prompt_text="Given the context please generate a question. Context: {documents}; Question:",
        ),
        PromptTemplate(
            name="conditioned-question-generation",
            prompt_text="Please come up with a question for the given context and the answer. "
            "Context: {documents}; Answer: {answers}; Question:",
        ),
        PromptTemplate(name="summarization", prompt_text="Summarize this document: {documents} Summary:"),
        PromptTemplate(
            name="question-answering-check",
            prompt_text="Does the following context contain the answer to the question? "
            "Context: {documents}; Question: {query}; Please answer yes or no! Answer:",
            output_parser=AnswerParser(),
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
            output_parser=AnswerParser(),
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
            "Thought: Let's think step-by-step, I first need to {transcript}",
        ),
        PromptTemplate(
            name="conversational-agent",
            prompt_text="The following is a conversation between a human and an AI.\n{history}\nHuman: {query}\nAI:",
        ),
        PromptTemplate(
            name="conversational-summary",
            prompt_text="Condense the following chat transcript by shortening and summarizing the content without losing important information:\n{chat_transcript}\nCondensed Transcript:",
        ),
    ]
