import logging
from typing import Dict, List, Optional, Tuple, Any

from haystack.preview import node
from haystack.preview.nodes.prompt.providers.base import get_model


logger = logging.getLogger(__name__)


PREDEFINED_TEMPLATES = {
    "question-answering": "Given the context please answer the question. Context: $documents; Question: $question; Answer:",
    "question-generation": "Given the context please generate a question. Context: $documents; Question:",
    "conditioned-question-generation": "Please come up with a question for the given context and the answer. Context: $documents; Answer: $answers; Question:",
    "summarization": "Summarize this document: $documents Summary:",
    "question-answering-check": "Does the following context contain the answer to the question? Context: $documents; Question: $questions; Please answer yes or no! Answer:",
    "sentiment-analysis": "Please give a sentiment for this context. Answer with positive, negative or neutral. Context: $documents; Answer:",
    "multiple-choice-question-answering": "Question:$questions ; Choose the most suitable option to answer the above question. Options: $options; Answer:",
    "topic-classification": "Categories: $options; What category best describes: $documents; Answer:",
    "language-detection": "Detect the language in the following context and answer with the name of the language. Context: $documents; Answer:",
    "translation": "Translate the following context to $target_language. Context: $documents; Translation:",
    "zero-shot-react": "You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions "
    "correctly, you have access to the following tools:\n\n"
    "$tool_names_with_descriptions\n\n"
    "To answer questions, you'll need to go through multiple steps involving step-by-step thinking and "
    "selecting appropriate tools and their inputs; tools will respond with observations. When you are ready "
    "for a final answer, respond with the `Final Answer:`\n\n"
    "Use the following format:\n\n"
    "Question: the question to be answered\n"
    "Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.\n"
    "Tool: pick one of $tool_names \n"
    "Tool Input: the input for the tool\n"
    "Observation: the tool will respond with the result\n"
    "...\n"
    "Final Answer: the final answer to the question, make it short (1-5 words)\n\n"
    "Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass\n"
    "---\n\n"
    "Question: $query\n"
    "Thought: Let's think step-by-step, I first need to ",
}


@node
class PromptNode:
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

    For more details, see [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node).
    """

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        template: str,
        model_name: str = "google/flan-t5-base",
        model_provider: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a PromptNode instance.

        :param inputs: the inputs that this PromptNode expects.
        :param outputs: the outputs that this PromptNode will generate.
        :param template: The prompt to use for the model. Might be the name of a predefined prompt template or your own template.
            NOTE: the variables that the template accepts must match with the variables you list in the `inputs` variable of
            this PromptNode. For example, if your prompt expects `$query` and `$documents`, make sure to initialize this node
            with `inputs=['query', 'documents']`.
        :param model_name: The name of the model to use, like a HF model identifier or an OpenAI model name.
        :param model_provider: force a specific provider for the model. If not given, Haystack will find a provider for your model automatically.
        """
        self.inputs = inputs
        self.outputs = outputs
        # TODO make a real prompt library system out of this.
        self.template = PREDEFINED_TEMPLATES[template] if template in PREDEFINED_TEMPLATES.keys() else template
        self.model = None
        self.model_name = model_name
        self.model_provider = model_provider
        self.model_kwargs = model_kwargs
        self.init_parameters = {
            "inputs": inputs,
            "outputs": outputs,
            "template": template,
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "mode_provider": model_provider,
        }

    def warm_up(self):
        if not self.model:
            self.model = get_model(
                model_name_or_path=self.model_name, model_provider=self.model_provider, model_kwargs=self.model_kwargs
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
        output = self.prompt(**data, **parameters.get(name, {}))
        return ({self.outputs[0]: output}, parameters)

    def prompt(self, **kwargs):
        """
        It takes in a prompt, and returns a list of responses using the underlying invocation layer.

        :param prompt: The prompt to use for the invocation. It can be a single prompt or a list of prompts.
        :param kwargs: Additional keyword arguments to pass to the invocation layer.
        :return: A list of model generated responses for the prompt or prompts.
        """
        self.warm_up()
        prompt = self._render_prompt(**kwargs)
        return self.model.invoke(prompt=prompt)

    def _render_prompt(self, **kwargs) -> str:
        """
        Replaces all variables ($query, $documents, etc) with the respective
        values found in **kwargs. If **kwargs comes from run(), it will contain
        both data and parameters.
        """
        prompt = self.template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"${key}", str(value))
        return prompt
