import copy
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from string import Template
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Iterator

import requests
import torch
from transformers import (
    pipeline,
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES

from haystack import MultiLabel
from haystack.environment import (
    HAYSTACK_REMOTE_API_BACKOFF_SEC,
    HAYSTACK_REMOTE_API_MAX_RETRIES,
    HAYSTACK_REMOTE_API_TIMEOUT_SEC,
)
from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from haystack.utils.reflection import retry_with_exponential_backoff

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


class PromptTemplate(BasePromptTemplate, ABC):
    """
    PromptTemplate is a template for a prompt you feed to the model to instruct it what to do. For example, if you want the model to perform sentiment analysis, you simply tell it to do that in a prompt. Here's what such prompt template may look like:

    ```python
        PromptTemplate(name="sentiment-analysis",
                   prompt_text="Give a sentiment for this context. Answer with positive, negative
                   or neutral. Context: $documents; Answer:")
    ```

    Optionally, you can declare prompt parameters in the PromptTemplate. Prompt parameters are input parameters that need to be filled in
    the prompt_text for the model to perform the task. For example, in the template above, there's one prompt parameter, `documents`. You declare prompt parameters by adding variables to the prompt text. These variables should be in the format: `$variable`. In the template above, the variable is `$documents`.

    At runtime, these variables are filled in with arguments passed to the `fill()` method of the PromptTemplate. So in the example above, the `$documents` variable will be filled with the Documents whose sentiment you want the model to analyze.

    For more details on how to use PromptTemplate, see
    [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node).
    """

    def __init__(self, name: str, prompt_text: str, prompt_params: Optional[List[str]] = None):
        """
         Creates a PromptTemplate instance.

        :param name: The name of the prompt template (for example, sentiment-analysis, question-generation). You can specify your own name but it must be unique.
        :param prompt_text: The prompt text, including prompt parameters.
        :param prompt_params: Optional parameters that need to be filled in the prompt text. If you don't specify them, they're inferred from the prompt text. Any variable in prompt text in the format `$variablename` is interpreted as a prompt parameter.
        """
        super().__init__()
        if not prompt_params:
            # Define the regex pattern to match the strings after the $ character
            pattern = r"\$([a-zA-Z0-9_]+)"
            prompt_params = re.findall(pattern, prompt_text)

        if prompt_text.count("$") != len(prompt_params):
            raise ValueError(
                f"The number of parameters in prompt text {prompt_text} for prompt template {name} "
                f"does not match the number of specified parameters {prompt_params}."
            )

        # use case when PromptTemplate is loaded from a YAML file, we need to start and end the prompt text with quotes
        prompt_text = prompt_text.strip("'").strip('"')

        t = Template(prompt_text)
        try:
            t.substitute(**{param: "" for param in prompt_params})
        except KeyError as e:
            raise ValueError(
                f"Invalid parameter {e} in prompt text "
                f"{prompt_text} for prompt template {name}, specified parameters are {prompt_params}"
            )

        self.name = name
        self.prompt_text = prompt_text
        self.prompt_params = prompt_params

    def prepare(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Prepares and verifies the prompt template with input parameters.

        :param args: Non-keyword arguments to fill the parameters in the prompt text of a PromptTemplate.
        :param kwargs: Keyword arguments to fill the parameters in the prompt text of a PromptTemplate.
        :return: A dictionary with the prompt text and the prompt parameters.
        """
        template_dict = {}
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
                template_dict[prompt_param] = [arg] if isinstance(arg, str) else arg
        # then attempt to resolve kwargs
        if kwargs:
            for param in self.prompt_params:
                if param in kwargs:
                    template_dict[param] = kwargs[param]

        if set(template_dict.keys()) != set(self.prompt_params):
            available_params = set(list(template_dict.keys()) + list(set(kwargs.keys())))
            raise ValueError(f"Expected prompt parameters {self.prompt_params} but got {list(available_params)}.")

        return template_dict

    def fill(self, *args, **kwargs) -> Iterator[str]:
        """
        Fills the parameters defined in the prompt text with the arguments passed to it and returns the iterator prompt text.

        You can pass non-keyword (args) or keyword (kwargs) arguments to this method. If you pass non-keyword arguments, their order must match the left-to-right
        order of appearance of the parameters in the prompt text. For example, if the prompt text is:
        `Come up with a question for the given context and the answer. Context: $documents;
        Answer: $answers; Question:`, then the first non-keyword argument fills the `$documents` variable
        and the second non-keyword argument fills the `$answers` variable.

        If you pass keyword arguments, the order of the arguments doesn't matter. Variables in the
        prompt text are filled with the corresponding keyword argument.

        :param args: Non-keyword arguments to fill the parameters in the prompt text. Their order must match the order of appearance of the parameters in prompt text.
        :param kwargs: Keyword arguments to fill the parameters in the prompt text.
        :return: An iterator of prompt texts.
        """
        template_dict = self.prepare(*args, **kwargs)
        template = Template(self.prompt_text)
        # the prompt context values should all be lists, as they will be split as one
        prompt_context_copy = {k: v if isinstance(v, list) else [v] for k, v in template_dict.items()}
        for prompt_context_values in zip(*prompt_context_copy.values()):
            template_input = {key: prompt_context_values[idx] for idx, key in enumerate(prompt_context_copy.keys())}
            prompt_prepared: str = template.substitute(template_input)
            yield prompt_prepared

    def __repr__(self):
        return f"PromptTemplate(name={self.name}, prompt_text={self.prompt_text}, prompt_params={self.prompt_params})"


class PromptModelInvocationLayer:
    """
    PromptModelInvocationLayer implementations execute a prompt on an underlying model.

    The implementation can be a simple invocation on the underlying model running in a local runtime, or
    could be even remote, for example, a call to a remote API endpoint.
    """

    def __init__(self, model_name_or_path: str, max_length: Optional[int] = 100, **kwargs):
        """
        Creates a new PromptModelInvocationLayer instance.


        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum length of output text.
        :param kwargs: Additional keyword arguments passed to the underlying model.

        """
        if model_name_or_path is None or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model_name_or_path = model_name_or_path
        self.max_length: Optional[int] = max_length

    @abstractmethod
    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated text using the underlying model.
        :return: A list of generated text.
        """
        pass

    @classmethod
    def supports(cls, model_name_or_path: str) -> bool:
        """
        Checks if the given model is supported by this invocation layer.

        :param model_name_or_path: The name or path of the model.
        :return: True if this invocation layer supports the model, False otherwise.
        """
        return False


class StopWordsCriteria(StoppingCriteria):
    """
    Stops text generation if any one of the stop words is generated.
    """

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], stop_words: List[str]):
        super().__init__()
        self.stop_words = tokenizer.encode(stop_words, add_special_tokens=False, return_tensors="pt")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(torch.isin(input_ids[-1], self.stop_words[-1]))


class HFLocalInvocationLayer(PromptModelInvocationLayer):
    """
    A subclass of the PromptModelInvocationLayer class. It loads a pre-trained model from Hugging Face and
    passes a prepared prompt into that model.

    Note: kwargs other than init parameter names are ignored to enable reflective construction of the class,
    as many variants of PromptModelInvocationLayer are possible and they may have different parameters.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-base",
        max_length: Optional[int] = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        **kwargs,
    ):
        """
        Creates an instance of HFLocalInvocationLayer used to invoke local Hugging Face models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum length of the output text.
        :param use_auth_token: The token to use as HTTP bearer authorization for remote files.
        :param use_gpu: Whether to use GPU for inference.
        :param device: The device to use for inference.
        :param kwargs: Additional keyword arguments passed to the underlying model. Due to reflective construction of
        all PromptModelInvocationLayer instances, this instance of HFLocalInvocationLayer might receive some unrelated
        kwargs. Only kwargs relevant to the HFLocalInvocationLayer are considered. The list of supported kwargs
        includes: trust_remote_code, revision, feature_extractor, tokenizer, config, use_fast, torch_dtype, device_map.
        For more details about these kwargs, see
        Hugging Face [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline).
        """
        super().__init__(model_name_or_path, max_length)
        self.use_auth_token = use_auth_token

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        # Due to reflective construction of all invocation layers we might receive some
        # unknown kwargs, so we need to take only the relevant.
        # For more details refer to Hugging Face pipeline documentation
        # Do not use `device_map` AND `device` at the same time as they will conflict
        model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "model_kwargs",
                "trust_remote_code",
                "revision",
                "feature_extractor",
                "tokenizer",
                "config",
                "use_fast",
                "torch_dtype",
                "device_map",
            ]
            if key in kwargs
        }
        # flatten model_kwargs one level
        if "model_kwargs" in model_input_kwargs:
            mkwargs = model_input_kwargs.pop("model_kwargs")
            model_input_kwargs.update(mkwargs)

        torch_dtype = model_input_kwargs.get("torch_dtype")
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if "torch." not in torch_dtype:
                    raise ValueError(
                        f"torch_dtype should be a torch.dtype or a string with 'torch.' prefix, got {torch_dtype}"
                    )
                torch_dtype_resolved = getattr(torch, torch_dtype.strip("torch."))
            elif isinstance(torch_dtype, torch.dtype):
                torch_dtype_resolved = torch_dtype
            else:
                raise ValueError(f"Invalid torch_dtype value {torch_dtype}")
            model_input_kwargs["torch_dtype"] = torch_dtype_resolved

        if len(model_input_kwargs) > 0:
            logger.info("Using model input kwargs %s in %s", model_input_kwargs, self.__class__.__name__)

        self.pipe = pipeline(
            "text2text-generation",
            model=model_name_or_path,
            device=self.devices[0] if "device_map" not in model_input_kwargs else None,
            use_auth_token=self.use_auth_token,
            model_kwargs=model_input_kwargs,
        )

    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated text using the local Hugging Face transformers model
        :return: A list of generated text.

        Note: Only kwargs relevant to Text2TextGenerationPipeline are passed to Hugging Face as model_input_kwargs.
        Other kwargs are ignored.
        """
        output: List[Dict[str, str]] = []
        stop_words = kwargs.pop("stop_words", None)
        top_k = kwargs.pop("top_k", None)
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")

            # Consider only Text2TextGenerationPipeline relevant, ignore others
            # For more details refer to Hugging Face Text2TextGenerationPipeline documentation
            # TODO resolve these kwargs from the pipeline signature
            model_input_kwargs = {
                key: kwargs[key]
                for key in ["return_tensors", "return_text", "clean_up_tokenization_spaces", "truncation"]
                if key in kwargs
            }
            if stop_words:
                sw = StopWordsCriteria(tokenizer=self.pipe.tokenizer, stop_words=stop_words)
                model_input_kwargs["stopping_criteria"] = StoppingCriteriaList([sw])
            if top_k:
                model_input_kwargs["num_return_sequences"] = top_k
                model_input_kwargs["num_beams"] = top_k
            output = self.pipe(prompt, max_length=self.max_length, **model_input_kwargs)
        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]

        if stop_words:
            # Although HF generates text until stop words are encountered unfortunately it includes the stop word
            # We want to exclude it to be consistent with other invocation layers
            for idx, _ in enumerate(generated_texts):
                for stop_word in stop_words:
                    generated_texts[idx] = generated_texts[idx].replace(stop_word, "").strip()
        return generated_texts

    @classmethod
    def supports(cls, model_name_or_path: str) -> bool:
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
        except OSError:
            # This is needed so OpenAI models are skipped over
            return False

        if not all(m in model_name_or_path for m in ["flan", "t5"]):
            logger.warning(
                "PromptNode has been potentially initialized with a language model not fine-tuned on instruction following tasks. "
                "Many of the default prompts and PromptTemplates will likely not work as intended. "
                "Use custom prompts and PromptTemplates specific to the %s model",
                model_name_or_path,
            )

        supported_models = list(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.values())
        return config.architectures[0] in supported_models


class OpenAIInvocationLayer(PromptModelInvocationLayer):
    """
    PromptModelInvocationLayer implementation for OpenAI's GPT-3 InstructGPT models. Invocations are made using REST API.
    See [OpenAI GPT-3](https://platform.openai.com/docs/models/gpt-3) for more details.

    Note: kwargs other than init parameter names are ignored to enable reflective construction of the class
    as many variants of PromptModelInvocationLayer are possible and they may have different parameters.
    """

    def __init__(
        self, api_key: str, model_name_or_path: str = "text-davinci-003", max_length: Optional[int] = 100, **kwargs
    ):
        """
         Creates an instance of OpenAIInvocationLayer for OpenAI's GPT-3 InstructGPT models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum length of the output text.
        :param api_key: The OpenAI API key.
        :param kwargs: Additional keyword arguments passed to the underlying model. Due to reflective construction of
        all PromptModelInvocationLayer instances, this instance of OpenAIInvocationLayer might receive some unrelated
        kwargs. Only the kwargs relevant to OpenAIInvocationLayer are considered. The list of OpenAI-relevant
        kwargs includes: suffix, temperature, top_p, presence_penalty, frequency_penalty, best_of, n, max_tokens,
        logit_bias, stop, echo, and logprobs. For more details about these kwargs, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/completions/create).

        """
        super().__init__(model_name_or_path, max_length)
        if not isinstance(api_key, str) or len(api_key) == 0:
            raise OpenAIError(
                f"api_key {api_key} must be a valid OpenAI key. Visit https://openai.com/api/ to get one."
            )
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/completions"

        # Due to reflective construction of all invocation layers we might receive some
        # unknown kwargs, so we need to take only the relevant.
        # For more details refer to OpenAI documentation
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "suffix",
                "max_tokens",
                "temperature",
                "top_p",
                "n",
                "logprobs",
                "echo",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "best_of",
                "logit_bias",
            ]
            if key in kwargs
        }

    @retry_with_exponential_backoff(
        backoff_in_seconds=float(os.environ.get(HAYSTACK_REMOTE_API_BACKOFF_SEC, 5)),
        max_retries=int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5)),
        errors=(OpenAIRateLimitError, OpenAIError),
    )
    def invoke(self, *args, **kwargs):
        """
        Invokes a prompt on the model. It takes in a prompt and returns a list of responses using a REST invocation.

        :return: The responses are being returned.

        Note: Only kwargs relevant to OpenAI are passed to OpenAI rest API. Others kwargs are ignored.
        For more details, see OpenAI [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                f"No prompt provided. Model {self.model_name_or_path} requires prompt."
                f"Make sure to provide prompt in kwargs."
            )

        kwargs_with_defaults = self.model_input_kwargs
        if kwargs:
            # we use keyword stop_words but OpenAI uses stop
            if "stop_words" in kwargs:
                kwargs["stop"] = kwargs.pop("stop_words")
            if "top_k" in kwargs:
                top_k = kwargs.pop("top_k")
                kwargs["n"] = top_k
                kwargs["best_of"] = top_k
            kwargs_with_defaults.update(kwargs)
        payload = {
            "model": self.model_name_or_path,
            "prompt": prompt,
            "suffix": kwargs_with_defaults.get("suffix", None),
            "max_tokens": kwargs_with_defaults.get("max_tokens", self.max_length),
            "temperature": kwargs_with_defaults.get("temperature", 0.7),
            "top_p": kwargs_with_defaults.get("top_p", 1),
            "n": kwargs_with_defaults.get("n", 1),
            "stream": False,  # no support for streaming
            "logprobs": kwargs_with_defaults.get("logprobs", None),
            "echo": kwargs_with_defaults.get("echo", False),
            "stop": kwargs_with_defaults.get("stop", None),
            "presence_penalty": kwargs_with_defaults.get("presence_penalty", 0),
            "frequency_penalty": kwargs_with_defaults.get("frequency_penalty", 0),
            "best_of": kwargs_with_defaults.get("best_of", 1),
            "logit_bias": kwargs_with_defaults.get("logit_bias", {}),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.request(
            "POST",
            self.url,
            headers=headers,
            data=json.dumps(payload),
            timeout=float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30)),
        )
        res = json.loads(response.text)

        if response.status_code != 200:
            openai_error: OpenAIError
            if response.status_code == 429:
                openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
            else:
                openai_error = OpenAIError(
                    f"OpenAI returned an error.\n"
                    f"Status code: {response.status_code}\n"
                    f"Response body: {response.text}",
                    status_code=response.status_code,
                )
            raise openai_error

        number_of_truncated_completions = sum(1 for ans in res["choices"] if ans["finish_reason"] == "length")
        if number_of_truncated_completions > 0:
            logger.warning(
                "%s out of the %s completions have been truncated before reaching a natural stopping point."
                "Consider increasing the max_tokens parameter to allow for longer completions.",
                number_of_truncated_completions,
                payload["n"],
            )

        responses = [ans["text"].strip() for ans in res["choices"]]
        return responses

    @classmethod
    def supports(cls, model_name_or_path: str) -> bool:
        return any(m for m in ["ada", "babbage", "davinci", "curie"] if m in model_name_or_path)


class PromptModel(BaseComponent):
    """
    The PromptModel class is a component that uses a pre-trained model to perform tasks based on a prompt. Out of
    the box, it supports two model invocation layers: Hugging Face transformers and OpenAI, with the ability to
    register additional custom invocation layers.

    Although it is possible to use PromptModel to make prompt invocations on the underlying model, use
    PromptNode to interact with the model. PromptModel instances are a way for multiple
    PromptNode instances to use a single PromptNode, and thus save computational resources.

    For more details, refer to [Promptnode](https://docs.haystack.deepset.ai/docs/prompt_node).
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
        :param model_kwargs: Additional keyword arguments passed to the underlying model.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.api_key = api_key
        self.use_auth_token = use_auth_token
        self.use_gpu = use_gpu
        self.devices = devices

        self.model_kwargs = model_kwargs if model_kwargs else {}

        self.invocation_layers: List[Type[PromptModelInvocationLayer]] = []

        self.register(HFLocalInvocationLayer)  # pylint: disable=W0108
        self.register(OpenAIInvocationLayer)  # pylint: disable=W0108

        self.model_invocation_layer = self.create_invocation_layer()

    def create_invocation_layer(self) -> PromptModelInvocationLayer:
        kwargs = {
            "api_key": self.api_key,
            "use_auth_token": self.use_auth_token,
            "use_gpu": self.use_gpu,
            "devices": self.devices,
        }
        all_kwargs = {**self.model_kwargs, **kwargs}

        for invocation_layer in self.invocation_layers:
            if invocation_layer.supports(self.model_name_or_path):
                return invocation_layer(
                    model_name_or_path=self.model_name_or_path, max_length=self.max_length, **all_kwargs
                )
        raise ValueError(
            f"Model {self.model_name_or_path} is not supported - no invocation layer found."
            f" Currently supported models are: {self.invocation_layers}"
            f" Register a new invocation layer for {self.model_name_or_path} using the register method."
        )

    def register(self, invocation_layer: Type[PromptModelInvocationLayer]):
        """
        Registers additional prompt model invocation layer. It takes a function that returns a boolean as a
        matching condition on `model_name_or_path` and a class that implements `PromptModelInvocationLayer` interface.
        """
        self.invocation_layers.append(invocation_layer)

    def invoke(self, prompt: Union[str, List[str]], **kwargs) -> List[str]:
        """
        It takes in a prompt, and returns a list of responses using the underlying invocation layer.

        :param prompt: The prompt to use for the invocation. It can be a single prompt or a list of prompts.
        :param kwargs: Additional keyword arguments to pass to the invocation layer.
        :return: A list of model generated responses for the prompt or prompts.
        """
        output = self.model_invocation_layer.invoke(prompt=prompt, **kwargs)
        return output

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


def get_predefined_prompt_templates() -> List[PromptTemplate]:
    return [
        PromptTemplate(
            name="question-answering",
            prompt_text="Given the context please answer the question. Context: $documents; Question: "
            "$questions; Answer:",
        ),
        PromptTemplate(
            name="question-generation",
            prompt_text="Given the context please generate a question. Context: $documents; Question:",
        ),
        PromptTemplate(
            name="conditioned-question-generation",
            prompt_text="Please come up with a question for the given context and the answer. "
            "Context: $documents; Answer: $answers; Question:",
        ),
        PromptTemplate(name="summarization", prompt_text="Summarize this document: $documents Summary:"),
        PromptTemplate(
            name="question-answering-check",
            prompt_text="Does the following context contain the answer to the question? "
            "Context: $documents; Question: $questions; Please answer yes or no! Answer:",
        ),
        PromptTemplate(
            name="sentiment-analysis",
            prompt_text="Please give a sentiment for this context. Answer with positive, "
            "negative or neutral. Context: $documents; Answer:",
        ),
        PromptTemplate(
            name="multiple-choice-question-answering",
            prompt_text="Question:$questions ; Choose the most suitable option to answer the above question. "
            "Options: $options; Answer:",
        ),
        PromptTemplate(
            name="topic-classification",
            prompt_text="Categories: $options; What category best describes: $documents; Answer:",
        ),
        PromptTemplate(
            name="language-detection",
            prompt_text="Detect the language in the following context and answer with the "
            "name of the language. Context: $documents; Answer:",
        ),
        PromptTemplate(
            name="translation",
            prompt_text="Translate the following context to $target_language. Context: $documents; Translation:",
        ),
        PromptTemplate(
            name="zero-shot-react",
            prompt_text="You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions "
            "correctly, you have access to the following tools:\n\n"
            "$tool_names_with_descriptions\n\n"
            "To answer questions, you'll need to go through multiple steps involving step-by-step thinking and "
            "selecting appropriate tools and their inputs; tools will respond with observations. When you are ready "
            "for a final answer, respond with the `Final Answer:`\n\n"
            "Use the following format:\n\n"
            "Question: the question to be answered\n"
            "Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.\n"
            "Tool: [$tool_names]\n"
            "Tool Input: the input for the tool\n"
            "Observation: the tool will respond with the result\n"
            "...\n"
            "Final Answer: the final answer to the question, make it short (1-5 words)\n\n"
            "Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass\n"
            "---\n\n"
            "Question: $query\n"
            "Thought: Let's think step-by-step, I first need to ",
        ),
    ]


class PromptNode(BaseComponent):
    """
    The PromptNode class is the central abstraction in Haystack's large language model (LLM) support. PromptNode
    supports multiple NLP tasks out of the box. You can use it to perform tasks, such as
    summarization, question answering, question generation, and more, using a single, unified model within the Haystack framework.

    One of the benefits of PromptNode is that you can use it to define and add additional prompt templates
     the model supports. Defining additional prompt templates makes it possible to extend the model's capabilities
    and use it for a broader range of NLP tasks in Haystack. Prompt engineers define templates
    for each NLP task and register them with PromptNode. The burden of defining templates for each task rests on
    the prompt engineers, not the users.

    Using an instance of the PromptModel class, you can create multiple PromptNodes that share the same model, saving
    the memory and time required to load the model multiple times.

    PromptNode also supports multiple model invocation layers: Hugging Face transformers and OpenAI with an
    ability to register additional custom invocation layers. However, we currently support only
    T5 Flan and OpenAI InstructGPT models.

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
        :param max_length: The maximum length of the generated text output.
        :param api_key: The API key to use for the model.
        :param use_auth_token: The authentication token to use for the model.
        :param use_gpu: Whether to use GPU or not.
        :param devices: The devices to use for the model.
        :param top_k: Number of independently generated text to return per prompt.
        :param stop_words: Stops text generation if any one of the stop words is generated.
        :param model_kwargs: Additional keyword arguments passed when loading the model specified by `model_name_or_path`.
        """
        super().__init__()
        self.prompt_templates: Dict[str, PromptTemplate] = {pt.name: pt for pt in get_predefined_prompt_templates()}  # type: ignore
        self.default_prompt_template: Union[str, PromptTemplate, None] = default_prompt_template
        self.output_variable: str = output_variable or "results"
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

    def __call__(self, *args, **kwargs) -> List[str]:
        """
        This method is invoked when the component is called directly, for example:
        ```python
            PromptNode pn = ...
            sa = pn.set_default_prompt_template("sentiment-analysis")
            sa(documents=[Document("I am in love and I feel great!")])
        ```
        """
        if "prompt_template_name" in kwargs:
            prompt_template_name = kwargs["prompt_template_name"]
            kwargs.pop("prompt_template_name")
            return self.prompt(prompt_template_name, *args, **kwargs)
        else:
            return self.prompt(self.default_prompt_template, *args, **kwargs)

    def prompt(self, prompt_template: Optional[Union[str, PromptTemplate]], *args, **kwargs) -> List[str]:
        """
        Prompts the model and represents the central API for the PromptNode. It takes a prompt template,
        a list of non-keyword and keyword arguments, and returns a list of strings - the responses from the underlying model.

        If you specify the optional prompt_template parameter, it takes precedence over the default PromptTemplate for this PromptNode.

        :param prompt_template: The name or object of the optional PromptTemplate to use.
        :return: A list of strings as model responses.
        """
        results = []
        # we pop the prompt_collector kwarg to avoid passing it to the model
        prompt_collector: List[str] = kwargs.pop("prompt_collector", [])
        if isinstance(prompt_template, str) and not self.is_supported_template(prompt_template):
            raise ValueError(
                f"{prompt_template} not supported, please select one of: {self.get_prompt_template_names()} "
                f"or pass a PromptTemplate instance for prompting."
            )

        # kwargs override model kwargs
        kwargs = {**self._prepare_model_kwargs(), **kwargs}
        prompt_template_used = prompt_template or self.default_prompt_template
        if prompt_template_used:
            if isinstance(prompt_template_used, PromptTemplate):
                template_to_fill = prompt_template_used
            elif isinstance(prompt_template_used, str):
                template_to_fill = self.get_prompt_template(prompt_template_used)
            else:
                raise ValueError(f"{prompt_template_used} with args {args} , and kwargs {kwargs} not supported")

            # prompt template used, yield prompts from inputs args
            for prompt in template_to_fill.fill(*args, **kwargs):
                kwargs_copy = copy.copy(kwargs)
                # and pass the prepared prompt and kwargs copy to the model
                prompt_collector.append(prompt)
                logger.debug("Prompt being sent to LLM with prompt %s and kwargs %s", prompt, kwargs_copy)
                output = self.prompt_model.invoke(prompt, **kwargs_copy)
                results.extend(output)
        else:
            # straightforward prompt, no templates used
            for prompt in list(args):
                kwargs_copy = copy.copy(kwargs)
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

    def get_prompt_template(self, prompt_template_name: str) -> PromptTemplate:
        """
        Returns a prompt template by name.
        :param prompt_template_name: The name of the prompt template to be returned.
        :return: The prompt template object.
        """
        if prompt_template_name not in self.prompt_templates:
            raise ValueError(f"Prompt template {prompt_template_name} not supported")
        return self.prompt_templates[prompt_template_name]

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

        if "documents" in invocation_context.keys():
            for doc in invocation_context.get("documents", []):
                if not isinstance(doc, str) and not isinstance(doc.content, str):
                    raise ValueError("PromptNode only accepts text documents.")
            invocation_context["documents"] = [
                doc.content if isinstance(doc, Document) else doc for doc in invocation_context.get("documents", [])
            ]

        results = self(prompt_collector=prompt_collector, **invocation_context)

        invocation_context[self.output_variable] = results
        final_result: Dict[str, Any] = {
            self.output_variable: results,
            "invocation_context": invocation_context,
            "_debug": {"prompts_used": prompt_collector},
        }
        return final_result, "output_1"

    def run_batch(  # type: ignore
        self,
        queries: Optional[List[str]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        invocation_contexts: Optional[List[Dict[str, Any]]] = None,
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
        inputs = PromptNode._flatten_inputs(queries, documents, invocation_contexts)
        all_results: Dict[str, List] = {self.output_variable: [], "invocation_contexts": [], "_debug": []}
        for query, docs, invocation_context in zip(
            inputs["queries"], inputs["documents"], inputs["invocation_contexts"]
        ):
            results = self.run(query=query, documents=docs, invocation_context=invocation_context)[0]
            all_results[self.output_variable].append(results[self.output_variable])
            all_results["invocation_contexts"].append(all_results["invocation_contexts"])
            all_results["_debug"].append(all_results["_debug"])
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

        multi_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list)
        single_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], Document)

        # Docs case 1: single list of Documents
        # -> apply each query (and invocation_contexts) to all Documents
        inputs: Dict[str, List] = {"queries": [], "invocation_contexts": [], "documents": []}
        if documents is not None:
            if single_docs_list:
                for query, invocation_context in zip(input_queries, input_invocation_contexts):
                    for doc in documents:
                        inputs["queries"].append(query)
                        inputs["invocation_contexts"].append(invocation_context)
                        inputs["documents"].append([doc])
            # Docs case 2: list of lists of Documents
            # -> apply each query (and invocation_context) to corresponding list of Documents,
            # if queries contains only one query, apply it to each list of Documents
            elif multi_docs_list:
                total_queries = input_queries.copy()
                total_invocation_contexts = input_invocation_contexts.copy()
                if len(total_queries) == 1 and len(total_invocation_contexts) == 1:
                    total_queries = input_queries * len(documents)
                    total_invocation_contexts = input_invocation_contexts * len(documents)
                if len(total_queries) != len(documents) or len(total_invocation_contexts) != len(documents):
                    raise ValueError("Number of queries must be equal to number of provided Document lists.")
                for query, invocation_context, cur_docs in zip(total_queries, total_invocation_contexts, documents):
                    inputs["queries"].append(query)
                    inputs["invocation_contexts"].append(invocation_context)
                    inputs["documents"].append(cur_docs)
        elif queries is not None or invocation_contexts is not None:
            for query, invocation_context in zip(input_queries, input_invocation_contexts):
                inputs["queries"].append(query)
                inputs["invocation_contexts"].append(invocation_context)
                inputs["documents"].append([None])
        return inputs
