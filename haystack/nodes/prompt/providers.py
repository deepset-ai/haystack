import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Union, Type

import torch
from transformers import (
    pipeline,
    AutoConfig,
    StoppingCriteriaList,
    StoppingCriteria,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES

from haystack.errors import OpenAIError
from haystack.modeling.utils import initialize_device_settings
from haystack.utils.openai_utils import (
    USE_TIKTOKEN,
    openai_request,
    _openai_text_completion_tokenization_details,
    load_openai_tokenizer,
    _check_openai_text_completion_answers,
    count_openai_tokens,
)

logger = logging.getLogger(__name__)


class PromptModelInvocationLayer:
    """
    PromptModelInvocationLayer implementations execute a prompt on an underlying model.

    The implementation can be a simple invocation on the underlying model running in a local runtime, or
    could be even remote, for example, a call to a remote API endpoint.
    """

    invocation_layer_providers: List[Type["PromptModelInvocationLayer"]] = []

    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Creates a new PromptModelInvocationLayer instance.

        :param model_name_or_path: The name or path of the underlying model.
        :param kwargs: Additional keyword arguments passed to the underlying model.
        """
        if model_name_or_path is None or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model_name_or_path = model_name_or_path

    def __init_subclass__(cls, **kwargs):
        """
        Used to register user-defined invocation layers.

        Called when a subclass of PromptModelInvocationLayer is imported.
        """
        super().__init_subclass__(**kwargs)
        cls.invocation_layer_providers.append(cls)

    @abstractmethod
    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated text using the underlying model.
        :return: A list of generated text.
        """
        pass

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Checks if the given model is supported by this invocation layer.

        :param model_name_or_path: The name or path of the model.
        :param kwargs: additional keyword arguments passed to the underlying model which might be used to determine
        if the model is supported.
        :return: True if this invocation layer supports the model, False otherwise.
        """
        return False

    @abstractmethod
    def _ensure_token_limit(self, prompt: str) -> str:
        """Ensure that length of the prompt and answer is within the maximum token length of the PromptModel.

        :param prompt: Prompt text to be sent to the generative model.
        """
        pass


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
        super().__init__(model_name_or_path)
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
                if "torch." in torch_dtype:
                    torch_dtype_resolved = getattr(torch, torch_dtype.strip("torch."))
                elif torch_dtype == "auto":
                    torch_dtype_resolved = torch_dtype
                else:
                    raise ValueError(
                        f"torch_dtype should be a torch.dtype, a string with 'torch.' prefix or the string 'auto', got {torch_dtype}"
                    )
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
        # This is how the default max_length is determined for Text2TextGenerationPipeline shown here
        # https://huggingface.co/transformers/v4.6.0/_modules/transformers/pipelines/text2text_generation.html
        # max_length must be set otherwise HFLocalInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or self.pipe.model.config.max_length

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

    def _ensure_token_limit(self, prompt: str) -> str:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        n_prompt_tokens = len(self.pipe.tokenizer.tokenize(prompt))
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.pipe.tokenizer.model_max_length:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens such that the prompt length and "
            "answer length (%s tokens) fits within the max token limit (%s tokens). "
            "Shorten the prompt to prevent it from being cut off",
            n_prompt_tokens,
            self.pipe.tokenizer.model_max_length - n_answer_tokens,
            n_answer_tokens,
            self.pipe.tokenizer.model_max_length,
        )

        tokenized_payload = self.pipe.tokenizer.tokenize(prompt)
        decoded_string = self.pipe.tokenizer.convert_tokens_to_string(
            tokenized_payload[: self.pipe.tokenizer.model_max_length - n_answer_tokens]
        )
        return decoded_string

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
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
        super().__init__(model_name_or_path)
        if not isinstance(api_key, str) or len(api_key) == 0:
            raise OpenAIError(
                f"api_key {api_key} must be a valid OpenAI key. Visit https://openai.com/api/ to get one."
            )
        self.api_key = api_key

        # 16 is the default length for answers from OpenAI shown in the docs
        # here, https://platform.openai.com/docs/api-reference/completions/create.
        # max_length must be set otherwise OpenAIInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or 16

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

        tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(
            model_name=self.model_name_or_path
        )
        self.max_tokens_limit = max_tokens_limit
        self._tokenizer = load_openai_tokenizer(tokenizer_name=tokenizer_name)

    @property
    def url(self) -> str:
        return "https://api.openai.com/v1/completions"

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

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
        res = openai_request(url=self.url, headers=self.headers, payload=payload)
        _check_openai_text_completion_answers(result=res, payload=payload)
        responses = [ans["text"].strip() for ans in res["choices"]]
        return responses

    def _ensure_token_limit(self, prompt: str) -> str:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        n_prompt_tokens = count_openai_tokens(prompt, self._tokenizer)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.max_tokens_limit:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens such that the prompt length and "
            "answer length (%s tokens) fits within the max token limit (%s tokens). "
            "Reduce the length of the prompt to prevent it from being cut off.",
            n_prompt_tokens,
            self.max_tokens_limit - n_answer_tokens,
            n_answer_tokens,
            self.max_tokens_limit,
        )

        if USE_TIKTOKEN:
            tokenized_payload = self._tokenizer.encode(prompt)
            decoded_string = self._tokenizer.decode(tokenized_payload[: self.max_tokens_limit - n_answer_tokens])
        else:
            tokenized_payload = self._tokenizer.tokenize(prompt)
            decoded_string = self._tokenizer.convert_tokens_to_string(
                tokenized_payload[: self.max_tokens_limit - n_answer_tokens]
            )
        return decoded_string

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        valid_model = any(m for m in ["ada", "babbage", "davinci", "curie"] if m in model_name_or_path)
        return valid_model and kwargs.get("azure_base_url") is None


class AzureOpenAIInvocationLayer(OpenAIInvocationLayer):
    """
    Azure OpenAI Invocation Layer

    This layer is used to invoke the OpenAI API on Azure. It is essentially the same as the OpenAIInvocationLayer
    with additional two parameters: azure_base_url and azure_deployment_name. The azure_base_url is the URL of the Azure OpenAI
    endpoint and the azure_deployment_name is the name of the deployment.
    """

    def __init__(
        self,
        azure_base_url: str,
        azure_deployment_name: str,
        api_key: str,
        api_version: str = "2022-12-01",
        model_name_or_path: str = "text-davinci-003",
        max_length: Optional[int] = 100,
        **kwargs,
    ):
        super().__init__(api_key, model_name_or_path, max_length, **kwargs)
        self.azure_base_url = azure_base_url
        self.azure_deployment_name = azure_deployment_name
        self.api_version = api_version

    @property
    def url(self) -> str:
        return f"{self.azure_base_url}/openai/deployments/{self.azure_deployment_name}/completions?api-version={self.api_version}"

    @property
    def headers(self) -> Dict[str, str]:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Ensures Azure OpenAI Invocation Layer is selected when azure_base_url and azure_deployment_name are provided in
        addition to a list of supported models.
        """
        valid_model = any(m for m in ["ada", "babbage", "davinci", "curie"] if m in model_name_or_path)
        return (
            valid_model and kwargs.get("azure_base_url") is not None and kwargs.get("azure_deployment_name") is not None
        )
