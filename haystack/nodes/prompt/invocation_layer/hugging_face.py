from typing import Optional, Union, List, Dict, Any
import logging
import os

from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer, TokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler
from haystack.nodes.prompt.invocation_layer.utils import get_task
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install farm-haystack[inference]'") as torch_and_transformers_import:
    import torch
    from transformers import (
        pipeline,
        StoppingCriteriaList,
        StoppingCriteria,
        GenerationConfig,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        PreTrainedModel,
        Pipeline,
        AutoTokenizer,
        AutoConfig,
        TOKENIZER_MAPPING,
    )
    from haystack.modeling.utils import initialize_device_settings  # pylint: disable=ungrouped-imports
    from haystack.nodes.prompt.invocation_layer.handlers import HFTokenStreamingHandler

    class StopWordsCriteria(StoppingCriteria):
        """
        Stops text generation if any one of the stop words is generated.

        Note: When a stop word is encountered, the generation of new text is stopped.
        However, if the stop word is in the prompt itself, it can stop generating new text
        prematurely after the first token. This is particularly important for LLMs designed
        for dialogue generation. For these models, like for example mosaicml/mpt-7b-chat,
        the output includes both the new text and the original prompt. Therefore, it's important
        to make sure your prompt has no stop words.
        """

        def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            stop_words: List[str],
            device: Union[str, torch.device] = "cpu",
        ):
            super().__init__()
            encoded_stop_words = tokenizer(stop_words, add_special_tokens=False, padding=True, return_tensors="pt")
            self.stop_words = encoded_stop_words.input_ids.to(device)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_word in self.stop_words:
                found_stop_word = self.is_stop_word_found(input_ids, stop_word)
                if found_stop_word:
                    return True
            return False

        def is_stop_word_found(self, generated_text_ids: torch.Tensor, stop_word: torch.Tensor) -> bool:
            generated_text_ids = generated_text_ids[-1]
            len_generated_text_ids = generated_text_ids.size(0)
            len_stop_word = stop_word.size(0)
            result = all(generated_text_ids[len_generated_text_ids - len_stop_word :].eq(stop_word))
            return result


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
        max_length: int = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        **kwargs,
    ):
        """
        Creates an instance of HFLocalInvocationLayer used to invoke local Hugging Face models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum number of tokens the output text can have.
        :param use_auth_token: The token to use as HTTP bearer authorization for remote files.
        :param use_gpu: Whether to use GPU for inference.
        :param device: The device to use for inference.
        :param kwargs: Additional keyword arguments passed to the underlying model. Due to reflective construction of
        all PromptModelInvocationLayer instances, this instance of HFLocalInvocationLayer might receive some unrelated
        kwargs. Only kwargs relevant to the HFLocalInvocationLayer are considered. The list of supported kwargs
        includes: "task", "model", "config", "tokenizer", "feature_extractor", "revision", "use_auth_token",
        "device_map", "device", "torch_dtype", "trust_remote_code", "model_kwargs", and "pipeline_class".
        For more details about pipeline kwargs in general, see
        Hugging Face [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline).

        This layer supports two additional kwargs: generation_kwargs and model_max_length.

        The generation_kwargs are used to customize text generation for the underlying pipeline. See Hugging
        Face [docs](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
        for more details.

        The model_max_length is used to specify the custom sequence length for the underlying pipeline.
        """
        torch_and_transformers_import.check()

        super().__init__(model_name_or_path)
        self.use_auth_token = use_auth_token

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )
        if "device" not in kwargs:
            kwargs["device"] = self.devices[0]

        # save stream settings and stream_handler for pipeline invocation
        self.stream_handler = kwargs.get("stream_handler", None)
        self.stream = kwargs.get("stream", False)

        # save generation_kwargs for pipeline invocation
        self.generation_kwargs = kwargs.get("generation_kwargs", {})

        # If task_name is not provided, get the task name from the model name or path (uses HFApi)
        self.task_name = (
            kwargs.get("task_name")
            if "task_name" in kwargs
            else get_task(model_name_or_path, use_auth_token=use_auth_token)
        )
        # we check in supports class method if task_name is supported but here we check again as
        # we could have gotten the task_name from kwargs
        if self.task_name not in ["text2text-generation", "text-generation"]:
            raise ValueError(
                f"Task name {self.task_name} is not supported. "
                f"We only support text2text-generation and text-generation tasks."
            )
        pipeline_kwargs = self._prepare_pipeline_kwargs(
            task=self.task_name, model_name_or_path=model_name_or_path, use_auth_token=use_auth_token, **kwargs
        )
        # create the transformer pipeline
        self.pipe: Pipeline = pipeline(**pipeline_kwargs)

        # This is how the default max_length is determined for Text2TextGenerationPipeline shown here
        # https://huggingface.co/transformers/v4.6.0/_modules/transformers/pipelines/text2text_generation.html
        # max_length must be set otherwise HFLocalInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or self.pipe.model.config.max_length

        model_max_length = kwargs.get("model_max_length", None)
        # we allow users to override the tokenizer's model_max_length because models like T5 have relative positional
        # embeddings and can accept sequences of more than 512 tokens
        if model_max_length is not None:
            self.pipe.tokenizer.model_max_length = model_max_length

        if self.max_length > self.pipe.tokenizer.model_max_length:
            logger.warning(
                "The max_length %s is greater than model_max_length %s. This might result in truncation of the "
                "generated text. Please lower the max_length (number of answer tokens) parameter!",
                self.max_length,
                self.pipe.tokenizer.model_max_length,
            )

    def _prepare_pipeline_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Sanitizes and prepares the kwargs passed to the transformers pipeline function.
        For more details about pipeline kwargs in general, see Hugging Face
        [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline).
        """
        # as device and device_map are mutually exclusive, we set device to None if device_map is provided
        device_map = kwargs.get("device_map", None)
        device = kwargs.get("device") if device_map is None else None
        # prepare torch_dtype for pipeline invocation
        torch_dtype = self._extract_torch_dtype(**kwargs)
        # and the model (prefer model instance over model_name_or_path str identifier)
        model = kwargs.get("model") or kwargs.get("model_name_or_path")
        trust_remote_code = kwargs.get("trust_remote_code", False)
        hub_kwargs = {
            "revision": kwargs.get("revision", None),
            "use_auth_token": kwargs.get("use_auth_token", None),
            "trust_remote_code": trust_remote_code,
        }
        model_kwargs = kwargs.get("model_kwargs", {})
        tokenizer = kwargs.get("tokenizer", None)

        if tokenizer is None and trust_remote_code:
            # For models not yet supported by the transformers library, we must set `trust_remote_code=True` within
            # the underlying pipeline to ensure the model's successful loading. However, this does not guarantee the
            # tokenizer will be loaded alongside. Therefore, we need to add additional logic here to manually load the
            # tokenizer and pass it to transformers' pipeline.
            # Otherwise, calling `self.pipe.tokenizer.model_max_length` will return an error.
            tokenizer = self._prepare_tokenizer(model, hub_kwargs, model_kwargs)

        pipeline_kwargs = {
            "task": kwargs.get("task", None),
            "model": model,
            "config": kwargs.get("config", None),
            "tokenizer": tokenizer,
            "feature_extractor": kwargs.get("feature_extractor", None),
            "device_map": device_map,
            "device": device,
            "torch_dtype": torch_dtype,
            "model_kwargs": model_kwargs,
            "pipeline_class": kwargs.get("pipeline_class", None),
            "use_fast": kwargs.get("use_fast", True),
            **hub_kwargs,
        }
        return pipeline_kwargs

    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated texts using the local Hugging Face transformers model
        :return: A list of generated texts.

        Note: Only kwargs relevant to Text2TextGenerationPipeline and TextGenerationPipeline are passed to
        Hugging Face as model_input_kwargs. Other kwargs are ignored.
        """
        output: List[Dict[str, str]] = []
        stop_words = kwargs.pop("stop_words", None)
        top_k = kwargs.pop("top_k", None)
        # either stream is True (will use default handler) or stream_handler is provided for custom handler
        stream = kwargs.get("stream", self.stream)
        stream_handler = kwargs.get("stream_handler", self.stream_handler)
        stream = stream or stream_handler is not None
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")

            # Consider only Text2TextGenerationPipeline and TextGenerationPipeline relevant, ignore others
            # For more details refer to Hugging Face Text2TextGenerationPipeline and TextGenerationPipeline
            # documentation
            model_input_kwargs = {
                key: kwargs[key]
                for key in [
                    "return_tensors",
                    "return_text",
                    "return_full_text",
                    "clean_up_tokenization_spaces",
                    "truncation",
                    "generation_kwargs",
                    "max_new_tokens",
                    "num_beams",
                    "do_sample",
                    "num_return_sequences",
                    "max_length",
                ]
                if key in kwargs
            }
            generation_kwargs = model_input_kwargs.pop("generation_kwargs", self.generation_kwargs)
            if isinstance(generation_kwargs, dict):
                model_input_kwargs.update(generation_kwargs)
            elif isinstance(generation_kwargs, GenerationConfig):
                gen_dict = generation_kwargs.to_diff_dict()
                gen_dict.pop("transformers_version", None)
                model_input_kwargs.update(gen_dict)

            is_text_generation = "text-generation" == self.task_name
            # Prefer return_full_text is False for text-generation (unless explicitly set)
            # Thus only generated text is returned (excluding prompt)
            if is_text_generation and "return_full_text" not in model_input_kwargs:
                model_input_kwargs["return_full_text"] = False
            if stop_words:
                sw = StopWordsCriteria(tokenizer=self.pipe.tokenizer, stop_words=stop_words, device=self.pipe.device)
                model_input_kwargs["stopping_criteria"] = StoppingCriteriaList([sw])
            if top_k:
                model_input_kwargs["num_return_sequences"] = top_k
                if "num_beams" not in model_input_kwargs or model_input_kwargs["num_beams"] < top_k:
                    if "num_beams" in model_input_kwargs:
                        logger.warning("num_beams should not be less than top_k, hence setting it to %s", top_k)
                    model_input_kwargs["num_beams"] = top_k
            # max_new_tokens is used for text-generation and max_length for text2text-generation
            if is_text_generation:
                model_input_kwargs["max_new_tokens"] = model_input_kwargs.pop("max_length", self.max_length)
            else:
                model_input_kwargs["max_length"] = model_input_kwargs.pop("max_length", self.max_length)

            if stream:
                stream_handler: TokenStreamingHandler = stream_handler or DefaultTokenStreamingHandler()
                model_input_kwargs["streamer"] = HFTokenStreamingHandler(self.pipe.tokenizer, stream_handler)

            output = self.pipe(prompt, **model_input_kwargs)
        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]

        if stop_words:
            # Although HF generates text until stop words are encountered unfortunately it includes the stop word
            # We want to exclude it to be consistent with other invocation layers
            for idx, _ in enumerate(generated_texts):
                for stop_word in stop_words:
                    generated_texts[idx] = generated_texts[idx].replace(stop_word, "").rstrip()
        return generated_texts

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        model_max_length = self.pipe.tokenizer.model_max_length
        tokenized_prompt = self.pipe.tokenizer.tokenize(prompt)
        n_prompt_tokens = len(tokenized_prompt)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= model_max_length:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
            "answer length (%s tokens) fit within the max token limit (%s tokens). "
            "Shorten the prompt to prevent it from being cut off",
            n_prompt_tokens,
            max(0, model_max_length - n_answer_tokens),
            n_answer_tokens,
            model_max_length,
        )

        decoded_string = self.pipe.tokenizer.convert_tokens_to_string(
            tokenized_prompt[: model_max_length - n_answer_tokens]
        )
        return decoded_string

    def _extract_torch_dtype(self, **kwargs) -> Optional["torch.dtype"]:
        torch_dtype_resolved = None
        torch_dtype = kwargs.get("torch_dtype", None)
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
        return torch_dtype_resolved

    def _prepare_tokenizer(
        self, model: Union[str, "PreTrainedModel"], hub_kwargs: Dict, model_kwargs: Optional[Dict] = None
    ) -> Union["PreTrainedTokenizer", "PreTrainedTokenizerFast", None]:
        """
        This method prepares the tokenizer before passing it to transformers' pipeline, so that the instantiated pipeline
        object has a working tokenizer.

        It checks whether the pipeline method in the transformers library will load the tokenizer.
        - If yes, None will be returned, because in this case, the pipeline is intelligent enough to load the tokenizer by itself.
        - If not, we will load the tokenizer and an tokenizer instance is returned.

        :param model: The name or path of the underlying model.
        :hub_kwargs: Keyword argument related to hugging face hub, including revision, trust_remote_code and use_auth_token.
        :model_kwargs: Keyword arguments passed to the underlying model.
        """

        if isinstance(model, str):
            model_config = AutoConfig.from_pretrained(model, **hub_kwargs, **model_kwargs)
        else:
            model_config = model.config
            model = model_config._name_or_path
        # the will_load_tokenizer logic corresponds to this line in transformers library
        # https://github.com/huggingface/transformers/blob/05cda5df3405e6a2ee4ecf8f7e1b2300ebda472e/src/transformers/pipelines/__init__.py#L805
        will_load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
        if not will_load_tokenizer:
            logger.warning(
                "The transformers library doesn't know which tokenizer class should be "
                "loaded for the model %s. Therefore, the tokenizer will be loaded in Haystack's "
                "invocation layer and then passed to the underlying pipeline. Alternatively, you could "
                "pass `tokenizer_class` to `model_kwargs` to workaround this, if your tokenizer is supported "
                "by the transformers library.",
                model,
            )
            tokenizer = AutoTokenizer.from_pretrained(model, **hub_kwargs, **model_kwargs)
        else:
            tokenizer = None
        return tokenizer

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        task_name: Optional[str] = kwargs.get("task_name", None)
        if os.path.exists(model_name_or_path):
            return True

        try:
            task_name = task_name or get_task(model_name_or_path, use_auth_token=kwargs.get("use_auth_token", None))
        except RuntimeError:
            # This will fail for all non-HF models
            return False
        # if we are using an api_key it could be HF inference point
        using_api_key = bool(kwargs.get("api_key", None))
        return not using_api_key and task_name in ["text2text-generation", "text-generation"]
