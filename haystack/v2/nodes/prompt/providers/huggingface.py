import logging
from typing import Dict, List, Optional, Union, Any

from haystack.modeling.utils import initialize_device_settings
from haystack.v2.nodes.prompt.providers import prompt_model_provider


logger = logging.getLogger(__name__)


TRANSFORMERS_IMPORTED = False
try:
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

    TRANSFORMERS_IMPORTED = False
except ImportError as exc:
    logger.debug("Either tranformers or torch could not be imported. HuggingFace models will fail to initialize.")


@prompt_model_provider
class HFLocalInvocationLayer:
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
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of HFLocalInvocationLayer used to invoke local Hugging Face models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum length of the output text.
        :param use_auth_token: The token to use as HTTP bearer authorization for remote files.
        :param use_gpu: Whether to use GPU for inference.
        :param device: The device to use for inference.
        :param model_kwargs: Additional keyword arguments passed to the underlying model.
        Only kwargs relevant to the HFLocalInvocationLayer are considered. The list of supported kwargs
        includes: `trust_remote_code`, `revision`, `feature_extractor`, `tokenizer`, `config`, `use_fast`,
        `torch_dtype`, `device_map`. For more details about these kwargs, see Hugging Face
        [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline).
        """
        self.model_name_or_path = model_name_or_path
        self.use_auth_token = use_auth_token

        if not TRANSFORMERS_IMPORTED:
            raise ImportError(
                "Either tranformers or torch could not be imported. "
                "HuggingFace models cannot be used without these dependencies. "
                "Run 'pip install transformers[torch]' to fix this issue."
            )

        # Initialize torch devices
        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        # Resolve torch_dtype
        model_kwargs["torch_dtype"] = self._resolve_dtype(dtype_name=model_kwargs.get("torch_dtype"))

        # Do not use `device_map` AND `device` at the same time as they will conflict
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name_or_path,
            device=self.devices[0] if "device_map" not in model_kwargs else None,
            use_auth_token=self.use_auth_token,
            model_kwargs=model_kwargs,
        )
        # This is how the default max_length is determined for Text2TextGenerationPipeline shown here
        # https://huggingface.co/transformers/v4.6.0/_modules/transformers/pipelines/text2text_generation.html
        # max_length must be set otherwise HFLocalInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or self.pipe.model.config.max_length

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Tries to load the model with AutoConfig to assess whether it is a HF model.

        :param model_name_or_path: the name of the model to load.
        :returns: True if the model is recognized by AutoConfig, False otherwise.
        """
        if not TRANSFORMERS_IMPORTED:
            logger.debug("Either tranformers or torch could not be imported. HuggingFace models can't run.")
            return False
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
        except OSError:
            # This is needed so other models are skipped over
            return False

        if not all(m in model_name_or_path for m in ["flan", "t5"]):
            logger.warning(
                "PromptNode might have been initialized with a language model not fine-tuned on instruction following tasks. "
                "Many of the default prompts and PromptTemplates will likely not work as intended. "
                "Use custom prompts and PromptTemplates specific to the %s model",
                model_name_or_path,
            )

        supported_models = list(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.values())
        return config.architectures[0] in supported_models

    def invoke(
        self,
        prompt: str,
        top_k: int = 1,
        stop_words: Optional[List[str]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        It takes a prompt and returns a list of generated text using the local Hugging Face transformers model
        :return: A list of generated text.
        """
        output: List[Dict[str, str]] = []

        prompt = self._ensure_token_limit(prompt=prompt)

        if stop_words:
            sw = StopWordsCriteria(tokenizer=self.pipe.tokenizer, stop_words=stop_words)
            model_kwargs["stopping_criteria"] = StoppingCriteriaList([sw])
        if top_k:
            model_kwargs["num_return_sequences"] = top_k
            model_kwargs["num_beams"] = top_k

        output = self.pipe(prompt, max_length=self.max_length, **model_kwargs)
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

    def _resolve_dtype(dtype_name: Union[str, torch.dtype]):
        """
        Resolve torch dtype if given in the model kwargs
        """
        if dtype_name is not None:
            if isinstance(dtype_name, str):
                if "torch." in dtype_name:
                    return getattr(torch, dtype_name.strip("torch."))
                elif dtype_name == "auto":
                    return dtype_name
                else:
                    raise ValueError(
                        "torch_dtype should be a torch.dtype, a string with 'torch.' "
                        f"prefix or the string 'auto', got '{dtype_name}'"
                    )
            elif isinstance(dtype_name, torch.dtype):
                return dtype_name
            else:
                raise ValueError(f"Invalid torch_dtype value {dtype_name}")


class StopWordsCriteria(StoppingCriteria):
    """
    Stops text generation if any one of the stop words is generated.
    """

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], stop_words: List[str]):
        super().__init__()
        self.stop_words = tokenizer.encode(stop_words, add_special_tokens=False, return_tensors="pt")

    def __call__(self, input_ids: torch.LongTensor, **kwargs) -> bool:
        return any(torch.isin(input_ids[-1], self.stop_words[-1]))
