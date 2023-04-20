import logging
from typing import Dict, List, Optional, Union, Any

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
from generalimport import is_imported

from haystack.preview.components.prompt.models.base import prompt_model


logger = logging.getLogger(__name__)


@prompt_model
class HFLocalModel:
    """
    It loads a pre-trained model from Hugging Face and passes a prepared prompt into that model.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-base",
        max_length: Optional[int] = 100,
        use_auth_token: Optional[Union[str, bool]] = None,
        device: Optional[Union[str, torch.device]] = None,
        default_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a model implementation for local Hugging Face models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum length of the output text.
        :param use_auth_token: The token to use as HTTP bearer authorization for remote files.
        :param use_gpu: Whether to use GPU for inference.
        :param device: The device to use for inference.
        :param default_model_params: Additional keyword arguments passed to the underlying model.
            The list of supported parameters includes:
             - `trust_remote_code`
             - `revision`
             - `feature_extractor`
             - `tokenizer`
             - `config`
             - `use_fast`
             - `torch_dtype`
             - `device_map`
            For more details about these parameters, see Hugging Face
            [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline).
        """
        self.model_name_or_path = model_name_or_path
        self.use_auth_token = use_auth_token
        self.device: torch.device = torch.device(device) if device else torch.device("cpu")

        if not default_model_params:
            default_model_params = {}

        if "torch_dtype" in default_model_params.keys():
            default_model_params["torch_dtype"] = self._resolve_dtype(dtype=default_model_params["torch_dtype"])

        # Do not use `device_map` AND `device` at the same time as they will conflict
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name_or_path,
            device=self.device if "device_map" not in default_model_params else None,
            use_auth_token=self.use_auth_token,
            model_kwargs=default_model_params,
        )
        # This is how the default max_length is determined for Text2TextGenerationPipeline shown here
        # https://huggingface.co/transformers/v4.6.0/_modules/transformers/pipelines/text2text_generation.html
        # max_length must be set otherwise ensure_token_limit() will fail.
        self.max_length = max_length or self.pipe.model.config.max_length

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Tries to load the model with AutoConfig to assess whether it is a HF model.

        :param model_name_or_path: the model identifier.
        :param **kwargs: any other argument needed to load this model.
        :returns: True if the model is compatible with this implementation, False otherwise.
        """
        if not is_imported("transformers"):
            logger.debug("Either tranformers or torch could not be imported. HuggingFace models can't run.")
            return False
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        except Exception as exc:
            logger.debug("This model doesn't seem to be a HuggingFace model. Exception: %s", str(exc))
            return False

        if not config or not config.architectures:
            return False

        if not all(m in model_name_or_path for m in ["flan", "t5"]):
            logger.warning(
                "This language model (%s) might not be fine-tuned on instruction-following tasks. "
                "Many of the default prompts will likely not work as intended. If you believe this is a false "
                "positive, open an issue on our GitHub: https://github.com/deepset-ai/haystack/issues/new",
                model_name_or_path,
            )
        supported_models = list(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.values())
        return config.architectures[0] in supported_models

    def invoke(self, prompt: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Sends a prompt the model.

        :param prompt: the prompt to send to the model
        :param model_params: any other parameter needed to invoke this model.
        :return: The responses from the model.
        """
        output: List[Dict[str, str]] = []
        if not model_params:
            model_params = {}

        model_params = self._translate_model_parameters(model_params=model_params)
        prompt = self.ensure_token_limit(prompt=prompt)
        output = self.pipe(prompt, max_length=self.max_length, **model_params)
        output = [o["generated_text"] for o in output if "generated_text" in o]

        if model_params.get("stopping_criteria"):
            output = self._exclude_stop_words(output)
        return output

    def ensure_token_limit(self, prompt: str) -> str:
        """
        Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        :returns: the same prompt, cut down to the maximum length if it was too long. The tail of the prompt is cut.
        """
        n_prompt_tokens = len(self.pipe.tokenizer.tokenize(prompt))
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.pipe.tokenizer.model_max_length:
            return prompt

        logger.error(
            "The prompt has been truncated from %s tokens to %s tokens such that the prompt length and "
            "answer length (%s tokens) fits within the max token limit (%s tokens). "
            "Shorten the prompt to avoid this error.",
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

    def _translate_model_parameters(self, model_params: Dict[str, Any]):
        """
        Some parameter names might need to be converted to be understood by the model.
        For example, we use 'stop_words' but HF models use `StopWordsCriteria` instances.
        """
        stop_words = model_params.pop("stop_words", None)
        if stop_words:
            sw = _StopWordsCriteria(tokenizer=self.pipe.tokenizer, stop_words=stop_words)
            model_params["stopping_criteria"] = StoppingCriteriaList([sw])
        top_k = model_params.pop("top_k", None)
        if top_k:
            model_params["num_return_sequences"] = top_k
            model_params["num_beams"] = top_k
        return model_params

    def _exclude_stop_words(self, generated_texts: List[str], stop_words: List[str]):
        """
        Removes the stopwords from the model's output for consistency with other implementations.
        """
        for idx, _ in enumerate(generated_texts):
            for stop_word in stop_words:
                generated_texts[idx] = generated_texts[idx].replace(stop_word, "").strip()

    def _resolve_dtype(self, dtype: Union[str, torch.dtype]):
        """
        Resolve torch dtype if given in the model kwargs.
        """
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            try:
                return getattr(torch, dtype.strip("torch.") if dtype.startswith("torch.") else dtype)
            except Exception as exc:
                raise ValueError(f"Invalid torch_dtype value {dtype}") from exc
        raise ValueError(f"Invalid torch_dtype value {dtype}, it must be either a torch.dtype or its name.")


class _StopWordsCriteria(StoppingCriteria):
    """
    Stops text generation if any one of the stop words is generated.
    """

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], stop_words: List[str]):
        super().__init__()
        self.stop_words = tokenizer.encode(stop_words, add_special_tokens=False, return_tensors="pt")

    def __call__(self, input_ids: torch.LongTensor, **kwargs) -> bool:
        return any(torch.isin(input_ids[-1], self.stop_words[-1]))
