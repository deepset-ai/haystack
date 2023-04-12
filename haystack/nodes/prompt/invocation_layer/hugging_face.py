from typing import Optional, Union, List, Dict
import logging

import torch
from transformers import pipeline, StoppingCriteriaList, StoppingCriteria, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.pipelines import get_task

from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer

logger = logging.getLogger(__name__)


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
        self.task_name = get_task(model_name_or_path, use_auth_token=use_auth_token)
        self.pipe = pipeline(
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
        It takes a prompt and returns a list of generated texts using the local Hugging Face transformers model
        :return: A list of generated texts.

        Note: Only kwargs relevant to Text2TextGenerationPipeline and TextGenerationPipeline are passed to
        Hugging Face as model_input_kwargs. Other kwargs are ignored.
        """
        output: List[Dict[str, str]] = []
        stop_words = kwargs.pop("stop_words", None)
        top_k = kwargs.pop("top_k", None)
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")

            # Consider only Text2TextGenerationPipeline and TextGenerationPipeline relevant, ignore others
            # For more details refer to Hugging Face Text2TextGenerationPipeline and TextGenerationPipeline
            # documentation
            # TODO resolve these kwargs from the pipeline signature
            model_input_kwargs = {
                key: kwargs[key]
                for key in [
                    "return_tensors",
                    "return_text",
                    "return_full_text",
                    "clean_up_tokenization_spaces",
                    "truncation",
                ]
                if key in kwargs
            }
            is_text_generation = "text-generation" == self.task_name
            # Prefer return_full_text is False for text-generation (unless explicitly set)
            # Thus only generated text is returned (excluding prompt)
            if is_text_generation and "return_full_text" not in model_input_kwargs:
                model_input_kwargs["return_full_text"] = False
                model_input_kwargs["max_new_tokens"] = self.max_length
            if stop_words:
                sw = StopWordsCriteria(tokenizer=self.pipe.tokenizer, stop_words=stop_words, device=self.pipe.device)
                model_input_kwargs["stopping_criteria"] = StoppingCriteriaList([sw])
            if top_k:
                model_input_kwargs["num_return_sequences"] = top_k
                model_input_kwargs["num_beams"] = top_k
            # max_new_tokens is used for text-generation and max_length for text2text-generation
            if is_text_generation:
                model_input_kwargs["max_new_tokens"] = self.max_length
            else:
                model_input_kwargs["max_length"] = self.max_length

            output = self.pipe(prompt, **model_input_kwargs)
        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]

        if stop_words:
            # Although HF generates text until stop words are encountered unfortunately it includes the stop word
            # We want to exclude it to be consistent with other invocation layers
            for idx, _ in enumerate(generated_texts):
                for stop_word in stop_words:
                    generated_texts[idx] = generated_texts[idx].replace(stop_word, "").strip()
        return generated_texts

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        n_prompt_tokens = len(self.pipe.tokenizer.tokenize(prompt))
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.pipe.tokenizer.model_max_length:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
            "answer length (%s tokens) fit within the max token limit (%s tokens). "
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
        task_name: Optional[str] = None
        try:
            task_name = get_task(model_name_or_path, use_auth_token=kwargs.get("use_auth_token", None))
        except RuntimeError:
            # This will fail for all non-HF models
            return False

        return task_name in ["text2text-generation", "text-generation"]


class StopWordsCriteria(StoppingCriteria):
    """
    Stops text generation if any one of the stop words is generated.
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        stop_words: List[str],
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.stop_words = tokenizer(stop_words, add_special_tokens=False, return_tensors="pt").to(device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(torch.isin(input_ids[-1], self.stop_words["input_ids"]))
