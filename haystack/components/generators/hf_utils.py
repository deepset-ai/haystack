import inspect
from typing import Any, Dict, List, Optional, Union, Callable

from haystack.dataclasses import StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import InferenceClient, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

PIPELINE_SUPPORTED_TASKS = ["text-generation", "text2text-generation"]


def check_generation_params(kwargs: Optional[Dict[str, Any]], additional_accepted_params: Optional[List[str]] = None):
    """
    Check the provided generation parameters for validity.

    :param kwargs: A dictionary containing the generation parameters.
    :param additional_accepted_params: An optional list of strings representing additional accepted parameters.
    :raises ValueError: If any unknown text generation parameters are provided.
    """
    transformers_import.check()

    if kwargs:
        accepted_params = {
            param
            for param in inspect.signature(InferenceClient.text_generation).parameters.keys()
            if param not in ["self", "prompt"]
        }
        if additional_accepted_params:
            accepted_params.update(additional_accepted_params)
        unknown_params = set(kwargs.keys()) - accepted_params
        if unknown_params:
            raise ValueError(
                f"Unknown text generation parameters: {unknown_params}. The valid parameters are: {accepted_params}."
            )


def check_valid_model(model_id: str, token: Optional[Secret]) -> None:
    """
    Check if the provided model ID corresponds to a valid model on HuggingFace Hub.
    Also check if the model is a text generation model.

    :param model_id: A string representing the HuggingFace model ID.
    :param token: An optional authentication token.
    :raises ValueError: If the model is not found or is not a text generation model.
    """
    transformers_import.check()

    api = HfApi()
    try:
        model_info = api.model_info(model_id, token=token.resolve_value() if token else None)
    except RepositoryNotFoundError as e:
        raise ValueError(
            f"Model {model_id} not found on HuggingFace Hub. Please provide a valid HuggingFace model_id."
        ) from e

    allowed_model = model_info.pipeline_tag in ["text-generation", "text2text-generation"]
    if not allowed_model:
        raise ValueError(f"Model {model_id} is not a text generation model. Please provide a text generation model.")


with LazyImport(message="Run 'pip install transformers[torch]'") as torch_and_transformers_import:
    import torch
    from transformers import StoppingCriteria, PreTrainedTokenizer, PreTrainedTokenizerFast, TextStreamer

    transformers_import.check()

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
            # check if tokenizer is a valid tokenizer
            if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                raise ValueError(
                    f"Invalid tokenizer provided for StopWordsCriteria - {tokenizer}. "
                    f"Please provide a valid tokenizer from the HuggingFace Transformers library."
                )
            if not tokenizer.pad_token:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            encoded_stop_words = tokenizer(stop_words, add_special_tokens=False, padding=True, return_tensors="pt")
            self.stop_ids = encoded_stop_words.input_ids.to(device)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in self.stop_ids:
                found_stop_word = self.is_stop_word_found(input_ids, stop_id)
                if found_stop_word:
                    return True
            return False

        def is_stop_word_found(self, generated_text_ids: torch.Tensor, stop_id: torch.Tensor) -> bool:
            generated_text_ids = generated_text_ids[-1]
            len_generated_text_ids = generated_text_ids.size(0)
            len_stop_id = stop_id.size(0)
            result = all(generated_text_ids[len_generated_text_ids - len_stop_id :].eq(stop_id))
            return result

    class HFTokenStreamingHandler(TextStreamer):
        def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            stream_handler: Callable[[StreamingChunk], None],
            stop_words: Optional[List[str]] = None,
        ):
            super().__init__(tokenizer=tokenizer, skip_prompt=True)  # type: ignore
            self.token_handler = stream_handler
            self.stop_words = stop_words or []

        def on_finalized_text(self, word: str, stream_end: bool = False):
            word_to_send = word + "\n" if stream_end else word
            if word_to_send.strip() not in self.stop_words:
                self.token_handler(StreamingChunk(content=word_to_send))
