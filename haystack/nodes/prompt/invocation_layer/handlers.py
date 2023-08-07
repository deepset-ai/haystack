from abc import abstractmethod, ABC
from typing import Union, Dict

from haystack.lazy_imports import LazyImport

TextStreamer = object
with LazyImport() as transformers_import:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, TextStreamer, AutoTokenizer  # type: ignore


class TokenStreamingHandler(ABC):
    """
    TokenStreamingHandler implementations handle the streaming of tokens from the stream.
    """

    DONE_MARKER = "[DONE]"

    @abstractmethod
    def __call__(self, token_received: str, **kwargs) -> str:
        """
        This callback method is called when a new token is received from the stream.

        :param token_received: The token received from the stream.
        :param kwargs: Additional keyword arguments passed to the handler.
        :return: The token to be sent to the stream.
        """
        pass


class DefaultTokenStreamingHandler(TokenStreamingHandler):
    def __call__(self, token_received, **kwargs) -> str:
        """
        This callback method is called when a new token is received from the stream.

        :param token_received: The token received from the stream.
        :param kwargs: Additional keyword arguments passed to the handler.
        :return: The token to be sent to the stream.
        """
        print(token_received, flush=True, end="")
        return token_received


class HFTokenStreamingHandler(TextStreamer):  # pylint: disable=useless-object-inheritance
    def __init__(
        self,
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        stream_handler: "TokenStreamingHandler",
    ):
        transformers_import.check()
        super().__init__(tokenizer=tokenizer, skip_prompt=True)  # type: ignore
        self.token_handler = stream_handler

    def on_finalized_text(self, token: str, stream_end: bool = False):
        token_to_send = token + "\n" if stream_end else token
        self.token_handler(token_received=token_to_send, **{})


class DefaultPromptHandler:
    """
    DefaultPromptHandler resizes the prompt to ensure that the prompt and answer token lengths together
    are within the model_max_length.
    """

    def __init__(self, model_name_or_path: str, model_max_length: int, max_length: int = 100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_max_length = model_max_length
        self.max_length = max_length

    def __call__(self, prompt: str, **kwargs) -> Dict[str, Union[str, int]]:
        """
        Resizes the prompt to ensure that the prompt and answer is within the model_max_length

        :param prompt: the prompt to be sent to the model.
        :param kwargs: Additional keyword arguments passed to the handler.
        :return: A dictionary containing the resized prompt and additional information.
        """
        resized_prompt = prompt
        prompt_length = 0
        new_prompt_length = 0

        if prompt:
            tokenized_prompt = self.tokenizer.tokenize(prompt)
            prompt_length = len(tokenized_prompt)
            if (prompt_length + self.max_length) <= self.model_max_length:
                resized_prompt = prompt
                new_prompt_length = prompt_length
            else:
                resized_prompt = self.tokenizer.convert_tokens_to_string(
                    tokenized_prompt[: self.model_max_length - self.max_length]
                )
                new_prompt_length = len(tokenized_prompt[: self.model_max_length - self.max_length])

        return {
            "resized_prompt": resized_prompt,
            "prompt_length": prompt_length,
            "new_prompt_length": new_prompt_length,
            "model_max_length": self.model_max_length,
            "max_length": self.max_length,
        }
