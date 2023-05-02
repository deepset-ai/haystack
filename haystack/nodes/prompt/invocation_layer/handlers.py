from abc import abstractmethod, ABC
from typing import Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, TextStreamer


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


class HFTokenStreamingHandler(TextStreamer):
    def __init__(
        self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], stream_handler: TokenStreamingHandler
    ):
        super().__init__(tokenizer=tokenizer)
        self.token_handler = stream_handler

    def on_finalized_text(self, token: str, stream_end: bool = False):
        token_to_send = token + "\n" if stream_end else token
        self.token_handler(token_received=token_to_send, **{})
