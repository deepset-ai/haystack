from abc import abstractmethod, ABC


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
