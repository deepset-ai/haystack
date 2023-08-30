from typing import Optional
from haystack.preview import ComponentError


class OpenAIError(ComponentError):
    """Exception for issues that occur in the OpenAI APIs"""

    def __init__(
        self, message: Optional[str] = None, status_code: Optional[int] = None, send_message_in_event: bool = False
    ):
        super().__init__(message=message, send_message_in_event=send_message_in_event)
        self.status_code = status_code


class OpenAIRateLimitError(OpenAIError):
    """
    Rate limit error for OpenAI API (status code 429)
    See https://help.openai.com/en/articles/5955604-how-can-i-solve-429-too-many-requests-errors
    See https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
    """

    def __init__(self, message: Optional[str] = None, send_message_in_event: bool = False):
        super().__init__(message=message, status_code=429, send_message_in_event=send_message_in_event)


class OpenAIUnauthorizedError(OpenAIError):
    """
    Unauthorized error for OpenAI API (status code 401)
    See https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    def __init__(self, message: Optional[str] = None, send_message_in_event: bool = False):
        super().__init__(message=message, status_code=401, send_message_in_event=send_message_in_event)
