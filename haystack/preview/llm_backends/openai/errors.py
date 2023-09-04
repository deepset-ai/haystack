from typing import Optional
from haystack.preview import ComponentError


class OpenAIError(ComponentError):
    """Exception for issues that occur in the OpenAI APIs"""

    def __init__(self, message: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__()
        self.message = message
        self.status_code = status_code

    def __str__(self):
        return self.message + f"(status code {self.status_code})" if self.status_code else ""


class OpenAIRateLimitError(OpenAIError):
    """
    Rate limit error for OpenAI API (status code 429)
    See https://help.openai.com/en/articles/5955604-how-can-i-solve-429-too-many-requests-errors
    See https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
    """

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message, status_code=429)


class OpenAIUnauthorizedError(OpenAIError):
    """
    Unauthorized error for OpenAI API (status code 401)
    See https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message, status_code=401)
