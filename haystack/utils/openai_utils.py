"""Utils for using OpenAI API"""
import os
import logging
import platform
import sys
import json
from typing import Dict, Union, Tuple
import requests

from transformers import GPT2TokenizerFast

from haystack.errors import OpenAIError, OpenAIRateLimitError
from haystack.utils.reflection import retry_with_exponential_backoff
from haystack.environment import (
    HAYSTACK_REMOTE_API_BACKOFF_SEC,
    HAYSTACK_REMOTE_API_MAX_RETRIES,
    HAYSTACK_REMOTE_API_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)


machine = platform.machine().lower()
system = platform.system()


OPENAI_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
OPENAI_BACKOFF = float(os.environ.get(HAYSTACK_REMOTE_API_BACKOFF_SEC, 10))
OPENAI_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


def get_use_tiktoken():
    """Return True if the tiktoken library is available and False if it is not."""
    use_tiktoken = False
    if sys.version_info >= (3, 8) and (machine in ["amd64", "x86_64"] or (machine == "arm64" and system == "Darwin")):
        use_tiktoken = True

    if not use_tiktoken:
        logger.warning(
            "OpenAI tiktoken module is not available for Python < 3.8,Linux ARM64 and AARCH64. Falling back to GPT2TokenizerFast."
        )
    return use_tiktoken


def get_openai_tokenizer(use_tiktoken: bool, tokenizer_name: str):
    """Load either the tokenizer from tiktoken (if the library is available) or fallback to the GPT2TokenizerFast
    from the transformers library.

    :param use_tiktoken: If True load the tokenizer from the tiktoken library.
                         Otherwise, load a GPT2 tokenizer from transformers.
    :param tokenizer_name: The name of the tokenizer to load.
    """
    if use_tiktoken:
        import tiktoken  # pylint: disable=import-error

        logger.debug("Using tiktoken %s tokenizer", tokenizer_name)
        tokenizer = tiktoken.get_encoding(tokenizer_name)
    else:
        logger.debug("Using GPT2TokenizerFast tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer


@retry_with_exponential_backoff(
    backoff_in_seconds=OPENAI_BACKOFF, max_retries=OPENAI_MAX_RETRIES, errors=(OpenAIRateLimitError, OpenAIError)
)
def openai_request(
    url: str, headers: Dict[str, str], payload: Dict, timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT
):
    """Make a request to the OpenAI API given a `url`, `headers`, `payload`, and `timeout`.

    :param url: The URL of the OpenAI API.
    :param headers: The headers to send with the request.
    :param payload: The payload to send with the request.
    :param timeout: The timeout length of the request. The default is 30s.
    """
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload), timeout=timeout)
    res = json.loads(response.text)

    if response.status_code != 200:
        openai_error: OpenAIError
        if response.status_code == 429:
            openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
        else:
            openai_error = OpenAIError(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )
        raise openai_error

    return res
