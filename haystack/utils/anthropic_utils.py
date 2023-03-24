"""Utils for using Anthropic API"""
import os
import logging
import json
from typing import Dict, Union, Tuple, Optional, List
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
)
from transformers import GPT2TokenizerFast

from haystack.errors import (
    AnthropicError,
    AnthropicRateLimitError,
    AnthropicUnauthorizedError,
)
from haystack.environment import (
    HAYSTACK_REMOTE_API_BACKOFF_SEC,
    HAYSTACK_REMOTE_API_MAX_RETRIES,
    HAYSTACK_REMOTE_API_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)

ANTHROPIC_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
ANTHROPIC_BACKOFF = int(os.environ.get(HAYSTACK_REMOTE_API_BACKOFF_SEC, 10))
ANTHROPIC_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))


def load_anthropic_tokenizer(tokenizer_name: str):
    """Load the GPT2TokenizerFast from the transformers library.

    :param tokenizer_name: The name of the tokenizer to load.
    """
    logger.debug("Using GPT2TokenizerFast tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer


def count_anthropic_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in `text` based on the `tokenizer`.

    :param text: A string to be tokenized.
    :param tokenizer: An OpenAI tokenizer.
    """
    return len(tokenizer.tokenize(text))


def count_anthropic_tokens_messages(messages: List[Dict[str, str]], tokenizer) -> int:
    """Count the number of tokens in `messages` based on the `tokenizer` provided.

    :param messages: The messages to be tokenized.
    :param tokenizer: An tokenizer.
    """
    num_tokens = 0
    for message in messages:
        num_tokens += (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        )
        for key, value in message.items():
            num_tokens += len(tokenizer.tokenize(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def _anthropic_text_completion_tokenization_details(model_name: str):
    """Return the tokenizer name and max tokens limit for a given Anthropic `model_name`.

    :param model_name: Name of the Anthropic model.
    """
    tokenizer_name = "gpt2"
    max_tokens_limit = 9000  # Based on this ref: https://cdn2.assets-servd.host/anthropic-website/production/images/FINAL-PRICING.pdf

    return tokenizer_name, max_tokens_limit


@retry(
    retry=retry_if_exception_type(AnthropicRateLimitError),
    wait=wait_exponential(multiplier=ANTHROPIC_BACKOFF),
    stop=stop_after_attempt(ANTHROPIC_MAX_RETRIES),
)
def anthropic_request(
    url: str,
    headers: Dict,
    payload: Dict,
    timeout: Union[float, Tuple[float, float]] = ANTHROPIC_TIMEOUT,
    read_response: Optional[bool] = True,
    **kwargs,
):
    """Make a request to the Anthropic API given a `url`, `headers`, `payload`, and `timeout`.

    :param url: The URL of the OpenAI API.
    :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
    :param payload: The payload to send with the request.
    :param timeout: The timeout length of the request. The default is 30s.
    :param read_response: Whether to read the response as JSON. The default is True.
    """
    response = requests.request(
        "POST",
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
        **kwargs,
    )
    if read_response:
        json_response = json.loads(response.text)

    if response.status_code != 200:
        openai_error: AnthropicError
        if response.status_code == 429:
            openai_error = AnthropicRateLimitError(
                f"API rate limit exceeded: {response.text}"
            )
        elif response.status_code == 401:
            openai_error = AnthropicUnauthorizedError(
                f"API key is invalid: {response.text}"
            )
        else:
            openai_error = AnthropicError(
                f"Anthropic returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )
        raise openai_error
    if read_response:
        return json_response
    else:
        return response
