"""Utils for using OpenAI API"""
import os
import logging
import platform
import sys
import json
from typing import Dict, Union, Tuple, List
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


USE_TIKTOKEN = False
if sys.version_info >= (3, 8) and (machine in ["amd64", "x86_64"] or (machine == "arm64" and system == "Darwin")):
    USE_TIKTOKEN = True

if USE_TIKTOKEN:
    import tiktoken  # pylint: disable=import-error
    from tiktoken.model import MODEL_TO_ENCODING
else:
    logger.warning(
        "OpenAI tiktoken module is not available for Python < 3.8,Linux ARM64 and AARCH64. Falling back to GPT2TokenizerFast."
    )


def load_openai_tokenizer(tokenizer_name: str):
    """Load either the tokenizer from tiktoken (if the library is available) or fallback to the GPT2TokenizerFast
    from the transformers library.

    :param tokenizer_name: The name of the tokenizer to load.
    """
    if USE_TIKTOKEN:
        logger.debug("Using tiktoken %s tokenizer", tokenizer_name)
        tokenizer = tiktoken.get_encoding(tokenizer_name)
    else:
        logger.debug("Using GPT2TokenizerFast tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer


def count_openai_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in `text` based on the provided OpenAI `tokenizer`.

    :param text: A string to be tokenized.
    :param tokenizer: An OpenAI tokenizer.
    """
    if USE_TIKTOKEN:
        return len(tokenizer.encode(text))
    else:
        return len(tokenizer.tokenize(text))


def count_openai_tokens_messages(messages: List[Dict[str, str]], tokenizer) -> int:
    """Count the number of tokens in `messages` based on the OpenAI `tokenizer` provided.

    :param messages: The messages to be tokenized.
    :param tokenizer: An OpenAI tokenizer.
    """
    # adapted from https://platform.openai.com/docs/guides/chat/introduction
    # should be kept up to date
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            if USE_TIKTOKEN:
                num_tokens += len(tokenizer.encode(value))
            else:
                num_tokens += len(tokenizer.tokenize(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def _openai_text_completion_tokenization_details(model_name: str):
    """Return the tokenizer name and max tokens limit for a given OpenAI `model_name`.

    :param model_name: Name of the OpenAI model.
    """
    tokenizer_name = "gpt2"
    if "davinci" in model_name:
        max_tokens_limit = 4000
        if USE_TIKTOKEN:
            tokenizer_name = MODEL_TO_ENCODING.get(model_name, "p50k_base")
    elif "gpt-3.5-turbo" in model_name:
        max_tokens_limit = 4096
        if USE_TIKTOKEN:
            tokenizer_name = MODEL_TO_ENCODING.get(model_name, "cl100k_base")
    else:
        max_tokens_limit = 2048
    return tokenizer_name, max_tokens_limit


@retry_with_exponential_backoff(
    backoff_in_seconds=OPENAI_BACKOFF, max_retries=OPENAI_MAX_RETRIES, errors=(OpenAIRateLimitError, OpenAIError)
)
def openai_request(
    url: str, headers: Dict, payload: Dict, timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT
) -> Dict:
    """Make a request to the OpenAI API given a `url`, `headers`, `payload`, and `timeout`.

    :param url: The URL of the OpenAI API.
    :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
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


def _check_openai_finish_reason(result: Dict, payload: Dict) -> None:
    """Check the `finish_reason` the answers returned by OpenAI completions endpoint.
    If the `finish_reason` is `length` or `content_filter`, log a warning to the user.

    :param result: The result returned from the OpenAI API.
    :param payload: The payload sent to the OpenAI API.
    """
    number_of_truncated_completions = sum(1 for ans in result["choices"] if ans["finish_reason"] == "length")
    if number_of_truncated_completions > 0:
        logger.warning(
            "%s out of the %s completions have been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions.",
            number_of_truncated_completions,
            payload["n"],
        )

    number_of_content_filtered_completions = sum(
        1 for ans in result["choices"] if ans["finish_reason"] == "content_filter"
    )
    if number_of_content_filtered_completions > 0:
        logger.warning(
            "%s out of the %s completions have omitted content due to a flag from OpenAI content filters.",
            number_of_truncated_completions,
            payload["n"],
        )
