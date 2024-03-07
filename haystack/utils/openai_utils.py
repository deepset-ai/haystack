"""Utils for using OpenAI API"""
import os
import logging
import platform
import json
from typing import Dict, Union, Tuple, Optional, List, cast

import httpx
import requests
import tenacity
import tiktoken

from haystack.errors import OpenAIError, OpenAIRateLimitError, OpenAIUnauthorizedError
from haystack.environment import (
    HAYSTACK_REMOTE_API_BACKOFF_SEC,
    HAYSTACK_REMOTE_API_MAX_RETRIES,
    HAYSTACK_REMOTE_API_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)

machine = platform.machine().lower()
system = platform.system()

OPENAI_TIMEOUT = float(os.environ.get(HAYSTACK_REMOTE_API_TIMEOUT_SEC, 30))
OPENAI_BACKOFF = int(os.environ.get(HAYSTACK_REMOTE_API_BACKOFF_SEC, 10))
OPENAI_MAX_RETRIES = int(os.environ.get(HAYSTACK_REMOTE_API_MAX_RETRIES, 5))

OPENAI_MODERATION_URL = "https://api.openai.com/v1/moderations"


def load_openai_tokenizer(tokenizer_name: str):
    """Load either the tokenizer from tiktoken (if the library is available) or fallback to the GPT2TokenizerFast
    from the transformers library.

    :param tokenizer_name: The name of the tokenizer to load.
    """

    logger.debug("Using tiktoken %s tokenizer", tokenizer_name)
    return tiktoken.get_encoding(tokenizer_name)


def count_openai_tokens_messages(messages: List[Dict[str, str]], tokenizer) -> int:
    """Count the number of tokens in `messages` based on the OpenAI `tokenizer` provided.

    :param messages: The messages to be tokenized.
    :param tokenizer: An OpenAI tokenizer.
    """
    # adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    # should be kept up to date
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 3  # every reply is primed with <im_start>assistant<|message|>
    return num_tokens


def _openai_text_completion_tokenization_details(model_name: str):
    """Return the tokenizer name and max tokens limit for a given OpenAI `model_name`.

    :param model_name: Name of the OpenAI model.
    """
    tokenizer_name = "cl100k_base"
    # It is the minimum max_tokens_limit value based on this ref: https://platform.openai.com/docs/models/overview
    max_tokens_limit = 4096
    try:
        model_tokenizer = tiktoken.encoding_name_for_model(model_name)
    except KeyError:
        model_tokenizer = None

    if model_tokenizer:
        tokenizer_name = model_tokenizer
        if model_name == "davinci-002" or model_name == "babbage-002":
            max_tokens_limit = 16384

        # GPT-3.5 models that have a different token limit than 4096
        # Handles default case for GPT-3.5 models
        if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-35-turbo"):
            max_tokens_limit = 16385
        # Handles two edge-cases where the value is 4096
        if (
            model_name == "gpt-3.5-turbo-instruct"
            or model_name == "gpt-3.5-turbo-0613"
            or model_name == "gpt-35-turbo-instruct"
            or model_name == "gpt-35-turbo-0613"
        ):
            max_tokens_limit = 4096
        if model_name == "gpt-3.5-turbo-16k" or model_name == "gpt-35-turbo-16k":
            max_tokens_limit = 16384

        # GPT-4 models that have a different token limit than 4096
        # Ref: https://platform.openai.com/docs/models/gpt-4
        if model_name.startswith("gpt-4-"):
            max_tokens_limit = 128000
        if model_name == "gpt-4" or model_name == "gpt-4-0613":
            max_tokens_limit = 8192
        if model_name == "gpt-4-32k" or model_name == "gpt-4-32k-0613":
            max_tokens_limit = 32768

    return tokenizer_name, max_tokens_limit


@tenacity.retry(
    reraise=True,
    retry=tenacity.retry_if_exception_type(OpenAIError)
    and tenacity.retry_if_not_exception_type(OpenAIUnauthorizedError),
    wait=tenacity.wait_exponential(multiplier=OPENAI_BACKOFF),
    stop=tenacity.stop_after_attempt(OPENAI_MAX_RETRIES),
)
def openai_request(
    url: str,
    headers: Dict,
    payload: Dict,
    timeout: Optional[Union[float, Tuple[float, float]]] = None,
    read_response: Optional[bool] = True,
    **kwargs,
):
    """Make a request to the OpenAI API given a `url`, `headers`, `payload`, and `timeout`.

    :param url: The URL of the OpenAI API.
    :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
    :param payload: The payload to send with the request.
    :param timeout: The timeout length of the request. The default is 30s.
    :param read_response: Whether to read the response as JSON. The default is True.
    """
    if timeout is None:
        timeout = OPENAI_TIMEOUT
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload), timeout=timeout, **kwargs)
    if read_response:
        json_response = json.loads(response.text)

    if response.status_code != 200:
        openai_error: OpenAIError
        if response.status_code == 429:
            openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
        elif response.status_code == 401:
            openai_error = OpenAIUnauthorizedError(f"API key is invalid: {response.text}")
        else:
            openai_error = OpenAIError(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )
        raise openai_error
    if read_response:
        return json_response
    else:
        return response


@tenacity.retry(
    reraise=True,
    retry=tenacity.retry_if_exception_type(OpenAIError)
    and tenacity.retry_if_not_exception_type(OpenAIUnauthorizedError),
    wait=tenacity.wait_exponential(multiplier=OPENAI_BACKOFF),
    stop=tenacity.stop_after_attempt(OPENAI_MAX_RETRIES),
)
async def openai_async_request(
    url: str,
    headers: Dict,
    payload: Dict,
    timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT,
    read_response: bool = True,
    **kwargs,
):
    """Make a request to the OpenAI API given a `url`, `headers`, `payload`, and `timeout`.

    See `openai_request`.
    """
    async with httpx.AsyncClient() as client:
        response = await client.request(
            "POST", url, headers=headers, json=payload, timeout=cast(float, timeout), **kwargs
        )

    if read_response:
        json_response = json.loads(response.text)

    if response.status_code != 200:
        openai_error: OpenAIError
        if response.status_code == 429:
            openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
        elif response.status_code == 401:
            openai_error = OpenAIUnauthorizedError(f"API key is invalid: {response.text}")
        else:
            openai_error = OpenAIError(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )
        raise openai_error
    if read_response:
        return json_response
    else:
        return response


def check_openai_policy_violation(input: Union[List[str], str], headers: Dict) -> bool:
    """
    Calls the moderation endpoint to check if the text(s) violate the policy.
    See [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation) for more details.
    Returns true if any of the input is flagged as any of ['sexual', 'hate', 'violence', 'self-harm', 'sexual/minors', 'hate/threatening', 'violence/graphic'].
    """
    response = openai_request(url=OPENAI_MODERATION_URL, headers=headers, payload={"input": input})
    results = response["results"]
    flagged = any(res["flagged"] for res in results)
    if flagged:
        for result in results:
            if result["flagged"]:
                logger.debug(
                    "OpenAI Moderation API flagged the text '%s' as a potential policy violation of the following categories: %s",
                    input,
                    result["categories"],
                )
    return flagged


async def check_openai_async_policy_violation(input: Union[List[str], str], headers: Dict) -> bool:
    """
    Calls the moderation endpoint to check if the text(s) violate the policy.
    See [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation) for more details.
    Returns true if any of the input is flagged as any of ['sexual', 'hate', 'violence', 'self-harm', 'sexual/minors', 'hate/threatening', 'violence/graphic'].
    """
    response = await openai_async_request(url=OPENAI_MODERATION_URL, headers=headers, payload={"input": input})
    results = response["results"]
    flagged = any(res["flagged"] for res in results)
    if flagged:
        for result in results:
            if result["flagged"]:
                logger.debug(
                    "OpenAI Moderation API flagged the text '%s' as a potential policy violation of the following categories: %s",
                    input,
                    result["categories"],
                )
    return flagged


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
