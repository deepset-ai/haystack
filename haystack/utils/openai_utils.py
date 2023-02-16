"""Utils for using OpenAI API"""
import logging
import platform
import sys

from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


machine = platform.machine().lower()
system = platform.system()


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
        tokenizer: tiktoken.Encoding = tiktoken.get_encoding(tokenizer_name)
    else:
        logger.debug("Using GPT2TokenizerFast tokenizer")
        tokenizer: PreTrainedTokenizerFast = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer
