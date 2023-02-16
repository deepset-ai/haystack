"""Utils for using OpenAI API"""
import logging
import platform
import sys

from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


machine = platform.machine().lower()
system = platform.system()


def get_use_tiktoken():
    use_tiktoken = False
    if sys.version_info >= (3, 8) and (machine in ["amd64", "x86_64"] or (machine == "arm64" and system == "Darwin")):
        use_tiktoken = True

    if not use_tiktoken:
        logger.warning(
            "OpenAI tiktoken module is not available for Python < 3.8,Linux ARM64 and AARCH64. Falling back to GPT2TokenizerFast."
        )
    return use_tiktoken


def get_openai_tokenizer(use_tiktoken: bool, tokenizer_name: str):
    if use_tiktoken:
        import tiktoken  # pylint: disable=import-error

        logger.debug("Using tiktoken %s tokenizer", tokenizer_name)
        tokenizer: tiktoken.Encoding = tiktoken.get_encoding(tokenizer_name)
    else:
        logger.debug("Using GPT2TokenizerFast tokenizer")
        tokenizer: PreTrainedTokenizerFast = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer
