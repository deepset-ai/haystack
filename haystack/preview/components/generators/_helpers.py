from typing import List

import logging


logger = logging.getLogger(__name__)


def enforce_token_limit(prompt: str, tokenizer, max_tokens_limit: int) -> str:
    """
    Ensure that the length of the prompt is within the max tokens limit of the model.
    If needed, truncate the prompt text so that it fits within the limit.

    :param prompt: Prompt text to be sent to the generative model.
    :param tokenizer: The tokenizer used to encode the prompt.
    :param max_tokens_limit: The max tokens limit of the model.
    :return: The prompt text that fits within the max tokens limit of the model.
    """
    tokens = tokenizer.encode(prompt)
    tokens_count = len(tokens)
    if tokens_count > max_tokens_limit:
        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens to fit within the max token limit. "
            "Reduce the length of the prompt to prevent it from being cut off.",
            tokens_count,
            max_tokens_limit,
        )
        tokenized_payload = tokenizer.encode(prompt)
        prompt = tokenizer.decode(tokenized_payload[:max_tokens_limit])
    return prompt


def enforce_token_limit_chat(
    prompts: List[str], tokenizer, max_tokens_limit: int, tokens_per_message_overhead: int
) -> List[str]:
    """
    Ensure that the length of the list of prompts is within the max tokens limit of the model.
    If needed, truncate the prompts text and list so that it fits within the limit.

    :param prompts: Prompts text to be sent to the generative model.
    :param tokenizer: The tokenizer used to encode the prompt.
    :param max_tokens_limit: The max tokens limit of the model.
    :param tokens_per_message_overhead: The number of tokens that are added to the prompt text for each message.
    :return: A list of prompts that fits within the max tokens limit of the model.
    """
    prompts_lens = [len(tokenizer.encode(prompt)) for prompt in prompts]
    if (total_prompt_length := sum(prompts_lens) + (tokens_per_message_overhead * len(prompts))) <= max_tokens_limit:
        return prompts

    logger.warning(
        "The prompts have been truncated from %s tokens to %s tokens to fit within the max token limit. "
        "Reduce the length of the prompt to prevent it from being cut off.",
        total_prompt_length,
        max_tokens_limit,
    )
    cut_prompts = []
    cut_prompts_lens = []
    for prompt, prompt_len in zip(prompts, prompts_lens):
        prompt_len = prompt_len + sum(cut_prompts_lens) + (tokens_per_message_overhead * (len(cut_prompts) + 1))
        if prompt_len <= max_tokens_limit:
            cut_prompts.append(prompt)
            cut_prompts_lens.append(prompt_len)
        else:
            remaining_tokens = (
                max_tokens_limit - sum(cut_prompts_lens) - (tokens_per_message_overhead * (len(cut_prompts) + 1))
            )
            cut_prompts.append(enforce_token_limit(prompt, tokenizer, remaining_tokens))
            return cut_prompts
