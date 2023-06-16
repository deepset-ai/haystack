from typing import List
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import logging

import tiktoken
from tiktoken.model import MODEL_TO_ENCODING
from banks import Prompt

from haystack.utils.openai_utils import openai_request
from haystack.preview import component, ComponentInput, ComponentOutput
from haystack.preview.components.prompt.base import PromptInputMixin

from haystack.preview.components.prompt.extensions.datetime_ext import TimeExtension


PAYLOAD_KEYS = [
    "suffix",
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "logprobs",
    "echo",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "best_of",
    "logit_bias",
    "stream",
]


@component
class PromptOpenAI(PromptInputMixin):
    @dataclass
    class Output(ComponentOutput):
        responses: List[str]

    def __init__(self, api_key: str, prompt: str, model: str = "text-davinci-003", **kwargs):
        # TODO figure a way to make all PromptComponents have this
        PromptOpenAI.env.add_extension(TimeExtension)

        if not api_key:
            raise ValueError("api_key is required.")
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self.api_base = kwargs.get("api_base", "https://api.openai.com/v1")
        self.payload = {"model": self.model}

        for key in PAYLOAD_KEYS:
            if key in kwargs:
                self.payload[key] = kwargs.pop(key)

        # either stream is True (will use default handler) or stream_handler is provided
        if "stream_handler" in kwargs:
            self.payload["stream"] = True

        # Tokenization params
        self._tokenizer_name = kwargs.get("tokenizer")
        self._max_tokens_limit = kwargs.get(
            "max_tokens_limit", 2049
        )  # Based on this ref: https://platform.openai.com/docs/models/gpt-3

    def warm_up(self):
        if not self._tokenizer_name:
            tokenizer_name = "gpt2"
            if self.model in MODEL_TO_ENCODING:
                tokenizer_name = MODEL_TO_ENCODING[self.model]
                # Based on OpenAI models page, 'davinci' considers have 2049 tokens
                # see https://platform.openai.com/docs/models/gpt-3
                # if "text-davinci" in self.model:
                #     max_tokens_limit = 4097

        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

    def run(self, data) -> Output:
        prompt = self._render_prompt(data)
        logging.warning(prompt)

        res = openai_request(
            url=f"{self.api_base}/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            payload={"prompt": prompt, **self.payload},
        )
        # TODO raise warning in case output is cut
        responses = [ans["text"].strip() for ans in res["choices"]]
        return PromptOpenAI.Output(responses)
