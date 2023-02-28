import os
import sys
from typing import List

import pytest

from haystack.schema import Document
from haystack.nodes.answer_generator import Seq2SeqGenerator, OpenAIAnswerGenerator
from haystack.pipelines import TranslationWrapperPipeline, GenerativeQAPipeline
from haystack.nodes import PromptTemplate

import logging


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_openai_answer_generator(openai_generator, docs):
    prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1
    assert "Carla" in prediction["answers"][0].answer


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_openai_answer_generator_custom_template(docs):
    lfqa_prompt = PromptTemplate(
        name="lfqa",
        prompt_text="""
        Synthesize a comprehensive answer from your knowledge and the following topk most relevant paragraphs and the given question.
        \n===\Paragraphs: $context\n===\n$query""",
        prompt_params=["context", "query"],
    )
    node = OpenAIAnswerGenerator(
        api_key=os.environ.get("OPENAI_API_KEY", ""), model="text-babbage-001", top_k=1, prompt_template=lfqa_prompt
    )
    prediction = node.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_openai_answer_generator_max_token(docs, caplog):
    openai_generator = OpenAIAnswerGenerator(
        api_key=os.environ.get("OPENAI_API_KEY", ""), model="text-babbage-001", top_k=1
    )
    openai_generator.MAX_TOKENS_LIMIT = 116
    with caplog.at_level(logging.INFO):
        prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
        assert "Skipping all of the provided Documents" in caplog.text
        assert len(prediction["answers"]) == 1
        # Can't easily check content of answer since it is generative and can change between runs
