import os

import pytest
import numpy as np

from haystack.schema import Document
from haystack.nodes.answer_generator import OpenAIAnswerGenerator
from haystack.nodes import PromptTemplate, RAGenerator


from ..conftest import SAMPLES_PATH


NO_KEY = not bool(os.environ.get("OPENAI_API_KEY", False))
NO_KEY_MSG = "No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test."


@pytest.fixture
def docs_with_true_emb():
    return [
        Document(
            content="The capital of Germany is the city state of Berlin.",
            embedding=np.loadtxt(SAMPLES_PATH / "embeddings" / "embedding_1.txt"),
        ),
        Document(
            content="Berlin is the capital and largest city of Germany by both area and population.",
            embedding=np.loadtxt(SAMPLES_PATH / "embeddings" / "embedding_2.txt"),
        ),
    ]


def test_rag_token_generator(docs_with_true_emb):
    rag_generator = RAGenerator(model_name_or_path="facebook/rag-token-nq", generator_type="token", max_length=20)
    query = "What is capital of the Germany?"
    generated_docs = rag_generator.predict(query=query, documents=docs_with_true_emb, top_k=1)
    answers = generated_docs["answers"]
    assert len(answers) == 1
    assert "berlin" in answers[0].answer


@pytest.mark.skipif(NO_KEY, reason=NO_KEY_MSG)
def test_openai_answer_generator(openai_generator, docs):
    openai_generator = OpenAIAnswerGenerator(
        api_key=os.environ.get("OPENAI_API_KEY", ""), model="text-babbage-001", top_k=1
    )
    prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1
    assert "Carla" in prediction["answers"][0].answer


@pytest.mark.skipif(NO_KEY, reason=NO_KEY_MSG)
def test_openai_answer_generator_custom_template(docs):
    lfqa_prompt = PromptTemplate(
        name="lfqa",
        prompt_text="""
        Synthesize a comprehensive answer from your knowledge and the following topk most relevant paragraphs and the given question.
        \n===\Paragraphs: $context\n===\n$query""",
        prompt_params=["context", "query"],
    )
    openai_generator = OpenAIAnswerGenerator(
        api_key=os.environ.get("OPENAI_API_KEY", ""), model="text-babbage-001", top_k=1, prompt_template=lfqa_prompt
    )
    prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1
