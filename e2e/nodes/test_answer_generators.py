import os

import pytest
import numpy as np

from haystack.schema import Document
from haystack.nodes.answer_generator import OpenAIAnswerGenerator
from haystack.nodes import PromptTemplate, RAGenerator


from ..conftest import SAMPLES_PATH


@pytest.fixture
def openai_generator(request):
    if request.param == "azure":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        azure_base_url = os.environ.get("AZURE_OPENAI_BASE_URL", None)
        azure_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", None)
        if api_key and azure_base_url and azure_deployment_name:
            return OpenAIAnswerGenerator(
                api_key=api_key,
                azure_base_url=azure_base_url,
                azure_deployment_name=azure_deployment_name,
                model="text-babbage-001",
                top_k=1,
            )
        pytest.skip("No OpenAI API keys provided. Check 'e2e/nodes/test_answer_generators.py' to see what's required.")

    elif request.param == "openai":
        if bool(os.environ.get("OPENAI_API_KEY", False)):
            return OpenAIAnswerGenerator(
                api_key=os.environ.get("OPENAI_API_KEY", ""), model="text-babbage-001", top_k=1
            )
        pytest.skip("No Azure keys provided. Check 'e2e/nodes/test_answer_generators.py' to see what's required.")


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


@pytest.mark.parametrize("openai_generator", ["openai", "azure"], indirect=True)
def test_openai_answer_generator(openai_generator, docs):
    prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1
    assert "Carla" in prediction["answers"][0].answer


@pytest.mark.parametrize("openai_generator", ["openai", "azure"], indirect=True)
def test_openai_answer_generator_custom_template(openai_generator, docs):
    lfqa_prompt = PromptTemplate(
        name="lfqa",
        prompt_text="""
        Synthesize a comprehensive answer from your knowledge and the following topk most relevant paragraphs and the given question.
        ===
        Paragraphs: $context
        ===
        $query
        """,
        prompt_params=["context", "query"],
    )
    prediction = openai_generator.predict(query=lfqa_prompt, documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1
