from unittest.mock import patch, create_autospec

import pytest
from haystack import Pipeline
from haystack.schema import Document, Answer
from haystack.nodes.answer_generator import OpenAIAnswerGenerator
from haystack.nodes import PromptTemplate

import logging


@pytest.mark.unit
@patch("haystack.nodes.answer_generator.openai.openai_request")
def test_no_openai_organization(mock_request):
    with patch("haystack.nodes.answer_generator.openai.load_openai_tokenizer"):
        generator = OpenAIAnswerGenerator(api_key="fake_api_key")
    assert generator.openai_organization is None

    generator.predict(query="test query", documents=[Document(content="test document")])
    assert "OpenAI-Organization" not in mock_request.call_args.kwargs["headers"]


@pytest.mark.unit
@patch("haystack.nodes.answer_generator.openai.openai_request")
def test_openai_organization(mock_request):
    with patch("haystack.nodes.answer_generator.openai.load_openai_tokenizer"):
        generator = OpenAIAnswerGenerator(api_key="fake_api_key", openai_organization="fake_organization")
    assert generator.openai_organization == "fake_organization"

    generator.predict(query="test query", documents=[Document(content="test document")])
    assert mock_request.call_args.kwargs["headers"]["OpenAI-Organization"] == "fake_organization"


@pytest.mark.unit
@patch("haystack.nodes.answer_generator.openai.openai_request")
def test_openai_answer_generator_default_api_base(mock_request):
    with patch("haystack.nodes.answer_generator.openai.load_openai_tokenizer"):
        generator = OpenAIAnswerGenerator(api_key="fake_api_key")
    assert generator.api_base == "https://api.openai.com/v1"
    generator.predict(query="test query", documents=[Document(content="test document")])
    assert mock_request.call_args.kwargs["url"] == "https://api.openai.com/v1/completions"


@pytest.mark.unit
@patch("haystack.nodes.answer_generator.openai.openai_request")
def test_openai_answer_generator_custom_api_base(mock_request):
    with patch("haystack.nodes.answer_generator.openai.load_openai_tokenizer"):
        generator = OpenAIAnswerGenerator(api_key="fake_api_key", api_base="https://fake_api_base.com")
    assert generator.api_base == "https://fake_api_base.com"
    generator.predict(query="test query", documents=[Document(content="test document")])
    assert mock_request.call_args.kwargs["url"] == "https://fake_api_base.com/completions"


@pytest.mark.integration
@pytest.mark.parametrize("haystack_openai_config", ["openai", "azure"], indirect=True)
def test_openai_answer_generator(haystack_openai_config, docs):
    if not haystack_openai_config:
        pytest.skip("No API key found, skipping test")

    openai_generator = OpenAIAnswerGenerator(
        api_key=haystack_openai_config["api_key"],
        azure_base_url=haystack_openai_config.get("azure_base_url", None),
        azure_deployment_name=haystack_openai_config.get("azure_deployment_name", None),
        model="text-babbage-001",
        top_k=1,
    )
    prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1
    assert "Carla" in prediction["answers"][0].answer


@pytest.mark.integration
@pytest.mark.parametrize("haystack_openai_config", ["openai", "azure"], indirect=True)
def test_openai_answer_generator_custom_template(haystack_openai_config, docs):
    if not haystack_openai_config:
        pytest.skip("No API key found, skipping test")

    lfqa_prompt = PromptTemplate(
        """Synthesize a comprehensive answer from your knowledge and the following topk most relevant paragraphs and
        the given question.\n===\Paragraphs: {context}\n===\n{query}"""
    )
    node = OpenAIAnswerGenerator(
        api_key=haystack_openai_config["api_key"],
        azure_base_url=haystack_openai_config.get("azure_base_url", None),
        azure_deployment_name=haystack_openai_config.get("azure_deployment_name", None),
        model="text-babbage-001",
        top_k=1,
        prompt_template=lfqa_prompt,
    )
    prediction = node.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
    assert len(prediction["answers"]) == 1


@pytest.mark.integration
@pytest.mark.parametrize("haystack_openai_config", ["openai", "azure"], indirect=True)
def test_openai_answer_generator_max_token(haystack_openai_config, docs, caplog):
    if not haystack_openai_config:
        pytest.skip("No API key found, skipping test")

    openai_generator = OpenAIAnswerGenerator(
        api_key=haystack_openai_config["api_key"],
        azure_base_url=haystack_openai_config.get("azure_base_url", None),
        azure_deployment_name=haystack_openai_config.get("azure_deployment_name", None),
        model="text-babbage-001",
        top_k=1,
    )
    openai_generator.MAX_TOKENS_LIMIT = 116
    with caplog.at_level(logging.INFO):
        prediction = openai_generator.predict(query="Who lives in Berlin?", documents=docs, top_k=1)
        assert "Skipping all of the provided Documents" in caplog.text
        assert len(prediction["answers"]) == 1
        # Can't easily check content of answer since it is generative and can change between runs


# mock tokenizer that splits the string
class MockTokenizer:
    def encode(self, *args, **kwargs):
        return str.split(*args, **kwargs)

    def tokenize(self, *args, **kwargs):
        return str.split(*args, **kwargs)


@pytest.mark.unit
def test_build_prompt_within_max_length():
    with patch("haystack.nodes.answer_generator.openai.load_openai_tokenizer") as mock_load_tokenizer:
        mock_load_tokenizer.return_value = MockTokenizer()

        generator = OpenAIAnswerGenerator(api_key="fake_key", max_tokens=50)
        generator.MAX_TOKENS_LIMIT = 92
        query = "query"
        documents = [Document("most relevant document"), Document("less relevant document")]
        prompt_str, prompt_docs = generator._build_prompt_within_max_length(query=query, documents=documents)

        assert len(prompt_docs) == 1
        assert prompt_docs[0] == documents[0]


@pytest.mark.unit
def test_openai_answer_generator_pipeline_max_tokens():
    """
    tests that the max_tokens parameter is passed to the generator component in the pipeline
    """
    question = "What is New York City like?"
    mocked_response = "Forget NYC, I was generated by the mock method."
    nyc_docs = [Document(content="New York is a cool and amazing city to live in the United States of America.")]
    pipeline = Pipeline()

    # mock load_openai_tokenizer to avoid accessing the internet to init tiktoken
    with patch("haystack.nodes.answer_generator.openai.load_openai_tokenizer"):
        openai_generator = OpenAIAnswerGenerator(api_key="fake_api_key", model="text-babbage-001", top_k=1)

        pipeline.add_node(component=openai_generator, name="generator", inputs=["Query"])
        openai_generator.run = create_autospec(openai_generator.run)
        openai_generator.run.return_value = ({"answers": mocked_response}, "output_1")

        result = pipeline.run(query=question, documents=nyc_docs, params={"generator": {"max_tokens": 3}})
        assert result["answers"] == mocked_response
        openai_generator.run.assert_called_with(query=question, documents=nyc_docs, max_tokens=3)


@pytest.mark.unit
@patch("haystack.nodes.answer_generator.openai.OpenAIAnswerGenerator.predict")
def test_openai_answer_generator_run_with_labels_and_isolated_node_eval(patched_predict, eval_labels):
    label = eval_labels[0]
    query = label.query
    document = label.labels[0].document

    patched_predict.return_value = {
        "answers": [Answer(answer=label.labels[0].answer.answer, document_ids=[document.id])]
    }
    with patch("haystack.nodes.answer_generator.openai.load_openai_tokenizer"):
        openai_generator = OpenAIAnswerGenerator(api_key="fake_api_key", model="text-babbage-001", top_k=1)
        result, _ = openai_generator.run(query=query, documents=[document], labels=label, add_isolated_node_eval=True)

    assert "answers_isolated" in result


@pytest.mark.unit
@patch("haystack.nodes.answer_generator.base.BaseGenerator.predict_batch")
def test_openai_answer_generator_run_batch_with_labels_and_isolated_node_eval(patched_predict_batch, eval_labels):
    queries = [label.query for label in eval_labels]
    documents = [[label.labels[0].document] for label in eval_labels]

    patched_predict_batch.return_value = {
        "queries": queries,
        "answers": [
            [Answer(answer=label.labels[0].answer.answer, document_ids=[label.labels[0].document.id])]
            for label in eval_labels
        ],
    }
    with patch("haystack.nodes.answer_generator.openai.load_openai_tokenizer"):
        openai_generator = OpenAIAnswerGenerator(api_key="fake_api_key", model="text-babbage-001", top_k=1)
        result, _ = openai_generator.run_batch(
            queries=queries, documents=documents, labels=eval_labels, add_isolated_node_eval=True
        )

    assert "answers_isolated" in result
