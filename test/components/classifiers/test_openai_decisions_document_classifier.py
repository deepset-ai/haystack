# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError
from openai.types import Decision
from openai.types.decision_create_params import Question

from haystack import Document
from haystack.components.classifiers import OpenAIDecisionsDocumentClassifier
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret


def _decision(*answers):
    return Decision.model_validate(
        {
            "model": "gpt-6-luna",
            "answers": list(answers),
            "usage": {
                "input_tokens": 10,
                "output_tokens": 0,
                "total_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


@pytest.fixture
def client():
    with patch("haystack.components.classifiers.openai_decisions_document_classifier.OpenAI") as constructor:
        constructor.return_value.decisions.create.return_value = _decision()
        yield constructor.return_value


@pytest.fixture
def async_client():
    with patch("haystack.components.classifiers.openai_decisions_document_classifier.AsyncOpenAI") as constructor:
        constructor.return_value.decisions.create = AsyncMock(return_value=_decision())
        yield constructor.return_value


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestInit:
    def test_init_default(self):
        component = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}]
        )
        assert component.questions == [
            {"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}
        ]
        assert component.model == "gpt-6-luna"
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.api_base_url is None
        assert component.metadata_field == "openai_decisions"
        assert component._client is None
        assert component._async_client is None

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"questions": []}, "at least one question"),
            ({"questions": [{"type": "predicate", "instructions": "?"}]}, "non-empty string `name`"),
            ({"questions": [{"name": "", "type": "predicate", "instructions": "?"}]}, "non-empty string `name`"),
            ({"questions": [{"name": 1, "type": "predicate", "instructions": "?"}]}, "non-empty string `name`"),
            ({"questions": [{"name": "q", "type": "predicate", "instructions": "?"}] * 2}, "names must be unique"),
            ({"questions": [{"name": "q", "type": "yesno", "instructions": "?"}]}, "Question 'q' has type 'yesno'"),
            (
                {"questions": [{"name": "q", "type": "choice", "instructions": "?"}]},
                "Question 'q' of type 'choice' needs `choices`",
            ),
            (
                {"questions": [{"name": "q", "type": "score", "instructions": "?", "levels": []}]},
                "Question 'q' of type 'score' needs `levels`",
            ),
            (
                {
                    "questions": [{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
                    "max_workers": 0,
                },
                "at least 1",
            ),
        ],
    )
    def test_init_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            OpenAIDecisionsDocumentClassifier(**kwargs)


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.delenv("CUSTOM_OPENAI_KEY", raising=False)
        component = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
            model="custom-model",
            api_key=Secret.from_env_var("CUSTOM_OPENAI_KEY"),
            api_base_url="https://example.com/v1",
            classification_field="summary",
            metadata_field="decisions",
            timeout=30.0,
            max_retries=5,
            max_workers=8,
            raise_on_failure=True,
        )
        data = component_to_dict(component, name="classifier")
        restored = component_from_dict(OpenAIDecisionsDocumentClassifier, data, name="classifier")
        assert restored.questions == [
            {"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}
        ]
        assert restored.model == "custom-model"
        assert restored.api_key == Secret.from_env_var("CUSTOM_OPENAI_KEY")
        assert restored.api_base_url == "https://example.com/v1"
        assert restored.classification_field == "summary"
        assert restored.metadata_field == "decisions"
        assert restored.timeout == 30.0
        assert restored.max_retries == 5
        assert restored.max_workers == 8
        assert restored.raise_on_failure is True


@pytest.mark.usefixtures("api_key")
class TestLifecycle:
    @patch("haystack.components.classifiers.openai_decisions_document_classifier.OpenAI")
    def test_warm_up(self, mock_client, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "45.5")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
        component = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
            api_base_url="https://example.com/v1",
            timeout=30.0,
            max_retries=0,
        )
        component.warm_up()
        component.warm_up()
        mock_client.assert_called_once_with(
            api_key="test-key", base_url="https://example.com/v1", timeout=30.0, max_retries=0
        )
        assert component._client is mock_client.return_value

    @pytest.mark.asyncio
    @patch("haystack.components.classifiers.openai_decisions_document_classifier.AsyncOpenAI")
    async def test_warm_up_async(self, mock_client, monkeypatch):
        monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)
        monkeypatch.delenv("OPENAI_MAX_RETRIES", raising=False)
        component = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}]
        )
        await component.warm_up_async()
        await component.warm_up_async()
        mock_client.assert_called_once_with(api_key="test-key", base_url=None, timeout=30.0, max_retries=5)
        assert component._async_client is mock_client.return_value

    def test_close(self):
        component = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}]
        )
        client = MagicMock()
        component._client = client
        component.close()
        component.close()
        client.close.assert_called_once()
        assert component._client is None

    @pytest.mark.asyncio
    async def test_close_async(self):
        component = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}]
        )
        client = MagicMock(close=AsyncMock())
        component._async_client = client
        await component.close_async()
        await component.close_async()
        client.close.assert_awaited_once()
        assert component._async_client is None


@pytest.mark.usefixtures("api_key")
class TestRun:
    def test_run(self, client):
        questions: list[Question] = [
            {"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}
        ]
        classifier = OpenAIDecisionsDocumentClassifier(questions=questions)
        answer = {"type": "predicate", "probability": 0.8}
        client.decisions.create.return_value = _decision({"name": "refund", **answer})
        documents = [Document(content="Refund me.", meta={"source": "email"}), Document(content="The app crashes.")]

        result = classifier.run(documents=documents)

        assert result == {
            "documents": [
                Document(
                    id=documents[0].id,
                    content="Refund me.",
                    meta={"source": "email", "openai_decisions": {"refund": answer}},
                ),
                Document(id=documents[1].id, content="The app crashes.", meta={"openai_decisions": {"refund": answer}}),
            ],
            "failed_documents": [],
        }
        assert [document.meta for document in documents] == [{"source": "email"}, {}]
        assert client.decisions.create.call_count == 2
        client.decisions.create.assert_any_call(model="gpt-6-luna", input="Refund me.", questions=questions)
        client.decisions.create.assert_any_call(model="gpt-6-luna", input="The app crashes.", questions=questions)

    def test_run_classification_field(self, client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
            classification_field="summary",
        )
        result = classifier.run(documents=[Document(content="ignored", meta={"summary": "Refund me."})])
        client.decisions.create.assert_called_once_with(
            model="gpt-6-luna",
            input="Refund me.",
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
        )
        assert len(result["documents"]) == 1

    @pytest.mark.parametrize("summary", [None, "", ["not", "text"]], ids=["null", "empty", "non-text"])
    def test_run_without_text(self, client, summary):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
            classification_field="summary",
        )
        result = classifier.run(documents=[Document(meta={"summary": summary})])
        assert result["documents"] == []
        assert result["failed_documents"][0].meta["classification_error"] == "Document has no text to classify."
        client.decisions.create.assert_not_called()

    def test_run_preserves_refusals(self, client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[
                {"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"},
                {"name": "urgent", "type": "predicate", "instructions": "Is this urgent?"},
            ],
            metadata_field="decisions",
        )
        answers = [{"name": "refund", "type": "predicate", "probability": 0.8}, {"name": "urgent", "type": "refusal"}]
        client.decisions.create.return_value = _decision(*answers)
        result = classifier.run(documents=[Document(content="Refund me.")])
        assert result["documents"][0].meta["decisions"] == {
            "refund": {"type": "predicate", "probability": 0.8},
            "urgent": {"type": "refusal"},
        }
        assert result["failed_documents"] == []

    def test_run_empty_documents(self, client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}]
        )
        assert classifier.run(documents=[]) == {"documents": [], "failed_documents": []}
        client.decisions.create.assert_not_called()

    def test_run_records_failed_requests(self, client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}]
        )
        client.decisions.create.side_effect = APIConnectionError(message="connection refused", request=MagicMock())
        result = classifier.run(documents=[Document(content="Refund me.")])
        assert result["documents"] == []
        assert result["failed_documents"][0].meta["classification_error"] == "connection refused"

    def test_run_clears_previous_error(self, client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}]
        )
        document = Document(content="Refund me.", meta={"classification_error": "connection refused"})
        result = classifier.run(documents=[document])
        assert "classification_error" not in result["documents"][0].meta

    def test_run_raise_on_failure(self, client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
            raise_on_failure=True,
        )
        client.decisions.create.side_effect = APIConnectionError(message="connection refused", request=MagicMock())
        with pytest.raises(APIConnectionError, match="connection refused"):
            classifier.run(documents=[Document(content="Refund me.")])

    @pytest.mark.asyncio
    async def test_run_async(self, async_client):
        questions: list[Question] = [
            {"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}
        ]
        classifier = OpenAIDecisionsDocumentClassifier(questions=questions)
        answer = {"type": "predicate", "probability": 0.8}
        async_client.decisions.create.return_value = _decision({"name": "refund", **answer})
        result = await classifier.run_async(documents=[Document(content="Refund me."), Document()])
        async_client.decisions.create.assert_awaited_once_with(
            model="gpt-6-luna", input="Refund me.", questions=questions
        )
        assert [document.content for document in result["documents"]] == ["Refund me."]
        assert result["documents"][0].meta == {"openai_decisions": {"refund": answer}}
        assert [document.meta for document in result["failed_documents"]] == [
            {"classification_error": "Document has no text to classify."}
        ]

    @pytest.mark.asyncio
    async def test_run_async_records_failed_requests(self, async_client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}]
        )
        async_client.decisions.create.side_effect = [
            _decision(),
            APIConnectionError(message="connection refused", request=MagicMock()),
        ]
        result = await classifier.run_async(documents=[Document(content="Success"), Document(content="Failure")])
        assert [document.content for document in result["documents"]] == ["Success"]
        assert result["failed_documents"][0].content == "Failure"
        assert result["failed_documents"][0].meta["classification_error"] == "connection refused"

    @pytest.mark.asyncio
    async def test_run_async_raise_on_failure_cancels_remaining_requests(self, async_client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
            raise_on_failure=True,
        )
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def create(**kwargs):
            if kwargs["input"] == "fails":
                await started.wait()
                raise APIConnectionError(message="connection refused", request=MagicMock())
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async_client.decisions.create.side_effect = create
        with pytest.raises(APIConnectionError, match="connection refused"):
            await asyncio.wait_for(
                classifier.run_async(documents=[Document(content="slow"), Document(content="fails")]), timeout=1
            )
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_run_async_limits_concurrent_requests(self, async_client):
        classifier = OpenAIDecisionsDocumentClassifier(
            questions=[{"name": "refund", "type": "predicate", "instructions": "Is this a refund request?"}],
            max_workers=2,
        )
        in_flight = 0
        max_in_flight = 0

        async def create(**kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return _decision()

        async_client.decisions.create.side_effect = create
        result = await classifier.run_async(documents=[Document(content=f"Ticket {i}") for i in range(6)])
        assert max_in_flight == 2
        assert len(result["documents"]) == 6


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_run():
    questions: list[Question] = [
        {
            "name": "department",
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "choices": [{"value": "billing"}, {"value": "technical"}],
        },
        {
            "name": "urgency",
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "levels": [{"label": "can wait"}, {"label": "today"}],
        },
        {"name": "refund", "type": "predicate", "instructions": "Is the customer asking for a refund?"},
    ]
    component = OpenAIDecisionsDocumentClassifier(questions=questions)
    documents = [
        Document(content="I was charged twice for my subscription, please refund the second payment."),
        Document(content="The app crashes every time I open the settings page."),
    ]
    result = component.run(documents=documents)
    billing, technical = (document.meta["openai_decisions"] for document in result["documents"])
    assert result["failed_documents"] == []
    assert billing["department"]["choice"] == "billing"
    assert technical["department"]["choice"] == "technical"
    assert {item["value"] for item in billing["department"]["probabilities"]} == {"billing", "technical"}
    assert 0 <= billing["urgency"]["score"] <= 1
    assert billing["refund"]["probability"] > technical["refund"]["probability"]
