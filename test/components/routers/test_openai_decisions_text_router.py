# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types import Decision

from haystack import Pipeline
from haystack.components.routers import OpenAIDecisionsTextRouter
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

_LABELS: dict[str, str | None] = {
    "billing": "Payments, invoices and refunds",
    "technical": "Bugs and errors in the product",
}


def _response(choice: str | None, confidence: float = 0.9) -> Decision:
    return Decision.model_validate(
        {
            "model": "gpt-6-luna",
            "answers": [
                {"type": "refusal", "name": "route"}
                if choice is None
                else {
                    "type": "choice",
                    "name": "route",
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": [
                        {"value": "billing", "probability": 0.5},
                        {"value": "technical", "probability": 0.5},
                    ],
                }
            ],
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
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.mark.usefixtures("api_key")
class TestInit:
    def test_init_default(self):
        router = OpenAIDecisionsTextRouter(labels=["billing", "technical"])
        assert router.labels == ["billing", "technical"]
        assert router.instructions == "Which category does this text belong to?"
        assert router.model == "gpt-6-luna"
        assert router.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert router.min_confidence is None
        assert router._question == {
            "type": "choice",
            "name": "route",
            "instructions": "Which category does this text belong to?",
            "choices": [{"value": "billing"}, {"value": "technical"}],
        }
        assert hasattr(router, "__haystack_output__")
        assert set(router.__haystack_output__._sockets_dict) == {"billing", "technical", "refused"}

    def test_init_min_confidence_adds_low_confidence_output(self):
        router = OpenAIDecisionsTextRouter(labels=_LABELS, min_confidence=0.6)
        assert router._question["choices"] == [
            {"value": "billing", "description": "Payments, invoices and refunds"},
            {"value": "technical", "description": "Bugs and errors in the product"},
        ]
        assert hasattr(router, "__haystack_output__")
        assert set(router.__haystack_output__._sockets_dict) == {"billing", "technical", "refused", "low_confidence"}

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"labels": []}, "between 2 and 255"),
            ({"labels": ["one"]}, "between 2 and 255"),
            ({"labels": [str(i) for i in range(256)]}, "between 2 and 255"),
            ({"labels": ["", "billing"]}, "non-empty string"),
            ({"labels": [True, "billing"]}, "non-empty string"),
            ({"labels": ["billing", "billing"]}, "unique"),
            ({"labels": ["billing", "refused"]}, "reserved"),
            ({"labels": ["billing", "low_confidence"]}, "reserved"),
            ({"labels": ["a", "b"], "min_confidence": -0.1}, "between 0 and 1"),
            ({"labels": ["a", "b"], "min_confidence": 1.1}, "between 0 and 1"),
            ({"labels": ["a", "b"], "min_confidence": float("nan")}, "between 0 and 1"),
        ],
    )
    def test_init_invalid(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            OpenAIDecisionsTextRouter(**kwargs)


@pytest.mark.usefixtures("api_key")
class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_KEY", "test-key")
        router = OpenAIDecisionsTextRouter(
            labels=_LABELS,
            instructions="Which team should handle this?",
            model="gpt-6-luna",
            api_key=Secret.from_env_var("ROUTER_API_KEY"),
            api_base_url="https://example.com/v1",
            min_confidence=0.6,
            timeout=30.0,
            max_retries=5,
        )
        data = component_to_dict(obj=router, name="router")
        restored = component_from_dict(cls=OpenAIDecisionsTextRouter, data=data, name="router")
        assert restored.labels == _LABELS
        assert restored.instructions == "Which team should handle this?"
        assert restored.model == "gpt-6-luna"
        assert restored.api_key.resolve_value() == "test-key"
        assert restored.api_base_url == "https://example.com/v1"
        assert restored.min_confidence == 0.6
        assert restored.timeout == 30.0
        assert restored.max_retries == 5
        assert component_to_dict(obj=restored, name="router") == data

    def test_pipeline_round_trip(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        pipeline = Pipeline()
        pipeline.add_component("router", OpenAIDecisionsTextRouter(labels=_LABELS, min_confidence=0.6))
        restored = Pipeline.loads(pipeline.dumps())
        router = restored.get_component("router")
        assert isinstance(router, OpenAIDecisionsTextRouter)
        assert component_to_dict(obj=router, name="router") == component_to_dict(
            obj=pipeline.get_component("router"), name="router"
        )
        assert router._client is None
        assert router._async_client is None
        assert hasattr(router, "__haystack_output__")
        assert set(router.__haystack_output__._sockets_dict) == {"billing", "technical", "refused", "low_confidence"}


@pytest.mark.usefixtures("api_key")
class TestWarmUp:
    @patch("haystack.components.routers.openai_decisions_text_router.OpenAI")
    def test_warm_up(self, mock_client):
        router = OpenAIDecisionsTextRouter(
            labels=_LABELS, api_base_url="https://example.com/v1", timeout=30.0, max_retries=0
        )
        mock_client.assert_not_called()
        router.warm_up()
        router.warm_up()
        mock_client.assert_called_once_with(
            api_key="test-key", base_url="https://example.com/v1", timeout=30.0, max_retries=0
        )
        assert router._client is mock_client.return_value
        assert router._async_client is None

    @pytest.mark.asyncio
    @patch("haystack.components.routers.openai_decisions_text_router.AsyncOpenAI")
    async def test_warm_up_async(self, mock_client):
        router = OpenAIDecisionsTextRouter(labels=_LABELS)
        mock_client.assert_not_called()
        await router.warm_up_async()
        await router.warm_up_async()
        mock_client.assert_called_once_with(api_key="test-key", base_url=None)
        assert router._async_client is mock_client.return_value
        assert router._client is None

    def test_close(self):
        router = OpenAIDecisionsTextRouter(labels=_LABELS)
        client = MagicMock()
        router._client = client
        router.close()
        router.close()
        client.close.assert_called_once()
        assert router._client is None

    @pytest.mark.asyncio
    async def test_close_async(self):
        router = OpenAIDecisionsTextRouter(labels=_LABELS)
        client = MagicMock(close=AsyncMock())
        router._async_client = client
        await router.close_async()
        await router.close_async()
        client.close.assert_awaited_once()
        assert router._async_client is None


@pytest.mark.usefixtures("api_key")
class TestRun:
    @pytest.mark.parametrize(
        ("confidence", "expected_output"), [(0.9, "billing"), (0.3, "low_confidence"), (0.6, "billing")]
    )
    def test_run(self, confidence, expected_output):
        router = OpenAIDecisionsTextRouter(
            labels=_LABELS, instructions="Which team should handle this?", min_confidence=0.6
        )
        client = MagicMock()
        router._client = client
        client.decisions.create.return_value = _response("billing", confidence=confidence)
        result = router.run(text="I was charged twice.")
        client.decisions.create.assert_called_once_with(
            model="gpt-6-luna",
            input="I was charged twice.",
            questions=[
                {
                    "type": "choice",
                    "name": "route",
                    "instructions": "Which team should handle this?",
                    "choices": [
                        {"value": "billing", "description": "Payments, invoices and refunds"},
                        {"value": "technical", "description": "Bugs and errors in the product"},
                    ],
                }
            ],
        )
        assert result == {expected_output: "I was charged twice."}

    def test_run_without_min_confidence_ignores_confidence(self):
        router = OpenAIDecisionsTextRouter(labels=_LABELS)
        client = MagicMock()
        router._client = client
        client.decisions.create.return_value = _response("technical", confidence=0.01)
        assert router.run(text="Hello?") == {"technical": "Hello?"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("confidence", "expected_output"), [(0.9, "billing"), (0.3, "low_confidence")])
    async def test_run_async(self, confidence, expected_output):
        router = OpenAIDecisionsTextRouter(labels=_LABELS, min_confidence=0.6)
        client = MagicMock()
        router._async_client = client
        client.decisions.create = AsyncMock(return_value=_response("billing", confidence=confidence))
        assert await router.run_async(text="I was charged twice.") == {expected_output: "I was charged twice."}
        client.decisions.create.assert_awaited_once_with(
            model="gpt-6-luna", input="I was charged twice.", questions=[router._question]
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [["not", "a", "string"]])
    async def test_run_wrong_input_type(self, text):
        router = OpenAIDecisionsTextRouter(labels=_LABELS)
        with pytest.raises(TypeError, match="^OpenAIDecisionsTextRouter expects a str as input\\.$"):
            router.run(text=text)
        with pytest.raises(TypeError, match="expects a str"):
            await router.run_async(text=text)
        assert router._client is None
        assert router._async_client is None

    @pytest.mark.parametrize("min_confidence", [None, 0.6])
    def test_run_refusal(self, min_confidence):
        router = OpenAIDecisionsTextRouter(labels=_LABELS, min_confidence=min_confidence)
        client = MagicMock()
        router._client = client
        client.decisions.create.return_value = _response(None)
        assert router.run(text="Hello?") == {"refused": "Hello?"}

    @pytest.mark.asyncio
    async def test_run_async_refusal(self):
        router = OpenAIDecisionsTextRouter(labels=_LABELS, min_confidence=0.6)
        client = MagicMock()
        router._async_client = client
        client.decisions.create = AsyncMock(return_value=_response(None))
        assert await router.run_async(text="Hello?") == {"refused": "Hello?"}

    @pytest.mark.parametrize(
        "answer",
        [
            {"type": "predicate", "probability": 0.9},
            {"type": "choice", "choice": True, "confidence": 0.9, "probabilities": []},
            {"type": "choice", "choice": "unknown", "confidence": 0.9, "probabilities": []},
        ],
    )
    def test_run_unexpected_answer(self, answer):
        router = OpenAIDecisionsTextRouter(labels=_LABELS)
        client = MagicMock()
        router._client = client
        data = _response("billing").model_dump()
        data["answers"] = [answer]
        client.decisions.create.return_value = Decision.model_validate(data)
        with pytest.raises(ValueError, match="unexpected answer"):
            router.run(text="Hello?")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_live_run_in_pipeline(self):
    pipeline = Pipeline()
    router = OpenAIDecisionsTextRouter(labels=_LABELS, instructions="Which team should handle this support request?")
    pipeline.add_component("router", router)
    texts = {
        "billing": "I was charged twice, please refund the second payment.",
        "technical": "The app crashes every time I open the settings page.",
    }
    for label, text in texts.items():
        data = {"router": {"text": text}}
        assert await asyncio.to_thread(pipeline.run, data=data) == {"router": {label: text}}
        assert await pipeline.run_async(data=data) == {"router": {label: text}}
