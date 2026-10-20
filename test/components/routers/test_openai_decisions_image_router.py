# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types import Decision

from haystack.components.routers import OpenAIDecisionsImageRouter
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ImageContent
from haystack.utils import Secret

_LABELS: dict[str, str | None] = {"fruit": "Photographs of fruit", "logo": "Company logos"}


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
                    "probabilities": [{"value": "fruit", "probability": 0.5}, {"value": "logo", "probability": 0.5}],
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
def image(test_files_path):
    return ImageContent.from_file_path(
        file_path=test_files_path / "images" / "apple.jpg", detail="high", meta={"source": "apple.jpg"}
    )


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.mark.usefixtures("api_key")
class TestInitSerialization:
    def test_init_default(self):
        router = OpenAIDecisionsImageRouter(labels=["fruit", "logo"])
        assert router.labels == ["fruit", "logo"]
        assert router.instructions == "Which category does this image belong to?"
        assert router.model == "gpt-6-luna"
        assert router.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert router.min_confidence is None
        assert router._question == {
            "type": "choice",
            "name": "route",
            "instructions": "Which category does this image belong to?",
            "choices": [{"value": "fruit"}, {"value": "logo"}],
        }
        assert hasattr(router, "__haystack_output__")
        assert set(router.__haystack_output__._sockets_dict) == {"fruit", "logo", "refused"}
        assert all(socket.type is ImageContent for socket in router.__haystack_output__._sockets_dict.values())

    def test_init_min_confidence_adds_low_confidence_output(self):
        router = OpenAIDecisionsImageRouter(labels=_LABELS, min_confidence=0.6)
        assert router._question["choices"] == [
            {"value": "fruit", "description": "Photographs of fruit"},
            {"value": "logo", "description": "Company logos"},
        ]
        assert hasattr(router, "__haystack_output__")
        assert set(router.__haystack_output__._sockets_dict) == {"fruit", "logo", "refused", "low_confidence"}

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"labels": []}, "between 2 and 255"),
            ({"labels": ["one"]}, "between 2 and 255"),
            ({"labels": [str(i) for i in range(256)]}, "between 2 and 255"),
            ({"labels": ["", "fruit"]}, "non-empty string"),
            ({"labels": [True, "fruit"]}, "non-empty string"),
            ({"labels": ["fruit", "fruit"]}, "unique"),
            ({"labels": ["fruit", "refused"]}, "reserved"),
            ({"labels": ["fruit", "low_confidence"]}, "reserved"),
            ({"labels": ["a", "b"], "min_confidence": -0.1}, "between 0 and 1"),
            ({"labels": ["a", "b"], "min_confidence": 1.1}, "between 0 and 1"),
            ({"labels": ["a", "b"], "min_confidence": float("nan")}, "between 0 and 1"),
        ],
    )
    def test_init_invalid(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            OpenAIDecisionsImageRouter(**kwargs)

    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("ROUTER_API_KEY", "test-key")
        router = OpenAIDecisionsImageRouter(
            labels=_LABELS,
            instructions="What does this image show?",
            model="gpt-6-luna",
            api_key=Secret.from_env_var("ROUTER_API_KEY"),
            api_base_url="https://example.com/v1",
            min_confidence=0.6,
            timeout=30.0,
            max_retries=5,
        )
        data = component_to_dict(obj=router, name="router")
        restored = component_from_dict(cls=OpenAIDecisionsImageRouter, data=data, name="router")
        assert restored.labels == _LABELS
        assert restored.instructions == "What does this image show?"
        assert restored.model == "gpt-6-luna"
        assert restored.api_key.resolve_value() == "test-key"
        assert restored.api_base_url == "https://example.com/v1"
        assert restored.min_confidence == 0.6
        assert restored.timeout == 30.0
        assert restored.max_retries == 5
        assert component_to_dict(obj=restored, name="router") == data


@pytest.mark.usefixtures("api_key")
class TestLifecycle:
    @patch("haystack.components.routers.openai_decisions_image_router.OpenAI")
    def test_warm_up(self, mock_client):
        router = OpenAIDecisionsImageRouter(
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
    @patch("haystack.components.routers.openai_decisions_image_router.AsyncOpenAI")
    async def test_warm_up_async(self, mock_client, monkeypatch):
        monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)
        monkeypatch.delenv("OPENAI_MAX_RETRIES", raising=False)
        router = OpenAIDecisionsImageRouter(labels=_LABELS)
        mock_client.assert_not_called()
        await router.warm_up_async()
        await router.warm_up_async()
        mock_client.assert_called_once_with(api_key="test-key", base_url=None, timeout=30.0, max_retries=5)
        assert router._async_client is mock_client.return_value
        assert router._client is None

    def test_close(self):
        router = OpenAIDecisionsImageRouter(labels=_LABELS)
        client = MagicMock()
        router._client = client
        router.close()
        router.close()
        client.close.assert_called_once()
        assert router._client is None

    @pytest.mark.asyncio
    async def test_close_async(self):
        router = OpenAIDecisionsImageRouter(labels=_LABELS)
        client = MagicMock(close=AsyncMock())
        router._async_client = client
        await router.close_async()
        await router.close_async()
        client.close.assert_awaited_once()
        assert router._async_client is None


@pytest.mark.usefixtures("api_key")
class TestRun:
    @pytest.mark.parametrize(("confidence", "expected_output"), [(0.9, "fruit"), (0.3, "low_confidence")])
    def test_run(self, confidence, expected_output, image):
        router = OpenAIDecisionsImageRouter(
            labels=_LABELS, instructions="What does this image show?", min_confidence=0.6
        )
        client = MagicMock()
        router._client = client
        client.decisions.create.return_value = _response("fruit", confidence=confidence)
        original_meta = image.meta.copy()
        result = router.run(image=image, text="An apple.")
        client.decisions.create.assert_called_once_with(
            model="gpt-6-luna",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "An apple."},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image.base64_image}",
                            "detail": "high",
                        },
                    ],
                }
            ],
            questions=[
                {
                    "type": "choice",
                    "name": "route",
                    "instructions": "What does this image show?",
                    "choices": [
                        {"value": "fruit", "description": "Photographs of fruit"},
                        {"value": "logo", "description": "Company logos"},
                    ],
                }
            ],
        )
        assert result == {expected_output: image}
        assert result[expected_output] is image
        assert image.meta == original_meta

    def test_run_without_min_confidence_ignores_confidence(self, image):
        router = OpenAIDecisionsImageRouter(labels=_LABELS)
        client = MagicMock()
        router._client = client
        client.decisions.create.return_value = _response("logo", confidence=0.01)
        assert router.run(image=image) == {"logo": image}

    @pytest.mark.asyncio
    async def test_run_async(self, test_files_path):
        image = ImageContent.from_file_path(file_path=test_files_path / "images" / "apple.jpg")
        router = OpenAIDecisionsImageRouter(labels=_LABELS)
        client = MagicMock()
        router._async_client = client
        client.decisions.create = AsyncMock(return_value=_response("fruit"))
        assert await router.run_async(image=image) == {"fruit": image}
        client.decisions.create.assert_awaited_once_with(
            model="gpt-6-luna",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{image.base64_image}"}],
                }
            ],
            questions=[router._question],
        )

    @pytest.mark.parametrize("image", ["image.jpg"])
    def test_run_wrong_input_type(self, image):
        router = OpenAIDecisionsImageRouter(labels=_LABELS)
        with pytest.raises(TypeError, match=r"^OpenAIDecisionsImageRouter expects an ImageContent as input\.$"):
            router.run(image=image)
        assert router._client is None

    @pytest.mark.parametrize("min_confidence", [None, 0.6])
    def test_run_refusal(self, min_confidence, image):
        router = OpenAIDecisionsImageRouter(labels=_LABELS, min_confidence=min_confidence)
        client = MagicMock()
        router._client = client
        client.decisions.create.return_value = _response(None)
        assert router.run(image=image) == {"refused": image}

    def test_run_unexpected_answer(self, image):
        router = OpenAIDecisionsImageRouter(labels=_LABELS)
        client = MagicMock()
        router._client = client
        client.decisions.create.return_value = _response("unknown")
        with pytest.raises(ValueError, match="unexpected answer"):
            router.run(image=image)

    @pytest.mark.parametrize("text", [["context"]])
    def test_run_wrong_text_type(self, image, text):
        router = OpenAIDecisionsImageRouter(labels=_LABELS)
        with pytest.raises(TypeError, match=r"^OpenAIDecisionsImageRouter expects text to be a str or None\.$"):
            router.run(image=image, text=text)
        assert router._client is None


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_run(test_files_path):
    router = OpenAIDecisionsImageRouter(labels=_LABELS, instructions="What does this image show?")
    image = ImageContent.from_file_path(file_path=test_files_path / "images" / "apple.jpg", detail="low")
    assert router.run(image=image) == {"fruit": image}
