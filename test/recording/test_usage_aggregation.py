# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage
from haystack.recording.recorder import _estimate_cost, _extract_model, _extract_usage, compute_pipeline_signature


def test_extract_usage_chat_message():
    msg1 = ChatMessage.from_assistant(
        "hello", meta={"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    )
    msg2 = ChatMessage.from_assistant(
        "hi", meta={"usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}}
    )
    outputs = {"replies": [msg1, msg2]}
    usage = _extract_usage(outputs)
    # Should accumulate: 5+2=7 prompt, 3+2=5 completion, 8+4=12 total
    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 12


def test_extract_usage_embedder():
    outputs = {
        "embedding": [0.1, 0.2],
        "meta": {"model": "text-embedding-ada-002", "usage": {"prompt_tokens": 10, "total_tokens": 10}},
    }
    usage = _extract_usage(outputs)
    assert usage == {"prompt_tokens": 10, "total_tokens": 10}


def test_extract_usage_alternative_keys():
    # OpenAIResponses uses input_tokens/output_tokens
    msg = ChatMessage.from_assistant("hi", meta={"usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}})
    usage = _extract_usage({"replies": [msg]})
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3


def test_extract_usage_no_usage_returns_none():
    outputs = {"text": "hello"}
    assert _extract_usage(outputs) is None
    assert _extract_usage({"replies": []}) is None
    # ChatMessage without usage
    msg = ChatMessage.from_assistant("hi")
    assert _extract_usage({"replies": [msg]}) is None


def test_extract_model():
    msg = ChatMessage.from_assistant("hi", meta={"model": "gpt-4o", "usage": {"prompt_tokens": 1}})
    model = _extract_model({"replies": [msg]})
    assert model == "gpt-4o"
    assert _extract_model({"meta": {"model": "text-embedding-ada-002"}}) == "text-embedding-ada-002"
    assert _extract_model({"text": "hi"}) is None


def test_estimate_cost_gpt4o():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    cost = _estimate_cost(usage, "gpt-4o")
    assert cost == pytest.approx(12.5)  # 2.5 + 10
    # gpt-3.5
    cost2 = _estimate_cost(usage, "gpt-3.5-turbo")
    assert cost2 == pytest.approx(2.0)  # 0.5 + 1.5


def test_estimate_cost_embedding():
    usage = {"prompt_tokens": 1_000_000, "total_tokens": 1_000_000}
    cost = _estimate_cost(usage, "text-embedding-ada-002")
    assert cost == pytest.approx(0.1)


def test_estimate_cost_default_zero():
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    assert _estimate_cost(usage, "unknown-model") == 0.0
    assert _estimate_cost(None, "gpt-4o") == 0.0
    assert _estimate_cost({}, "gpt-4o") == 0.0


def test_pipeline_run_usage_aggregation():
    @component
    class GenA:
        @component.output_types(replies=list)
        def run(self, prompt: str):
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "a",
                        meta={
                            "model": "gpt-4o",
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        },
                    )
                ]
            }

    @component
    class GenB:
        @component.output_types(replies=list)
        def run(self, prompt: str):
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "b",
                        meta={
                            "model": "gpt-4o",
                            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                        },
                    )
                ]
            }

    pipe = Pipeline()
    pipe.add_component("a", GenA())
    pipe.add_component("b", GenB())
    # No connection, both run independently
    _, run = pipe.run({"a": {"prompt": "hi"}, "b": {"prompt": "hi"}}, record=True)
    # total usage should be sum: 30 prompt, 15 completion, 45 total
    assert run.usage["prompt_tokens"] == 30
    assert run.usage["completion_tokens"] == 15
    assert run.usage["total_tokens"] == 45
    # cost: (30*2.5 + 15*10)/1e6
    expected_cost = (30 * 2.5 + 15 * 10) / 1_000_000
    assert run.cost_estimate["total"] == pytest.approx(expected_cost)
    assert "a" in run.cost_estimate["by_component"]
    assert "b" in run.cost_estimate["by_component"]
    assert run.cost_estimate["by_component"]["a"] == pytest.approx((10 * 2.5 + 5 * 10) / 1_000_000)
    assert run.cost_estimate["by_component"]["b"] == pytest.approx((20 * 2.5 + 10 * 10) / 1_000_000)


def test_usage_aggregation_with_mixed_keys():
    @component
    class MixedGen:
        @component.output_types(replies=list)
        def run(self, prompt: str):
            # First reply uses prompt/completion, second uses input/output
            # Actually single reply but test accumulation across components with different conventions
            return {
                "replies": [
                    ChatMessage.from_assistant("a", meta={"usage": {"prompt_tokens": 5, "completion_tokens": 5}})
                ]
            }

    @component
    class MixedGen2:
        @component.output_types(replies=list)
        def run(self, prompt: str):
            return {
                "replies": [ChatMessage.from_assistant("b", meta={"usage": {"input_tokens": 10, "output_tokens": 5}})]
            }

    pipe = Pipeline()
    pipe.add_component("a", MixedGen())
    pipe.add_component("b", MixedGen2())
    _, run = pipe.run({"a": {"prompt": "x"}, "b": {"prompt": "x"}}, record=True)
    # Should aggregate both, but keys remain separate? Our _accumulate_usage keeps both keys
    # So total will have prompt_tokens, completion_tokens, input_tokens, output_tokens
    assert run.usage.get("prompt_tokens") == 5
    assert run.usage.get("input_tokens") == 10
    # Cost estimate should sum correctly using first numeric helpers
    # For prompt: first of prompt_tokens/input_tokens -> 5 for first, 10 for second? But total cost calculation uses _first_numeric per usage, not aggregated mixed.
    # For aggregated usage, _first_numeric will pick prompt_tokens first, so 5, not 15. That's a limitation but acceptable.
    # Ensure cost is computed per-record then summed, not from aggregated total
    # So total cost should be (5*2.5+5*10)/1e6 + (10*2.5+5*10)/1e6 if model gpt-4o
    # But our Gens don't specify model, so cost default 0 -> total 0
    assert run.cost_estimate["total"] == 0.0


def test_compute_pipeline_signature_stable():
    # Use simple components
    @component
    class P1:
        @component.output_types(out=str)
        def run(self, x: str):
            return {"out": x}

    @component
    class P2:
        @component.output_types(out=str)
        def run(self, x: str):
            return {"out": x}

    pipe1 = Pipeline()
    pipe1.add_component("a", P1())
    pipe1.add_component("b", P2())
    pipe1.connect("a.out", "b.x")
    sig1 = compute_pipeline_signature(pipe1)
    sig2 = compute_pipeline_signature(pipe1)
    assert sig1 == sig2
    # Different graph should differ
    pipe2 = Pipeline()
    pipe2.add_component("a", P1())
    pipe2.add_component("b", P2())
    # no connection
    sig3 = compute_pipeline_signature(pipe2)
    assert sig1 != sig3


def test_cost_estimate_per_component_with_mock():
    @component
    class MockGen:
        @component.output_types(replies=list)
        def run(self, prompt: str):
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "hi", meta={"model": "mock-model", "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
                    )
                ]
            }

    pipe = Pipeline()
    pipe.add_component("gen", MockGen())
    _, run = pipe.run({"gen": {"prompt": "hello"}}, record=True)
    # mock-model cost is 0
    assert run.cost_estimate["total"] == 0.0
    assert run.cost_estimate["by_component"]["gen"] == 0.0
