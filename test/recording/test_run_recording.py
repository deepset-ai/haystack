# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import time
from pathlib import Path

import pytest

from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage, Document
from haystack.recording import PipelineRun, load_run


@component
class TextProducer:
    @component.output_types(text=str)
    def run(self, value: str) -> dict[str, str]:
        return {"text": f"produced:{value}"}


@component
class EchoComponent:
    @component.output_types(echo=str)
    def run(self, text: str) -> dict[str, str]:
        return {"echo": text}


@component
class MockChatGen:
    """Simple chat generator returning usage."""

    @component.output_types(replies=list)
    def run(self, prompt: str) -> dict[str, list]:
        msg = ChatMessage.from_assistant(
            f"reply to {prompt}",
            meta={"model": "mock-model", "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
        )
        return {"replies": [msg]}


@component
class CounterLoop:
    def __init__(self, limit: int = 3) -> None:
        self.limit = limit

    @component.output_types(next_val=int, done=str)
    def run(self, value: int) -> dict[str, object]:
        if value < self.limit:
            return {"next_val": value + 1}
        return {"done": f"finished {value}"}


def test_record_returns_tuple_and_preserves_result():
    pipe = Pipeline()
    pipe.add_component("p", TextProducer())
    pipe.add_component("e", EchoComponent())
    pipe.connect("p", "e")
    data = {"p": {"value": "hello"}}
    result_plain = pipe.run(data)
    result, run = pipe.run(data, record=True)
    assert result == result_plain
    assert isinstance(run, PipelineRun)
    assert run.output_data == result
    assert run.input_data == data
    assert run.format == "v1"
    assert run.pipeline_signature != ""
    assert run.haystack_version != ""


def test_record_captures_per_component_io_and_visits(tmp_path):
    pipe = Pipeline()
    pipe.add_component("a", TextProducer())
    pipe.add_component("b", EchoComponent())
    pipe.add_component("c", EchoComponent())
    pipe.connect("a", "b")
    pipe.connect("b", "c")
    result, run = pipe.run({"a": {"value": "x"}}, record=True)
    # three components executed
    assert len(run.timeline) == 3
    assert len(run.components) == 3
    assert "a" in run.components
    assert run.components["a"][0].inputs == {"value": "x"}
    assert run.components["a"][0].outputs == {"text": "produced:x"}
    assert run.components["a"][0].visit_index == 0
    # ensure save/load roundtrip preserves ChatMessage etc
    path = tmp_path / "run.json"
    run.save(path)
    assert path.exists()
    loaded = PipelineRun.load(path)
    assert loaded.run_id == run.run_id
    assert loaded.pipeline_signature == run.pipeline_signature
    # also via module helper
    loaded2 = load_run(path)
    assert loaded2.run_id == run.run_id


def test_record_with_chat_message_serialization(tmp_path):
    pipe = Pipeline()
    pipe.add_component("gen", MockChatGen())
    result, run = pipe.run({"gen": {"prompt": "hello"}}, record=True)
    # check usage extraction
    assert "gen" in run.components
    rec = run.components["gen"][0]
    assert rec.usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    # save/load preserves ChatMessage
    path = tmp_path / "chat_run.json"
    run.save(path)
    loaded = PipelineRun.load(path)
    # outputs should deserialize to ChatMessage
    replies = loaded.components["gen"][0].outputs["replies"]
    assert len(replies) == 1
    assert isinstance(replies[0], ChatMessage)
    assert replies[0].text == "reply to hello"
    # also check input_data deserialization
    assert loaded.output_data["gen"]["replies"][0].text == "reply to hello"


def test_timeline_captures_durations():
    pipe = Pipeline()
    pipe.add_component("p", TextProducer())
    pipe.add_component("e", EchoComponent())
    pipe.connect("p", "e")
    _, run = pipe.run({"p": {"value": "t"}}, record=True)
    assert len(run.timeline) == 2
    for entry in run.timeline:
        assert entry.duration_s >= 0
        assert entry.ended_at >= entry.started_at
        assert entry.component_name in ("p", "e")
    # timeline sorted by started_at
    starts = [e.started_at for e in run.timeline]
    assert starts == sorted(starts)


def test_record_loop_multiple_visits():
    from haystack.components.joiners import BranchJoiner

    pipe = Pipeline(max_runs_per_component=10)
    pipe.add_component("joiner", BranchJoiner(int))
    pipe.add_component("counter", CounterLoop(limit=3))
    pipe.connect("joiner.value", "counter.value")
    pipe.connect("counter.next_val", "joiner.value")
    _, run = pipe.run({"joiner": {"value": 0}}, record=True)
    # joiner visited 4 times (0,1,2,3), counter 4 times
    assert len(run.components["joiner"]) == 4
    assert len(run.components["counter"]) == 4
    assert [r.visit_index for r in run.components["joiner"]] == [0, 1, 2, 3]
    assert len(run.timeline) == 8


def test_save_load_preserves_documents(tmp_path):
    @component
    class DocProducer:
        @component.output_types(documents=list)
        def run(self, query: str) -> dict[str, list]:
            return {"documents": [Document(content=f"doc for {query}", meta={"query": query})]}

    pipe = Pipeline()
    pipe.add_component("retriever", DocProducer())
    _, run = pipe.run({"retriever": {"query": "hello"}}, record=True)
    path = tmp_path / "doc_run.json"
    run.save(path)
    loaded = PipelineRun.load(path)
    docs = loaded.components["retriever"][0].outputs["documents"]
    assert isinstance(docs[0], Document)
    assert docs[0].content == "doc for hello"


def test_record_without_content_tracing_env(monkeypatch):
    # Ensure recording works even when content tracing disabled
    monkeypatch.delenv("HAYSTACK_CONTENT_TRACING_ENABLED", raising=False)
    pipe = Pipeline()
    pipe.add_component("p", TextProducer())
    _, run = pipe.run({"p": {"value": "x"}}, record=True)
    assert run.components["p"][0].inputs == {"value": "x"}


def test_to_dict_from_dict_roundtrip():
    pipe = Pipeline()
    pipe.add_component("p", TextProducer())
    _, run = pipe.run({"p": {"value": "a"}}, record=True)
    d = run.to_dict()
    assert d["format"] == "v1"
    assert "run_id" in d
    restored = PipelineRun.from_dict(d)
    assert restored.run_id == run.run_id
    assert restored.usage == run.usage
    # json roundtrip
    j = run.to_json()
    assert json.loads(j)["run_id"] == run.run_id


def test_invalid_format_raises(tmp_path):
    pipe = Pipeline()
    pipe.add_component("p", TextProducer())
    _, run = pipe.run({"p": {"value": "a"}}, record=True)
    d = run.to_dict()
    d["format"] = "v2"
    with pytest.raises(Exception, match="Incompatible recording format"):
        PipelineRun.from_dict(d)
