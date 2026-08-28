# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage
from haystack.recording import PipelineRun, ReplayMode
from haystack.recording.replay import DEFAULT_SIDE_EFFECTING_QUALNAMES, ReplayMismatchError, is_side_effecting


@component
class PureBuilder:
    @component.output_types(prompt=str)
    def run(self, x: str) -> dict[str, str]:
        return {"prompt": f"prompt:{x}"}


@component
class SideEffectLLM:
    __haystack_side_effecting__ = True

    @component.output_types(replies=list)
    def run(self, prompt: str) -> dict[str, list]:
        # If this is called during replay, we would know replay failed
        assert False, "SideEffectLLM should not be called during replay"
        return {"replies": [ChatMessage.from_assistant("should not happen")]}


@component
class CountingPure:
    def __init__(self):
        self.calls = 0

    @component.output_types(out=str)
    def run(self, x: str) -> dict[str, str]:
        self.calls += 1
        return {"out": f"{x}-{self.calls}"}


@component
class CountingSide:
    __haystack_side_effecting__ = True

    def __init__(self):
        self.calls = 0

    @component.output_types(out=str)
    def run(self, x: str) -> dict[str, str]:
        self.calls += 1
        return {"out": f"side-{x}-{self.calls}"}


def test_replay_strict_reexecutes_routers_and_replays_side_effecting():
    # Build pipeline: pure -> side (should replay)
    pipe = Pipeline()
    builder = PureBuilder()
    side = SideEffectLLM()
    # Use a variant that actually returns without assertion for recording phase
    # so we need a recordable LLM that doesn't assert

    @component
    class RecordableLLM:
        __haystack_side_effecting__ = True

        @component.output_types(replies=list)
        def run(self, prompt: str) -> dict[str, list]:
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        f"answer for {prompt}",
                        meta={
                            "model": "mock-model",
                            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                        },
                    )
                ]
            }

    pipe2 = Pipeline()
    pipe2.add_component("builder", PureBuilder())
    pipe2.add_component("llm", RecordableLLM())
    pipe2.connect("builder", "llm")
    result, run = pipe2.run({"builder": {"x": "hello"}}, record=True)
    original_reply = result["llm"]["replies"][0].text

    # Now create replay pipeline with same structure but side component that would fail if called
    replay_pipe = Pipeline()
    replay_pipe.add_component("builder", PureBuilder())
    replay_pipe.add_component("llm", SideEffectLLM())
    replay_pipe.connect("builder", "llm")
    # Use replay with strict mode: should replay llm (no network, no assertion)
    # To avoid signature mismatch, we need same qualnames. SideEffectLLM vs RecordableLLM have different qualnames,
    # so strict would fail. Use loose for this test or make qualnames match.
    # Instead use same pipeline instance for strict test.

    # Strict with same pipeline instance (signature matches)
    result_replay = pipe2.run({"builder": {"x": "hello"}}, replay=run, replay_mode="strict")
    assert result_replay["llm"]["replies"][0].text == original_reply

    # Verify builder re-executed but llm replayed: use counting components
    pipe3 = Pipeline()
    pure = CountingPure()
    side_c = CountingSide()
    pipe3.add_component("pure", pure)
    pipe3.add_component("side", side_c)
    pipe3.connect("pure", "side")
    _, rec = pipe3.run({"pure": {"x": "a"}}, record=True)
    assert pure.calls == 1
    assert side_c.calls == 1
    assert rec.components["side"][0].outputs["out"] == "side-a-1-1"
    # reset counters but keep same pipeline instance for strict
    pure.calls = 0
    side_c.calls = 0
    res = pipe3.run({"pure": {"x": "a"}}, replay=rec, replay_mode="strict")
    # pure should have run live again, side should be replayed
    assert pure.calls == 1
    assert side_c.calls == 0
    assert res["side"]["out"] == "side-a-1-1"


def test_replay_loose_ignores_input_changes():
    pipe = Pipeline()
    pure = CountingPure()
    side = CountingSide()
    pipe.add_component("pure", pure)
    pipe.add_component("side", side)
    pipe.connect("pure", "side")
    _, rec = pipe.run({"pure": {"x": "first"}}, record=True)
    assert rec.components["side"][0].outputs["out"] == "side-first-1-1"
    pure.calls = 0
    side.calls = 0
    # loose replay with different input: side still returns recorded value
    res = pipe.run({"pure": {"x": "different"}}, replay=rec, replay_mode="loose")
    assert pure.calls == 1
    assert side.calls == 0
    assert res["side"]["out"] == "side-first-1-1"


def test_replay_explicit_override():
    # Pure component marked as replayed via explicit set
    pipe = Pipeline()
    a = CountingPure()
    b = CountingPure()
    pipe.add_component("a", a)
    pipe.add_component("b", b)
    pipe.connect("a", "b")
    _, rec = pipe.run({"a": {"x": "hi"}}, record=True)
    a.calls = 0
    b.calls = 0
    # explicit set says replay only "b", even though neither is side_effecting by default
    # But "b" should be replayed, "a" live
    res = pipe.run({"a": {"x": "hi"}}, replay=rec, replay_mode="strict", replay_side_effecting_components={"b"})
    assert a.calls == 1
    assert b.calls == 0
    # empty set means nothing replayed
    a.calls = 0
    b.calls = 0
    res2 = pipe.run({"a": {"x": "hi"}}, replay=rec, replay_mode="strict", replay_side_effecting_components=set())
    assert a.calls == 1
    assert b.calls == 1


def test_replay_strict_signature_mismatch_raises():
    pipe1 = Pipeline()
    pipe1.add_component("a", PureBuilder())
    pipe1.add_component("b", CountingSide())
    pipe1.connect("a", "b")
    _, rec = pipe1.run({"a": {"x": "hi"}}, record=True)

    # Different pipeline graph: missing component
    pipe2 = Pipeline()
    pipe2.add_component("a", PureBuilder())
    # no b component, so signature mismatch
    with pytest.raises(ReplayMismatchError, match="Pipeline signature mismatch"):
        pipe2.run({"a": {"x": "hi"}}, replay=rec, replay_mode="strict")

    # Loose should not raise signature mismatch (but will have missing recorded component)
    # For loose, it will just run live since no side_effecting components to replay? But our pipe2 has only "a" which is not side_effecting,
    # so replay should just run normally without error
    res = pipe2.run({"a": {"x": "hi"}}, replay=rec, replay_mode="loose")
    assert "a" in res


def test_replay_no_recorded_output_strict_raises():
    # Record pipeline with one execution, then try to replay with pipeline that has loop requiring second visit
    from haystack.components.joiners import BranchJoiner

    @component
    class Counter:
        __haystack_side_effecting__ = True

        def __init__(self, limit=2):
            self.limit = limit

        @component.output_types(next_val=int, done=str)
        def run(self, value: int) -> dict[str, object]:
            if value < self.limit:
                return {"next_val": value + 1}
            return {"done": f"done {value}"}

    pipe = Pipeline(max_runs_per_component=5)
    pipe.add_component("joiner", BranchJoiner(int))
    pipe.add_component("counter", Counter(limit=2))
    pipe.connect("joiner.value", "counter.value")
    pipe.connect("counter.next_val", "joiner.value")
    _, rec = pipe.run({"joiner": {"value": 0}}, record=True)
    # Now replay with same pipe but strict: should work for 2 loop iterations
    # If we artificially truncate recorded components, strict should raise when missing
    # Simulate by manually removing one record
    rec.components["counter"].pop()
    with pytest.raises(ReplayMismatchError):
        pipe.run({"joiner": {"value": 0}}, replay=rec, replay_mode="strict")


def test_is_side_effecting_detection():
    assert is_side_effecting(CountingSide())
    assert not is_side_effecting(CountingPure())
    # check default list via fake instance with matching qualname
    from unittest.mock import MagicMock

    # Create mock with qualname in default list
    # Use actual OpenAIChatGenerator class if available, else simulate
    try:
        from haystack.components.generators.chat.openai import OpenAIChatGenerator

        # Don't instantiate (needs API key), just check qualname string
        qname = "haystack.components.generators.chat.openai.OpenAIChatGenerator"
        assert qname in DEFAULT_SIDE_EFFECTING_QUALNAMES
    except ImportError:
        pass

    # marker via class attribute
    @component
    class Marked:
        __haystack_side_effecting__ = True

        @component.output_types(out=str)
        def run(self, x: str):
            return {"out": x}

    assert is_side_effecting(Marked())


def test_replay_with_path_string(tmp_path):
    pipe = Pipeline()
    pure = CountingPure()
    side = CountingSide()
    pipe.add_component("pure", pure)
    pipe.add_component("side", side)
    pipe.connect("pure", "side")
    _, rec = pipe.run({"pure": {"x": "a"}}, record=True)
    path = tmp_path / "run.json"
    rec.save(path)
    pure.calls = 0
    side.calls = 0
    # replay via path string
    res = pipe.run({"pure": {"x": "a"}}, replay=str(path), replay_mode="strict")
    assert pure.calls == 1
    assert side.calls == 0
    # also via Path object
    pure.calls = 0
    side.calls = 0
    res2 = pipe.run({"pure": {"x": "a"}}, replay=path, replay_mode="strict")
    assert pure.calls == 1
    assert side.calls == 0


def test_replay_side_effecting_default_list():
    # Test that default side-effecting components are detected via qualname
    # Use InMemoryBM25Retriever which is in default list
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.dataclasses import Document

    store = InMemoryDocumentStore()
    store.write_documents([Document(content="hello world"), Document(content="haystack")])
    retriever = InMemoryBM25Retriever(document_store=store)
    # is_side_effecting should be True via qualname
    assert is_side_effecting(retriever) is True

    @component
    class QueryBuilder:
        @component.output_types(query=str)
        def run(self, x: str):
            return {"query": x}

    pipe = Pipeline()
    pipe.add_component("builder", QueryBuilder())
    pipe.add_component("retriever", retriever)
    pipe.connect("builder.query", "retriever.query")
    # Record
    _, rec = pipe.run({"builder": {"x": "hello"}}, record=True)
    original_docs = rec.components["retriever"][0].outputs["documents"]
    # Modify store to change retrieval results
    store.write_documents([Document(content="different")])
    # Replay should return original docs even though store changed, because retriever is side_effecting and replayed
    result = pipe.run({"builder": {"x": "hello"}}, replay=rec, replay_mode="strict")
    assert len(result["retriever"]["documents"]) == len(original_docs)
    assert result["retriever"]["documents"][0].content == original_docs[0].content
