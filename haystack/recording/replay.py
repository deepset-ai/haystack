# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from haystack import logging
from haystack.recording.run import PipelineRun, RecordingError

logger = logging.getLogger(__name__)


class ReplayMode(str, Enum):
    STRICT = "strict"
    LOOSE = "loose"


class ReplayMismatchError(RecordingError):
    """Raised when replay validation fails in STRICT mode."""


DEFAULT_SIDE_EFFECTING_QUALNAMES: frozenset[str] = frozenset(
    {
        "haystack.components.generators.chat.openai.OpenAIChatGenerator",
        "haystack.components.generators.openai.OpenAIGenerator",
        "haystack.components.generators.chat.openai_responses.OpenAIResponsesChatGenerator",
        "haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever",
        "haystack.components.retrievers.in_memory.embedding_retriever.InMemoryEmbeddingRetriever",
        "haystack.components.embedders.openai_text_embedder.OpenAITextEmbedder",
        "haystack.components.embedders.openai_document_embedder.OpenAIDocumentEmbedder",
        "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
        "haystack.components.fetchers.link_content.LinkContentFetcher",
        "haystack.components.websearch.serper_dev.SerperDevWebSearch",
        "haystack.components.agents.agent.Agent",
    }
)


def is_side_effecting(instance: Any) -> bool:
    """Hybrid check for side-effecting components."""
    # explicit instance marker
    if getattr(instance, "__haystack_side_effecting__", False):
        return True
    # also check type marker
    if getattr(type(instance), "__haystack_side_effecting__", False):
        return True
    # check is_side_effecting attribute
    try:
        val = getattr(instance, "is_side_effecting", None)
        if isinstance(val, bool) and val:
            return True
        if val is not None and not isinstance(val, bool) and not callable(val) and bool(val):
            return True
    except Exception:
        pass

    # qualname check
    try:
        from haystack.core.serialization import generate_qualified_class_name

        qname = generate_qualified_class_name(type(instance))
    except Exception:
        qname = f"{type(instance).__module__}.{type(instance).__name__}"
    return qname in DEFAULT_SIDE_EFFECTING_QUALNAMES


class ReplayStore:
    """Wraps PipelineRun for deterministic replay."""

    def __init__(
        self, run: PipelineRun, mode: ReplayMode = ReplayMode.STRICT, explicit_set: set[str] | None = None
    ) -> None:
        self.run = run
        self.mode = mode
        self.explicit_set = explicit_set
        # cursor per component
        self._cursors: dict[str, int] = {}
        # quick map for validation
        self._components = run.components

    @classmethod
    def from_run(
        cls,
        replay: PipelineRun | str | Path | dict[str, Any],
        mode: ReplayMode | str = ReplayMode.STRICT,
        explicit_set: set[str] | None = None,
    ) -> ReplayStore:
        """Create ReplayStore from run object or path."""
        # resolve mode
        if isinstance(mode, str):
            try:
                mode_enum = ReplayMode(mode.lower())
            except ValueError:
                raise ValueError(f"Invalid replay_mode '{mode}', expected 'strict' or 'loose'")
        else:
            mode_enum = mode

        # resolve run
        run_obj: PipelineRun
        if isinstance(replay, PipelineRun):
            run_obj = replay
        elif isinstance(replay, (str, Path)):
            p = Path(replay)
            if not p.exists():
                raise FileNotFoundError(f"Replay file not found: {p}")
            run_obj = PipelineRun.load(p)
        elif isinstance(replay, dict):
            run_obj = PipelineRun.from_dict(replay)
        else:
            raise ValueError(f"Unsupported replay type: {type(replay)}")  # noqa: TRY004
        return cls(run=run_obj, mode=mode_enum, explicit_set=explicit_set)

    def should_replay(self, component_name: str, qualname: str, instance: Any) -> bool:
        """Return True if component should be replayed."""
        if self.explicit_set is not None:
            return component_name in self.explicit_set
        # check instance markers
        if getattr(instance, "__haystack_side_effecting__", False):
            return True
        if getattr(type(instance), "__haystack_side_effecting__", False):
            return True
        # check is_side_effecting attribute
        try:
            v = getattr(instance, "is_side_effecting", None)
            if isinstance(v, bool) and v:
                return True
            if v is not None and not callable(v) and bool(v):
                return True
        except Exception:
            pass
        if qualname in DEFAULT_SIDE_EFFECTING_QUALNAMES:
            return True
        # fallback to is_side_effecting helper
        return is_side_effecting(instance)

    def pop(self, component_name: str) -> Any | None:
        """Get next recorded output for component, advancing cursor."""
        records = self._components.get(component_name)
        if not records:
            return None
        idx = self._cursors.get(component_name, 0)
        if idx >= len(records):
            return None
        rec = records[idx]
        self._cursors[component_name] = idx + 1
        return rec

    def get_replay_output(self, component_name: str) -> Mapping[str, Any] | None:
        """Get replay output for component."""
        rec = self.pop(component_name)
        if rec is None:
            return None
        return rec.outputs

    def peek(self, component_name: str) -> Any | None:
        """Peek next record without advancing cursor."""
        records = self._components.get(component_name)
        if not records:
            return None
        idx = self._cursors.get(component_name, 0)
        if idx >= len(records):
            return None
        return records[idx]

    def validate_signature(self, pipeline: Any) -> None:
        """Validate pipeline signature in STRICT mode."""
        if self.mode != ReplayMode.STRICT:
            return
        from haystack.recording.recorder import compute_pipeline_signature

        current = compute_pipeline_signature(pipeline)
        recorded = self.run.pipeline_signature
        if recorded and current != recorded:
            raise ReplayMismatchError(
                f"Pipeline signature mismatch: recorded={recorded} current={current}. "
                "Pipeline graph changed since recording. Use replay_mode='loose' to ignore."
            )
        # Also check component names set matches
        recorded_names = set(self.run.components.keys())
        # For pipelines that have components not executed (e.g., conditional branches not taken),
        # the recorded set may be subset of graph nodes. Only check that recorded components exist in graph.
        current_names = set(pipeline.graph.nodes.keys())
        missing = recorded_names - current_names
        if missing:
            raise ReplayMismatchError(
                f"Recorded components {missing} not found in current pipeline. "
                "Pipeline components changed since recording."
            )
