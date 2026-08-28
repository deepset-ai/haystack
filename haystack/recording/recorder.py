# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
import time
from collections.abc import Mapping
from typing import Any

from haystack import logging
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.dataclasses import ChatMessage
from haystack.recording.run import ComponentRecord, TimelineEntry

logger = logging.getLogger(__name__)

# ContextVar for async propagation
_recording_context_var: contextvars.ContextVar[Any] = contextvars.ContextVar("haystack_recording_context", default=None)


def _first_numeric(usage: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _extract_usage(outputs: Mapping[str, Any]) -> dict[str, Any] | None:  # noqa: C901 PLR0912
    """
    Extract usage dict from component outputs.

    Handles:
    - Chat generators: {"replies": [ChatMessage(meta={"usage": ...})]}
    - Embedders: {"meta": {"usage": {...}}}
    - Generic: {"usage": {...}}
    """
    if not isinstance(outputs, Mapping):
        return None

    # ChatMessage list paths
    for key in ("replies", "answers"):
        val = outputs.get(key)
        if isinstance(val, list) and val and isinstance(val[0], ChatMessage):
            from haystack.components.agents.utils import _accumulate_usage

            acc: dict[str, Any] | None = None
            for msg in val:
                if not isinstance(msg, ChatMessage):
                    continue
                meta = getattr(msg, "meta", None)
                if not isinstance(meta, dict):
                    continue
                u = meta.get("usage")
                if isinstance(u, dict):
                    if acc is None:
                        acc = dict(u)
                    else:
                        acc = _accumulate_usage(acc, u)
            if acc is not None:
                return acc

    # Embedder path: {"meta": {"usage": {...}}}
    meta = outputs.get("meta")
    if isinstance(meta, dict):
        u = meta.get("usage")
        if isinstance(u, dict):
            return dict(u)

    # Fallback top-level usage
    u = outputs.get("usage")
    if isinstance(u, dict):
        return dict(u)

    # Search any ChatMessage nested inside values
    for v in outputs.values():
        if isinstance(v, list) and v and isinstance(v[0], ChatMessage):
            from haystack.components.agents.utils import _accumulate_usage

            acc = None
            for msg in v:
                if isinstance(msg, ChatMessage):
                    meta = getattr(msg, "meta", None)
                    if isinstance(meta, dict) and isinstance(meta.get("usage"), dict):
                        u = meta["usage"]
                        if acc is None:
                            acc = dict(u)
                        else:
                            acc = _accumulate_usage(acc, u)
            if acc is not None:
                return acc

    return None


def _extract_model(outputs: Mapping[str, Any]) -> str | None:
    """Try to extract model name from outputs for cost estimation."""
    if not isinstance(outputs, Mapping):
        return None
    # ChatMessage path
    for key in ("replies", "answers"):
        val = outputs.get(key)
        if isinstance(val, list) and val and isinstance(val[0], ChatMessage):
            for msg in val:
                if isinstance(msg, ChatMessage):
                    meta = getattr(msg, "meta", None)
                    if isinstance(meta, dict) and isinstance(meta.get("model"), str):
                        return meta["model"]
    meta = outputs.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("model"), str):
        return meta["model"]
    # Also check usage dict model?
    return None


# Pricing per 1M tokens in USD
_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-3.5": {"input": 0.5, "output": 1.5},
    "text-embedding-ada-002": {"input": 0.1, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "mock-model": {"input": 0.0, "output": 0.0},
    "mock": {"input": 0.0, "output": 0.0},
}


def _estimate_cost(usage: dict[str, Any] | None, model: str | None = None) -> float:
    """Estimate cost in USD from usage dict and model."""
    if not isinstance(usage, dict):
        return 0.0
    # extract prompt/completion tokens with fallback keys
    prompt = _first_numeric(usage, ("prompt_tokens", "input_tokens"))
    completion = _first_numeric(usage, ("completion_tokens", "output_tokens"))
    # if both zero but total_tokens present, treat as prompt for embedder
    if prompt == 0 and completion == 0:
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            prompt = int(total)
    # Determine pricing
    model_l = (model or "").lower()
    pricing: dict[str, float] | None = None
    # exact match first
    for key, price in _PRICING_PER_1M.items():
        if key.lower() in model_l:
            pricing = price
            break
    if pricing is None:
        # check embedding generic
        if "embedding" in model_l or "ada" in model_l:
            pricing = {"input": 0.1, "output": 0.0}
        else:
            pricing = {"input": 0.0, "output": 0.0}
    return (prompt * pricing["input"] + completion * pricing["output"]) / 1_000_000


def compute_pipeline_signature(pipeline: Any) -> str:
    """Hash of sorted component qualnames + connections."""

    def _fallback_qname(cls: type) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    try:
        from haystack.core.serialization import generate_qualified_class_name
    except Exception:
        generate_qualified_class_name = _fallback_qname  # type: ignore

    nodes = []
    for name in sorted(pipeline.graph.nodes.keys()):
        instance = pipeline.graph.nodes[name].get("instance")
        if instance is not None:
            try:
                qname = generate_qualified_class_name(type(instance))
            except Exception:
                qname = f"{type(instance).__module__}.{type(instance).__name__}"
        else:
            qname = "unknown"
        nodes.append(f"{name}:{qname}")
    edges = []
    for sender, receiver, edge_data in sorted(pipeline.graph.edges(data=True)):
        from_socket = edge_data.get("from_socket")
        to_socket = edge_data.get("to_socket")
        from_name = getattr(from_socket, "name", str(from_socket)) if from_socket else "unknown"
        to_name = getattr(to_socket, "name", str(to_socket)) if to_socket else "unknown"
        edges.append(f"{sender}.{from_name}->{receiver}.{to_name}")
    raw = json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class RecordingContext:
    """Thread-safe collector for recording."""

    def __init__(self) -> None:  # noqa: D107
        self.records: list[ComponentRecord] = []
        self.timeline: list[TimelineEntry] = []
        self._lock = threading.Lock()
        self.start_perf = time.perf_counter()
        self.start_wall = time.time()
        # also set contextvar
        self._token = _recording_context_var.set(self)

    def add(self, record: ComponentRecord, timeline_entry: TimelineEntry) -> None:
        """Add a record and timeline entry."""
        with self._lock:
            self.records.append(record)
            self.timeline.append(timeline_entry)

    def add_from_parts(
        self,
        component_name: str,
        visit_index: int,
        inputs: dict[str, Any],
        outputs: Mapping[str, Any],
        duration_s: float,
        started_at: float,
        ended_at: float,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Add from individual parts, constructing record and entry."""
        rec = ComponentRecord(
            component_name=component_name,
            visit_index=visit_index,
            inputs=_deepcopy_with_exceptions(inputs),
            outputs=_deepcopy_with_exceptions(dict(outputs)),
            duration_s=duration_s,
            usage=usage,
        )
        entry = TimelineEntry(
            component_name=component_name,
            visit_index=visit_index,
            duration_s=duration_s,
            started_at=started_at,
            ended_at=ended_at,
        )
        self.add(rec, entry)

    def close(self) -> None:
        """Reset the contextvar."""
        import contextlib

        with contextlib.suppress(Exception):
            _recording_context_var.reset(self._token)

    @staticmethod
    def current() -> RecordingContext | None:
        """Return current context from ContextVar."""
        return _recording_context_var.get()

    def get_components_dict(self) -> dict[str, list[ComponentRecord]]:
        """Return components dict sorted by visit_index."""
        d: dict[str, list[ComponentRecord]] = {}
        with self._lock:
            for rec in self.records:
                d.setdefault(rec.component_name, []).append(rec)
        # sort each list by visit_index
        for lst in d.values():
            lst.sort(key=lambda r: r.visit_index)
        return d

    def get_timeline_sorted(self) -> list[TimelineEntry]:
        """Return timeline sorted by started_at."""
        with self._lock:
            return sorted(self.timeline, key=lambda e: e.started_at)
