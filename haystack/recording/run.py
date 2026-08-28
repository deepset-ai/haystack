# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from haystack import logging
from haystack.utils.base_serialization import _deserialize_value_with_schema, _serialize_value_with_schema

try:
    from haystack.version import __version__ as haystack_version
except Exception:
    haystack_version = "unknown"

logger = logging.getLogger(__name__)

RECORDING_FORMAT = "v1"


class RecordingError(Exception):
    """Base error for recording operations."""


@dataclass
class ComponentRecord:
    """Record for a single component execution."""

    component_name: str
    visit_index: int
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    duration_s: float
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "component_name": self.component_name,
            "visit_index": self.visit_index,
            "inputs": _serialize_value_with_schema(self.inputs),
            "outputs": _serialize_value_with_schema(dict(self.outputs)),
            "duration_s": self.duration_s,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComponentRecord:
        """Deserialize from dict."""
        inputs = data.get("inputs")
        outputs = data.get("outputs")
        # inputs/outputs are stored as serialized envelope
        if isinstance(inputs, dict) and "serialization_schema" in inputs and "serialized_data" in inputs:
            inputs = _deserialize_value_with_schema(inputs)
        if isinstance(outputs, dict) and "serialization_schema" in outputs and "serialized_data" in outputs:
            outputs = _deserialize_value_with_schema(outputs)
        return cls(
            component_name=data["component_name"],
            visit_index=data["visit_index"],
            inputs=inputs if isinstance(inputs, dict) else {},
            outputs=outputs if isinstance(outputs, dict) else {},
            duration_s=data.get("duration_s", 0.0),
            usage=data.get("usage"),
        )


@dataclass
class TimelineEntry:
    """Timeline entry for a component execution."""

    component_name: str
    visit_index: int
    duration_s: float
    started_at: float
    ended_at: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimelineEntry:
        """Deserialize from dict."""
        return cls(**data)


def _aggregate_usage(usages: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Aggregate list of usage dicts using recursive accumulation."""
    from haystack.components.agents.utils import _accumulate_usage  # lazy to avoid circular

    total: dict[str, Any] = {}
    first = True
    for u in usages:
        if not isinstance(u, dict):
            continue
        if first:
            # deep copy to avoid mutating original
            total = dict(u)
            first = False
        else:
            total = _accumulate_usage(total, u)
    return total


def aggregate_usage(usages: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Public helper to aggregate usage dicts."""
    return _aggregate_usage(usages)


@dataclass
class PipelineRun:
    """Shareable artifact of a pipeline execution."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_signature: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    components: dict[str, list[ComponentRecord]] = field(default_factory=dict)
    timeline: list[TimelineEntry] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    cost_estimate: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    haystack_version: str = haystack_version
    format: str = RECORDING_FORMAT  # noqa: A003

    def __post_init__(self) -> None:
        if not self.created_at:
            from datetime import datetime, timezone

            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.run_id:
            self.run_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict with JSON-serializable leaves."""
        return {
            "format": self.format,
            "run_id": self.run_id,
            "pipeline_signature": self.pipeline_signature,
            "created_at": self.created_at,
            "haystack_version": self.haystack_version,
            "input_data": _serialize_value_with_schema(self.input_data),
            "output_data": _serialize_value_with_schema(self.output_data),
            "components": {name: [rec.to_dict() for rec in records] for name, records in self.components.items()},
            "timeline": [entry.to_dict() for entry in self.timeline],
            "usage": self.usage,
            "cost_estimate": self.cost_estimate,
        }

    def to_json(self) -> str:
        """Return JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineRun:  # noqa: D102
        """Deserialize from dict."""
        fmt = data.get("format", RECORDING_FORMAT)
        if fmt != RECORDING_FORMAT:
            raise RecordingError(f"Incompatible recording format '{fmt}', expected '{RECORDING_FORMAT}'")
        input_data = data.get("input_data", {})
        output_data = data.get("output_data", {})
        if isinstance(input_data, dict) and "serialization_schema" in input_data and "serialized_data" in input_data:
            input_data = _deserialize_value_with_schema(input_data)
        if isinstance(output_data, dict) and "serialization_schema" in output_data and "serialized_data" in output_data:
            output_data = _deserialize_value_with_schema(output_data)
        components_raw = data.get("components", {})
        components: dict[str, list[ComponentRecord]] = {}
        for name, recs in components_raw.items():
            components[name] = [ComponentRecord.from_dict(r) for r in recs]
        timeline_raw = data.get("timeline", [])
        timeline = [TimelineEntry.from_dict(t) for t in timeline_raw]
        return cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            pipeline_signature=data.get("pipeline_signature", ""),
            input_data=input_data if isinstance(input_data, dict) else {},
            output_data=output_data if isinstance(output_data, dict) else {},
            components=components,
            timeline=timeline,
            usage=data.get("usage", {}),
            cost_estimate=data.get("cost_estimate", {}),
            created_at=data.get("created_at", ""),
            haystack_version=data.get("haystack_version", haystack_version),
            format=fmt,
        )

    def save(self, path: str | Path) -> str:
        """Save to JSON file. Creates parent directories."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> PipelineRun:
        """Load from JSON file."""
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # convenience alias used by tests
    @property
    def total_usage(self) -> dict[str, Any]:  # noqa: D102
        """Return total usage."""
        return self.usage

    @property
    def total_cost(self) -> float:  # noqa: D102
        """Return total cost."""
        if isinstance(self.cost_estimate, dict):
            return float(self.cost_estimate.get("total", 0.0))
        return 0.0


def load_run(path: str | Path) -> PipelineRun:
    """Module-level helper to load a PipelineRun from path."""
    return PipelineRun.load(path)
