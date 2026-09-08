# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from haystack import Pipeline
from haystack.core.serialization import generate_qualified_class_name
from haystack.testing.sample_components import AddFixedValue


def test_pipeline_run_reports_tagged_event_name(block_telemetry_network_calls: Mock) -> None:
    """
    Covers the autouse fixtures from `haystack.testing.telemetry`: Pipeline runs in this test suite should
    report a "(tests)"-tagged event name instead of the production "Pipeline run (3.x)" one, and the event must
    reach `posthog.capture` through the real `Telemetry.send_event()` code path end-to-end.

    `block_telemetry_network_calls` is the `posthog.capture` mock installed for every test - the only thing between
    the real telemetry code and PostHog - so asserting on it also proves nothing is ever sent.
    """
    mock_capture = block_telemetry_network_calls

    pipe = Pipeline()
    pipe.add_component("add", AddFixedValue())
    pipe.run({"add": {"value": 1}})

    mock_capture.assert_called_once()
    assert mock_capture.call_args.kwargs["event"] == "Pipeline run (3.x) (tests)"

    properties = mock_capture.call_args.kwargs["properties"]
    assert properties["pipeline_type"] == generate_qualified_class_name(Pipeline)
    assert properties["components"] == {generate_qualified_class_name(AddFixedValue): [{"name": "add"}]}
