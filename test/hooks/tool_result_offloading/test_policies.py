# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.agents.state.state import State
from haystack.hooks.tool_result_offloading import AlwaysOffload, CallableOffloadPolicy, NeverOffload, OffloadOverChars


# Module-level so it is serializable by CallableOffloadPolicy.
def _offload_web_results(tool_name: str, result: str, state: State) -> bool:
    return tool_name == "web_search" and len(result) > 5


class TestOffloadPolicies:
    def test_always_offload(self):
        assert AlwaysOffload().should_offload("t", "anything", State(schema={})) is True

    def test_never_offload(self):
        assert NeverOffload().should_offload("t", "x" * 10_000, State(schema={})) is False

    def test_offload_over_chars_is_strictly_greater(self):
        policy = OffloadOverChars(threshold=10)
        assert policy.should_offload("t", "x" * 10, State(schema={})) is False
        assert policy.should_offload("t", "x" * 11, State(schema={})) is True

    def test_offload_over_chars_roundtrip(self):
        restored = OffloadOverChars.from_dict(OffloadOverChars(threshold=42).to_dict())
        assert restored.threshold == 42

    def test_callable_policy_delegates(self):
        policy = CallableOffloadPolicy(condition=_offload_web_results)
        assert policy.should_offload("web_search", "long result", State(schema={})) is True
        assert policy.should_offload("other", "long result", State(schema={})) is False

    def test_callable_policy_roundtrip(self):
        restored = CallableOffloadPolicy.from_dict(CallableOffloadPolicy(condition=_offload_web_results).to_dict())
        assert restored.should_offload("web_search", "long result", State(schema={})) is True
