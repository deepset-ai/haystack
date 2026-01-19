# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ConfirmationUIResult
from haystack.human_in_the_loop import AlwaysAskPolicy, AskOncePolicy, NeverAskPolicy
from haystack.tools import Tool, create_tool_from_function


def addition(x: int, y: int) -> int:
    return x + y


@pytest.fixture
def addition_tool() -> Tool:
    return create_tool_from_function(function=addition, name="Addition tool", description="Adds two integers together.")


class TestAlwaysAskPolicy:
    def test_should_ask_always_true(self, addition_tool):
        policy = AlwaysAskPolicy()
        assert policy.should_ask(addition_tool.name, addition_tool.description, {"x": 1, "y": 2}) is True

    def test_to_dict(self):
        policy = AlwaysAskPolicy()
        policy_dict = policy.to_dict()
        assert policy_dict["type"] == "haystack.human_in_the_loop.policies.AlwaysAskPolicy"
        assert policy_dict["init_parameters"] == {}

    def test_from_dict(self):
        policy_dict = {"type": "haystack.human_in_the_loop.policies.AlwaysAskPolicy", "init_parameters": {}}
        policy = AlwaysAskPolicy.from_dict(policy_dict)
        assert isinstance(policy, AlwaysAskPolicy)


class TestAskOncePolicy:
    def test_should_ask_first_time_true(self, addition_tool):
        policy = AskOncePolicy()
        assert policy.should_ask(addition_tool.name, addition_tool.description, {"x": 1, "y": 2}) is True

    def test_should_ask_second_time_false(self, addition_tool):
        policy = AskOncePolicy()
        params = {"x": 1, "y": 2}
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is True
        # Simulate the update after confirmation that occurs in HumanInTheLoopStrategy
        policy.update_after_confirmation(
            addition_tool.name, addition_tool.description, params, ConfirmationUIResult(action="confirm", feedback=None)
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, params) is False

    def test_should_ask_different_params_true(self, addition_tool):
        policy = AskOncePolicy()
        params1 = {"x": 1, "y": 2}
        params2 = {"x": 3, "y": 4}
        assert policy.should_ask(addition_tool.name, addition_tool.description, params1) is True
        # Simulate the update after confirmation that occurs in HumanInTheLoopStrategy
        policy.update_after_confirmation(
            addition_tool.name,
            addition_tool.description,
            params1,
            ConfirmationUIResult(action="confirm", feedback=None),
        )
        assert policy.should_ask(addition_tool.name, addition_tool.description, params2) is True

    def test_to_dict(self):
        policy = AskOncePolicy()
        policy_dict = policy.to_dict()
        assert policy_dict["type"] == "haystack.human_in_the_loop.policies.AskOncePolicy"
        assert policy_dict["init_parameters"] == {}

    def test_from_dict(self):
        policy_dict = {"type": "haystack.human_in_the_loop.policies.AskOncePolicy", "init_parameters": {}}
        policy = AskOncePolicy.from_dict(policy_dict)
        assert isinstance(policy, AskOncePolicy)


class TestNeverAskPolicy:
    def test_should_ask_always_false(self, addition_tool):
        policy = NeverAskPolicy()
        assert policy.should_ask(addition_tool.name, addition_tool.description, {"x": 1, "y": 2}) is False

    def test_to_dict(self):
        policy = NeverAskPolicy()
        policy_dict = policy.to_dict()
        assert policy_dict["type"] == "haystack.human_in_the_loop.policies.NeverAskPolicy"
        assert policy_dict["init_parameters"] == {}

    def test_from_dict(self):
        policy_dict = {"type": "haystack.human_in_the_loop.policies.NeverAskPolicy", "init_parameters": {}}
        policy = NeverAskPolicy.from_dict(policy_dict)
        assert isinstance(policy, NeverAskPolicy)
