# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest

from haystack.dataclasses.breakpoints import AgentBreakpoint, AgentSnapshot, Breakpoint, PipelineSnapshot, PipelineState


def test_agent_snapshot_no_warning_on_init():
    bp = AgentBreakpoint(agent_name="agent", break_point=Breakpoint(component_name="chat_generator"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        AgentSnapshot(component_inputs={}, component_visits={}, break_point=bp)


def test_agent_snapshot_warn_on_inplace_mutation():
    bp = AgentBreakpoint(agent_name="agent", break_point=Breakpoint(component_name="chat_generator"))
    snap = AgentSnapshot(component_inputs={}, component_visits={}, break_point=bp)
    with pytest.warns(DeprecationWarning, match="dataclasses.replace"):
        snap.component_inputs = {"new": "value"}


def test_pipeline_state_no_warning_on_init():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        PipelineState(inputs={}, component_visits={}, pipeline_outputs={})


def test_pipeline_state_warn_on_inplace_mutation():
    state = PipelineState(inputs={}, component_visits={}, pipeline_outputs={})
    with pytest.warns(DeprecationWarning, match="dataclasses.replace"):
        state.inputs = {"new": "value"}


def test_pipeline_snapshot_no_warning_on_init():
    state = PipelineState(inputs={}, component_visits={"comp": 1}, pipeline_outputs={})
    bp = Breakpoint(component_name="comp")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        PipelineSnapshot(original_input_data={}, ordered_component_names=["comp"], pipeline_state=state, break_point=bp)


def test_pipeline_snapshot_warn_on_inplace_mutation():
    state = PipelineState(inputs={}, component_visits={"comp": 1}, pipeline_outputs={})
    bp = Breakpoint(component_name="comp")
    snap = PipelineSnapshot(
        original_input_data={}, ordered_component_names=["comp"], pipeline_state=state, break_point=bp
    )
    with pytest.warns(DeprecationWarning, match="dataclasses.replace"):
        snap.original_input_data = {"new": "data"}
