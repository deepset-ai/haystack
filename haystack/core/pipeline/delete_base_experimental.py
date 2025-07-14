# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from haystack.core.pipeline.base import PipelineBase as HaystackPipelineBase
from haystack.core.pipeline.component_checks import _NO_OUTPUT_PRODUCED, is_socket_lazy_variadic


class PipelineBase(HaystackPipelineBase):
    @staticmethod
    def _consume_component_inputs(
        component_name: str, component: Dict, inputs: Dict, is_resume: bool = False
    ) -> Dict[str, Any]:
        """
        Extracts the inputs needed to run for the component and removes them from the global inputs state.

        :param component_name: The name of a component.
        :param component: Component with component metadata.
        :param inputs: Global inputs state.
        :returns: The inputs for the component.
        """
        component_inputs = inputs.get(component_name, {})
        consumed_inputs = {}
        greedy_inputs_to_remove = set()
        for socket_name, socket in component["input_sockets"].items():
            socket_inputs = component_inputs.get(socket_name, [])
            socket_inputs = [sock["value"] for sock in socket_inputs if sock["value"] is not _NO_OUTPUT_PRODUCED]

            # if we are resuming a component, the inputs are already consumed, so we just return the first input
            if is_resume:
                consumed_inputs[socket_name] = socket_inputs[0]
                continue
            if socket_inputs:
                if not socket.is_variadic:
                    # We only care about the first input provided to the socket.
                    consumed_inputs[socket_name] = socket_inputs[0]
                elif socket.is_greedy:
                    # We need to keep track of greedy inputs because we always remove them, even if they come from
                    # outside the pipeline. Otherwise, a greedy input from the user would trigger a pipeline to run
                    # indefinitely.
                    greedy_inputs_to_remove.add(socket_name)
                    consumed_inputs[socket_name] = [socket_inputs[0]]
                elif is_socket_lazy_variadic(socket):
                    # We use all inputs provided to the socket on a lazy variadic socket.
                    consumed_inputs[socket_name] = socket_inputs

        # We prune all inputs except for those that were provided from outside the pipeline (e.g. user inputs).
        pruned_inputs = {
            socket_name: [
                sock for sock in socket if sock["sender"] is None and not socket_name in greedy_inputs_to_remove
            ]
            for socket_name, socket in component_inputs.items()
        }
        pruned_inputs = {socket_name: socket for socket_name, socket in pruned_inputs.items() if len(socket) > 0}

        inputs[component_name] = pruned_inputs

        return consumed_inputs
