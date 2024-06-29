# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from haystack import logging, tracing
from haystack.core.component import Component
from haystack.core.errors import PipelineMaxLoops, PipelineRuntimeError
from haystack.telemetry import pipeline_running

from .base import PipelineBase

logger = logging.getLogger(__name__)


class Pipeline(PipelineBase):
    """
    Synchronous version of the orchestration engine.

    Orchestrates component execution according to the execution graph, one after the other.
    """

    def _component_has_enough_inputs_to_run(self, name: str, inputs: Dict[str, Dict[str, Any]]) -> bool:
        """
        Returns True if the Component has all the inputs it needs to run.

        :param name: Name of the Component as defined in the Pipeline.
        :param inputs: The current state of the inputs divided by Component name.

        :return: Whether the Component can run or not.
        """
        instance: Component = self.graph.nodes[name]["instance"]
        if name not in inputs:
            return False
        expected_inputs = instance.__haystack_input__._sockets_dict.keys()  # type: ignore
        current_inputs = inputs[name].keys()
        return expected_inputs == current_inputs

    def _run_component(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs a Component with the given inputs.

        :param name: Name of the Component as defined in the Pipeline.
        :param inputs: Inputs for the Component.
        :raises PipelineRuntimeError: If Component doesn't return a dictionary.
        :return: The output of the Component.
        """
        instance: Component = self.graph.nodes[name]["instance"]

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={
                "haystack.component.name": name,
                "haystack.component.type": instance.__class__.__name__,
                "haystack.component.input_types": {k: type(v).__name__ for k, v in inputs.items()},
                "haystack.component.input_spec": {
                    key: {
                        "type": value.type.__name__ if isinstance(value.type, type) else str(value.type),
                        "senders": value.senders,
                    }
                    for key, value in instance.__haystack_input__._sockets_dict.items()  # type: ignore
                },
                "haystack.component.output_spec": {
                    key: {
                        "type": value.type.__name__ if isinstance(value.type, type) else str(value.type),
                        "receivers": value.receivers,
                    }
                    for key, value in instance.__haystack_output__._sockets_dict.items()  # type: ignore
                },
            },
        ) as span:
            span.set_content_tag("haystack.component.input", inputs)
            logger.info("Running component {component_name}", component_name=name)
            res: Dict[str, Any] = instance.run(**inputs)
            self.graph.nodes[name]["visits"] += 1

            # After a Component that has variadic inputs is run, we need to reset the variadic inputs that were consumed
            for socket in instance.__haystack_input__._sockets_dict.values():  # type: ignore
                if socket.name not in inputs:
                    continue
                if socket.is_variadic:
                    inputs[socket.name] = []

            if not isinstance(res, Mapping):
                raise PipelineRuntimeError(
                    f"Component '{name}' didn't return a dictionary. "
                    "Components must always return dictionaries: check the the documentation."
                )
            span.set_tag("haystack.component.visits", self.graph.nodes[name]["visits"])
            span.set_content_tag("haystack.component.output", res)

            return res

    # TODO: We're ignoring these linting rules for the time being, after we properly optimize this function we'll
    # remove the noqa
    def run(  # noqa: C901, PLR0912, PLR0915 pylint: disable=too-many-branches,too-many-locals
        self, data: Dict[str, Any], debug: bool = False, include_outputs_from: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs the pipeline with given input data.

        :param data:
            A dictionary of inputs for the pipeline's components. Each key is a component name
            and its value is a dictionary of that component's input parameters:
            ```
            data = {
                "comp1": {"input1": 1, "input2": 2},
            }
            ```
            For convenience, this format is also supported when input names are unique:
            ```
            data = {
                "input1": 1, "input2": 2,
            }
            ```

        :param debug:
            Set to True to collect and return debug information.
        :param include_outputs_from:
            Set of component names whose individual outputs are to be
            included in the pipeline's output. For components that are
            invoked multiple times (in a loop), only the last-produced
            output is included.
        :returns:
            A dictionary where each entry corresponds to a component name
            and its output. If `include_outputs_from` is `None`, this dictionary
            will only contain the outputs of leaf components, i.e., components
            without outgoing connections.

        :raises PipelineRuntimeError:
            If a component fails or returns unexpected output.

        Example a - Using named components:
        Consider a 'Hello' component that takes a 'word' input and outputs a greeting.

        ```python
        @component
        class Hello:
            @component.output_types(output=str)
            def run(self, word: str):
                return {"output": f"Hello, {word}!"}
        ```

        Create a pipeline with two 'Hello' components connected together:

        ```python
        pipeline = Pipeline()
        pipeline.add_component("hello", Hello())
        pipeline.add_component("hello2", Hello())
        pipeline.connect("hello.output", "hello2.word")
        result = pipeline.run(data={"hello": {"word": "world"}})
        ```

        This runs the pipeline with the specified input for 'hello', yielding
        {'hello2': {'output': 'Hello, Hello, world!!'}}.

        Example b - Using flat inputs:
        You can also pass inputs directly without specifying component names:

        ```python
        result = pipeline.run(data={"word": "world"})
        ```

        The pipeline resolves inputs to the correct components, returning
        {'hello2': {'output': 'Hello, Hello, world!!'}}.
        """
        pipeline_running(self)

        # Reset the visits count for each component
        self._init_graph()

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        # normalize `data`
        data = self._prepare_component_input_data(data)

        # Raise if input is malformed in some way
        self._validate_input(data)

        # Initialize the inputs state
        last_inputs: Dict[str, Dict[str, Any]] = self._init_inputs_state(data)

        # Take all components that:
        # - have no inputs
        # - receive input from the user
        # - have at least one input not connected
        # - have at least one input that is variadic
        to_run: List[Tuple[str, Component]] = self._init_to_run(data)

        # These variables are used to detect when we're stuck in a loop.
        # Stuck loops can happen when one or more components are waiting for input but
        # no other component is going to run.
        # This can happen when a whole branch of the graph is skipped for example.
        # When we find that two consecutive iterations of the loop where the waiting_for_input list is the same,
        # we know we're stuck in a loop and we can't make any progress.
        before_last_waiting_for_input: Optional[Set[str]] = None
        last_waiting_for_input: Optional[Set[str]] = None

        # The waiting_for_input list is used to keep track of components that are waiting for input.
        waiting_for_input: List[Tuple[str, Component]] = []

        include_outputs_from = set() if include_outputs_from is None else include_outputs_from

        # This is what we'll return at the end
        final_outputs: Dict[Any, Any] = {}

        with tracing.tracer.trace(
            "haystack.pipeline.run",
            tags={
                "haystack.pipeline.input_data": data,
                "haystack.pipeline.output_data": final_outputs,
                "haystack.pipeline.debug": debug,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_loops_allowed": self.max_loops_allowed,
            },
        ):
            # Cache for extra outputs, if enabled.
            extra_outputs: Dict[Any, Any] = {}

            while len(to_run) > 0:
                name, comp = to_run.pop(0)

                if any(socket.is_variadic for socket in comp.__haystack_input__._sockets_dict.values()) and not getattr(  # type: ignore
                    comp, "is_greedy", False
                ):
                    there_are_non_variadics = False
                    for _, other_comp in to_run:
                        if not any(
                            socket.is_variadic
                            for socket in other_comp.__haystack_input__._sockets_dict.values()  # type: ignore
                        ):
                            there_are_non_variadics = True
                            break

                    if there_are_non_variadics:
                        if (name, comp) not in waiting_for_input:
                            waiting_for_input.append((name, comp))
                        continue

                if self._component_has_enough_inputs_to_run(name, last_inputs):
                    if self.graph.nodes[name]["visits"] > self.max_loops_allowed:
                        msg = f"Maximum loops count ({self.max_loops_allowed}) exceeded for component '{name}'"
                        raise PipelineMaxLoops(msg)

                    res: Dict[str, Any] = self._run_component(name, last_inputs[name])

                    if name in include_outputs_from:
                        # Deepcopy the outputs to prevent downstream nodes from modifying them
                        # We don't care about loops - Always store the last output.
                        extra_outputs[name] = deepcopy(res)

                    # Reset the waiting for input previous states, we managed to run a component
                    before_last_waiting_for_input = None
                    last_waiting_for_input = None

                    if (name, comp) in waiting_for_input:
                        # We manage to run this component that was in the waiting list, we can remove it.
                        # This happens when a component was put in the waiting list but we reached it from another edge.
                        waiting_for_input.remove((name, comp))

                    self._dequeue_components_that_received_no_input(name, res, to_run, waiting_for_input)
                    res = self._distribute_output(name, res, last_inputs, to_run, waiting_for_input)

                    if len(res) > 0:
                        final_outputs[name] = res
                else:
                    # This component doesn't have enough inputs so we can't run it yet
                    if (name, comp) not in waiting_for_input:
                        waiting_for_input.append((name, comp))

                if len(to_run) == 0 and len(waiting_for_input) > 0:
                    # Check if we're stuck in a loop.
                    # It's important to check whether previous waitings are None as it could be that no
                    # Component has actually been run yet.
                    if (
                        before_last_waiting_for_input is not None
                        and last_waiting_for_input is not None
                        and before_last_waiting_for_input == last_waiting_for_input
                    ):
                        # Are we actually stuck or there's a lazy variadic or a component with has only default inputs
                        # waiting for input?
                        # This is our last resort, if there's no lazy variadic or component with only default inputs
                        # waiting for input we're stuck for real and we can't make any progress.
                        for name, comp in waiting_for_input:
                            is_variadic = any(
                                socket.is_variadic
                                for socket in comp.__haystack_input__._sockets_dict.values()  # type: ignore
                            )
                            has_only_defaults = all(
                                not socket.is_mandatory
                                for socket in comp.__haystack_input__._sockets_dict.values()  # type: ignore
                            )
                            if is_variadic and not comp.__haystack_is_greedy__ or has_only_defaults:  # type: ignore[attr-defined]
                                break
                        else:
                            # We're stuck in a loop for real, we can't make any progress.
                            # BAIL!
                            break

                        if len(waiting_for_input) == 1:
                            # We have a single component with variadic input or only default inputs waiting for input.
                            # If we're at this point it means it has been waiting for input for at least 2 iterations.
                            # This will never run.
                            # BAIL!
                            break

                        # There was a lazy variadic or a component with only default waiting for input, we can run it
                        waiting_for_input.remove((name, comp))
                        to_run.append((name, comp))

                        # Let's use the default value for the inputs that are still missing, or the component
                        # won't run and will be put back in the waiting list, causing an infinite loop.
                        for input_socket in comp.__haystack_input__._sockets_dict.values():  # type: ignore
                            if input_socket.is_mandatory:
                                continue
                            if input_socket.name not in last_inputs[name]:
                                last_inputs[name][input_socket.name] = input_socket.default_value

                        continue

                    before_last_waiting_for_input = (
                        last_waiting_for_input.copy() if last_waiting_for_input is not None else None
                    )
                    last_waiting_for_input = {item[0] for item in waiting_for_input}

                    self._enqueue_next_runnable_component(last_inputs, to_run, waiting_for_input)

            if len(include_outputs_from) > 0:
                for name, output in extra_outputs.items():
                    inner = final_outputs.get(name)
                    if inner is None:
                        final_outputs[name] = output
                    else:
                        # Let's not override any keys that are already
                        # in the final_outputs as they might be different
                        # from what we cached in extra_outputs, e.g. when loops
                        # are involved.
                        for k, v in output.items():
                            if k not in inner:
                                inner[k] = v

            return final_outputs
