from collections import defaultdict
from typing import Any, Dict, Optional, Union, TextIO, Tuple
from pathlib import Path
import datetime
import logging

from haystack.core.pipeline import Pipeline as _pipeline
from haystack.telemetry import pipeline_running
from haystack.marshal import Marshaller, YamlMarshaller


DEFAULT_MARSHALLER = YamlMarshaller()
logger = logging.getLogger(__name__)


class Pipeline(_pipeline):
    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        max_loops_allowed: int = 100,
        debug_path: Union[Path, str] = Path(".haystack_debug/"),
    ):
        """
        Creates the Pipeline.

        Args:
            metadata: arbitrary dictionary to store metadata about this pipeline. Make sure all the values contained in
                this dictionary can be serialized and deserialized if you wish to save this pipeline to file with
                `save_pipelines()/load_pipelines()`.
            max_loops_allowed: how many times the pipeline can run the same node before throwing an exception.
            debug_path: when debug is enabled in `run()`, where to save the debug data.
        """
        self._telemetry_runs = 0
        self._last_telemetry_sent: Optional[datetime.datetime] = None
        super().__init__(metadata=metadata, max_loops_allowed=max_loops_allowed, debug_path=debug_path)

    def run(self, data: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        """
        Runs the pipeline with given input data.

        :param data: A dictionary of inputs for the pipeline's components. Each key is a component name
        and its value is a dictionary of that component's input parameters.
        :param debug: Set to True to collect and return debug information.
        :return: A dictionary containing the pipeline's output.
        :raises PipelineRuntimeError: If a component fails or returns unexpected output.

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
        # check whether the data is a nested dictionary of component inputs where each key is a component name
        # and each value is a dictionary of input parameters for that component
        is_nested_component_input = all(isinstance(value, dict) for value in data.values())
        if is_nested_component_input:
            return self._run_internal(data=data, debug=debug)
        else:
            # flat input, a dict where keys are input names and values are the corresponding values
            # we need to convert it to a nested dictionary of component inputs and then run the pipeline
            # just like in the previous case
            pipeline_inputs, unresolved_inputs = self._prepare_component_input_data(data)
            if unresolved_inputs:
                logger.warning(
                    "Inputs %s were not matched to any component inputs, please check your run parameters.",
                    list(unresolved_inputs.keys()),
                )

            return self._run_internal(data=pipeline_inputs, debug=debug)

    def _run_internal(self, data: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        """
        Runs the pipeline by invoking the underlying run to initiate the pipeline execution.

        :params data: the inputs to give to the input components of the Pipeline.
        :params debug: whether to collect and return debug information.

        :returns: A dictionary with the outputs of the output components of the Pipeline.

        :raises PipelineRuntimeError: if any of the components fail or return unexpected output.
        """
        pipeline_running(self)
        return super().run(data=data, debug=debug)

    def dumps(self, marshaller: Marshaller = DEFAULT_MARSHALLER) -> str:
        """
        Returns the string representation of this pipeline according to the
        format dictated by the `Marshaller` in use.

        :params marshaller: The Marshaller used to create the string representation. Defaults to
                            `YamlMarshaller`

        :returns: A string representing the pipeline.
        """
        return marshaller.marshal(self.to_dict())

    def dump(self, fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER):
        """
        Writes the string representation of this pipeline to the file-like object
        passed in the `fp` argument.

        :params fp: A file-like object ready to be written to.
        :params marshaller: The Marshaller used to create the string representation. Defaults to
                            `YamlMarshaller`.
        """
        fp.write(marshaller.marshal(self.to_dict()))

    @classmethod
    def loads(cls, data: Union[str, bytes, bytearray], marshaller: Marshaller = DEFAULT_MARSHALLER) -> "Pipeline":
        """
        Creates a `Pipeline` object from the string representation passed in the `data` argument.

        :params data: The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
        :params marshaller: the Marshaller used to create the string representation. Defaults to
                            `YamlMarshaller`

        :returns: A `Pipeline` object.
        """
        return cls.from_dict(marshaller.unmarshal(data))

    @classmethod
    def load(cls, fp: TextIO, marshaller: Marshaller = DEFAULT_MARSHALLER) -> "Pipeline":
        """
        Creates a `Pipeline` object from the string representation read from the file-like
        object passed in the `fp` argument.

        :params data: The string representation of the pipeline, can be `str`, `bytes` or `bytearray`.
        :params fp: A file-like object ready to be read from.
        :params marshaller: the Marshaller used to create the string representation. Defaults to
                            `YamlMarshaller`

        :returns: A `Pipeline` object.
        """
        return cls.from_dict(marshaller.unmarshal(fp.read()))

    def _prepare_component_input_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Organizes input data for pipeline components and identifies any inputs that are not matched to any
        component's input slots.

        This method processes a flat dictionary of input data, where each key-value pair represents an input name
        and its corresponding value. It distributes these inputs to the appropriate pipeline components based on
        their input requirements. Inputs that don't match any component's input slots are classified as unresolved.

        :param data: A dictionary with input names as keys and input values as values.
        :type data: Dict[str, Any]
        :return: A tuple containing two elements:
             1. A dictionary mapping component names to their respective matched inputs.
             2. A dictionary of inputs that were not matched to any component, termed as unresolved keyword arguments.
        :rtype: Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]
        """
        pipeline_input_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        unresolved_kwargs = {}

        # Retrieve the input slots for each component in the pipeline
        available_inputs: Dict[str, Dict[str, Any]] = self.inputs()

        # Go through all provided to distribute them to the appropriate component inputs
        for input_name, input_value in data.items():
            resolved_at_least_once = False

            # Check each component to see if it has a slot for the current kwarg
            for component_name, component_inputs in available_inputs.items():
                if input_name in component_inputs:
                    # If a match is found, add the kwarg to the component's input data
                    pipeline_input_data[component_name][input_name] = input_value
                    resolved_at_least_once = True

            if not resolved_at_least_once:
                unresolved_kwargs[input_name] = input_value

        return pipeline_input_data, unresolved_kwargs
