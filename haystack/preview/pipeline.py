from collections import defaultdict
from typing import Any, Dict, Optional, Union, TextIO
from pathlib import Path
import datetime
import logging
import canals

from haystack.preview.telemetry import pipeline_running
from haystack.preview.marshal import Marshaller, YamlMarshaller


DEFAULT_MARSHALLER = YamlMarshaller()
logger = logging.getLogger(__name__)


class Pipeline(canals.Pipeline):
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
        Runs the pipeline

        :param data: the inputs to give to the input components of the Pipeline. data is a dictionary
        where each key is a component name and each value is a dictionary of input parameter of that component.
        :param debug: whether to collect and return debug information.
        :return: A dictionary with the outputs of the Pipeline.
        :raises PipelineRuntimeError: if any of the components fail or return unexpected output.

        Here is an example using a simple Hello component having an input 'word', a string, and output
        named 'output', also a string. The component just returns the input string with a greeting, i.e:

        ```python
        @component
        class Hello:
            @component.output_types(output=str)
            def run(self, word: str):
                return {"output": f"Hello, {word}!"}
        ```

        We can create a pipeline connecting two instances of this component together, and run it like so:

        ```python
        pipeline = Pipeline()
        pipeline.add_component("hello", Hello())
        pipeline.add_component("hello2", Hello())

        pipeline.connect("hello.output", "hello2.word")
        result = pipeline.run(data={"hello": {"word": "world"}})
        assert result == {'hello2': {'output': 'Hello, Hello, world!!'}}
        ```

        Notice how run method takes a data dictionary with the inputs for each component. The keys of the
        dictionary are the component names and the values are dictionaries with the input parameters of
        that component.

        We can also run the pipeline by passing the inputs directly to the run method, omitting component names,
        as in the following example:

        ```python
        result = pipeline.run(data={"word": "world"})
        assert result == {"hello2": {"output": "Hello, Hello, world!!"}}
        ```
        In this case, the pipeline will try to resolve the input parameters to the appropriate components and
        if successful, it will run the pipeline and return the outputs.
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
            pipeline_inputs, unresolved_inputs = self._prepare_pipeline_component_input_data(data)
            if unresolved_inputs:
                logger.warning(
                    "Inputs %s were not matched to any component inputs, please check your " "pipeline run parameters.",
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

    def _prepare_pipeline_component_input_data(self, data: Dict[str, Any]):
        """
        Prepares the input data for the pipeline components and identifies any unresolved parameters.

        This method organizes the inputs for each component in the pipeline based on the provided
        data arguments. It also keeps track of any inputs that do not correspond to input slots of
        any component.

        :param data: flat dict of inputs where the keys are the input names and the values are the input values.
        :return: A tuple containing the organized input data for the pipeline components (as a dictionary) and
                 a dictionary of any unresolved keyword arguments.
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
