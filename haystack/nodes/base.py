from __future__ import annotations
from typing import Any, Callable, Optional, Dict, List, Tuple, Optional

import io
from functools import wraps
from copy import deepcopy
from abc import abstractmethod
import inspect
import logging

from haystack.schema import Document, MultiLabel


logger = logging.getLogger(__name__)


class BaseComponent:
    """
    A base class for implementing nodes in a Pipeline.
    """
    outgoing_edges: int
    subclasses: dict = {}
    pipeline_config: dict = {}
    name: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """ 
        Automatically keeps track of all available subclasses.
        Enables generic load() for all specific component implementations.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, component_type: str):
        if component_type not in cls.subclasses.keys():
            raise Exception(f"Haystack component with the name '{component_type}' does not exist.")
        subclass = cls.subclasses[component_type]
        return subclass

    @classmethod
    def load_from_args(cls, component_type: str, **kwargs):
        """
        Load a component instance of the given type using the kwargs.
        
        :param component_type: name of the component class to load.
        :param kwargs: parameters to pass to the __init__() for the component. 
        """
        subclass = cls.get_subclass(component_type)
        instance = subclass(**kwargs)
        return instance

    @classmethod
    def load_from_pipeline_config(cls, pipeline_config: dict, component_name: str):
        """
        Load an individual component from a YAML config for Pipelines.

        :param pipeline_config: the Pipelines YAML config parsed as a dict.
        :param component_name: the name of the component to load.
        """
        if pipeline_config:
            all_component_configs = pipeline_config["components"]
            all_component_names = [comp["name"] for comp in all_component_configs]
            component_config = next(comp for comp in all_component_configs if comp["name"] == component_name)
            component_params = component_config["params"]

            for key, value in component_params.items():
                if value in all_component_names:  # check if the param value is a reference to another component
                    component_params[key] = cls.load_from_pipeline_config(pipeline_config, value)

            component_instance = cls.load_from_args(component_config["type"], **component_params)
        else:
            component_instance = cls.load_from_args(component_name)
        return component_instance

    @abstractmethod
    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None
    ) -> Tuple[Dict, str]:
        """
        Method that will be executed when the node in the graph is called.

        The argument that are passed can vary between different types of nodes
        (e.g. retriever nodes expect different args than a reader node)


        See an example for an implementation in haystack/reader/base/BaseReader.py
        :return:
        """
        pass

    def _dispatch_run(self, **kwargs) -> Tuple[Dict, str]:
        """
        The Pipelines call this method which in turn executes the run() method of Component.

        It takes care of the following:
          - inspect run() signature to validate if all necessary arguments are available
          - pop `debug` and sets them on the instance to control debug output
          - call run() with the corresponding arguments and gather output
          - collate `_debug` information if present
          - merge component output with the preceding output and pass it on to the subsequent Component in the Pipeline
        """
        arguments = deepcopy(kwargs)
        params = arguments.get("params") or {}

        run_signature_args = inspect.signature(self.run).parameters.keys()

        run_params: Dict[str, Any] = {}
        for key, value in params.items():
            if key == self.name:  # targeted params for this node
                if isinstance(value, dict):

                    # Extract debug attributes
                    if "debug" in value.keys():
                        self.debug = value.pop("debug")

                    for _k, _v in value.items():
                        if _k not in run_signature_args:
                            raise Exception(f"Invalid parameter '{_k}' for the node '{self.name}'.")

                run_params.update(**value)
            elif key in run_signature_args:  # global params
                run_params[key] = value

        run_inputs = {}
        for key, value in arguments.items():
            if key in run_signature_args:
                run_inputs[key] = value

        output, stream = self.run(**run_inputs, **run_params)

        # Collect debug information
        debug_info = {}
        if getattr(self, "debug", None):
            # Include input
            debug_info["input"] = {**run_inputs, **run_params}
            debug_info["input"]["debug"] = self.debug
            # Include output
            filtered_output = {key: value for key, value in output.items() if key != "_debug"}  # Exclude _debug to avoid recursion
            debug_info["output"] = filtered_output
        # Include custom debug info
        custom_debug = output.get("_debug", {})
        if custom_debug:
            debug_info["runtime"] = custom_debug

        # append _debug information from nodes
        all_debug = arguments.get("_debug", {})
        if debug_info:
            all_debug[self.name] = debug_info
        if all_debug:
            output["_debug"] = all_debug

        # add "extra" args that were not used by the node
        for k, v in arguments.items():
            if k not in output.keys():
                output[k] = v

        output["params"] = params
        return output, stream

    def set_config(self, **kwargs):
        """
        Save the init parameters of a component that later can be used with exporting
        YAML configuration of a Pipeline.

        :param kwargs: all parameters passed to the __init__() of the Component.
        """
        if not self.pipeline_config:
            self.pipeline_config = {"params": {}, "type": type(self).__name__}
            for k, v in kwargs.items():
                if isinstance(v, BaseComponent):
                    self.pipeline_config["params"][k] = v.pipeline_config
                elif v is not None:
                    self.pipeline_config["params"][k] = v
