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


class InMemoryLogger(io.TextIOBase):
    """
    Implementation of a logger that keeps track
    of the log lines in a list called `logs`,
    from where they can be accessed freely.
    """
    def __init__(self, *args):
        io.TextIOBase.__init__(self, *args)
        self.logs = []

    def write(self, x):
        self.logs.append(x)


def record_debug_logs(func: Callable, node_name: str, logs: bool) -> Callable:
    """
    Captures the debug logs of the wrapped function and
    saves them in the `_debug` key of the output dictionary.
    If `logs` is True, dumps the same logs to the console as well.

    Used in `BaseComponent.__getattribute__()` to wrap `run()` functions.
    This makes sure that every implementation of `run()` by a subclass will
    be automagically decorated with this method when requested.

    :param func: the function to decorate (must be an implementation of
                 `BaseComponent.run()`).
    :param logs: whether the captured logs should also be displayed
                 in the console during the execution of the pipeline.
    """
    @wraps(func)
    def inner(*args, **kwargs) -> Tuple[Dict[str, Any], str]:

        with InMemoryLogger() as logs_container:
            logger = logging.getLogger()

            # Adds a handler that stores the logs in a variable
            handler = logging.StreamHandler(logs_container)
            handler.setLevel(logger.level or logging.DEBUG)
            logger.addHandler(handler)

            # Add a handler that prints log messages in the console
            # to the specified level for the node
            if logs:
                handler_console = logging.StreamHandler()
                handler_console.setLevel(logging.DEBUG)
                formatter = logging.Formatter(f'[{node_name} logs] %(message)s')
                handler_console.setFormatter(formatter)
                logger.addHandler(handler_console)

            output, stream = func(*args, **kwargs)

            if not "_debug" in output.keys():
                output["_debug"] = {}
            output["_debug"]["logs"] = logs_container.logs

            # Remove both handlers
            logger.removeHandler(handler)
            if logs:
                logger.removeHandler(handler_console)

            return output, stream

    return inner


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

    def __getattribute__(self, name):
        """
        This modified `__getattribute__` method automagically decorates
        every `BaseComponent.run()` implementation with the
        `record_debug_logs` decorator defined above.

        This decorator makes the function collect its debug logs into a
        `_debug` key of the output dictionary.

        The logs collection is not always performed. Before applying the decorator,
        it checks for an instance attribute called `debug` to know
        whether it should or not. The decorator is applied if the attribute is
        defined and True.

        In addition, the value of the instance attribute `debug_logs` is
        passed to the decorator. If it's True, it will print the
        logs in the console as well.
        """
        if name == "run" and self.debug:
            func = getattr(type(self), "run")
            return record_debug_logs(func=func, node_name=self.__class__.__name__, logs=self.debug_logs).__get__(self)
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        """
        Ensures that `debug` and `debug_logs` are always defined.
        """
        if name in ["debug", "debug_logs"]:
            return None
        raise AttributeError(name)

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
          - pop `debug` and `debug_logs` and sets them on the instance to control debug output
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
                    if "debug_logs" in value.keys():
                        self.debug_logs = value.pop("debug_logs")

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
        current_debug = output.get("_debug", {})
        if self.debug:
            current_debug["input"] = {**run_inputs, **run_params}
            if self.debug:
                current_debug["input"]["debug"] = self.debug
            if self.debug_logs:
                current_debug["input"]["debug_logs"] = self.debug_logs
            filtered_output = {key: value for key, value in output.items() if key != "_debug"} # Exclude _debug to avoid recursion
            current_debug["output"] = filtered_output

        # append _debug information from nodes
        all_debug = arguments.get("_debug", {})
        if current_debug:
            all_debug[self.name] = current_debug
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
