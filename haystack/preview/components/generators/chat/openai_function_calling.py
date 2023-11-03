import inspect
import json
from typing import Callable, Any, List, Dict, Type

import docstring_parser

from haystack.preview import component
from haystack.preview.dataclasses import ChatMessage


class DocumentationError(Exception):
    """Exception raised for errors in the pydoc documentation."""

    def __init__(self, class_name: str, function_name: str, message: str = "Invalid pydoc documentation"):
        detailed_message = f"{class_name}: {function_name} - {message}"
        super().__init__(detailed_message)


@component
class FunctionCallingServiceContainer:
    """
    A container for enabling OpenAI function calling within Haystack 2.x.

    This container acts as a seamless bridge between the Haystack framework and an arbitrary
    service instance, allowing for dynamic method invocation based on incoming messages.

    :param service_instance: An instance of the service which methods should be invoked.
    """

    def __init__(self, service_instance: Any):
        """
        Initialize the container with the given service instance.
        :param service_instance: An instance of the service which methods should be invoked.
        """
        self.service_instance = service_instance

    @component.output_types(messages=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        """
        Processes the last message in the list, invoking a method on the service instance
        based on the message content, and returning the result.

        The method to be invoked, along with its arguments, is expected to be specified
        within the message content in a JSON format.

        :param messages: A list of ChatMessage instances to be processed.
        :return: A dictionary containing the original list of messages, appended with a
                 new ChatMessage instance representing the result of the method invocation.
        """
        # last message is the function call
        message = messages[-1]
        payload = json.loads(message.content)
        function_name = payload["name"]
        function_args = payload["arguments"]

        # reflectively call the function
        method = getattr(self.service_instance, function_name, None)
        if method is not None and callable(method):
            # prepare arguments for invocation
            invocation_args = dict(function_args) if function_args else {}
            try:
                invocation_result = method(**invocation_args)
            except Exception as e:
                raise RuntimeError(
                    f"{self.__name__} received invocation for {function_name} but the function invocation "
                    f"raised an exception: {e}"
                ) from e
            return {"messages": messages + [ChatMessage.from_function(str(invocation_result), function_name)]}
        else:
            raise RuntimeError(
                f"{self.__name__} received invocation for {function_name} but that function "
                f"is not in {self.service_instance.__class__.__name__}"
            )

    def generate_openai_function_descriptions(self) -> List[Dict[str, Any]]:
        """
        Generates a list of OpenAI-compatible JSON descriptions for the public methods
        of the service instance class.

        :return: A list of dictionaries, each representing a JSON description of a method.
        """
        clazz = type(self.service_instance)
        descriptions = []
        # iterate over all functions of the class
        for name, func in inspect.getmembers(clazz, predicate=inspect.isfunction):
            # only include public functions
            if not name.startswith("_"):
                json_repr = self._generate_description(func)
                descriptions.append(json_repr)
        return descriptions

    def _generate_description(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """
        Generates a JSON description for a specified method according to the OpenAI
        functional API specification.

        :param func: The method for which to generate a description.
        :return: A dictionary representing the JSON description of the method.
        """
        docstring = inspect.getdoc(func)
        if docstring is None:
            raise DocumentationError(
                type(self.service_instance),
                func.__name__,
                f" has no pydocs documentation. " f"Please add pydoc comments to {func.__name__}.",
            )

        try:
            parsed_docstring = docstring_parser.parse(docstring)
        except RuntimeError as e:
            raise DocumentationError(
                type(self.service_instance), func.__name__, " has pydocs documentation that cannot be parsed."
            ) from e

        function_description = parsed_docstring.short_description
        param_descriptions = {param.arg_name: param.description for param in parsed_docstring.params}

        return {
            "name": func.__name__,
            "description": function_description,
            "parameters": {
                "type": "object",
                "properties": self._generate_args_description(func, param_descriptions),
                "required": self._get_required_args(func),
            },
        }

    def _get_arg_properties(
        self, func_name: str, arg_name: str, arg_type: Type, arg_descriptions: dict
    ) -> Dict[str, Any]:
        """
        Generate JSON Schema properties for a single function argument.

        :param arg_name: The name of the argument.
        :param arg_type: The type of the argument.
        :param arg_descriptions: A dictionary of argument descriptions.
        :return: A dictionary representing the JSON Schema properties of the argument.
        """
        simple_object_mapping = {"str": "string", "int": "number", "float": "number", "bool": "boolean"}
        if arg_type.__name__ not in simple_object_mapping:
            raise NotImplementedError(
                f"Only primitive function parameters are supported at the moment. "
                f"Function {func_name} has an argument {arg_name} whose type is {arg_type.__name__}."
            )
        return {"type": simple_object_mapping[arg_type.__name__], "description": arg_descriptions[arg_name]}

    def _generate_args_description(self, func: Callable[..., Any], arg_descriptions: dict) -> Dict[str, Any]:
        """
        Generate a JSON Schema description for all arguments of a function.

        :param func: The function whose arguments are to be described.
        :param arg_descriptions: A dictionary of argument descriptions.
        :return: A dictionary representing the JSON Schema description of the arguments.
        """
        # Get the annotations from the function, excluding the return annotation.
        func_spec: inspect.FullArgSpec = inspect.getfullargspec(func)
        args = [arg for arg in func_spec.args if arg not in ["self"]]

        annotations = inspect.getfullargspec(func).annotations
        annotations.pop("return", None)

        if len(args) != len(annotations):
            raise DocumentationError(
                type(self.service_instance),
                func.__name__,
                f" has a different number of arguments ({len(args)}) than pydoc params ({len(annotations)}). "
                f"Please check the pydoc comments for {func.__name__}.",
            )

        return {
            arg: self._get_arg_properties(func.__name__, arg, arg_type, arg_descriptions)
            for arg, arg_type in annotations.items()
        }

    def _get_required_args(self, func: Callable[..., Any]) -> List[str]:
        """
        Get the names of the required arguments for a Python function.

        :param func: The function whose required arguments are to be retrieved.
        :return: A list of strings, each representing the name of a required argument.
        """
        spec = inspect.getfullargspec(func)
        required_args = [arg for arg in spec.args if arg not in ["self"]]

        # Check if the function has default values for some arguments
        if spec.defaults:
            args_with_defaults = spec.args[-len(spec.defaults) :]
            required_args = [arg for arg in required_args if arg not in args_with_defaults]

        # handle keyword-only arguments
        for kwonlyarg in spec.kwonlyargs:
            if spec.kwonlydefaults is None or kwonlyarg not in spec.kwonlydefaults:
                required_args.append(kwonlyarg)

        return required_args
