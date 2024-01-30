from typing import Dict, Union

from ..type_utils import _type_name
from .component import Component
from .sockets import InputSocket, OutputSocket


class InputOutput:
    """
    This class is used to represent the inputs or outputs of a `Component`.
    Depending on the type of sockets passed to the constructor, it will represent either the inputs or the outputs of
    the `Component`.

    Usage:
    ```python
    from haystack.components.builders.prompt_builder import PromptBuilder

    prompt_template = \"""
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    \"""

    prompt_builder = PromptBuilder(template=prompt_template)

    inputs = InputOutput(component=prompt_builder, sockets=prompt_builder.__haystack_input__)
    inputs
    >>> PromptBuilder inputs:
    >>>   - question: Any
    >>>   - documents: Any

    inputs.question
    >>> PromptBuilder.inputs.question
    ```

    The behaviour changes a bit if the `Component` has been added to a `Pipeline`.
    In that case we get the name of the `Component` from the `Pipeline`. These ease creating connections with `Pipeline.connect()`.

    ```
    from haystack.core.pipeline import Pipeline

    pipeline = Pipeline()
    pipeline.add_component("my_prompt_builder", prompt_builder)

    inputs.question
    >>> my_prompt_builder.inputs.question
    ```
    """

    def __init__(self, component: Component, sockets: Union[Dict[str, InputSocket], Dict[str, OutputSocket]]):
        if sockets is None:
            msg = f"InputOutput must be initialized with sockets. Got {sockets}"
            raise ValueError(msg)

        socket_type = {type(s) for s in sockets.values()}
        if len(socket_type) > 1:
            msg = f"InputOutput must be initialized with sockets of the same type. Got {socket_type}"
            raise ValueError(msg)

        self._sockets_type = socket_type.pop() if len(socket_type) == 1 else None
        if self._sockets_type not in (InputSocket, OutputSocket, None):
            msg = f"InputOutput must be initialized with InputSocket or OutputSocket. Got {self._sockets_type}"
            raise ValueError(msg)

        if not isinstance(component, Component):
            msg = f"InputOutput must be initialized with a component. Got {component}"
            raise ValueError(msg)

        self._component = component
        self._sockets = sockets
        self.__dict__.update(sockets)

    def _component_name(self) -> str:
        if pipeline := getattr(self._component, "__haystack_added_to_pipeline__"):
            # This Component has been added in a Pipeline, let's get the name from there.
            return pipeline.get_component_name(self._component)

        # This Component has not been added to a Pipeline yet, so we can't know its name.
        # Let's use the class name instead.
        return self._component.__class__.__name__

    def _socket_repr(self, socket_name: str) -> str:
        parts = [self._component_name()]

        if self._sockets_type == InputSocket:
            parts.append("inputs")
        elif self._sockets_type == OutputSocket:
            parts.append("outputs")

        parts.append(socket_name)

        return ".".join(parts)

    def __getattribute__(self, name):
        try:
            sockets = object.__getattribute__(self, "_sockets")
            if name in sockets:
                return object.__getattribute__(self, "_socket_repr")(sockets[name].name)
        except AttributeError:
            pass

        return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        result = self._component_name()
        if self._sockets_type == InputSocket:
            result += " inputs:\n"
        elif self._sockets_type == OutputSocket:
            result += " outputs:\n"

        result += "\n".join([f"  - {n}: {_type_name(s.type)}" for n, s in self._sockets.items()])

        return result
