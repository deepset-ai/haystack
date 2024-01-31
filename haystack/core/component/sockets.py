# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Type, Union

from haystack.core.type_utils import _type_name

from .types import InputSocket, OutputSocket

logger = logging.getLogger(__name__)

SocketsType = Union[Type[InputSocket], Type[OutputSocket]]


class Sockets:
    """
    This class is used to represent the inputs or outputs of a `Component`.
    Depending on the type passed to the constructor, it will represent either the inputs or the outputs of
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
    sockets = {"question": InputSocket("question", Any), "documents": InputSocket("documents", Any)}
    inputs = Sockets(component=prompt_builder, sockets=sockets, sockets_type=InputSocket)
    inputs
    >>> PromptBuilder inputs:
    >>>   - question: Any
    >>>   - documents: Any

    inputs.question
    >>> InputSocket(name='question', type=typing.Any, default_value=<class 'haystack.core.component.types._empty'>, is_variadic=False, senders=[])
    ```
    """

    # We're using a forward declaration here to avoid a circular import.
    def __init__(
        self, component: "Component", sockets: Dict[str, SocketsType], sockets_type: SocketsType  # noqa: F821
    ):
        """
        Create a new Sockets object.
        We don't do any enforcement on the types of the sockets here, the `sockets_type` is only used for
        the `__repr__` method.
        We could do without it and use the type of a random value in the `sockets` dict, but that wouldn't
        work for components that have no sockets at all. Either input or output.
        """
        self._sockets_type = sockets_type
        self._component = component
        self._sockets = sockets
        self.__dict__.update(sockets)

    def __setitem__(self, key: str, socket: SocketsType):
        """
        Adds a new socket to this Sockets object.
        This eases a bit updating the list of sockets after Sockets has been created.
        That should happen only in the `component` decorator.
        """
        self._sockets[key] = socket
        self.__dict__[key] = socket

    def _component_name(self) -> str:
        if pipeline := getattr(self._component, "__haystack_added_to_pipeline__"):
            # This Component has been added in a Pipeline, let's get the name from there.
            return pipeline.get_component_name(self._component)

        # This Component has not been added to a Pipeline yet, so we can't know its name.
        # Let's use the class name instead.
        return self._component.__class__.__name__

    def __getattribute__(self, name):
        try:
            sockets = object.__getattribute__(self, "_sockets")
            if name in sockets:
                return sockets[name]
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
