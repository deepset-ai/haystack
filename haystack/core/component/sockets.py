# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Type, Union

from haystack import logging
from haystack.core.type_utils import _type_name

from .types import InputSocket, OutputSocket

logger = logging.getLogger(__name__)

SocketsDict = Dict[str, Union[InputSocket, OutputSocket]]
SocketsIOType = Union[Type[InputSocket], Type[OutputSocket]]


class Sockets:
    """
    Represents the inputs or outputs of a `Component`.

    Depending on the type passed to the constructor, it will represent either the inputs or the outputs of
    the `Component`.

    Usage:
    ```python
    from typing import Any
    from haystack.components.builders.prompt_builder import PromptBuilder
    from haystack.core.component.sockets import Sockets
    from haystack.core.component.types import InputSocket, OutputSocket


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
    inputs = Sockets(component=prompt_builder, sockets_dict=sockets, sockets_io_type=InputSocket)
    inputs
    >>> Inputs:
    >>>   - question: Any
    >>>   - documents: Any

    inputs.question
    >>> InputSocket(name='question', type=typing.Any, default_value=<class 'haystack.core.component.types._empty'>, ...
    ```
    """

    # We're using a forward declaration here to avoid a circular import.
    def __init__(
        self,
        component: "Component",  # type: ignore[name-defined] # noqa: F821
        sockets_dict: SocketsDict,
        sockets_io_type: SocketsIOType,
    ):
        """
        Create a new Sockets object.

        We don't do any enforcement on the types of the sockets here, the `sockets_type` is only used for
        the `__repr__` method.
        We could do without it and use the type of a random value in the `sockets` dict, but that wouldn't
        work for components that have no sockets at all. Either input or output.

        :param component:
            The component that these sockets belong to.
        :param sockets_dict:
            A dictionary of sockets.
        :param sockets_io_type:
            The type of the sockets.
        """
        self._sockets_io_type = sockets_io_type
        self._component = component
        self._sockets_dict = sockets_dict
        self.__dict__.update(sockets_dict)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Sockets):
            return False

        return (
            self._sockets_io_type == value._sockets_io_type
            and self._component == value._component
            and self._sockets_dict == value._sockets_dict
        )

    def __setitem__(self, key: str, socket: Union[InputSocket, OutputSocket]):
        """
        Adds a new socket to this Sockets object.

        This eases a bit updating the list of sockets after Sockets has been created.
        That should happen only in the `component` decorator.
        """
        self._sockets_dict[key] = socket
        self.__dict__[key] = socket

    def __contains__(self, key: str) -> bool:
        return key in self._sockets_dict

    def get(
        self, key: str, default: Optional[Union[InputSocket, OutputSocket]] = None
    ) -> Optional[Union[InputSocket, OutputSocket]]:
        """
        Get a socket from the Sockets object.

        :param key:
            The name of the socket to get.
        :param default:
            The value to return if the key is not found.
        :returns:
            The socket with the given key or `default` if the key is not found.
        """
        return self._sockets_dict.get(key, default)

    def _component_name(self) -> str:
        if pipeline := getattr(self._component, "__haystack_added_to_pipeline__"):
            # This Component has been added in a Pipeline, let's get the name from there.
            return pipeline.get_component_name(self._component)

        # This Component has not been added to a Pipeline yet, so we can't know its name.
        # Let's use default __repr__. We don't call repr() directly as Components have a custom
        # __repr__ method and that would lead to infinite recursion since we call Sockets.__repr__ in it.
        return object.__repr__(self._component)

    def __getattribute__(self, name):
        try:
            sockets = object.__getattribute__(self, "_sockets")
            if name in sockets:
                return sockets[name]
        except AttributeError:
            pass

        return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        result = ""
        if self._sockets_io_type == InputSocket:
            result = "Inputs:\n"
        elif self._sockets_io_type == OutputSocket:
            result = "Outputs:\n"

        return result + "\n".join([f"  - {n}: {_type_name(s.type)}" for n, s in self._sockets_dict.items()])
