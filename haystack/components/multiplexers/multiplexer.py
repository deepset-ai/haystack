import sys
import logging
from typing import Any, Dict

from haystack.core.component.types import Variadic
from haystack import component, default_to_dict, default_from_dict
from haystack.components.routers.conditional_router import serialize_type, deserialize_type

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias


logger = logging.getLogger(__name__)


@component
class Multiplexer:
    """
    This component is used to distribute a single value to many components that may need it.
    It can take such value from different sources (the user's input, or another component), so
    it's only input is of Variadic type.

    The type of the expected input (and therefore of the output as well) must be given at init time.

    Example usage:

    ```python
    >>> mp = Multiplexer(str)
    >>> mp.run(value=["hello"])
    {"value" : "hello"}

    >>> mp = Multiplexer(int)
    >>> mp.run(value=[3])
    {"value": 3}
    ```

    This component won't handle several inputs at the same time: it always only expects one.
    If more than one input is received when run is invoked, it will raise a ValueError.

    ```python
    >>> mp = Multiplexer(int)
    >>> mp.run([2, 4])
    ValueError: Multiplexer expects only one input, but 2 were received.

    >>> mp = Multiplexer(int)
    >>> mp.run([2, None])
    ValueError: Multiplexer expects only one input, but 2 were received.
    ```
    """

    def __init__(self, type_: TypeAlias):
        self.type_ = type_
        component.set_input_types(self, value=Variadic[type_])
        component.set_output_types(self, value=type_)

    def to_dict(self):
        return default_to_dict(self, type_=serialize_type(self.type_))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Multiplexer":
        data["init_parameters"]["type_"] = deserialize_type(data["init_parameters"]["type_"])
        return default_from_dict(cls, data)

    def run(self, **kwargs):
        if (inputs_count := len(kwargs["value"])) != 1:
            raise ValueError(f"Multiplexer expects only one input, but {inputs_count} were received.")
        return {"value": kwargs["value"][0]}
