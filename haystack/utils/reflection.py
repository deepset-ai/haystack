import inspect
import logging
from typing import Any, Dict, Tuple, Callable

logger = logging.getLogger(__name__)


def args_to_kwargs(args: Tuple, func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    arg_names = list(sig.parameters.keys())
    # skip self and cls args for instance and class methods
    if any(arg_names) and arg_names[0] in ["self", "cls"]:
        arg_names = arg_names[1 : 1 + len(args)]
    args_as_kwargs = {arg_name: arg for arg, arg_name in zip(args, arg_names)}
    return args_as_kwargs
