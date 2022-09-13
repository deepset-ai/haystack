import inspect
import functools
from typing import Any, Dict, Tuple, Callable


def args_to_kwargs(args: Tuple, func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    arg_names = list(sig.parameters.keys())
    # skip self and cls args for instance and class methods
    if any(arg_names) and arg_names[0] in ["self", "cls"]:
        arg_names = arg_names[1 : 1 + len(args)]
    args_as_kwargs = {arg_name: arg for arg, arg_name in zip(args, arg_names)}
    return args_as_kwargs


def pipeline_invocation_counter(func):
    @functools.wraps(func)
    def wrapper_invocation_counter(*args, **kwargs):
        # single query
        this_invocation_count = 1
        # were named arguments used?
        if "queries" in kwargs:
            this_invocation_count = len(kwargs["queries"]) if kwargs["queries"] else 1
        elif "documents" in kwargs:
            this_invocation_count = len(kwargs["documents"]) if kwargs["documents"] else 1
        else:
            # positional arguments used? try to infer count from the first parameter in args
            if args[0] and isinstance(args[0], list):
                this_invocation_count = len(args[0])

        wrapper_invocation_counter.counter += this_invocation_count
        return func(*args, **kwargs)

    wrapper_invocation_counter.counter = 0
    return wrapper_invocation_counter
