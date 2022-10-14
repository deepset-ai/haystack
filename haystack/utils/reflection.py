import inspect
import functools
import time
from random import random
from typing import Any, Dict, Tuple, Callable

from haystack.errors import OpenAIRateLimitError


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


def retry_with_exponential_backoff(
    function,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (OpenAIRateLimitError,),
):
    """Retry a function with exponential backoff."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator
