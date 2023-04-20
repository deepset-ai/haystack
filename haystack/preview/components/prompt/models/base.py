from typing import Type, Optional, Dict, Any, List

import logging

from canals import _find_decorated_classes


logger = logging.getLogger(__name__)


class PromotModelError(Exception):
    pass


def prompt_model(class_):
    """
    Prompt model implementations execute a prompt on an underlying model.

    The implementation can be a simple invocation on the underlying model running in a local runtime, or
    could be even remote, for example, a call to a remote API endpoint.

    `def __init__(self, model_name_or_path: str, **kwargs)`

        Creates a new PromptModelInvocationLayer instance.

        :param model_name_or_path: The name or path of the underlying model.
        :param kwargs: Additional keyword arguments passed to the underlying model.

    `def invoke(self, *args, **kwargs)`

        It takes a prompt and returns a list of generated text using the underlying model.
        :return: A list of generated text.

    `def supports(cls, model_name_or_path: str, **kwargs) -> bool`

        Checks if the given model is supported by this invocation layer.

        :param model_name_or_path: The name or path of the model.
        :param kwargs: additional keyword arguments passed to the underlying model which might be used to determine
        if the model is supported.
        :return: True if this invocation layer supports the model, False otherwise.

    """
    logger.debug("Registering %s as a prompt model implementation", class_)

    # '__haystack_prompt_model__' is used to distinguish prompt model implementations from regular classes.
    # Its value is set to the desired implementation name: normally it is the class name, but it can technically be customized.
    class_.__haystack_prompt_model__ = class_.__name__

    # Check for invoke()
    if not hasattr(class_, "invoke"):
        # TODO check the node signature too
        raise PromotModelError(
            "Prompt model implementations must have an 'invoke()' method. See the docs for more information."
        )

    # Check for supports()
    if not hasattr(class_, "supports"):
        # TODO check the node signature too
        raise PromotModelError(
            "Prompt model implementations  must have a 'supports()' method. See the docs for more information."
        )

    return class_


def get_model(
    model_name_or_path: str,
    model_implementation: Optional[Type] = None,
    model_params: Optional[Dict[str, Any]] = None,
    modules_to_search: Optional[List[str]] = None,
) -> object:
    """
    Returns an instance of the proper model implementation for the given
    model name.

    :param model_name_or_path: the name or path of the model to load
    :param model_implementation: if set, either returns an instance of the given implementation or fails.
    :param model_kwargs: any kwargs that the model and/ore the implementation might require for initialization
    :returns: an instance of the model, ready to use. See `base.py`.
    """
    if not modules_to_search:
        modules_to_search = ["haystack.preview"]

    implementations = _find_decorated_classes(
        modules_to_search=modules_to_search, decorator="__haystack_prompt_model__"
    )
    if model_implementation:
        return model_implementation(model_name_or_path=model_name_or_path, **(model_params or {}))

    # search all implementations and find the first one that supports the model,
    # then create an instance of a mode with that implementation.
    for implementation in implementations.values():
        if implementation.supports(model_name_or_path, **(model_params or {})):
            return implementation(model_name_or_path=model_name_or_path, **(model_params or {}))

    raise ValueError(
        f"Model '{model_name_or_path}' is not supported - no matching implementations found. "
        f"Currently supported implementations are: {', '.join([imp for imp in implementations.keys()])} Make sure all "
        "the dependencies needed by your implementation are met, or enable DEBUG logs to check for failed imports. "
        f"You can implement a custom implementation for {model_name_or_path} by creating a class that respects "
        "the @prompt_model contract. See the documentation for details."
    )
