# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.generators.chat.types import ChatGenerator

# The provider-neutral generation parameters that Haystack components can request from Chat Generators.
# The chosen name is based on OpenAI's Responses API.
_HAYSTACK_GENERATION_PARAMETERS = frozenset({"max_output_tokens"})


def _convert_haystack_generation_kwargs(
    chat_generator: ChatGenerator, haystack_generation_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    Convert provider-neutral Haystack generation parameters for a Chat Generator.

    Chat Generators advertise supported parameters through a private class-level mapping from the canonical Haystack
    name to the provider-specific name. Parameters not advertised by the generator are omitted, allowing callers to
    provide a fallback for generators that do not expose this optional capability.

    :param chat_generator: The Chat Generator that will receive the converted parameters.
    :param haystack_generation_kwargs: Generation parameters using Haystack's canonical names.
    :returns: The supported parameters converted to their provider-specific names.
    :raises ValueError: If a parameter is not part of Haystack's canonical vocabulary.
    """
    unknown_parameters = haystack_generation_kwargs.keys() - _HAYSTACK_GENERATION_PARAMETERS
    if unknown_parameters:
        unknown = ", ".join(sorted(unknown_parameters))
        msg = f"Unknown Haystack generation parameter(s): {unknown}"
        raise ValueError(msg)

    parameter_mapping = getattr(chat_generator, "_HAYSTACK_TO_PROVIDER_GENERATION_KWARGS", {})

    return {
        provider_name: haystack_generation_kwargs[haystack_name]
        for haystack_name, provider_name in parameter_mapping.items()
        if haystack_name in haystack_generation_kwargs
    }
