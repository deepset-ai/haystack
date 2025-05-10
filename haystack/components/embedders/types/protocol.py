# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Protocol, TypeVar

T = TypeVar("T", bound="TextEmbedder")

# See https://github.com/pylint-dev/pylint/issues/9319.
# pylint: disable=unnecessary-ellipsis


class TextEmbedder(Protocol):
    """
    Protocol for Text Embedders.
    """

    def run(self, text: str) -> Dict[str, Any]:
        """
        Generate embeddings for the input text.

        Implementing classes may accept additional optional parameters in their run method.
        For example: `def run (self, text: str, param_a="default", param_b="another_default")`.

        :param text:
            The input text to be embedded.
        :returns:
            A dictionary containing the keys:
                - 'embedding', which is expected to be a List[float] representing the embedding.
                - any optional keys such as 'metadata'.
        """
        ...
