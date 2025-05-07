# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Protocol, TypeVar

T = TypeVar("T", bound="TextEmbedder")


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
            A dictionary containing the embedding and metadata.
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the component to a dictionary representation.

        :returns:
            A dictionary representation of the component.
        """
        ...

    def from_dict(self, data: Dict[str, Any]) -> T:
        """
        Create a component instance from a dictionary representation.

        :param data:
            A dictionary representation of the component.
        :returns:
            An instance of the component.
        """
        ...
