# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class SparseEmbedding:
    """
    Class representing a sparse embedding.

    :param indices: List of indices of non-zero elements in the embedding.
    :param values: List of values of non-zero elements in the embedding.
    """

    indices: List[int]
    values: List[float]

    def __post_init__(self):
        """
        Checks if the indices and values lists are of the same length.

        Raises a ValueError if they are not.
        """
        if len(self.indices) != len(self.values):
            raise ValueError("Length of indices and values must be the same.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the SparseEmbedding object to a dictionary.

        :returns:
            Serialized sparse embedding.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, sparse_embedding_dict: Dict[str, Any]) -> "SparseEmbedding":
        """
        Deserializes the sparse embedding from a dictionary.

        :param sparse_embedding_dict:
            Dictionary to deserialize from.
        :returns:
            Deserialized sparse embedding.
        """
        return cls(**sparse_embedding_dict)
