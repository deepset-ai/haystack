# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List


class SparseEmbedding:
    """
    Class representing a sparse embedding.
    """

    def __init__(self, indices: List[int], values: List[float]):
        """
        Initialize a SparseEmbedding object.

        :param indices: List of indices of non-zero elements in the embedding.
        :param values: List of values of non-zero elements in the embedding.

        :raises ValueError: If the indices and values lists are not of the same length.
        """
        if len(indices) != len(values):
            raise ValueError("Length of indices and values must be the same.")
        self.indices = indices
        self.values = values

    def __eq__(self, other):
        return self.indices == other.indices and self.values == other.values

    def to_dict(self):
        """
        Convert the SparseEmbedding object to a dictionary.

        :returns:
            Serialized sparse embedding.
        """
        return {"indices": self.indices, "values": self.values}

    @classmethod
    def from_dict(cls, sparse_embedding_dict):
        """
        Deserializes the sparse embedding from a dictionary.

        :param sparse_embedding_dict:
            Dictionary to deserialize from.
        :returns:
            Deserialized sparse embedding.
        """
        return cls(indices=sparse_embedding_dict["indices"], values=sparse_embedding_dict["values"])
