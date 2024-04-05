from typing import List


class SparseEmbedding:
    """
    Class representing a sparse embedding.
    """

    def __init__(self, indices: List[int], values: List[float]):
        """
        :param indices: List of indices of non-zero elements in the embedding.
        :param values: List of values of non-zero elements in the embedding.

        :raises ValueError: If the indices and values lists are not of the same length.
        """
        if len(indices) != len(values):
            raise ValueError("Length of indices and values must be the same.")
        self.indices = indices
        self.values = values

    def to_dict(self):
        return {"indices": self.indices, "values": self.values}

    @classmethod
    def from_dict(cls, sparse_embedding_dict):
        return cls(indices=sparse_embedding_dict["indices"], values=sparse_embedding_dict["values"])
