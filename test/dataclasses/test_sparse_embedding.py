# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.dataclasses.sparse_embedding import SparseEmbedding


class TestSparseEmbedding:
    def test_init(self):
        se = SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3])
        assert se.indices == [0, 2, 4]
        assert se.values == [0.1, 0.2, 0.3]

    def test_init_with_wrong_parameters(self):
        with pytest.raises(ValueError):
            SparseEmbedding(indices=[0, 2], values=[0.1, 0.2, 0.3, 0.4])

    def test_to_dict(self):
        se = SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3])
        assert se.to_dict() == {"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]}

    def test_from_dict(self):
        se = SparseEmbedding.from_dict({"indices": [0, 2, 4], "values": [0.1, 0.2, 0.3]})
        assert se.indices == [0, 2, 4]
        assert se.values == [0.1, 0.2, 0.3]

    def test_eq(self):
        se1 = SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3])
        se2 = SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.3])
        assert se1 == se2

        se3 = SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.2, 0.4])
        assert se1 != se3
