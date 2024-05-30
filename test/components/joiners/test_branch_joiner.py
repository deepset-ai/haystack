# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.components.joiners import BranchJoiner


class TestBranchJoiner:
    def test_one_value(self):
        joiner = BranchJoiner(int)
        output = joiner.run(value=[2])
        assert output == {"value": 2}

    def test_one_value_of_wrong_type(self):
        # BranchJoiner does not type check the input
        joiner = BranchJoiner(int)
        output = joiner.run(value=["hello"])
        assert output == {"value": "hello"}

    def test_one_value_of_none_type(self):
        # BranchJoiner does not type check the input
        joiner = BranchJoiner(int)
        output = joiner.run(value=[None])
        assert output == {"value": None}

    def test_more_values_of_expected_type(self):
        joiner = BranchJoiner(int)
        with pytest.raises(ValueError, match="BranchJoiner expects only one input, but 3 were received."):
            joiner.run(value=[2, 3, 4])

    def test_no_values(self):
        joiner = BranchJoiner(int)
        with pytest.raises(ValueError, match="BranchJoiner expects only one input, but 0 were received."):
            joiner.run(value=[])
