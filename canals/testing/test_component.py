# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json

from canals.pipeline import Pipeline


#
# TODO right now the tests iterate on components, so it's not always clear which component fails the tests.
# Can we instead parametrize the tests on the output of the components fixture?
#
class BaseTestComponent:  # pylint: disable=too-few-public-methods
    """
    Base tests for components.
    """

    def assert_can_be_saved_and_loaded_in_pipeline(self, component, tmp_path):
        """
        Verifies whether the components can be properly serialized and de-serialized.
        """
        pipe = Pipeline()
        pipe.add_component("component", component)
        data = pipe.to_dict()
        test_file = tmp_path / "test_pipe.json"
        test_file.write_text(json.dumps(data))
        data = json.loads(test_file.read_text())
        assert pipe == Pipeline.from_dict(data)
