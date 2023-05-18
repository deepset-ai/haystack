# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.pipeline import Pipeline, save_pipelines, load_pipelines

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
        save_pipelines({"pipe": pipe}, tmp_path / "test_pipe.json")

        loaded_pipes = load_pipelines(tmp_path / "test_pipe.json")
        new_pipe = loaded_pipes["pipe"]

        assert pipe == new_pipe
