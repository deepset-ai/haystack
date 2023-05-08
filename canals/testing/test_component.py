from dataclasses import is_dataclass

import pytest

from canals import Pipeline, save_pipelines, load_pipelines, marshal_pipelines

#
# TODO right now the tests iterate on components, so it's not always clear which component fails the tests.
# Can we instead parametrize the tests on the output of the components fixture?
#
class BaseTestComponent:
    """
    Base tests for components.
    """

    @pytest.fixture
    def components(self):
        """
        Provide here a list of differnetly instantiated components to be tested by this suite.
        Should cover most relevant combinations to get a good coverage of your component's basic features.
        """

    def test_component_contract(self, components):
        """
        Very similar checks are done by the decorator itself,
        but for complex components we might mess with these things after instantiation.
        Let's double check it's all good on this front.
        """
        for component in components:
            assert hasattr(component, "__canals_component__")
            assert hasattr(component, "run")
            if hasattr(component, "Output"):
                assert is_dataclass(component.Output)
            else:
                assert hasattr(component, "output_type")

    def test_save_and_load_in_pipeline(self, tmp_path, components):
        """
        Verifies whether the components can be properly serialized and de-serialized.
        """
        for component in components:
            pipe = Pipeline()
            pipe.add_component("component", component)
            save_pipelines({"pipe": pipe}, tmp_path / "test_pipe.json")

            print(marshal_pipelines({"pipe": pipe}))

            loaded_pipes = load_pipelines(tmp_path / "test_pipe.json")
            new_pipe = loaded_pipes["pipe"]

            assert pipe == new_pipe
