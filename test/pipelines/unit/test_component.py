import pytest
from dataclasses import is_dataclass
from canals import component


class BaseTestComponent:
    """
    Base tests for components.
    """

    @pytest.fixture
    def components(self):
        """
        Provide here all the combinations of (component instance, data) to be tested by this suite.
        Should cover most relevant combinations to get a good coverage of your component's basic features.
        """
        pass

    @pytest.mark.unit
    def test_component_contract(self, components):
        class_ = components[0].__class__

        assert class_(component, "run")
        # TODO test run() signature

        if hasattr(class_, "Output"):
            assert is_dataclass(class_.Output)
        else:
            assert hasattr(class_, "output_type")

    @pytest.mark.unit
    def test_warm_up_works(self, components):
        for component in components:
            if hasattr(component, "warm_up"):
                component.warm_up()

    @pytest.mark.unit
    def test_run_output_matches_declared_output(self, components):
        pass

    @pytest.mark.unit
    def test_save_and_load_in_pipeline(self, components):
        pass
