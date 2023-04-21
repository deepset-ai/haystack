from abc import ABC, abstractmethod

import pytest


class _BaseTestComponent(ABC):
    @pytest.fixture
    @abstractmethod
    def components(self):
        pass

    def test_component_contract(self, components):
        for component in components:
            assert hasattr(component, "inputs")
            assert hasattr(component, "outputs")
            assert hasattr(component, "init_parameters")
            assert hasattr(component, "run")

    def test_save_and_load(self, components):
        # TODO
        pass
