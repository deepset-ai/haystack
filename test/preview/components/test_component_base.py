from typing import List, Tuple, Dict, Any

from abc import ABC, abstractmethod

from unittest.mock import MagicMock

import pytest


class BaseTestComponent(ABC):
    """
    Base tests for Haystack components.
    """

    @pytest.fixture
    def request_mock(self, monkeypatch):
        request_mock = MagicMock()
        monkeypatch.setattr("requests.request", MagicMock())
        return request_mock

    @pytest.fixture
    @abstractmethod
    def components(self) -> List[Tuple[object, Dict[str, Dict[str, Any]]]]:
        """
        Provide here all the combinations of (component instance, {data and parameters})
        to be tested by this suite. Should cover most relevant combinations to get a good coverage of your component's
        basic features.
        """
        pass

    @pytest.mark.unit
    def test_component_contract(self, components):
        for component, _ in components:
            assert hasattr(component, "inputs")
            assert hasattr(component, "outputs")
            assert hasattr(component, "init_parameters")
            assert hasattr(component, "run")
            # TODO test run() signature

    @pytest.mark.unit
    def test_warm_up_works(self, components):
        for component in components:
            if hasattr(component, "warm_up"):
                component.warm_up()

    #
    # TODO
    #

    @pytest.mark.unit
    def test_run_output_matches_declared_output(self, components):
        pass

    @pytest.mark.unit
    def test_save_and_load_in_pipeline(self, components):
        pass

    @pytest.mark.unit
    def test_api_calls_handle_exceptions(self, request_mock, components):
        # TODO
        pass

    @pytest.mark.unit
    def test_api_calls_attempt_retries(self, request_mock, components):
        for component, run_params in components:
            component.run("test_component", **run_params)
            assert request_mock.call_count > 1
