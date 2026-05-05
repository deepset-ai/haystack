# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import component
from haystack.utils.experimental import ExperimentalWarning, _experimental


class TestExperimentalDecorator:
    def test_emits_experimental_warning_on_init(self):
        @_experimental
        @component
        class MyComponent:
            @component.output_types(value=int)
            def run(self, value: int) -> dict:
                return {"value": value}

        with pytest.warns(ExperimentalWarning):
            MyComponent()

    def test_warning_message_contains_class_name(self):
        @_experimental
        @component
        class MyComponent:
            @component.output_types(value=int)
            def run(self, value: int) -> dict:
                return {"value": value}

        with pytest.warns(ExperimentalWarning, match="MyComponent"):
            MyComponent()

    def test_sets_experimental_attribute(self):
        @_experimental
        @component
        class MyComponent:
            @component.output_types(value=int)
            def run(self, value: int) -> dict:
                return {"value": value}

        assert MyComponent.__experimental__ is True

    def test_passes_args_and_kwargs_to_init(self):
        @_experimental
        @component
        class MyComponent:
            def __init__(self, value: int, label: str = "default"):
                self.value = value
                self.label = label

            @component.output_types(value=int)
            def run(self, value: int) -> dict:
                return {"value": value}

        with pytest.warns(ExperimentalWarning):
            instance = MyComponent(42, label="custom")

        assert instance.value == 42
        assert instance.label == "custom"

    def test_preserves_init_name(self):
        @_experimental
        @component
        class MyComponent:
            @component.output_types(value=int)
            def run(self, value: int) -> dict:
                return {"value": value}

        assert MyComponent.__init__.__name__ == "__init__"

    def test_experimental_warning_is_user_warning_subclass(self):
        assert issubclass(ExperimentalWarning, UserWarning)

    def test_warning_emitted_on_every_instantiation(self):
        @_experimental
        @component
        class MyComponent:
            @component.output_types(value=int)
            def run(self, value: int) -> dict:
                return {"value": value}

        with pytest.warns(ExperimentalWarning):
            MyComponent()

        with pytest.warns(ExperimentalWarning):
            MyComponent()
