# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from jinja2 import Environment
import arrow
from haystack.utils import Jinja2TimeExtension


class TestJinja2TimeExtension:
    @pytest.fixture
    def jinja_env(self) -> Environment:
        return Environment(extensions=[Jinja2TimeExtension])

    @pytest.fixture
    def jinja_extension(self, jinja_env: Environment) -> Jinja2TimeExtension:
        return Jinja2TimeExtension(jinja_env)

    @patch("haystack.utils.jinja2_extensions.arrow_import")
    def test_init_fails_without_arrow(self, arrow_import_mock) -> None:
        arrow_import_mock.check.side_effect = ImportError
        with pytest.raises(ImportError):
            Jinja2TimeExtension(Environment())

    def test_valid_datetime(self, jinja_extension: Jinja2TimeExtension) -> None:
        result = jinja_extension._get_datetime(
            "UTC", operator="+", offset="hours=2", datetime_format="%Y-%m-%d %H:%M:%S"
        )
        assert isinstance(result, str)
        assert len(result) == 19

    def test_parse_valid_expression(self, jinja_env: Environment) -> None:
        template = "{% now 'UTC' + 'hours=2', '%Y-%m-%d %H:%M:%S' %}"
        result = jinja_env.from_string(template).render()
        assert isinstance(result, str)
        assert len(result) == 19

    def test_get_datetime_no_offset(self, jinja_extension: Jinja2TimeExtension) -> None:
        result = jinja_extension._get_datetime("UTC")
        expected = arrow.now("UTC").strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_get_datetime_with_offset_add(self, jinja_extension: Jinja2TimeExtension) -> None:
        result = jinja_extension._get_datetime("UTC", operator="+", offset="hours=1")
        expected = arrow.now("UTC").shift(hours=1).strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_get_datetime_with_offset_subtract(self, jinja_extension: Jinja2TimeExtension) -> None:
        result = jinja_extension._get_datetime("UTC", operator="-", offset="days=1")
        expected = arrow.now("UTC").shift(days=-1).strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_get_datetime_with_offset_subtract_days_hours(self, jinja_extension: Jinja2TimeExtension) -> None:
        result = jinja_extension._get_datetime("UTC", operator="-", offset="days=1, hours=2")
        expected = arrow.now("UTC").shift(days=-1, hours=-2).strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_get_datetime_with_custom_format(self, jinja_extension: Jinja2TimeExtension) -> None:
        result = jinja_extension._get_datetime("UTC", datetime_format="%d-%m-%Y")
        expected = arrow.now("UTC").strftime("%d-%m-%Y")
        assert result == expected

    def test_get_datetime_new_york_timezone(self, jinja_env: Environment) -> None:
        template = jinja_env.from_string("{% now 'America/New_York' %}")
        result = template.render()
        expected = arrow.now("America/New_York").strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_parse_no_operator(self, jinja_env: Environment) -> None:
        template = jinja_env.from_string("{% now 'UTC' %}")
        result = template.render()
        expected = arrow.now("UTC").strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_parse_with_add(self, jinja_env: Environment) -> None:
        template = jinja_env.from_string("{% now 'UTC' + 'hours=2' %}")
        result = template.render()
        expected = arrow.now("UTC").shift(hours=2).strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_parse_with_subtract(self, jinja_env: Environment) -> None:
        template = jinja_env.from_string("{% now 'UTC' - 'days=1' %}")
        result = template.render()
        expected = arrow.now("UTC").shift(days=-1).strftime("%Y-%m-%d %H:%M:%S")
        assert result == expected

    def test_parse_with_custom_format(self, jinja_env: Environment) -> None:
        template = jinja_env.from_string("{% now 'UTC', '%d-%m-%Y' %}")
        result = template.render()
        expected = arrow.now("UTC").strftime("%d-%m-%Y")
        assert result == expected

    def test_default_format(self, jinja_env: Environment) -> None:
        template = jinja_env.from_string("{% now 'UTC'%}")
        result = template.render()
        expected = arrow.now("UTC").strftime("%Y-%m-%d %H:%M:%S")  # default format
        assert result == expected

    def test_invalid_timezone(self, jinja_extension: Jinja2TimeExtension) -> None:
        with pytest.raises(ValueError, match="Invalid timezone"):
            jinja_extension._get_datetime("Invalid/Timezone")

    def test_invalid_offset(self, jinja_extension: Jinja2TimeExtension) -> None:
        with pytest.raises(ValueError, match="Invalid offset or operator"):
            jinja_extension._get_datetime("UTC", operator="+", offset="invalid_format")

    def test_invalid_operator(self, jinja_extension: Jinja2TimeExtension) -> None:
        with pytest.raises(ValueError, match="Invalid offset or operator"):
            jinja_extension._get_datetime("UTC", operator="*", offset="hours=2")
