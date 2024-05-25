# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.core.pipeline.utils import parse_connect_string


def test_parse_connection():
    assert parse_connect_string("foobar") == ("foobar", None)
    assert parse_connect_string("foo.bar") == ("foo", "bar")
